"""
Run calibration analysis for five Transformer models on the SNLI/NLI task.

This script is anonymized and path-neutral. It does not contain user-specific
file paths, usernames, private links, or machine-specific directories.

Default model keys:
  - bert_base
  - distilbert
  - bert_medium
  - bert_mini
  - bert_tiny

Expected checkpoint layout by default:
  <checkpoint_root>/<model_key>/checkpoint_best/

The checkpoint directory should contain:
  - pytorch_model.bin
  - tokenizer files
  - metadata.json, if produced by the companion fine-tuning script

If metadata.json is missing, the script falls back to the model registry.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_REGISTRY: Dict[str, str] = {
    "bert_base": "bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "bert_medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert_mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert_tiny": "google/bert_uncased_L-2_H-128_A-2",
}

LABEL2ID: Dict[str, int] = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


@dataclass
class CalibrationBin:
    bin_id: int
    lower: float
    upper: float
    count: int
    accuracy: Optional[float]
    confidence: Optional[float]
    gap: Optional[float]
    ci95: Optional[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_model_key(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").replace(".", "_")


def read_snli_file(file_path: Path, max_samples: Optional[int] = None) -> List[Tuple[str, str, int]]:
    """
    Read SNLI-style files.

    Compatible with original SNLI 1.0 tab-separated files and simple three-column
    files with columns: label, sentence1, sentence2.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    examples: List[Tuple[str, str, int]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            label = parts[0].strip()
            if label not in LABEL2ID:
                continue

            # SNLI 1.0 files normally have many columns. In the user's original
            # scripts, the premise and hypothesis were read from negative indices.
            if len(parts) >= 14:
                sentence1 = parts[-9].strip()
                sentence2 = parts[-8].strip()
            elif len(parts) == 13:
                sentence1 = parts[-8].strip()
                sentence2 = parts[-7].strip()
            elif len(parts) == 12:
                sentence1 = parts[-7].strip()
                sentence2 = parts[-6].strip()
            elif len(parts) == 10:
                sentence1 = parts[-5].strip()
                sentence2 = parts[-4].strip()
            else:
                sentence1 = parts[1].strip()
                sentence2 = parts[2].strip()

            if not sentence1 or not sentence2:
                continue

            examples.append((sentence1, sentence2, LABEL2ID[label]))
            if max_samples is not None and len(examples) >= max_samples:
                break

    if not examples:
        raise ValueError(f"No valid NLI examples were found in: {file_path}")

    return examples


class NLIPairDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Tuple[str, str, int]],
        tokenizer,
        max_len: int,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sentence1, sentence2, label = self.examples[index]
        encoded = self.tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["index"] = torch.tensor(index, dtype=torch.long)
        return item


class NLIClassifier(nn.Module):
    """
    Generic NLI classifier for BERT-base, DistilBERT, BERT-Medium, BERT-Mini,
    and BERT-Tiny.

    The classification head mirrors the earlier BERT-base code pattern but uses
    the encoder hidden size dynamically so that all five models are supported.
    """

    def __init__(self, pretrained_model_name: str, num_labels: int = 3) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden = nn.Linear(hidden_size, 2 * num_labels)
        self.classifier = nn.Linear(2 * num_labels, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.encoder.config.model_type != "distilbert":
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        hidden_output = self.hidden(cls_output)
        logits = self.classifier(hidden_output)
        return logits


def load_metadata(checkpoint_dir: Path) -> Dict[str, object]:
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def resolve_checkpoint_dir(checkpoint_root: Path, model_key: str, override_path: Optional[str] = None) -> Path:
    if override_path:
        path = Path(override_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided checkpoint path does not exist for {model_key}: {path}")
        return path

    candidates = [
        checkpoint_root / model_key / "checkpoint_best",
        checkpoint_root / model_key,
    ]
    for candidate in candidates:
        if (candidate / "pytorch_model.bin").exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a checkpoint for model '{model_key}'. Checked:\n{checked}"
    )


def load_model_and_tokenizer(
    model_key: str,
    checkpoint_dir: Path,
    device: torch.device,
) -> Tuple[NLIClassifier, object, Dict[str, object]]:
    metadata = load_metadata(checkpoint_dir)
    pretrained_model_name = str(metadata.get("pretrained_model_name", MODEL_REGISTRY.get(model_key, model_key)))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = NLIClassifier(pretrained_model_name=pretrained_model_name, num_labels=len(LABEL2ID))

    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as strict_error:
        # Helpful fallback for checkpoints created with module prefixes.
        cleaned_state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError:
            raise RuntimeError(
                f"Unable to load checkpoint for {model_key} from {checkpoint_dir}. "
                "The checkpoint architecture may not match this calibration script."
            ) from strict_error

    model.to(device)
    model.eval()
    return model, tokenizer, metadata


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key != "index"}


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="calibrating", leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch)
        probabilities = F.softmax(logits, dim=-1)

        all_logits.append(logits.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def multiclass_brier_score(labels: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float64)[labels.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int,
    binning: str,
) -> Tuple[List[CalibrationBin], float, float]:
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.unique(np.quantile(confidences, quantiles))
        if len(bin_edges) <= 1:
            bin_edges = np.array([0.0, 1.0])
    elif binning == "fixed":
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        raise ValueError("binning must be either 'fixed' or 'quantile'")

    actual_bins = len(bin_edges) - 1
    bin_indices = np.digitize(confidences, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, actual_bins) - 1

    bins: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    total = len(labels)

    for i in range(actual_bins):
        selected = bin_indices == i
        count = int(np.sum(selected))
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])

        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_id=i,
                    lower=lower,
                    upper=upper,
                    count=0,
                    accuracy=None,
                    confidence=None,
                    gap=None,
                    ci95=None,
                )
            )
            continue

        accuracy = float(np.mean(correctness[selected]))
        confidence = float(np.mean(confidences[selected]))
        gap = abs(accuracy - confidence)
        ci95 = float(1.96 * np.sqrt((accuracy * (1.0 - accuracy)) / count)) if count > 0 else 0.0

        ece += (count / total) * gap
        mce = max(mce, gap)

        bins.append(
            CalibrationBin(
                bin_id=i,
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                gap=float(gap),
                ci95=ci95,
            )
        )

    return bins, float(ece), float(mce)


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_bins_csv(bins: Iterable[CalibrationBin], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_id",
                "lower",
                "upper",
                "count",
                "accuracy",
                "confidence",
                "gap",
                "ci95",
            ],
        )
        writer.writeheader()
        for item in bins:
            writer.writerow({
                "bin_id": item.bin_id,
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "accuracy": item.accuracy,
                "confidence": item.confidence,
                "gap": item.gap,
                "ci95": item.ci95,
            })


def save_probabilities_csv(
    labels: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "gold_id",
                "gold_label",
                "pred_id",
                "pred_label",
                "confidence",
                "prob_contradiction",
                "prob_neutral",
                "prob_entailment",
            ],
        )
        writer.writeheader()
        for i in range(len(labels)):
            gold_id = int(labels[i])
            pred_id = int(predictions[i])
            writer.writerow({
                "index": i,
                "gold_id": gold_id,
                "gold_label": ID2LABEL[gold_id],
                "pred_id": pred_id,
                "pred_label": ID2LABEL[pred_id],
                "confidence": float(confidences[i]),
                "prob_contradiction": float(probabilities[i, 0]),
                "prob_neutral": float(probabilities[i, 1]),
                "prob_entailment": float(probabilities[i, 2]),
            })


def plot_reliability_diagram(
    bins: Sequence[CalibrationBin],
    model_key: str,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_bins = [item for item in bins if item.count > 0 and item.accuracy is not None and item.confidence is not None]

    plt.figure(figsize=(7.5, 6.0))
    if valid_bins:
        x_values = [float(item.confidence) for item in valid_bins]
        y_values = [float(item.accuracy) for item in valid_bins]
        y_errors = [float(item.ci95 or 0.0) for item in valid_bins]
        plt.errorbar(x_values, y_values, yerr=y_errors, fmt="o", capsize=4, label="Observed bins")

    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Average confidence")
    plt.ylabel("Accuracy")
    plt.title(
        f"Calibration: {model_key}\n"
        f"Acc={metrics['accuracy']:.4f}, ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate_calibration(
    model_key: str,
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    binning: str,
) -> Tuple[Dict[str, float], List[CalibrationBin]]:
    del logits
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == labels))
    brier = multiclass_brier_score(labels, probabilities, num_classes=len(LABEL2ID))

    # Clip to avoid log-loss numerical issues when probabilities are exactly 0.
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)
    clipped_probabilities = clipped_probabilities / clipped_probabilities.sum(axis=1, keepdims=True)
    nll = float(log_loss(labels, clipped_probabilities, labels=[0, 1, 2]))

    bins, ece, mce = compute_calibration_bins(labels, probabilities, num_bins=num_bins, binning=binning)
    metrics = {
        "model_key": model_key,
        "num_examples": int(len(labels)),
        "accuracy": accuracy,
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "log_loss": nll,
    }
    return metrics, bins


def resolve_models(model_args: List[str]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for item in model_args:
        if item == "all":
            selected.update(MODEL_REGISTRY)
        elif item in MODEL_REGISTRY:
            selected[item] = MODEL_REGISTRY[item]
        else:
            selected[safe_model_key(item)] = item
    return selected


def get_checkpoint_override(args: argparse.Namespace, model_key: str) -> Optional[str]:
    return getattr(args, f"checkpoint_path_{model_key}", None)


def run_for_model(
    model_key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    start_time = time.time()
    checkpoint_override = get_checkpoint_override(args, model_key)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_root, model_key, checkpoint_override)

    print("=" * 80)
    print(f"Model: {model_key}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 80)

    model, tokenizer, metadata = load_model_and_tokenizer(model_key, checkpoint_dir, device=device)
    max_len = int(args.max_len or metadata.get("max_len", 512))

    examples = read_snli_file(args.test_file, max_samples=args.max_samples)
    dataset = NLIPairDataset(examples, tokenizer=tokenizer, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logits, probabilities, labels = collect_predictions(model, dataloader, device=device)
    metrics, bins = evaluate_calibration(
        model_key=model_key,
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        num_bins=args.num_bins,
        binning=args.binning,
    )
    metrics["inference_time_seconds"] = float(time.time() - start_time)
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["test_file"] = str(args.test_file)
    metrics["binning"] = args.binning
    metrics["num_bins_requested"] = int(args.num_bins)
    metrics["num_bins_used"] = int(len(bins))

    model_output_dir = args.output_dir / model_key
    save_json(metrics, model_output_dir / "calibration_metrics.json")
    save_bins_csv(bins, model_output_dir / "calibration_bins.csv")

    if args.save_probabilities:
        save_probabilities_csv(labels, probabilities, model_output_dir / "prediction_probabilities.csv")

    plot_reliability_diagram(
        bins=bins,
        model_key=model_key,
        metrics=metrics,
        path=model_output_dir / "reliability_diagram.png",
    )

    print(
        f"{model_key}: accuracy={metrics['accuracy']:.6f}, "
        f"ECE={metrics['ece']:.6f}, MCE={metrics['mce']:.6f}, "
        f"Brier={metrics['brier_score']:.6f}, LogLoss={metrics['log_loss']:.6f}"
    )
    return metrics


def write_summary_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "num_examples",
        "accuracy",
        "ece",
        "mce",
        "brier_score",
        "log_loss",
        "inference_time_seconds",
        "binning",
        "num_bins_used",
        "checkpoint_dir",
        "test_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NLI calibration analysis for five BERT-family models."
    )

    parser.add_argument(
        "--test_file",
        type=Path,
        default=Path("./snli_1.0/snli_1.0_test.txt"),
        help="Path to the SNLI/NLI test file.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("./outputs/snli_nli_models"),
        help="Root directory containing model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/nli_calibration"),
        help="Directory where calibration results are saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys to evaluate. Use 'all' for all five models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Maximum sequence length. If omitted, uses checkpoint metadata when available; otherwise 512.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of calibration bins.",
    )
    parser.add_argument(
        "--binning",
        choices=["fixed", "quantile"],
        default="quantile",
        help="Calibration binning strategy.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on test examples for quick debugging.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Save per-example probabilities as CSV.",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List supported model keys and exit.",
    )

    # Optional explicit checkpoint overrides. These keep the public script clean
    # while still letting users point to arbitrary local checkpoint folders.
    for key in MODEL_REGISTRY:
        parser.add_argument(
            f"--checkpoint_path_{key}",
            type=str,
            default=None,
            help=f"Optional explicit checkpoint directory for {key}.",
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available model keys:")
        for key, model_name in MODEL_REGISTRY.items():
            print(f"  {key}: {model_name}")
        return

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    selected_models = resolve_models(args.models)
    if not selected_models:
        raise ValueError("No models selected. Use --models all or one or more model keys.")

    summary_rows: List[Dict[str, object]] = []
    failures: Dict[str, str] = {}

    for model_key in selected_models:
        try:
            metrics = run_for_model(model_key, args, device=device)
            summary_rows.append(metrics)
        except Exception as exc:
            failures[model_key] = str(exc)
            print(f"Failed for {model_key}: {exc}")
            if len(selected_models) == 1:
                raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_rows:
        save_json({"results": summary_rows, "failures": failures}, args.output_dir / "summary.json")
        write_summary_csv(summary_rows, args.output_dir / "summary.csv")
        print(f"Summary saved to: {args.output_dir / 'summary.json'}")
        print(f"Summary CSV saved to: {args.output_dir / 'summary.csv'}")

    if failures:
        print("Some models failed. See summary.json for details.")


if __name__ == "__main__":
    main()
