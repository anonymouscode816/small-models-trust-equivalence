"""
Run calibration analysis for five Transformer models on a Paraphrase
Identification (PI) task such as QQP or PAWS.

This script is anonymized and path-neutral. It does not contain user-specific
file paths, usernames, private links, machine-specific directories, or keys.
All data/checkpoint/output locations are supplied through command-line
arguments.

Default model keys:
  - bert_base
  - distilbert
  - bert_medium
  - bert_mini
  - bert_tiny

Expected checkpoint layout by default:
  <checkpoint_root>/<model_key>/

The checkpoint directories should be Hugging Face compatible, for example those
created by AutoModelForSequenceClassification.save_pretrained().
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_REGISTRY: Dict[str, str] = {
    "bert_base": "bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "bert_medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert_mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert_tiny": "google/bert_uncased_L-2_H-128_A-2",
}

ID2LABEL: Dict[int, str] = {
    0: "not_paraphrase",
    1: "paraphrase",
}
LABEL2ID: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1,
    "not_paraphrase": 0,
    "non_paraphrase": 0,
    "not_duplicate": 0,
    "duplicate": 1,
    "paraphrase": 1,
}


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_model_key(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").replace(".", "_")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_models(model_args: Sequence[str]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for item in model_args:
        if item == "all":
            selected.update(MODEL_REGISTRY)
        elif item in MODEL_REGISTRY:
            selected[item] = MODEL_REGISTRY[item]
        else:
            # Allow a custom Hugging Face model path/name without hard-coding it.
            selected[safe_model_key(item)] = item
    return selected


def infer_separator(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    return ","


def find_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def infer_split_file(data_dir: Path, dataset_type: str) -> Path:
    names: List[str] = []
    prefixes = [""] if dataset_type == "custom" else ["", f"{dataset_type}_"]
    for prefix in prefixes:
        names.extend(
            [
                f"{prefix}test.csv",
                f"{prefix}test.tsv",
                f"{prefix}test.txt",
                f"{prefix}dev.csv",
                f"{prefix}dev.tsv",
                f"{prefix}validation.csv",
                f"{prefix}validation.tsv",
            ]
        )

    # Compatibility with common QQP-style examples.
    if dataset_type == "qqp":
        names.extend(["questions.csv", "qqp_questions.csv"])

    path = find_first_existing(data_dir / name for name in names)
    if path is None:
        searched = ", ".join(names)
        raise FileNotFoundError(
            f"Could not infer a test/evaluation file inside {data_dir}. "
            f"Searched: {searched}. Pass --test_file explicitly."
        )
    return path


def default_columns(dataset_type: str) -> Tuple[List[str], List[str], List[str]]:
    if dataset_type == "qqp":
        return (
            ["question1", "sentence1", "sent1", "text1"],
            ["question2", "sentence2", "sent2", "text2"],
            ["is_duplicate", "label", "target"],
        )
    if dataset_type == "paws":
        return (
            ["sentence1", "question1", "sent1", "text1"],
            ["sentence2", "question2", "sent2", "text2"],
            ["label", "is_duplicate", "target"],
        )
    return (
        ["sentence1", "question1", "sent1", "text1"],
        ["sentence2", "question2", "sent2", "text2"],
        ["label", "is_duplicate", "target"],
    )


def choose_column(
    dataframe: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit:
        if explicit not in dataframe.columns:
            raise ValueError(
                f"Column '{explicit}' for {role} was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return explicit

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not infer the {role} column. Tried {list(candidates)}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def normalize_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Encountered a missing label value.")
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if ivalue in (0, 1):
            return ivalue
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    if text in LABEL2ID:
        return LABEL2ID[text]
    raise ValueError(f"Unsupported PI label value: {value!r}. Expected binary labels 0/1.")


class PairClassificationDataset(Dataset):
    """Sentence-pair dataset for binary paraphrase identification."""

    def __init__(
        self,
        sentence1: Sequence[str],
        sentence2: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.sentence1 = [str(x) for x in sentence1]
        self.sentence2 = [str(x) for x in sentence2]
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not (len(self.sentence1) == len(self.sentence2) == len(self.labels)):
            raise ValueError("sentence1, sentence2, and labels must have the same length.")
        if not self.labels:
            raise ValueError("The evaluation dataset is empty.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.sentence1[idx],
            self.sentence2[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


def read_raw_examples(
    file_path: Path,
    dataset_type: str,
    sentence1_column: Optional[str],
    sentence2_column: Optional[str],
    label_column: Optional[str],
    max_samples: Optional[int],
) -> Tuple[List[str], List[str], List[int]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    sep = infer_separator(file_path)
    dataframe = pd.read_csv(file_path, sep=sep)
    sentence1_candidates, sentence2_candidates, label_candidates = default_columns(dataset_type)
    col1 = choose_column(dataframe, sentence1_column, sentence1_candidates, "first sentence/question")
    col2 = choose_column(dataframe, sentence2_column, sentence2_candidates, "second sentence/question")
    col_label = choose_column(dataframe, label_column, label_candidates, "label")

    if max_samples is not None:
        dataframe = dataframe.head(max_samples)

    sentence1 = dataframe[col1].astype(str).tolist()
    sentence2 = dataframe[col2].astype(str).tolist()
    labels = [normalize_label(value) for value in dataframe[col_label].tolist()]
    return sentence1, sentence2, labels


def read_pickle_object(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def coerce_pickle_to_dataset(
    obj,
    tokenizer,
    max_len: int,
    max_samples: Optional[int],
) -> Dataset:
    """
    Support common anonymized pickle layouts while avoiding assumptions about
    private folder structures.

    Accepted pickle formats:
      1. A torch Dataset object.
      2. A pandas DataFrame with sentence-pair columns.
      3. A dict containing sentence1/sentence2/label-like keys.
      4. A list of dicts or tuples: (sentence1, sentence2, label).
    """
    if isinstance(obj, Dataset):
        return obj

    if isinstance(obj, pd.DataFrame):
        dataframe = obj
        if max_samples is not None:
            dataframe = dataframe.head(max_samples)
        sentence1_candidates, sentence2_candidates, label_candidates = default_columns("custom")
        col1 = choose_column(dataframe, None, sentence1_candidates, "first sentence/question")
        col2 = choose_column(dataframe, None, sentence2_candidates, "second sentence/question")
        col_label = choose_column(dataframe, None, label_candidates, "label")
        return PairClassificationDataset(
            dataframe[col1].astype(str).tolist(),
            dataframe[col2].astype(str).tolist(),
            [normalize_label(value) for value in dataframe[col_label].tolist()],
            tokenizer=tokenizer,
            max_len=max_len,
        )

    if isinstance(obj, dict):
        sentence1_candidates, sentence2_candidates, label_candidates = default_columns("custom")
        col1 = next((key for key in sentence1_candidates if key in obj), None)
        col2 = next((key for key in sentence2_candidates if key in obj), None)
        col_label = next((key for key in label_candidates if key in obj), None)
        if col1 is None or col2 is None or col_label is None:
            raise ValueError(
                "Pickle dict must contain sentence-pair and label keys, e.g. "
                "sentence1, sentence2, label."
            )
        sentence1 = list(obj[col1])
        sentence2 = list(obj[col2])
        labels = [normalize_label(value) for value in list(obj[col_label])]
        if max_samples is not None:
            sentence1 = sentence1[:max_samples]
            sentence2 = sentence2[:max_samples]
            labels = labels[:max_samples]
        return PairClassificationDataset(sentence1, sentence2, labels, tokenizer=tokenizer, max_len=max_len)

    if isinstance(obj, list):
        records = obj[:max_samples] if max_samples is not None else obj
        sentence1: List[str] = []
        sentence2: List[str] = []
        labels: List[int] = []
        for record in records:
            if isinstance(record, dict):
                sentence1_candidates, sentence2_candidates, label_candidates = default_columns("custom")
                key1 = next((key for key in sentence1_candidates if key in record), None)
                key2 = next((key for key in sentence2_candidates if key in record), None)
                key_label = next((key for key in label_candidates if key in record), None)
                if key1 is None or key2 is None or key_label is None:
                    raise ValueError("List-of-dicts pickle contains an unsupported record format.")
                sentence1.append(str(record[key1]))
                sentence2.append(str(record[key2]))
                labels.append(normalize_label(record[key_label]))
            elif isinstance(record, (tuple, list)) and len(record) >= 3:
                sentence1.append(str(record[0]))
                sentence2.append(str(record[1]))
                labels.append(normalize_label(record[2]))
            else:
                raise ValueError("List pickle must contain dict records or tuple records.")
        return PairClassificationDataset(sentence1, sentence2, labels, tokenizer=tokenizer, max_len=max_len)

    raise ValueError(
        "Unsupported pickle format. Use raw CSV/TSV input or a pickle containing "
        "a Dataset, DataFrame, dict, or list of sentence-pair records."
    )


def make_dataset(args: argparse.Namespace, tokenizer, max_len: int) -> Dataset:
    if args.input_format == "raw":
        test_file = Path(args.test_file) if args.test_file else infer_split_file(Path(args.data_dir), args.dataset_type)
        sentence1, sentence2, labels = read_raw_examples(
            test_file,
            dataset_type=args.dataset_type,
            sentence1_column=args.sentence1_column,
            sentence2_column=args.sentence2_column,
            label_column=args.label_column,
            max_samples=args.max_samples,
        )
        return PairClassificationDataset(sentence1, sentence2, labels, tokenizer=tokenizer, max_len=max_len)

    if args.input_format == "pickle":
        test_file = Path(args.test_file) if args.test_file else Path(args.data_dir) / "test.pkl"
        if not test_file.exists():
            raise FileNotFoundError(f"Pickle evaluation file not found: {test_file}")
        obj = read_pickle_object(test_file)
        return coerce_pickle_to_dataset(obj, tokenizer=tokenizer, max_len=max_len, max_samples=args.max_samples)

    raise ValueError("input_format must be either raw or pickle")


def resolve_checkpoint_dir(checkpoint_root: Path, model_key: str, override_path: Optional[str] = None) -> Path:
    if override_path:
        path = Path(override_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided checkpoint path does not exist for {model_key}: {path}")
        return path

    candidates = [
        checkpoint_root / model_key,
        checkpoint_root / model_key / "checkpoint_best",
        checkpoint_root / model_key / "best_model",
        checkpoint_root / model_key / "checkpoint-best",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() or (candidate / "pytorch_model.bin").exists() or (candidate / "model.safetensors").exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a checkpoint for model '{model_key}'. Checked:\n{checked}")


def get_checkpoint_override(args: argparse.Namespace, model_key: str) -> Optional[str]:
    return getattr(args, f"checkpoint_path_{model_key}", None)


def load_model_and_tokenizer(
    model_key: str,
    model_name: str,
    checkpoint_dir: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=2)
    model.to(device)
    model.eval()
    return model, tokenizer


def normalize_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    model_type: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if "input_ids" in batch:
        input_ids = batch["input_ids"]
    elif "ids" in batch:
        input_ids = batch["ids"]
    else:
        raise KeyError("Batch does not contain input_ids or ids.")

    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
    elif "mask" in batch:
        attention_mask = batch["mask"]
    else:
        raise KeyError("Batch does not contain attention_mask or mask.")

    if "labels" in batch:
        labels = batch["labels"]
    elif "target" in batch:
        labels = batch["target"]
    else:
        raise KeyError("Batch does not contain labels or target.")

    labels = labels.view(-1).long().to(device)

    model_inputs: Dict[str, torch.Tensor] = {
        "input_ids": input_ids.long().to(device),
        "attention_mask": attention_mask.long().to(device),
    }

    # DistilBERT does not accept token_type_ids.
    if model_type != "distilbert" and "token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch["token_type_ids"].long().to(device)

    return model_inputs, labels


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    model_type = model.config.model_type

    for batch in tqdm(dataloader, desc="calibrating", leave=False):
        model_inputs, labels = normalize_batch(batch, device, model_type=model_type)
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probabilities = F.softmax(logits, dim=-1)

        all_logits.append(logits.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def binary_brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_probabilities = probabilities[:, 1]
    return float(np.mean((positive_probabilities - labels.astype(np.float64)) ** 2))


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
            bins.append(CalibrationBin(i, lower, upper, 0, None, None, None, None))
            continue

        accuracy = float(np.mean(correctness[selected]))
        confidence = float(np.mean(confidences[selected]))
        gap = abs(accuracy - confidence)
        ci95 = float(1.96 * np.sqrt((accuracy * (1.0 - accuracy)) / count))
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


def evaluate_calibration(
    model_key: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
    binning: str,
) -> Tuple[Dict[str, float], List[CalibrationBin]]:
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == labels))
    brier = binary_brier_score(labels, probabilities)

    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)
    clipped_probabilities = clipped_probabilities / clipped_probabilities.sum(axis=1, keepdims=True)
    nll = float(log_loss(labels, clipped_probabilities, labels=[0, 1]))

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


def save_json(data: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_bins_csv(bins: Iterable[CalibrationBin], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin_id", "lower", "upper", "count", "accuracy", "confidence", "gap", "ci95"],
        )
        writer.writeheader()
        for item in bins:
            writer.writerow(
                {
                    "bin_id": item.bin_id,
                    "lower": item.lower,
                    "upper": item.upper,
                    "count": item.count,
                    "accuracy": item.accuracy,
                    "confidence": item.confidence,
                    "gap": item.gap,
                    "ci95": item.ci95,
                }
            )


def save_probabilities_csv(labels: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
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
                "prob_not_paraphrase",
                "prob_paraphrase",
            ],
        )
        writer.writeheader()
        for i in range(len(labels)):
            gold_id = int(labels[i])
            pred_id = int(predictions[i])
            writer.writerow(
                {
                    "index": i,
                    "gold_id": gold_id,
                    "gold_label": ID2LABEL[gold_id],
                    "pred_id": pred_id,
                    "pred_label": ID2LABEL[pred_id],
                    "confidence": float(confidences[i]),
                    "prob_not_paraphrase": float(probabilities[i, 0]),
                    "prob_paraphrase": float(probabilities[i, 1]),
                }
            )


def plot_reliability_diagram(
    bins: Sequence[CalibrationBin],
    model_key: str,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    ensure_dir(path.parent)
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
        f"PI calibration: {model_key}\n"
        f"Acc={metrics['accuracy']:.4f}, ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_comparison(all_bins: Dict[str, List[CalibrationBin]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8.0, 6.0))
    for model_key, bins in all_bins.items():
        valid_bins = [item for item in bins if item.count > 0 and item.accuracy is not None and item.confidence is not None]
        if not valid_bins:
            continue
        plt.plot(
            [float(item.confidence) for item in valid_bins],
            [float(item.accuracy) for item in valid_bins],
            marker="o",
            label=model_key,
        )
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Average confidence")
    plt.ylabel("Accuracy")
    plt.title("PI calibration comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_summary_csv(summary: Sequence[Dict[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "model_key",
        "checkpoint_dir",
        "num_examples",
        "accuracy",
        "ece",
        "mce",
        "brier_score",
        "log_loss",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_for_model(
    model_key: str,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[CalibrationBin]]:
    start_time = time.time()
    checkpoint_override = get_checkpoint_override(args, model_key)
    checkpoint_dir = resolve_checkpoint_dir(Path(args.checkpoint_root), model_key, checkpoint_override)

    print("=" * 80)
    print(f"Model: {model_key}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(model_key, model_name, checkpoint_dir, device=device)
    dataset = make_dataset(args, tokenizer=tokenizer, max_len=args.max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logits, probabilities, labels = collect_predictions(model, dataloader, device=device)
    del logits

    metrics, bins = evaluate_calibration(
        model_key=model_key,
        probabilities=probabilities,
        labels=labels,
        num_bins=args.num_bins,
        binning=args.binning,
    )
    metrics["checkpoint_dir"] = str(checkpoint_dir)
    metrics["runtime_seconds"] = float(time.time() - start_time)

    model_output_dir = Path(args.output_dir) / model_key
    ensure_dir(model_output_dir)

    save_json(metrics, model_output_dir / "calibration_metrics.json")
    save_bins_csv(bins, model_output_dir / "calibration_bins.csv")
    if args.save_probabilities:
        save_probabilities_csv(labels, probabilities, model_output_dir / "prediction_probabilities.csv")
    plot_reliability_diagram(bins, model_key, metrics, model_output_dir / "reliability_diagram.png")

    print(
        f"{model_key}: accuracy={metrics['accuracy']:.6f}, "
        f"ECE={metrics['ece']:.6f}, MCE={metrics['mce']:.6f}, "
        f"Brier={metrics['brier_score']:.6f}, log_loss={metrics['log_loss']:.6f}"
    )
    return metrics, bins


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibration analysis for all five BERT-family PI models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys to evaluate: all, bert_base, distilbert, bert_medium, bert_mini, bert_tiny, or custom HF path.",
    )
    parser.add_argument("--list_models", action="store_true", help="Print available model keys and exit.")

    parser.add_argument(
        "--input_format",
        choices=["raw", "pickle"],
        default="raw",
        help="Evaluation data format. Use raw for CSV/TSV files and pickle for test.pkl-style files.",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["qqp", "paws", "custom"],
        default="qqp",
        help="Dataset column convention used when --input_format raw.",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the evaluation dataset.")
    parser.add_argument("--test_file", type=str, default=None, help="Optional explicit evaluation file path.")
    parser.add_argument("--sentence1_column", type=str, default=None, help="Optional first text column name for raw input.")
    parser.add_argument("--sentence2_column", type=str, default=None, help="Optional second text column name for raw input.")
    parser.add_argument("--label_column", type=str, default=None, help="Optional label column name for raw input.")

    parser.add_argument("--checkpoint_root", type=str, default="./outputs/pi_models", help="Root directory containing model checkpoints.")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_calibration", help="Directory to save calibration outputs.")

    for key in MODEL_REGISTRY:
        parser.add_argument(
            f"--checkpoint_path_{key}",
            type=str,
            default=None,
            help=f"Optional explicit checkpoint directory for {key}.",
        )

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of evaluation examples.")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--binning", choices=["fixed", "quantile"], default="quantile")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_probabilities", action="store_true", help="Save per-example probabilities as CSV.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        print("Available model keys:")
        for key, value in MODEL_REGISTRY.items():
            print(f"  {key}: {value}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dir(Path(args.output_dir))
    selected_models = resolve_models(args.models)

    print(f"Using device: {device}")
    print(f"Selected models: {', '.join(selected_models.keys())}")

    summary: List[Dict[str, object]] = []
    all_bins: Dict[str, List[CalibrationBin]] = {}
    for model_key, model_name in selected_models.items():
        metrics, bins = run_for_model(model_key, model_name, args, device=device)
        summary.append(metrics)
        all_bins[model_key] = bins

    output_dir = Path(args.output_dir)
    save_json({"summary": summary}, output_dir / "summary.json")
    save_summary_csv(summary, output_dir / "summary.csv")
    plot_comparison(all_bins, output_dir / "calibration_comparison.png")

    print("\nCalibration completed.")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Summary CSV:  {output_dir / 'summary.csv'}")
    print(f"Comparison plot: {output_dir / 'calibration_comparison.png'}")


if __name__ == "__main__":
    main()
