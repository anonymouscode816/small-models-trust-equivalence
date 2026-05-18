"""
Combined SNLI fine-tuning and testing script for five Transformer models.

This script is intentionally path-neutral: pass dataset and output locations through
command-line arguments instead of hard-coding user-specific paths.

Expected SNLI files by default:
  <data_dir>/snli_1.0_train.txt
  <data_dir>/snli_1.0_dev.txt
  <data_dir>/snli_1.0_test.txt

Default models:
  - bert-base-uncased
  - distilbert/distilbert-base-uncased
  - google/bert_uncased_L-8_H-512_A-8
  - google/bert_uncased_L-4_H-256_A-4
  - google/bert_uncased_L-2_H-128_A-2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_VERSION = "snli_5models_v2_bert_base_included"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").replace(".", "_")


def infer_snli_paths(data_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        data_dir / "snli_1.0_train.txt",
        data_dir / "snli_1.0_dev.txt",
        data_dir / "snli_1.0_test.txt",
    )


def read_snli_file(file_path: Path, max_samples: Optional[int] = None) -> List[Tuple[str, str, int]]:
    """
    Read an SNLI-style tab-separated file.

    The original scripts used:
        label = parts[0]
        sentence1 = parts[-9]
        sentence2 = parts[-8]

    That indexing is kept for compatibility with SNLI 1.0 files. A fallback is
    included for simple three-column files: label, sentence1, sentence2.
    """
    examples: List[Tuple[str, str, int]] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            label = parts[0].strip()
            if label not in LABEL2ID:
                continue

            if len(parts) >= 9:
                sentence1 = parts[-9].strip()
                sentence2 = parts[-8].strip()
            else:
                sentence1 = parts[1].strip()
                sentence2 = parts[2].strip()

            if not sentence1 or not sentence2:
                continue

            examples.append((sentence1, sentence2, LABEL2ID[label]))
            if max_samples is not None and len(examples) >= max_samples:
                break

    if not examples:
        raise ValueError(f"No valid SNLI examples were read from: {file_path}")

    return examples


class SNLIPairDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, str, int]],
        tokenizer,
        max_len: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sentence1, sentence2, label = self.examples[idx]
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
        item["index"] = torch.tensor(idx, dtype=torch.long)
        return item


class NLIClassifier(nn.Module):
    """
    Architecture-compatible NLI classifier.

    The old BERT-base script used a custom two-layer classification head:
        encoder CLS vector -> Linear(hidden_size, 2*num_labels) -> Linear(2*num_labels, num_labels)

    This class keeps the same head pattern but obtains hidden_size dynamically,
    so it works for BERT-base, DistilBERT, BERT-Medium, BERT-Mini, and BERT-Tiny.
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
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # DistilBERT does not accept token_type_ids. BERT-family models do.
        if token_type_ids is not None and self.encoder.config.model_type != "distilbert":
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        hidden_output = self.hidden(cls_output)
        logits = self.classifier(hidden_output)
        probabilities = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key != "index"}


def make_dataloader(
    file_path: Path,
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> DataLoader:
    examples = read_snli_file(file_path, max_samples=max_samples)
    dataset = SNLIPairDataset(examples, tokenizer, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc=f"epoch {epoch} train", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        predictions = torch.argmax(outputs["logits"], dim=-1)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

        progress.set_postfix(
            loss=total_loss / max(total_examples, 1),
            acc=total_correct / max(total_examples, 1),
        )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_outputs: bool = False,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_predictions: List[str] = []
    all_probabilities: List[List[float]] = []
    all_labels: List[str] = []

    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")

        outputs = model(**batch, labels=labels)
        logits = outputs["logits"]
        probabilities = outputs["probabilities"]
        predictions = torch.argmax(logits, dim=-1)

        batch_size = labels.size(0)
        total_loss += float(outputs["loss"].item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

        if save_outputs:
            all_predictions.extend(ID2LABEL[int(x)] for x in predictions.cpu().tolist())
            all_labels.extend(ID2LABEL[int(x)] for x in labels.cpu().tolist())
            all_probabilities.extend(probabilities.detach().cpu().tolist())

    result: Dict[str, object] = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }

    if save_outputs:
        result["predictions"] = all_predictions
        result["labels"] = all_labels
        result["probabilities"] = all_probabilities

    return result


def save_checkpoint(
    model: NLIClassifier,
    tokenizer,
    checkpoint_dir: Path,
    metadata: Dict[str, object],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> Tuple[NLIClassifier, object, Dict[str, object]]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    pretrained_model_name = str(metadata["pretrained_model_name"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = NLIClassifier(pretrained_model_name=pretrained_model_name, num_labels=len(LABEL2ID))
    state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, tokenizer, metadata


def write_lines(lines: Iterable[str], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def write_probability_file(probabilities: List[List[float]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for row in probabilities:
            fh.write(" ".join(f"{value:.8f}" for value in row) + "\n")


def write_json(data: Dict[str, object], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def resolve_models(model_args: List[str]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for item in model_args:
        if item == "all":
            selected.update(MODEL_REGISTRY)
        elif item in MODEL_REGISTRY:
            selected[item] = MODEL_REGISTRY[item]
        else:
            # Allows passing an additional model identifier without editing the file.
            selected[safe_name(item)] = item
    return selected


def run_for_model(
    model_key: str,
    pretrained_model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    model_output_dir = args.output_dir / model_key
    checkpoint_dir = model_output_dir / "checkpoint_best"
    predictions_dir = model_output_dir / "predictions"
    logs_dir = model_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Model key: {model_key}")
    print(f"Model identifier: {pretrained_model_name}")
    print("=" * 80)

    history: Dict[str, object] = {
        "model_key": model_key,
        "pretrained_model_name": pretrained_model_name,
        "epochs": [],
    }

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)
    model = NLIClassifier(pretrained_model_name=pretrained_model_name, num_labels=len(LABEL2ID))
    model.to(device)

    if args.mode in {"train", "train_test"}:
        train_loader = make_dataloader(
            args.train_file,
            tokenizer,
            args.max_len,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.max_train_samples,
        )
        dev_loader = make_dataloader(
            args.dev_file,
            tokenizer,
            args.max_len,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.max_eval_samples,
        )

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_dev_accuracy = -1.0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_clip=args.grad_clip,
            )
            dev_metrics = evaluate(model, dev_loader, device, save_outputs=False)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "dev_loss": dev_metrics["loss"],
                "dev_accuracy": dev_metrics["accuracy"],
            }
            history["epochs"].append(epoch_record)

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.6f}, "
                f"train_acc={train_metrics['accuracy']:.6f}, "
                f"dev_loss={dev_metrics['loss']:.6f}, "
                f"dev_acc={dev_metrics['accuracy']:.6f}"
            )

            if float(dev_metrics["accuracy"]) > best_dev_accuracy:
                best_dev_accuracy = float(dev_metrics["accuracy"])
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir=checkpoint_dir,
                    metadata={
                        "model_key": model_key,
                        "pretrained_model_name": pretrained_model_name,
                        "label2id": LABEL2ID,
                        "id2label": ID2LABEL,
                        "best_epoch": epoch,
                        "best_dev_accuracy": best_dev_accuracy,
                        "max_len": args.max_len,
                    },
                )

        history["training_time_seconds"] = time.time() - start_time
        history["best_dev_accuracy"] = best_dev_accuracy
        write_json(history, logs_dir / "training_history.json")

        print(f"Best dev accuracy for {model_key}: {best_dev_accuracy:.6f}")
        print(f"Best checkpoint saved to: {checkpoint_dir}")

    if args.mode in {"test", "train_test"}:
        if checkpoint_dir.exists():
            model, tokenizer, _ = load_checkpoint(checkpoint_dir, device=device)
        elif args.mode == "test":
            raise FileNotFoundError(
                f"Checkpoint not found for {model_key}: {checkpoint_dir}. "
                "Run training first or provide the correct output directory."
            )

        test_loader = make_dataloader(
            args.test_file,
            tokenizer,
            args.max_len,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.max_eval_samples,
        )
        test_result = evaluate(model, test_loader, device, save_outputs=True)

        predictions = test_result.pop("predictions")
        labels = test_result.pop("labels")
        probabilities = test_result.pop("probabilities")

        write_lines(predictions, predictions_dir / f"pred_{model_key}_snli.txt")
        write_lines(labels, predictions_dir / f"gold_{model_key}_snli.txt")
        write_probability_file(probabilities, predictions_dir / f"prob_{model_key}_snli.txt")
        write_json(test_result, logs_dir / "test_metrics.json")

        history["test_metrics"] = test_result
        print(f"SNLI test accuracy for {model_key}: {test_result['accuracy']:.6f}")
        print(f"Predictions saved to: {predictions_dir}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and test BERT-base and compact Transformer models on SNLI."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./snli_1.0"),
        help="Directory containing SNLI train/dev/test text files.",
    )
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--dev_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/snli_nli_models"),
        help="Directory where model checkpoints, logs, and predictions are saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Use one or more keys: all, bert_base, distilbert, bert_medium, bert_mini, bert_tiny. Additional model identifiers are also accepted.",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print the built-in model keys and Hugging Face identifiers, then exit.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
        help="Whether to train, test an existing checkpoint, or train then test.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device selection. Use auto to prefer CUDA when available.",
    )

    args = parser.parse_args()

    default_train, default_dev, default_test = infer_snli_paths(args.data_dir)
    args.train_file = args.train_file or default_train
    args.dev_file = args.dev_file or default_dev
    args.test_file = args.test_file or default_test

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Using device: {device}")
    print(f"Train file: {args.train_file}")
    print(f"Dev file: {args.dev_file}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")

    if args.mode in {"train", "train_test"} and not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if args.mode in {"train", "train_test"} and not args.dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {args.dev_file}")
    if args.mode in {"test", "train_test"} and not args.test_file.exists():
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    if args.list_models:
        print("Built-in model keys:")
        for key, identifier in MODEL_REGISTRY.items():
            print(f"  {key}: {identifier}")
        return

    selected_models = resolve_models(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "mode": args.mode,
        "models": {},
    }

    for model_key, pretrained_model_name in selected_models.items():
        result = run_for_model(model_key, pretrained_model_name, args, device)
        summary["models"][model_key] = result

    write_json(summary, args.output_dir / "summary.json")
    print(f"Summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
