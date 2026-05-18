"""
Combined fine-tuning and testing script for Paraphrase Identification (PI).

This script is path-neutral and suitable for anonymous review. Provide all data and
output locations through command-line arguments instead of hard-coding local paths.

Supported models:
  - bert-base-uncased
  - distilbert/distilbert-base-uncased
  - google/bert_uncased_L-8_H-512_A-8
  - google/bert_uncased_L-4_H-256_A-4
  - google/bert_uncased_L-2_H-128_A-2

Supported input styles:
  1. Raw QQP/PAWS/custom CSV or TSV files.
  2. Existing pickle splits compatible with the original PI scripts:
       <data_dir>/train.pkl, <data_dir>/val.pkl, <data_dir>/test.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_VERSION = "pi_5models_v1"

MODEL_REGISTRY: Dict[str, str] = {
    "bert_base": "bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "bert_medium": "google/bert_uncased_L-8_H-512_A-8",
    "bert_mini": "google/bert_uncased_L-4_H-256_A-4",
    "bert_tiny": "google/bert_uncased_L-2_H-128_A-2",
}

DEFAULT_MODEL_KEYS: List[str] = list(MODEL_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Reproducibility and utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)



def select_model_keys(requested_models: Optional[Sequence[str]]) -> List[str]:
    if not requested_models:
        return DEFAULT_MODEL_KEYS

    selected: List[str] = []
    for key in requested_models:
        if key not in MODEL_REGISTRY:
            valid = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model key '{key}'. Valid choices: {valid}")
        selected.append(key)
    return selected


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


class PairClassificationDataset(Dataset):
    """Raw sentence-pair dataset for binary PI classification."""

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



def read_pickle_dataset(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)



def infer_separator(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    return ","



def find_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None



def infer_split_file(data_dir: Path, split: str, dataset_type: str) -> Path:
    """Infer a split file for common QQP/PAWS/custom layouts."""
    split_aliases = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation", "dev"],
        "test": ["test", "testing"],
    }

    names: List[str] = []
    prefixes = ["", f"{dataset_type}_"] if dataset_type != "custom" else [""]
    for prefix in prefixes:
        for alias in split_aliases[split]:
            names.extend([
                f"{prefix}{alias}.csv",
                f"{prefix}{alias}.tsv",
                f"{prefix}{alias}.txt",
            ])

    # Compatibility with the older QQP script comment.
    if split == "train" and dataset_type == "qqp":
        names.append("questions.csv")

    path = find_first_existing(data_dir / name for name in names)
    if path is None:
        searched = ", ".join(names)
        raise FileNotFoundError(
            f"Could not infer the {split} file inside {data_dir}. "
            f"Searched: {searched}. You can pass --{split}_file explicitly."
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



def choose_column(dataframe: pd.DataFrame, explicit: Optional[str], candidates: List[str], role: str) -> str:
    if explicit:
        if explicit not in dataframe.columns:
            raise ValueError(f"Column '{explicit}' for {role} was not found. Available columns: {list(dataframe.columns)}")
        return explicit

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not infer the {role} column. Tried {candidates}. "
        f"Available columns: {list(dataframe.columns)}. Pass the column name explicitly."
    )



def read_pair_file(
    path: Path,
    dataset_type: str,
    sentence1_col: Optional[str],
    sentence2_col: Optional[str],
    label_col: Optional[str],
    max_samples: Optional[int],
) -> Tuple[List[str], List[str], List[int]]:
    separator = infer_separator(path)
    dataframe = pd.read_csv(path, sep=separator)

    sent1_candidates, sent2_candidates, label_candidates = default_columns(dataset_type)
    sent1_column = choose_column(dataframe, sentence1_col, sent1_candidates, "first sentence")
    sent2_column = choose_column(dataframe, sentence2_col, sent2_candidates, "second sentence")
    label_column = choose_column(dataframe, label_col, label_candidates, "label")

    dataframe = dataframe[[sent1_column, sent2_column, label_column]].dropna()
    dataframe[label_column] = dataframe[label_column].astype(int)
    dataframe = dataframe[dataframe[label_column].isin([0, 1])]

    if max_samples is not None:
        dataframe = dataframe.head(max_samples)

    if dataframe.empty:
        raise ValueError(f"No valid binary PI examples found in {path}.")

    return (
        dataframe[sent1_column].astype(str).tolist(),
        dataframe[sent2_column].astype(str).tolist(),
        dataframe[label_column].astype(int).tolist(),
    )



def resolve_split_path(args: argparse.Namespace, split: str) -> Path:
    explicit = getattr(args, f"{split}_file")
    if explicit:
        return Path(explicit)
    return infer_split_file(Path(args.data_dir), split, args.dataset_type)



def make_raw_dataloader(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    shuffle: bool,
    max_samples: Optional[int],
) -> DataLoader:
    split_path = resolve_split_path(args, split)
    sentence1, sentence2, labels = read_pair_file(
        path=split_path,
        dataset_type=args.dataset_type,
        sentence1_col=args.sentence1_col,
        sentence2_col=args.sentence2_col,
        label_col=args.label_col,
        max_samples=max_samples,
    )
    dataset = PairClassificationDataset(sentence1, sentence2, labels, tokenizer, args.max_len)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )



def make_pickle_dataloader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    filename = {"train": "train.pkl", "val": "val.pkl", "test": "test.pkl"}[split]
    path = Path(args.data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected pickle file not found: {path}")

    dataset = read_pickle_dataset(path)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )



def make_dataloader(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    shuffle: bool,
    max_samples: Optional[int] = None,
) -> DataLoader:
    if args.input_format == "pickle":
        return make_pickle_dataloader(args, split, shuffle)
    return make_raw_dataloader(args, tokenizer, split, shuffle, max_samples)


# -----------------------------------------------------------------------------
# Batch normalization for both raw and old pickle datasets
# -----------------------------------------------------------------------------



def normalize_batch(batch: Dict[str, torch.Tensor], device: torch.device, model_type: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Convert both new raw batches and old pickle batches to HF model inputs.

    Old pickle batches from the original PI code use:
      ids, mask, token_type_ids, target

    New raw batches use:
      input_ids, attention_mask, token_type_ids, labels
    """
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

    # DistilBERT does not use token type ids.
    if model_type != "distilbert" and "token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch["token_type_ids"].long().to(device)

    return model_inputs, labels


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------



def train_one_epoch(
    model: torch.nn.Module,
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
    model_type = model.config.model_type

    progress = tqdm(dataloader, desc=f"epoch {epoch} train", leave=False)
    for batch in progress:
        model_inputs, labels = normalize_batch(batch, device, model_type)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        predictions = torch.argmax(outputs.logits, dim=-1)
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



def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str,
    collect_outputs: bool = False,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_predictions: List[int] = []
    all_probabilities: List[List[float]] = []
    all_labels: List[int] = []
    model_type = model.config.model_type

    with torch.no_grad():
        progress = tqdm(dataloader, desc=desc, leave=False)
        for batch in progress:
            model_inputs, labels = normalize_batch(batch, device, model_type)
            outputs = model(**model_inputs, labels=labels)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(outputs.loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_examples += batch_size

            if collect_outputs:
                all_predictions.extend(predictions.detach().cpu().tolist())
                all_probabilities.extend(probabilities.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

            progress.set_postfix(
                loss=total_loss / max(total_examples, 1),
                acc=total_correct / max(total_examples, 1),
            )

    result: Dict[str, object] = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }
    if collect_outputs:
        result["predictions"] = all_predictions
        result["probabilities"] = all_probabilities
        result["labels"] = all_labels
    return result



def save_json(data: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)



def save_predictions(result: Dict[str, object], prediction_dir: Path, model_key: str) -> None:
    ensure_dir(prediction_dir)

    predictions = result.get("predictions", [])
    probabilities = result.get("probabilities", [])
    labels = result.get("labels", [])

    pred_path = prediction_dir / f"test_pred_{model_key}.txt"
    prob_path = prediction_dir / f"test_prob_{model_key}.txt"
    label_path = prediction_dir / f"test_labels_{model_key}.txt"

    with pred_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")

    with prob_path.open("w", encoding="utf-8") as file:
        for row in probabilities:
            file.write(" ".join(f"{float(value):.8f}" for value in row) + "\n")

    with label_path.open("w", encoding="utf-8") as file:
        for label in labels:
            file.write(f"{label}\n")



def train_model(args: argparse.Namespace, model_key: str, model_name: str, device: torch.device) -> Dict[str, object]:
    start_time = time.time()
    print(f"\n===== Training {model_key}: {model_name} =====")

    model_output_dir = Path(args.output_dir) / model_key
    ensure_dir(model_output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_loader = make_dataloader(args, tokenizer, split="train", shuffle=True, max_samples=args.max_train_samples)
    val_loader = make_dataloader(args, tokenizer, split="val", shuffle=False, max_samples=args.max_eval_samples)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_accuracy = -1.0
    best_epoch = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args.grad_clip)
        val_metrics = evaluate(model, val_loader, device, desc=f"epoch {epoch} val", collect_outputs=False)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"train_acc={train_metrics['accuracy']:.6f}, "
            f"val_loss={val_metrics['loss']:.6f}, "
            f"val_acc={val_metrics['accuracy']:.6f}"
        )

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            print(f"Saved best checkpoint for {model_key} at epoch {epoch}.")

    elapsed = time.time() - start_time
    summary = {
        "script_version": SCRIPT_VERSION,
        "model_key": model_key,
        "pretrained_model": model_name,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_time_seconds": elapsed,
        "history": history,
    }
    save_json(summary, model_output_dir / "training_summary.json")
    print(f"Training completed for {model_key}. Best val accuracy: {best_val_accuracy:.6f}")
    return summary



def test_model(args: argparse.Namespace, model_key: str, model_name: str, device: torch.device) -> Dict[str, object]:
    start_time = time.time()
    print(f"\n===== Testing {model_key} =====")

    model_output_dir = Path(args.output_dir) / model_key
    if not model_output_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found for {model_key}: {model_output_dir}. "
            "Run with --mode train or --mode train_test first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_output_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_output_dir)
    model.to(device)

    test_loader = make_dataloader(args, tokenizer, split="test", shuffle=False, max_samples=args.max_eval_samples)
    result = evaluate(model, test_loader, device, desc=f"{model_key} test", collect_outputs=True)

    elapsed = time.time() - start_time
    metrics = {
        "script_version": SCRIPT_VERSION,
        "model_key": model_key,
        "pretrained_model": model_name,
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
        "num_examples": result["num_examples"],
        "test_time_seconds": elapsed,
    }

    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else Path(args.output_dir) / "predictions"
    save_predictions(result, prediction_dir, model_key)
    save_json(metrics, model_output_dir / "test_metrics.json")

    print(f"Test accuracy for {model_key}: {float(result['accuracy']):.6f}")
    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune and test five Transformer models for binary Paraphrase Identification."
    )

    parser.add_argument("--list_models", action="store_true", help="Print supported model keys and exit.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to run. Default: all models.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
        help="Whether to train, test, or train and test.",
    )

    parser.add_argument("--data_dir", type=str, default="./dataset", help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_models", help="Output directory.")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Directory for prediction/probability files. Default: <output_dir>/predictions.",
    )

    parser.add_argument(
        "--input_format",
        choices=["raw", "pickle"],
        default="raw",
        help="Use raw CSV/TSV files or existing train.pkl/val.pkl/test.pkl splits.",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["qqp", "paws", "custom"],
        default="qqp",
        help="Used for automatic column and file-name detection in raw mode.",
    )
    parser.add_argument("--train_file", type=str, default=None, help="Explicit raw training file path.")
    parser.add_argument("--val_file", type=str, default=None, help="Explicit raw validation file path.")
    parser.add_argument("--test_file", type=str, default=None, help="Explicit raw test file path.")
    parser.add_argument("--sentence1_col", type=str, default=None, help="Column name for first sentence/question.")
    parser.add_argument("--sentence2_col", type=str, default=None, help="Column name for second sentence/question.")
    parser.add_argument("--label_col", type=str, default=None, help="Column name for binary label.")

    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm. Use 0 to disable.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, or a device string.")

    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap for quick training checks.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for quick validation/test checks.")

    return parser



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for key, value in MODEL_REGISTRY.items():
            print(f"  {key}: {value}")
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    ensure_dir(Path(args.output_dir))
    if args.prediction_dir:
        ensure_dir(Path(args.prediction_dir))

    selected_models = select_model_keys(args.models)
    print(f"Using device: {device}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Input format: {args.input_format}")

    all_results: Dict[str, Dict[str, object]] = {}

    for model_key in selected_models:
        model_name = MODEL_REGISTRY[model_key]
        model_result: Dict[str, object] = {}

        if args.mode in {"train", "train_test"}:
            model_result["training"] = train_model(args, model_key, model_name, device)

        if args.mode in {"test", "train_test"}:
            model_result["testing"] = test_model(args, model_key, model_name, device)

        all_results[model_key] = model_result

        # Help release memory before moving to the next model.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(all_results, Path(args.output_dir) / "combined_results.json")
    print(f"\nAll requested runs completed. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
