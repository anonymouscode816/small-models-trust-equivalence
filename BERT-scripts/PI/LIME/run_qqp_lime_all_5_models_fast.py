#!/usr/bin/env python3
"""
Run LIME explanations for paraphrase identification models.

This script is suitable for anonymized submission:
- no hard-coded usernames, machine-specific paths, private links, or keys;
- all data, checkpoint, and output locations are controlled by CLI arguments;
- supports five BERT-family PI models through a single script.

Expected input CSV columns by default:
    question1, question2, is_duplicate

The script writes one LIME CSV per model per run. These CSV files can be used by
`create_pi_lime_jaccard_table.py` to compute pairwise Jaccard similarity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover - optional dependency
    safe_load_file = None


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "bert_base": {
        "display_name": "BERT-base",
        "checkpoint_subdir": "bert_base",
        "output_folder": "BERT-base",
        "file_stem": "BERT-base",
    },
    "distilbert": {
        "display_name": "Distil-BERT",
        "checkpoint_subdir": "distilbert",
        "output_folder": "Distil-BERT",
        "file_stem": "Distil-BERT",
    },
    "bert_medium": {
        "display_name": "BERT-Medium",
        "checkpoint_subdir": "bert_medium",
        "output_folder": "BERT-Medium",
        "file_stem": "BERT-Medium",
    },
    "bert_mini": {
        "display_name": "BERT-Mini",
        "checkpoint_subdir": "bert_mini",
        "output_folder": "BERT-Mini",
        "file_stem": "BERT-Mini",
    },
    "bert_tiny": {
        "display_name": "BERT-Tiny",
        "checkpoint_subdir": "bert_tiny",
        "output_folder": "BERT-Tiny",
        "file_stem": "BERT-Tiny",
    },
}

DEFAULT_MODEL_KEYS = list(MODEL_SPECS.keys())
CLASS_NAMES = ["not duplicate", "duplicate"]
NUM_LABELS = 2
PAIR_SEPARATOR = "  "


# -----------------------------------------------------------------------------
# Optional custom checkpoint compatibility
# -----------------------------------------------------------------------------

class LossFunction(nn.Module):
    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        return -torch.log(probability).mean()


class CustomMainModel(PreTrainedModel):
    """
    Compatibility model for older checkpoints saved with a custom class whose
    encoder module was stored under `self.bert` for BERT and DistilBERT.

    For checkpoints saved using Hugging Face AutoModelForSequenceClassification,
    keep the default --checkpoint_format auto or use --checkpoint_format
    sequence_classification.
    """

    def __init__(self, config, loss_fn: Optional[nn.Module] = None):
        super().__init__(config)
        self.num_labels = NUM_LABELS
        self.loss_fn = loss_fn or LossFunction()
        self.config = config
        config.output_hidden_states = True

        self.bert = AutoModel.from_config(config)

        if hasattr(config, "hidden_size"):
            classifier_input_size = int(config.hidden_size)
        elif hasattr(config, "dim"):
            classifier_input_size = int(config.dim)
        else:
            raise ValueError("Could not determine classifier input size from model config.")

        self.classifier = nn.Linear(classifier_input_size, self.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.config.model_type != "distilbert" and token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        output = self.bert(**model_inputs)
        cls_output = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        probs = F.softmax(logits, dim=1)

        if labels is not None:
            gold_probs = torch.gather(probs, 1, labels)
            loss = self.loss_fn(gold_probs)
            return loss, probs

        return probs


# -----------------------------------------------------------------------------
# CLI and utilities
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LIME explanations for PI/QQP models in an anonymized way."
    )

    parser.add_argument("--input_csv", type=str, default="./data/questions.csv",
                        help="Input CSV containing sentence/question pairs.")
    parser.add_argument("--question1_col", type=str, default="question1",
                        help="Column name for the first question/sentence.")
    parser.add_argument("--question2_col", type=str, default="question2",
                        help="Column name for the second question/sentence.")
    parser.add_argument("--label_col", type=str, default="is_duplicate",
                        help="Gold-label column. Expected binary labels: 0/1.")

    parser.add_argument("--checkpoint_root", type=str, default="./outputs/pi_models",
                        help="Root directory containing one fine-tuned model folder per model key.")
    parser.add_argument("--model_paths_json", type=str, default=None,
                        help=(
                            "Optional JSON file mapping model keys to checkpoint directories. "
                            "This overrides --checkpoint_root for listed keys."
                        ))
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_lime",
                        help="Directory where LIME CSV files will be saved.")

    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS,
                        help=f"Model keys to run. Choices: {', '.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--list_models", action="store_true",
                        help="Print available model keys and exit.")
    parser.add_argument(
        "--checkpoint_format",
        type=str,
        default="auto",
        choices=["auto", "sequence_classification", "custom"],
        help=(
            "Checkpoint format. Use 'sequence_classification' for models saved with "
            "AutoModelForSequenceClassification. Use 'custom' for older MainModel-style "
            "checkpoints. 'auto' tries to infer the format."
        ),
    )

    parser.add_argument("--num_samples", type=int, default=15000,
                        help="Number of input rows to explain after dropping missing pairs. Use -1 for all rows.")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of repeated LIME runs per model.")
    parser.add_argument("--start_run", type=int, default=1,
                        help="First run index. Usually 1.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed. Actual run seed is seed + run_id.")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum tokenizer sequence length.")
    parser.add_argument("--pred_batch_size", type=int, default=512,
                        help="Prediction batch size used inside LIME.")
    parser.add_argument("--lime_num_samples", type=int, default=100,
                        help="Number of perturbed samples used by LIME per explanation.")
    parser.add_argument("--lime_num_features", type=int, default=20,
                        help="Maximum number of words returned by LIME per explanation.")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint CSV after this many samples.")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase question text before LIME explanation.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a model/run if its final LIME CSV already exists.")

    return parser.parse_args()


def validate_model_keys(model_keys: Iterable[str]) -> List[str]:
    clean_keys: List[str] = []
    for key in model_keys:
        if key not in MODEL_SPECS:
            valid = ", ".join(DEFAULT_MODEL_KEYS)
            raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
        clean_keys.append(key)
    return clean_keys


def print_available_models() -> None:
    print("Available model keys:")
    for key, spec in MODEL_SPECS.items():
        print(f"  {key:<12} -> {spec['display_name']}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA version:", torch.version.cuda)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def load_model_path_map(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--model_paths_json must contain a JSON object mapping model keys to paths.")
    return {str(k): str(v) for k, v in data.items()}


def resolve_checkpoint_path(args: argparse.Namespace, model_key: str, model_path_map: Dict[str, str]) -> Path:
    if model_key in model_path_map:
        path = Path(model_path_map[model_key])
    else:
        path = Path(args.checkpoint_root) / MODEL_SPECS[model_key]["checkpoint_subdir"]

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found for '{model_key}': {path}\n"
            "Use --checkpoint_root or --model_paths_json to provide the correct relative path."
        )
    return path


def get_state_dict_keys(model_dir: Path) -> List[str]:
    """Read checkpoint keys without exposing or requiring private paths."""
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists() and safe_load_file is not None:
        return list(safe_load_file(str(safetensors_path), device="cpu").keys())

    if bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            return list(state.keys())

    return []


def infer_checkpoint_format(model_dir: Path, config) -> str:
    keys = get_state_dict_keys(model_dir)
    if config.model_type == "distilbert":
        if any(key.startswith("bert.") for key in keys):
            return "custom"
        return "sequence_classification"
    return "sequence_classification"


def load_tokenizer_and_model(model_dir: Path, checkpoint_format: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    config = AutoConfig.from_pretrained(str(model_dir))

    if checkpoint_format == "auto":
        checkpoint_format = infer_checkpoint_format(model_dir, config)

    if checkpoint_format == "custom":
        model = CustomMainModel.from_pretrained(
            str(model_dir),
            config=config,
            loss_fn=LossFunction(),
        )
    elif checkpoint_format == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), config=config)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")

    model.to(device)
    model.eval()
    return tokenizer, model, config, checkpoint_format


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = [args.question1_col, args.question2_col, args.label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[args.question1_col, args.question2_col, args.label_col]).reset_index(drop=True)
    if args.num_samples is not None and args.num_samples > 0:
        df = df.head(args.num_samples).reset_index(drop=True)

    if args.lowercase:
        df[args.question1_col] = df[args.question1_col].astype(str).str.lower()
        df[args.question2_col] = df[args.question2_col].astype(str).str.lower()

    print("Total samples after dropna/filtering:", len(df))
    if len(df) == 0:
        raise ValueError("No valid input examples found after filtering missing values.")
    return df


def build_lime_texts(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    return [
        str(q1).strip() + PAIR_SEPARATOR + str(q2).strip()
        for q1, q2 in zip(df[args.question1_col], df[args.question2_col])
    ]


def split_question_pair(text: str) -> Tuple[str, str]:
    text = str(text)
    if PAIR_SEPARATOR in text:
        q1, q2 = text.split(PAIR_SEPARATOR, 1)
    else:
        q1, q2 = text, ""
    return q1.strip(), q2.strip()


def model_predict_proba(model, config, encoded: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    model_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    if config.model_type != "distilbert" and "token_type_ids" in encoded:
        model_inputs["token_type_ids"] = encoded["token_type_ids"]

    with torch.inference_mode():
        output = model(**model_inputs)

    if isinstance(output, torch.Tensor):
        probs = output
    elif hasattr(output, "logits"):
        probs = F.softmax(output.logits, dim=1)
    elif isinstance(output, (tuple, list)):
        last_item = output[-1]
        if isinstance(last_item, torch.Tensor) and last_item.ndim == 2:
            probs = last_item
        else:
            raise ValueError("Could not interpret model output as probabilities or logits.")
    else:
        raise ValueError("Could not interpret model output as probabilities or logits.")

    return probs


def make_predict_function(tokenizer, model, config, device: str, args: argparse.Namespace):
    def predict_f(texts_old: List[str]) -> np.ndarray:
        all_probs: List[np.ndarray] = []

        for start in range(0, len(texts_old), args.pred_batch_size):
            batch_texts = texts_old[start:start + args.pred_batch_size]

            q1_list: List[str] = []
            q2_list: List[str] = []
            for text in batch_texts:
                q1, q2 = split_question_pair(text)
                q1_list.append(q1)
                q2_list.append(q2)

            encoded = tokenizer(
                q1_list,
                q2_list,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            probs = model_predict_proba(model, config, encoded, device)
            all_probs.append(probs.detach().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    return predict_f


def get_class_name(prob_array: np.ndarray) -> str:
    predicted_class_idx = int(np.argmax(prob_array))
    return CLASS_NAMES[predicted_class_idx]


def clean_lime_word(word) -> str:
    return str(word).strip()


def list_to_clean_string(words: List[str]) -> str:
    return str([str(w) for w in words])


def save_checkpoint(results: List[dict], path: Path) -> None:
    pd.DataFrame(results).to_csv(path, index=False)


def output_paths(args: argparse.Namespace, model_key: str, run_id: int) -> Tuple[Path, Path]:
    spec = MODEL_SPECS[model_key]
    model_output_dir = Path(args.output_dir) / spec["output_folder"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = model_output_dir / f"{spec['file_stem']}_QQP_lime_run_{run_id}.csv"
    checkpoint_csv = model_output_dir / f"{spec['file_stem']}_QQP_lime_run_{run_id}_checkpoint.csv"
    return final_csv, checkpoint_csv


def run_lime_for_model_run(
    args: argparse.Namespace,
    model_key: str,
    run_id: int,
    df: pd.DataFrame,
    texts: List[str],
    tokenizer,
    model,
    config,
    device: str,
) -> Path:
    spec = MODEL_SPECS[model_key]
    output_csv, checkpoint_csv = output_paths(args, model_key, run_id)

    if args.skip_existing and output_csv.exists():
        print(f"Skipping existing output: {output_csv}")
        return output_csv

    run_seed = args.seed + run_id
    set_seed(run_seed)

    predict_f = make_predict_function(tokenizer, model, config, device, args)
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=run_seed)

    print("\n" + "=" * 80)
    print(f"Model: {spec['display_name']} | LIME run {run_id}")
    print("=" * 80)
    print("MAX_LEN:", args.max_len)
    print("PRED_BATCH_SIZE:", args.pred_batch_size)
    print("LIME_NUM_SAMPLES:", args.lime_num_samples)
    print("LIME_NUM_FEATURES:", args.lime_num_features)
    print("Run seed:", run_seed)
    print("Output CSV:", output_csv)
    print("Checkpoint CSV:", checkpoint_csv)

    results: List[dict] = []
    begin = time.time()

    for idx, item in enumerate(tqdm(texts, total=len(texts))):
        predicted_prob = predict_f([item])[0]
        predicted_prob_list = [float(predicted_prob[0]), float(predicted_prob[1])]
        predicted_label = get_class_name(predicted_prob)

        explanation = explainer.explain_instance(
            item,
            predict_f,
            labels=[1],
            num_features=args.lime_num_features,
            num_samples=args.lime_num_samples,
        )

        # Interpret weights for the duplicate class. Positive weights support
        # duplicate; negative weights support non-duplicate. The order returned by
        # LIME is already importance-oriented, so it is preserved here.
        lime_values = explanation.as_list(label=1)
        duplicate_words: List[str] = []
        non_duplicate_words: List[str] = []

        for word, weight in lime_values:
            token = clean_lime_word(word)
            if not token:
                continue
            if float(weight) > 0:
                duplicate_words.append(token)
            elif float(weight) < 0:
                non_duplicate_words.append(token)

        row = {
            "model_key": model_key,
            "model_name": spec["display_name"],
            "run_id": run_id,
            "sample_index": idx,
            "sent1": str(df.loc[idx, args.question1_col]),
            "sent2": str(df.loc[idx, args.question2_col]),
            "true_out": int(df.loc[idx, args.label_col]),
            "d_words": list_to_clean_string(duplicate_words),
            "nd_words": list_to_clean_string(non_duplicate_words),
            "predicted_out": predicted_label,
            "predicted_prob": str(predicted_prob_list),
        }
        results.append(row)

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            save_checkpoint(results, checkpoint_csv)
            elapsed = time.time() - begin
            print(f"\n{spec['display_name']} run {run_id}: checkpoint at sample {idx + 1}. "
                  f"Elapsed: {elapsed:.2f} sec\n")
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    final_result = pd.DataFrame(results)
    final_result.to_csv(output_csv, index=False)

    print(f"Finished {spec['display_name']} LIME run {run_id}.")
    print(f"Runtime: {time.time() - begin:.2f} seconds")
    print("Saved:", output_csv)
    return output_csv


def run_all(args: argparse.Namespace) -> None:
    if args.list_models:
        print_available_models()
        return

    model_keys = validate_model_keys(args.models)
    model_path_map = load_model_path_map(args.model_paths_json)
    df = load_input_dataframe(args)
    texts = build_lime_texts(df, args)

    print("First sample:", texts[0])
    device = resolve_device()

    all_outputs: List[Path] = []
    total_begin = time.time()

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        checkpoint_path = resolve_checkpoint_path(args, model_key, model_path_map)
        print("\n" + "#" * 80)
        print(f"Loading {spec['display_name']} from: {checkpoint_path}")
        print("#" * 80)

        tokenizer, model, config, loaded_format = load_tokenizer_and_model(
            checkpoint_path, args.checkpoint_format, device
        )
        print(f"Loaded checkpoint format: {loaded_format}")

        for run_id in range(args.start_run, args.start_run + args.num_runs):
            output_file = run_lime_for_model_run(
                args=args,
                model_key=model_key,
                run_id=run_id,
                df=df,
                texts=texts,
                tokenizer=tokenizer,
                model=model,
                config=config,
                device=device,
            )
            all_outputs.append(output_file)
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("All requested LIME runs completed.")
    print("=" * 80)
    for output_file in all_outputs:
        print(output_file)
    print(f"\nTotal runtime: {time.time() - total_begin:.2f} seconds")


if __name__ == "__main__":
    run_all(parse_args())
