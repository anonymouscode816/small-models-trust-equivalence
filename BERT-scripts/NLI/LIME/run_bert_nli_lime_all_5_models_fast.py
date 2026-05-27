#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LIME explanations for five BERT-family NLI models.

Anonymized submission-ready script:
- no hard-coded usernames, private paths, URLs, or keys;
- all data/checkpoint/output paths are supplied by CLI arguments;
- supports BERT-base, DistilBERT, BERT-Medium, BERT-Mini, and BERT-Tiny.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    from lime.lime_text import LimeTextExplainer
except ImportError as exc:
    raise ImportError("Please install LIME first: pip install lime") from exc


CLASS_NAMES = ["contradiction", "neutral", "entailment"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "bert_base": {
        "display_name": "BERT-base",
        "pretrained_id": "bert-base-uncased",
        "checkpoint_subdir": "bert_base",
    },
    "distilbert": {
        "display_name": "DistilBERT",
        "pretrained_id": "distilbert/distilbert-base-uncased",
        "checkpoint_subdir": "distilbert",
    },
    "bert_medium": {
        "display_name": "BERT-Medium",
        "pretrained_id": "google/bert_uncased_L-8_H-512_A-8",
        "checkpoint_subdir": "bert_medium",
    },
    "bert_mini": {
        "display_name": "BERT-Mini",
        "pretrained_id": "google/bert_uncased_L-4_H-256_A-4",
        "checkpoint_subdir": "bert_mini",
    },
    "bert_tiny": {
        "display_name": "BERT-Tiny",
        "pretrained_id": "google/bert_uncased_L-2_H-128_A-2",
        "checkpoint_subdir": "bert_tiny",
    },
}

DEFAULT_MODEL_KEYS = list(MODEL_SPECS.keys())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_json_mapping(value: Optional[str]) -> Dict[str, str]:
    """Parse either a JSON string or a path to a JSON file containing model_key -> checkpoint_dir."""
    if not value:
        return {}
    candidate = Path(value)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint mapping must be a JSON object: {model_key: checkpoint_dir}")
    return {str(k): str(v) for k, v in data.items()}


def resolve_checkpoint_dir(model_key: str, checkpoint_root: Path, checkpoint_map: Dict[str, str]) -> Path:
    if model_key in checkpoint_map:
        return Path(checkpoint_map[model_key])
    return checkpoint_root / MODEL_SPECS[model_key]["checkpoint_subdir"]


def find_checkpoint_file(model_dir: Path) -> Optional[Path]:
    for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.bin", "model.pt", "checkpoint.pt"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        files = sorted(model_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_state_dict_file(checkpoint_file: Path) -> Dict[str, torch.Tensor]:
    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load .safetensors checkpoints: pip install safetensors") from exc
        state = load_file(str(checkpoint_file))
    else:
        state = torch.load(str(checkpoint_file), map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_file}")

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def normalize_custom_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Support checkpoints saved with either encoder.* or bert.* key prefixes."""
    has_encoder = any(key.startswith("encoder.") for key in state_dict)
    has_bert = any(key.startswith("bert.") for key in state_dict)
    if has_encoder and not has_bert:
        return state_dict
    if has_bert and not has_encoder:
        converted = {}
        for key, value in state_dict.items():
            if key.startswith("bert."):
                key = "encoder." + key[len("bert."):]
            converted[key] = value
        return converted
    return state_dict


class CustomNLIModel(nn.Module):
    """Custom NLI head used by the accompanying fine-tuning script."""

    def __init__(self, pretrained_id: str, num_labels: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_id)
        hidden_size = int(getattr(self.encoder.config, "hidden_size", 768))
        self.hidden = nn.Linear(hidden_size, 2 * num_labels)
        self.classifier = nn.Linear(2 * num_labels, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.encoder.config, "model_type", "") != "distilbert":
            encoder_inputs["token_type_ids"] = token_type_ids
        output = self.encoder(**encoder_inputs)
        cls_output = output.last_hidden_state[:, 0, :]
        hidden_output = self.hidden(cls_output)
        logits = self.classifier(hidden_output)
        return logits


def read_metadata(checkpoint_dir: Path) -> Dict[str, object]:
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def load_nli_model(model_key: str, checkpoint_dir: Path, device: torch.device):
    spec = MODEL_SPECS[model_key]
    metadata = read_metadata(checkpoint_dir)
    pretrained_id = str(metadata.get("pretrained_model_name", spec["pretrained_id"]))

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found for {model_key}: {checkpoint_dir}\n"
            "Use --checkpoint_root or --checkpoint_map to provide the correct anonymized checkpoint path."
        )

    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else pretrained_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    checkpoint_file = find_checkpoint_file(checkpoint_dir)
    if checkpoint_file is None:
        raise FileNotFoundError(f"No checkpoint file found inside: {checkpoint_dir}")

    # First try the custom architecture used by the provided fine-tuning script.
    model = CustomNLIModel(pretrained_id=pretrained_id, num_labels=len(CLASS_NAMES))
    state_dict = normalize_custom_state_dict(load_state_dict_file(checkpoint_file))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    total_keys = len(model.state_dict())
    loaded_ratio = (total_keys - len(missing)) / max(total_keys, 1)

    if loaded_ratio < 0.40:
        # Fallback for standard Hugging Face sequence-classification checkpoints.
        del model
        gc.collect()
        config = AutoConfig.from_pretrained(checkpoint_dir if (checkpoint_dir / "config.json").exists() else pretrained_id)
        config.num_labels = len(CLASS_NAMES)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model._is_standard_sequence_classifier = True
        config = model.config
        print(f"[{spec['display_name']}] Loaded as AutoModelForSequenceClassification.")
    else:
        model._is_standard_sequence_classifier = False
        config = model.encoder.config
        print(
            f"[{spec['display_name']}] Loaded custom checkpoint from {checkpoint_dir} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return tokenizer, model, config


def parse_snli_file(file_path: Path) -> pd.DataFrame:
    sent1: List[str] = []
    sent2: List[str] = []
    labels: List[str] = []

    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            label = parts[0].strip()
            if label not in LABEL_TO_ID:
                continue

            # SNLI 1.0 files usually allow this indexing. Three-column files are also supported.
            if len(parts) >= 9:
                s1 = parts[-9].strip()
                s2 = parts[-8].strip()
            else:
                s1 = parts[1].strip()
                s2 = parts[2].strip()

            if not s1 or not s2:
                continue
            sent1.append(s1)
            sent2.append(s2)
            labels.append(label)

    df = pd.DataFrame({"sent1": sent1, "sent2": sent2, "label": labels})
    if df.empty:
        raise ValueError(f"No valid NLI examples were read from: {file_path}")
    return df


def build_text_rows(df: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    if limit and limit > 0:
        df = df.head(limit).copy()
    texts = (df["sent1"].astype(str) + " [SEP] " + df["sent2"].astype(str)).tolist()
    return pd.DataFrame(
        {
            "sent": texts,
            "sent1": df["sent1"].astype(str).tolist(),
            "sent2": df["sent2"].astype(str).tolist(),
            "label": df["label"].astype(str).tolist(),
        }
    )


def split_pair_text(text: str) -> Tuple[str, str]:
    text = str(text)
    if " [SEP] " in text:
        left, right = text.split(" [SEP] ", 1)
        return left.strip(), right.strip()
    if "[SEP]" in text:
        left, right = text.split("[SEP]", 1)
        return left.strip(), right.strip()
    return text, ""


def make_predict_fn(tokenizer, model, config, device, batch_size: int, max_len: int, fp16: bool):
    use_amp = bool(fp16 and device.type == "cuda")

    def predict_proba(texts: Iterable[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts_list = texts.tolist()
        elif isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        all_probs: List[np.ndarray] = []
        for start in range(0, len(texts_list), batch_size):
            batch_texts = texts_list[start:start + batch_size]
            sent1_batch, sent2_batch = zip(*(split_pair_text(item) for item in batch_texts))

            encoded = tokenizer(
                list(sent1_batch),
                list(sent2_batch),
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}

            if getattr(config, "model_type", "") == "distilbert":
                encoded.pop("token_type_ids", None)

            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(**encoded).logits if getattr(model, "_is_standard_sequence_classifier", False) else model(**encoded)
                else:
                    logits = model(**encoded).logits if getattr(model, "_is_standard_sequence_classifier", False) else model(**encoded)

                probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    return predict_proba


def predicted_class(prob_row: np.ndarray) -> str:
    return CLASS_NAMES[int(np.argmax(prob_row))]


def sort_weight_list(items):
    return sorted(items, key=lambda item: abs(float(item[1])) if len(item) > 1 else 0.0, reverse=True)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate_models(model_keys: List[str]) -> List[str]:
    for key in model_keys:
        if key not in MODEL_SPECS:
            valid = ", ".join(DEFAULT_MODEL_KEYS)
            raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
    return model_keys



def clean_lime_list(values):
    return [(str(word), float(score)) for word, score in values]


def run_lime_for_model_run(
    dx: pd.DataFrame,
    predict_proba,
    model_key: str,
    run_id: int,
    output_dir: Path,
    num_features: int,
    num_samples: int,
    seed: int,
) -> Path:
    spec = MODEL_SPECS[model_key]
    out_path = output_dir / model_key / f"run_{run_id:02d}" / f"{model_key}_nli_lime_run_{run_id:02d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=seed)

    rows = []
    begin = time.time()

    for _, row in tqdm(dx.iterrows(), total=len(dx), desc=f"LIME {model_key} run {run_id:02d}"):
        text = row["sent"]
        probs = predict_proba([text])[0]
        explanation = explainer.explain_instance(
            text,
            predict_proba,
            labels=[0, 1, 2],
            num_features=num_features,
            num_samples=num_samples,
        )

        rows.append(
            {
                "sent1": row["sent1"],
                "sent2": row["sent2"],
                "true_out": row["label"],
                "contradiction": sort_weight_list(clean_lime_list(explanation.as_list(label=0))),
                "neutral": sort_weight_list(clean_lime_list(explanation.as_list(label=1))),
                "entailment": sort_weight_list(clean_lime_list(explanation.as_list(label=2))),
                "predicted_out": predicted_class(probs),
                "predicted_prob": [float(value) for value in probs],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)

    elapsed = time.time() - begin
    print(f"[DONE] {spec['display_name']} LIME run {run_id:02d}: {out_path} | seconds={elapsed:.2f}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIME explanations for NLI models.")
    parser.add_argument("--snli_test_file", type=str, default="./snli_1.0/snli_1.0_test.txt",
                        help="Path to the SNLI test file or a three-column label/sentence1/sentence2 file.")
    parser.add_argument("--checkpoint_root", type=str, default="./outputs/snli_nli_models",
                        help="Root folder containing one checkpoint subfolder per model.")
    parser.add_argument("--checkpoint_map", type=str, default=None,
                        help="Optional JSON string or JSON file mapping model keys to checkpoint directories.")
    parser.add_argument("--output_dir", type=str, default="./outputs/nli_lime",
                        help="Directory where LIME CSV files will be saved.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS,
                        help=f"Model keys to run. Choices: {', '.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--num_runs", "--runs", dest="num_runs", type=int, default=3,
                        help="Number of repeated LIME runs.")
    parser.add_argument("--limit_samples", type=int, default=0,
                        help="Use only the first N examples. 0 means use all examples.")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--predict_batch_size", type=int, default=256,
                        help="Prediction batch size used inside LIME.")
    parser.add_argument("--lime_num_features", type=int, default=20,
                        help="Number of LIME features to save per class.")
    parser.add_argument("--lime_num_samples", type=int, default=1000,
                        help="Number of LIME perturbation samples per example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true", help="Disable CUDA fp16 autocast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = validate_models(args.models)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_map = parse_json_mapping(args.checkpoint_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Arguments:")
    print(json.dumps(vars(args), indent=2))

    snli_test_file = Path(args.snli_test_file)
    if not snli_test_file.exists():
        raise FileNotFoundError(f"SNLI test file not found: {snli_test_file}")

    df = parse_snli_file(snli_test_file)
    dx = build_text_rows(df, limit=args.limit_samples)
    print(f"Loaded {len(dx)} NLI examples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    summary_records = []
    for model_key in model_keys:
        checkpoint_dir = resolve_checkpoint_dir(model_key, checkpoint_root, checkpoint_map)
        print(f"\nLoading {MODEL_SPECS[model_key]['display_name']} from: {checkpoint_dir}")
        tokenizer, model, config = load_nli_model(model_key, checkpoint_dir, device)
        predict_proba = make_predict_fn(
            tokenizer=tokenizer,
            model=model,
            config=config,
            device=device,
            batch_size=args.predict_batch_size,
            max_len=args.max_len,
            fp16=not args.no_fp16,
        )

        for run_id in range(1, args.num_runs + 1):
            out_path = run_lime_for_model_run(
                dx=dx,
                predict_proba=predict_proba,
                model_key=model_key,
                run_id=run_id,
                output_dir=output_dir,
                num_features=args.lime_num_features,
                num_samples=args.lime_num_samples,
                seed=args.seed + run_id - 1,
            )
            summary_records.append({"method": "lime", "model": model_key, "run": run_id, "output": str(out_path)})

        del model, tokenizer, predict_proba
        free_memory()

    summary_path = output_dir / "nli_lime_summary_outputs.csv"
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)
    print(f"\nAll LIME runs completed. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
