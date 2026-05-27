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
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file
from torch import cuda
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)


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
SHAP_TOKEN_PATTERN = r"\w+|[^\w\s]"


# -----------------------------------------------------------------------------
# Optional custom checkpoint class
# -----------------------------------------------------------------------------

class LossFunction(nn.Module):
    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        return -torch.log(probability).mean()


class CustomMainModel(PreTrainedModel):
    """
    Compatibility model for older checkpoints saved with a custom class whose
    encoder module was stored under `self.bert` for both BERT and DistilBERT.

    For newer checkpoints saved with Hugging Face
    AutoModelForSequenceClassification, use --checkpoint_format sequence_classification
    or the default --checkpoint_format auto.
    """

    def __init__(self, config, loss_fn: Optional[nn.Module] = None):
        super().__init__(config)
        self.num_labels = NUM_LABELS
        self.loss_fn = loss_fn or LossFunction()
        self.config = config
        config.output_hidden_states = True

        self.bert = AutoModel.from_config(config)

        if hasattr(config, "hidden_size"):
            classifier_input_size = config.hidden_size
        elif hasattr(config, "dim"):
            classifier_input_size = config.dim
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
# Utilities
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP explanations for PI/QQP models in an anonymized way."
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
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_shap",
                        help="Directory where SHAP CSV files will be saved.")

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
                        help="Number of repeated SHAP runs per model.")
    parser.add_argument("--start_run", type=int, default=1,
                        help="First run index. Usually 1.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed. Actual run seed is seed + run_id.")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum tokenizer sequence length.")
    parser.add_argument("--pred_batch_size", type=int, default=512,
                        help="Prediction batch size used inside SHAP.")
    parser.add_argument("--shap_min_evals", type=int, default=100,
                        help="Minimum SHAP max_evals value.")
    parser.add_argument("--shap_batch_size", type=int, default=512,
                        help="Batch size passed to SHAP explainer.")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint CSV after this many samples.")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase question text before SHAP explanation.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a model/run if its final SHAP CSV already exists.")

    return parser.parse_args()


def validate_model_keys(model_keys: Iterable[str]) -> List[str]:
    clean_keys = []
    for key in model_keys:
        if key not in MODEL_SPECS:
            valid = ", ".join(DEFAULT_MODEL_KEYS)
            raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
        clean_keys.append(key)
    return clean_keys


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

    if safetensors_path.exists():
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
        # Custom checkpoints from older code stored DistilBERT under `bert.*`,
        # whereas Hugging Face sequence-classification checkpoints use `distilbert.*`.
        if any(key.startswith("bert.") for key in keys):
            return "custom"
        return "sequence_classification"

    # For BERT-family checkpoints, Hugging Face sequence-classification loading
    # works for standard checkpoints and for many older custom checkpoints because
    # both use `bert.*` and `classifier.*` names.
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


def split_question_pair(text: str, separator: str = "  ") -> Tuple[str, str]:
    text = str(text)
    if separator in text:
        q1, q2 = text.split(separator, 1)
    else:
        q1, q2 = text, ""
    return q1.strip(), q2.strip()


def clean_token(token: str) -> str:
    token = str(token).strip()
    return token if token else "[EMPTY]"


def sort_list(items: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    return sorted(items, key=lambda item: item[1], reverse=True)


def list_to_clean_string(items: List[Tuple[str, float, float]]) -> str:
    clean_items = [(str(token), float(shap_value), float(base_value))
                   for token, shap_value, base_value in items]
    return str(clean_items)


def save_checkpoint(results: List[dict], path: Path) -> None:
    pd.DataFrame(results).to_csv(path, index=False)


def get_required_shap_max_evals(text: str, shap_min_evals: int) -> int:
    num_features = len(re.findall(SHAP_TOKEN_PATTERN, str(text)))
    required = 2 * num_features + 1
    return max(shap_min_evals, required)


def parse_required_max_evals_from_error(error_message: str) -> Optional[int]:
    message = str(error_message)

    match = re.search(r"=\s*(\d+)\s*!", message)
    if match:
        return int(match.group(1))

    match = re.search(r"at least\s+(\d+)", message)
    if match:
        return int(match.group(1))

    return None


def extract_shap_words(shap_values) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """
    Convert SHAP output into lists for duplicate and non-duplicate classes.
    Each element is (token, shap_value, base_value).
    """
    tokens = shap_values.data[0]
    values = np.array(shap_values.values[0])
    base_values = np.array(shap_values.base_values[0])

    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}. Expected [num_tokens, 2].")

    duplicate_words = []
    non_duplicate_words = []

    for token, token_values in zip(tokens, values):
        token = clean_token(token)

        nd_value = round(float(token_values[0]), 4)
        d_value = round(float(token_values[1]), 4)

        nd_base = round(float(base_values[0]), 4)
        d_base = round(float(base_values[1]), 4)

        non_duplicate_words.append((token, nd_value, nd_base))
        duplicate_words.append((token, d_value, d_base))

    return sort_list(duplicate_words), sort_list(non_duplicate_words)


def load_input_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[str]]:
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [args.question1_col, args.question2_col, args.label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    df = df.dropna(subset=[args.question1_col, args.question2_col, args.label_col]).reset_index(drop=True)

    if args.num_samples is not None and args.num_samples > 0:
        df = df.head(args.num_samples).reset_index(drop=True)

    texts = [
        str(q1).strip() + "  " + str(q2).strip()
        for q1, q2 in zip(df[args.question1_col], df[args.question2_col])
    ]

    if args.lowercase:
        texts = [text.lower() for text in texts]

    return df, texts


def make_predict_function(tokenizer, model, config, device: str, args: argparse.Namespace):
    def predict_f(texts_old, batch_size=args.pred_batch_size):
        all_probs = []
        texts_old = list(texts_old)

        for start in range(0, len(texts_old), batch_size):
            batch_texts = texts_old[start:start + batch_size]

            q1_list = []
            q2_list = []
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

            with torch.inference_mode():
                if isinstance(model, CustomMainModel):
                    probs = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                        labels=None,
                        device=device,
                    )
                else:
                    model_inputs = {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    }
                    if config.model_type != "distilbert" and "token_type_ids" in encoded:
                        model_inputs["token_type_ids"] = encoded["token_type_ids"]
                    output = model(**model_inputs)
                    probs = F.softmax(output.logits, dim=1)

            all_probs.append(probs.detach().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    return predict_f


def run_for_one_model(
    model_key: str,
    checkpoint_path: Path,
    df: pd.DataFrame,
    texts: List[str],
    args: argparse.Namespace,
    device: str,
) -> List[Path]:
    spec = MODEL_SPECS[model_key]
    display_name = spec["display_name"]
    output_folder = spec["output_folder"]
    file_stem = spec["file_stem"]

    print("\n" + "#" * 80)
    print(f"Starting model: {display_name}")
    print(f"Checkpoint directory: {checkpoint_path}")
    print("#" * 80)

    model_output_dir = Path(args.output_dir) / output_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, config, resolved_format = load_tokenizer_and_model(
        checkpoint_path,
        args.checkpoint_format,
        device,
    )

    print("Model type:", config.model_type)
    print("Checkpoint format:", resolved_format)
    if hasattr(config, "hidden_size"):
        print("Hidden size:", config.hidden_size)
    if hasattr(config, "dim"):
        print("Hidden dim:", config.dim)

    predict_f = make_predict_function(tokenizer, model, config, device, args)
    masker = shap.maskers.Text(tokenizer=SHAP_TOKEN_PATTERN)

    def explain_one_text_with_dynamic_max_evals(explainer, text):
        sample_max_evals = get_required_shap_max_evals(text, args.shap_min_evals)
        try:
            return explainer([text], max_evals=sample_max_evals, batch_size=args.shap_batch_size)
        except ValueError as error:
            message = str(error)
            if "max_evals" in message and "too low" in message:
                required = parse_required_max_evals_from_error(message)
                if required is None:
                    required = sample_max_evals * 2
                required = max(required, sample_max_evals + 1)
                print(f"Retrying SHAP with larger max_evals: {sample_max_evals} -> {required}")
                return explainer([text], max_evals=required, batch_size=args.shap_batch_size)
            raise

    output_files = []
    final_run = args.start_run + args.num_runs - 1

    for run_id in range(args.start_run, final_run + 1):
        print("\n" + "=" * 70)
        print(f"Model: {display_name} | SHAP run {run_id}")
        print("=" * 70)

        output_csv = model_output_dir / f"{file_stem}_QQP_shap_run_{run_id}.csv"
        checkpoint_csv = model_output_dir / f"{file_stem}_QQP_shap_run_{run_id}_checkpoint.csv"

        if args.skip_existing and output_csv.exists():
            print(f"Skipping existing output: {output_csv}")
            output_files.append(output_csv)
            continue

        run_seed = args.seed + run_id
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(run_seed)

        explainer = shap.Explainer(
            predict_f,
            masker,
            output_names=CLASS_NAMES,
            algorithm="permutation",
            seed=run_seed,
        )

        results = []
        begin = time.time()

        print("MAX_LEN:", args.max_len)
        print("PRED_BATCH_SIZE:", args.pred_batch_size)
        print("SHAP_MIN_EVALS:", args.shap_min_evals)
        print("SHAP_BATCH_SIZE:", args.shap_batch_size)
        print("Run seed:", run_seed)
        print("Output CSV:", output_csv)
        print("Checkpoint CSV:", checkpoint_csv)

        for idx, item in enumerate(tqdm(texts, total=len(texts))):
            predicted_prob = predict_f([item])[0]
            predicted_prob_list = [float(predicted_prob[0]), float(predicted_prob[1])]
            predicted_label = CLASS_NAMES[int(np.argmax(predicted_prob))]

            shap_values = explain_one_text_with_dynamic_max_evals(explainer, item)
            d_arr, nd_arr = extract_shap_words(shap_values)

            row = {
                "model_name": display_name,
                "run_id": run_id,
                "sent1": str(df.loc[idx, args.question1_col]),
                "sent2": str(df.loc[idx, args.question2_col]),
                "true_out": int(df.loc[idx, args.label_col]),
                "d_words": list_to_clean_string(d_arr),
                "nd_words": list_to_clean_string(nd_arr),
                "predicted_out": predicted_label,
                "predicted_prob": str(predicted_prob_list),
            }
            results.append(row)

            if args.save_every > 0 and (idx + 1) % args.save_every == 0:
                save_checkpoint(results, checkpoint_csv)
                elapsed = time.time() - begin
                print(
                    f"\n{display_name}, run {run_id}: checkpoint saved at sample {idx + 1}. "
                    f"Elapsed time: {elapsed:.2f} sec\n"
                )
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        pd.DataFrame(results).to_csv(output_csv, index=False)
        print("Final output saved to:", output_csv)
        print(f"Runtime: {time.time() - begin:.2f} seconds")
        output_files.append(output_csv)

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del model
    del tokenizer
    del masker
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return output_files


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for key, spec in MODEL_SPECS.items():
            print(f"  {key}: {spec['display_name']}  [default checkpoint subdir: {spec['checkpoint_subdir']}]")
        return

    model_keys = validate_model_keys(args.models)
    model_path_map = load_model_path_map(args.model_paths_json)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if cuda.is_available() else "cpu"
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

    df, texts = load_input_dataframe(args)
    print("Total samples after dropna/sampling:", len(texts))
    if len(texts) > 0:
        print("First sample:", texts[0])

    all_output_files = []
    total_begin = time.time()

    for model_key in model_keys:
        checkpoint_path = resolve_checkpoint_path(args, model_key, model_path_map)
        files = run_for_one_model(model_key, checkpoint_path, df, texts, args, device)
        all_output_files.extend(files)

    total_end = time.time()

    print("\n" + "=" * 80)
    print("All requested SHAP runs completed.")
    print("=" * 80)
    print("Generated files:")
    for file_path in all_output_files:
        print(file_path)
    print(f"Total runtime: {total_end - total_begin:.2f} seconds")


if __name__ == "__main__":
    main()
