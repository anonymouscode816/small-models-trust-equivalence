import argparse
import ast
import itertools
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Model registry matching the SHAP extraction script
# -----------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "bert_base": {
        "display_name": "BERT-base",
        "output_folder": "BERT-base",
        "file_stem": "BERT-base",
    },
    "distilbert": {
        "display_name": "Distil-BERT",
        "output_folder": "Distil-BERT",
        "file_stem": "Distil-BERT",
    },
    "bert_medium": {
        "display_name": "BERT-Medium",
        "output_folder": "BERT-Medium",
        "file_stem": "BERT-Medium",
    },
    "bert_mini": {
        "display_name": "BERT-Mini",
        "output_folder": "BERT-Mini",
        "file_stem": "BERT-Mini",
    },
    "bert_tiny": {
        "display_name": "BERT-Tiny",
        "output_folder": "BERT-Tiny",
        "file_stem": "BERT-Tiny",
    },
}

DEFAULT_MODEL_KEYS = list(MODEL_SPECS.keys())

DEFAULT_MODEL_PAIRS = [
    ("bert_base", "distilbert"),
    ("bert_base", "bert_medium"),
    ("bert_base", "bert_mini"),
    ("bert_base", "bert_tiny"),
    ("distilbert", "bert_medium"),
    ("distilbert", "bert_mini"),
    ("distilbert", "bert_tiny"),
    ("bert_medium", "bert_mini"),
    ("bert_medium", "bert_tiny"),
    ("bert_tiny", "bert_mini"),
]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PI SHAP Jaccard similarity tables from SHAP CSV outputs."
    )

    parser.add_argument("--input_root", type=str, default="./outputs/pi_shap",
                        help="Root directory containing model-wise SHAP CSV folders.")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_shap_jaccard_tables",
                        help="Directory where Jaccard tables will be saved.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS,
                        help=f"Model keys to include. Choices: {', '.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--pair_mode", type=str, default="default", choices=["default", "all"],
                        help="Use the paper-style default pair order or all pairwise combinations.")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of SHAP runs to aggregate.")
    parser.add_argument("--start_run", type=int, default=1,
                        help="First SHAP run index. Usually 1.")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-K words used for set-based Jaccard similarity.")
    parser.add_argument("--var_ddof", type=int, default=1,
                        help="Delta degrees of freedom for variance. 1=sample variance, 0=population variance.")
    parser.add_argument("--keep_only_word_tokens", action="store_true",
                        help="Remove punctuation-only tokens before selecting top-K words.")
    parser.add_argument("--keep_empty_token", action="store_true",
                        help="Keep the SHAP placeholder token [EMPTY]. By default it is removed.")
    parser.add_argument("--allow_repeated_tokens", action="store_true",
                        help="Use top-K list positions directly instead of top-K unique tokens.")
    parser.add_argument("--list_models", action="store_true",
                        help="Print available model keys and exit.")

    return parser.parse_args()


def validate_model_keys(model_keys: Iterable[str]) -> List[str]:
    clean_keys = []
    for key in model_keys:
        if key not in MODEL_SPECS:
            valid = ", ".join(DEFAULT_MODEL_KEYS)
            raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
        clean_keys.append(key)
    return clean_keys


def get_model_pairs(model_keys: Sequence[str], pair_mode: str) -> List[Tuple[str, str]]:
    selected = set(model_keys)

    if pair_mode == "all":
        return list(itertools.combinations(model_keys, 2))

    pairs = [pair for pair in DEFAULT_MODEL_PAIRS if pair[0] in selected and pair[1] in selected]

    if len(pairs) == 0 and len(model_keys) >= 2:
        pairs = list(itertools.combinations(model_keys, 2))

    return pairs


# -----------------------------------------------------------------------------
# Parsing and scoring helpers
# -----------------------------------------------------------------------------

def display_name(model_key: str) -> str:
    return MODEL_SPECS[model_key]["display_name"]


def get_csv_path(input_root: Path, model_key: str, run_id: int) -> Path:
    spec = MODEL_SPECS[model_key]
    return input_root / spec["output_folder"] / f"{spec['file_stem']}_QQP_shap_run_{run_id}.csv"


def clean_old_numpy_strings(text: str) -> str:
    """Make parser robust to older files containing np.str_ or np.float32 wrappers."""
    text = str(text)
    text = re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", text)
    text = re.sub(r'np\.str_\("([^"]*)"\)', r'"\1"', text)
    text = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", text)
    text = re.sub(r"np\.int\d*\(([^)]*)\)", r"\1", text)
    return text


def is_valid_token(token: str, keep_only_word_tokens: bool, remove_empty_token: bool) -> bool:
    token = str(token).strip()

    if token == "":
        return False

    if remove_empty_token and token == "[EMPTY]":
        return False

    if keep_only_word_tokens and re.search(r"\w", token) is None:
        return False

    return True


def parse_shap_word_list(cell) -> List[Tuple[str, float, float]]:
    """
    Input cell example:
        [('pay', 0.0017, 0.9932), ('debit', 0.0017, 0.9932)]
    """
    if isinstance(cell, list):
        items = cell
    else:
        cell = clean_old_numpy_strings(cell)
        try:
            items = ast.literal_eval(cell)
        except Exception:
            print("Could not parse cell:")
            print(cell)
            return []

    parsed_items = []

    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token = str(item[0]).strip()
            try:
                score = float(item[1])
            except Exception:
                score = 0.0
            try:
                base_value = float(item[2]) if len(item) > 2 else 0.0
            except Exception:
                base_value = 0.0
            parsed_items.append((token, score, base_value))
        else:
            token = str(item).strip()
            parsed_items.append((token, 0.0, 0.0))

    return parsed_items


def get_top_k_words_from_row(row, args: argparse.Namespace) -> Set[str]:
    """
    Gold-label-based selection:
        true_out = 1 => duplicate => use d_words
        true_out = 0 => non-duplicate => use nd_words
    """
    true_label = int(row["true_out"])
    selected_col = "d_words" if true_label == 1 else "nd_words"
    items = parse_shap_word_list(row[selected_col])

    words = []
    seen = set()
    remove_empty_token = not args.keep_empty_token
    top_k_unique = not args.allow_repeated_tokens

    for token, score, base_value in items:
        token = str(token).strip()

        if not is_valid_token(token, args.keep_only_word_tokens, remove_empty_token):
            continue

        if top_k_unique:
            if token in seen:
                continue
            seen.add(token)

        words.append(token)

        if len(words) >= args.k:
            break

    return set(words)


def load_model_run_dataframe(input_root: Path, model_key: str, run_id: int, args: argparse.Namespace) -> pd.DataFrame:
    csv_path = get_csv_path(input_root, model_key, run_id)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing SHAP CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["true_out", "d_words", "nd_words"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df["top_words_gold"] = df.apply(lambda row: get_top_k_words_from_row(row, args), axis=1)
    return df


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    set_a = set(set_a)
    set_b = set(set_b)
    union = set_a | set_b

    if len(union) == 0:
        return 0.0

    return len(set_a & set_b) / len(union)


def compute_pairwise_jaccard(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    n = min(len(df1), len(df2))
    if n == 0:
        return 0.0

    scores = []
    for i in range(n):
        words_1 = df1.iloc[i]["top_words_gold"]
        words_2 = df2.iloc[i]["top_words_gold"]
        scores.append(jaccard_similarity(words_1, words_2))

    return float(np.mean(scores))


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def generate_latex_table(df: pd.DataFrame, run_cols: List[str], k: int) -> str:
    lines = []
    num_metric_cols = len(run_cols) + 2
    col_spec = "|l|l|" + "c|" * num_metric_cols

    lines.append(r"\begin{table}[hbt]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    run_headers = " & ".join([rf"\textbf{{Run {col.split('_')[-1]}}}" for col in run_cols])
    lines.append(
        rf"\textbf{{$M_1$}} & \textbf{{$M_2$}} & {run_headers} & "
        rf"\textbf{{Mean}} & \textbf{{Variance}} \\" 
    )
    lines.append(r"\hline")

    group_counts = df.groupby("M1").size().to_dict()
    used_m1 = set()

    for idx, row in df.iterrows():
        m1 = row["M1"]
        m2 = row["M2"]

        if m1 not in used_m1:
            m1_cell = rf"\multirow{{{group_counts[m1]}}}{{*}}{{\textbf{{{latex_escape(m1)}}}}}"
            used_m1.add(m1)
        else:
            m1_cell = ""

        run_values = " & ".join([f"{row[col]:.3f}" for col in run_cols])
        line = (
            f"{m1_cell} & {latex_escape(m2)} & {run_values} & "
            f"{row['mean']:.3f} & {row['variance']:.6f} \\\\" 
        )
        lines.append(line)

        current_m1_rows = df[df["M1"] == m1]
        if idx == current_m1_rows.index[-1]:
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{PI interpretability alignment between models using SHAP-based "
        rf"Jaccard coefficient with $K={k}$. The mean and variance are computed "
        rf"across repeated SHAP runs.}}"
    )
    lines.append(r"\label{tab:pi_shap_jaccard}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for key, spec in MODEL_SPECS.items():
            print(f"  {key}: {spec['display_name']}")
        return

    model_keys = validate_model_keys(args.models)
    model_pairs = get_model_pairs(model_keys, args.pair_mode)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_run = args.start_run + args.num_runs - 1
    run_ids = list(range(args.start_run, final_run + 1))

    all_run_tables = []

    for run_id in run_ids:
        print(f"\nProcessing SHAP run {run_id}")
        model_dfs = {}

        for model_key in model_keys:
            print(f"Loading {display_name(model_key)}, run {run_id}")
            model_dfs[model_key] = load_model_run_dataframe(input_root, model_key, run_id, args)

        run_rows = []

        for m1_key, m2_key in model_pairs:
            score = compute_pairwise_jaccard(model_dfs[m1_key], model_dfs[m2_key])

            run_rows.append({
                "M1": display_name(m1_key),
                "M2": display_name(m2_key),
                "PI_SHAP_Jaccard": score,
            })

            print(f"Run {run_id}: {display_name(m1_key)} vs {display_name(m2_key)} = {score:.4f}")

        run_df = pd.DataFrame(run_rows)
        run_df["run_id"] = run_id

        run_output_path = output_dir / f"PI_SHAP_jaccard_run_{run_id}.csv"
        run_df.to_csv(run_output_path, index=False)
        print(f"Saved run table: {run_output_path}")

        all_run_tables.append(run_df)

    combined_df = pd.concat(all_run_tables, ignore_index=True)

    wide_df = combined_df.pivot_table(
        index=["M1", "M2"],
        columns="run_id",
        values="PI_SHAP_Jaccard",
    ).reset_index()
    wide_df.columns.name = None

    rename_map = {run_id: f"run_{run_id}" for run_id in run_ids}
    wide_df = wide_df.rename(columns=rename_map)
    run_cols = [f"run_{run_id}" for run_id in run_ids]

    wide_df["mean"] = wide_df[run_cols].mean(axis=1)
    wide_df["variance"] = wide_df[run_cols].var(axis=1, ddof=args.var_ddof)

    pair_order_df = pd.DataFrame(
        [(display_name(m1), display_name(m2)) for m1, m2 in model_pairs],
        columns=["M1", "M2"],
    )
    summary_df = pair_order_df.merge(wide_df, on=["M1", "M2"], how="left")

    summary_output_path = output_dir / "PI_SHAP_jaccard_summary_mean_variance.csv"
    summary_df.to_csv(summary_output_path, index=False)
    print(f"\nSaved summary table: {summary_output_path}")

    rounded_df = summary_df.copy()
    for col in run_cols + ["mean"]:
        rounded_df[col] = rounded_df[col].round(3)
    rounded_df["variance"] = rounded_df["variance"].round(6)

    rounded_output_path = output_dir / "PI_SHAP_jaccard_summary_mean_variance_rounded.csv"
    rounded_df.to_csv(rounded_output_path, index=False)
    print(f"Saved rounded summary table: {rounded_output_path}")

    latex_code = generate_latex_table(summary_df, run_cols, args.k)
    latex_output_path = output_dir / "PI_SHAP_jaccard_summary_latex.txt"
    latex_output_path.write_text(latex_code, encoding="utf-8")
    print(f"Saved LaTeX table: {latex_output_path}")

    print("\nFinal rounded PI SHAP Jaccard table:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()
