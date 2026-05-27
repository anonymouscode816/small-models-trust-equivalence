#!/usr/bin/env python3
"""
Create pairwise Jaccard similarity tables from PI LIME output CSV files.

This script is suitable for anonymized submission:
- no usernames, private paths, external links, or tokens;
- all input/output locations are provided through CLI arguments;
- reads the neutral output layout produced by run_qqp_lime_all_5_models_fast.py.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


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
        description="Create PI LIME Jaccard similarity tables from model/run CSV files."
    )
    parser.add_argument("--input_root", type=str, default="./outputs/pi_lime",
                        help="Root directory containing LIME output folders for each model.")
    parser.add_argument("--output_dir", type=str, default="./outputs/pi_lime_jaccard_tables",
                        help="Directory where Jaccard tables will be saved.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS,
                        help=f"Model keys to include. Choices: {', '.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of repeated LIME runs to aggregate.")
    parser.add_argument("--start_run", type=int, default=1,
                        help="First run index. Usually 1.")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-K words used for Jaccard similarity.")
    parser.add_argument("--var_ddof", type=int, default=1,
                        help="Delta degrees of freedom for variance across runs. Use 1 for sample variance.")
    parser.add_argument("--keep_only_word_tokens", action="store_true",
                        help="Ignore tokens without word characters.")
    parser.add_argument("--keep_empty_token", action="store_true",
                        help="Keep [EMPTY] tokens if present.")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Skip missing files/pairs instead of raising an error.")
    parser.add_argument("--no_latex", action="store_true",
                        help="Do not generate the LaTeX table text file.")
    return parser.parse_args()


def validate_model_keys(model_keys: Iterable[str]) -> List[str]:
    clean_keys: List[str] = []
    for key in model_keys:
        if key not in MODEL_SPECS:
            valid = ", ".join(DEFAULT_MODEL_KEYS)
            raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
        clean_keys.append(key)
    return clean_keys


def selected_model_pairs(model_keys: Sequence[str]) -> List[Tuple[str, str]]:
    selected = set(model_keys)
    return [(m1, m2) for m1, m2 in DEFAULT_MODEL_PAIRS if m1 in selected and m2 in selected]


# -----------------------------------------------------------------------------
# Parsing and scoring helpers
# -----------------------------------------------------------------------------

def get_csv_path(input_root: Path, model_key: str, run_id: int) -> Path:
    spec = MODEL_SPECS[model_key]
    return input_root / spec["output_folder"] / f"{spec['file_stem']}_QQP_lime_run_{run_id}.csv"


def clean_old_numpy_strings(text: object) -> str:
    """Make parser robust to older cells such as np.str_('word')."""
    text = str(text)
    text = re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", text)
    text = re.sub(r'np\.str_\("([^"]*)"\)', r'"\1"', text)
    text = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", text)
    text = re.sub(r"np\.int\d*\(([^)]*)\)", r"\1", text)
    return text


def is_valid_token(token: object, keep_only_word_tokens: bool, remove_empty_token: bool) -> bool:
    token = str(token).strip()
    if token == "":
        return False
    if remove_empty_token and token == "[EMPTY]":
        return False
    if keep_only_word_tokens and re.search(r"\w", token) is None:
        return False
    return True


def parse_lime_word_list(cell: object, keep_only_word_tokens: bool, remove_empty_token: bool) -> List[str]:
    """
    Parse LIME word-list cells such as:
        ['can', 'card', 'credit']
    """
    if isinstance(cell, list):
        items = cell
    else:
        cell_text = clean_old_numpy_strings(cell)
        try:
            items = ast.literal_eval(cell_text)
        except Exception:
            print("Could not parse cell; returning empty list:")
            print(cell_text)
            return []

    words: List[str] = []
    for item in items:
        token = str(item).strip()
        if is_valid_token(token, keep_only_word_tokens, remove_empty_token):
            words.append(token)
    return words


def get_top_k_words_from_row(
    row: pd.Series,
    k: int,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
) -> Set[str]:
    """
    Gold-label-based selection:
      true_out = 1 -> duplicate -> use d_words
      true_out = 0 -> non-duplicate -> use nd_words
    """
    true_label = int(row["true_out"])
    selected_col = "d_words" if true_label == 1 else "nd_words"
    words_raw = parse_lime_word_list(row[selected_col], keep_only_word_tokens, remove_empty_token)

    words: List[str] = []
    for token in words_raw:
        token = str(token).strip()
        if not is_valid_token(token, keep_only_word_tokens, remove_empty_token):
            continue
        if token not in words:
            words.append(token)
        if len(words) >= k:
            break
    return set(words)


def load_model_run_dataframe(
    input_root: Path,
    model_key: str,
    run_id: int,
    k: int,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
) -> pd.DataFrame:
    csv_path = get_csv_path(input_root, model_key, run_id)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["true_out", "d_words", "nd_words"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    df["top_words_gold"] = df.apply(
        lambda row: get_top_k_words_from_row(
            row=row,
            k=k,
            keep_only_word_tokens=keep_only_word_tokens,
            remove_empty_token=remove_empty_token,
        ),
        axis=1,
    )
    return df


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_pairwise_jaccard(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Average Jaccard similarity across aligned examples."""
    n = min(len(df1), len(df2))
    if n == 0:
        return float("nan")
    scores: List[float] = []
    for i in range(n):
        scores.append(jaccard_similarity(df1.iloc[i]["top_words_gold"], df2.iloc[i]["top_words_gold"]))
    return float(np.mean(scores))


def display_name(model_key: str) -> str:
    return MODEL_SPECS[model_key]["display_name"]


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def generate_latex_table(df: pd.DataFrame, run_cols: List[str], k: int) -> str:
    lines: List[str] = []
    col_spec = "|l|l|" + "c|" * (len(run_cols) + 2)

    lines.append(r"\begin{table}[hbt]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    run_header = " & ".join([rf"\textbf{{Run {col.split('_')[-1]}}}" for col in run_cols])
    lines.append(rf"\textbf{{$M_1$}} & \textbf{{$M_2$}} & {run_header} & \textbf{{Mean}} & \textbf{{Variance}} \\")
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
        lines.append(
            f"{m1_cell} & {latex_escape(m2)} & {run_values} & "
            f"{row['mean']:.3f} & {row['variance']:.6f} \\\\"
        )

        current_m1_rows = df[df["M1"] == m1]
        if idx == current_m1_rows.index[-1]:
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{PI interpretability alignment between models using LIME-based "
        rf"Jaccard coefficient with $K={k}$. The mean and variance are computed "
        rf"across repeated LIME runs.}}"
    )
    lines.append(r"\label{tab:pi_lime_jaccard}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main computation
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = validate_model_keys(args.models)
    model_pairs = selected_model_pairs(model_keys)
    remove_empty_token = not args.keep_empty_token

    if not model_pairs:
        raise ValueError("No valid model pairs are available for the selected model set.")

    all_run_tables: List[pd.DataFrame] = []

    for run_id in range(args.start_run, args.start_run + args.num_runs):
        print(f"\nProcessing LIME run {run_id}")
        model_dfs: Dict[str, pd.DataFrame] = {}

        for model_key in model_keys:
            try:
                print(f"Loading {display_name(model_key)}, run {run_id}")
                model_dfs[model_key] = load_model_run_dataframe(
                    input_root=input_root,
                    model_key=model_key,
                    run_id=run_id,
                    k=args.k,
                    keep_only_word_tokens=args.keep_only_word_tokens,
                    remove_empty_token=remove_empty_token,
                )
            except FileNotFoundError as exc:
                if args.allow_missing:
                    print(f"Warning: {exc}")
                    continue
                raise

        run_rows: List[dict] = []
        for m1, m2 in model_pairs:
            if m1 not in model_dfs or m2 not in model_dfs:
                if args.allow_missing:
                    print(f"Skipping pair with missing data: {display_name(m1)} vs {display_name(m2)}")
                    continue
                raise FileNotFoundError(f"Missing data for pair: {m1}, {m2}")

            score = compute_pairwise_jaccard(model_dfs[m1], model_dfs[m2])
            run_rows.append({
                "M1": display_name(m1),
                "M2": display_name(m2),
                "PI_LIME_Jaccard": score,
                "run_id": run_id,
            })
            print(f"Run {run_id}: {display_name(m1)} vs {display_name(m2)} = {score:.4f}")

        if not run_rows:
            print(f"No rows created for run {run_id}; skipping.")
            continue

        run_df = pd.DataFrame(run_rows)
        run_output_path = output_dir / f"PI_LIME_jaccard_run_{run_id}.csv"
        run_df.to_csv(run_output_path, index=False)
        print(f"Saved run table: {run_output_path}")
        all_run_tables.append(run_df)

    if not all_run_tables:
        raise RuntimeError("No Jaccard tables were generated. Check input paths and run IDs.")

    combined_df = pd.concat(all_run_tables, ignore_index=True)
    combined_output_path = output_dir / "PI_LIME_jaccard_all_runs_long.csv"
    combined_df.to_csv(combined_output_path, index=False)

    wide_df = combined_df.pivot_table(
        index=["M1", "M2"],
        columns="run_id",
        values="PI_LIME_Jaccard",
    ).reset_index()
    wide_df.columns.name = None

    actual_run_ids = sorted(combined_df["run_id"].unique())
    rename_map = {run_id: f"run_{run_id}" for run_id in actual_run_ids}
    wide_df = wide_df.rename(columns=rename_map)
    run_cols = [f"run_{run_id}" for run_id in actual_run_ids]

    wide_df["mean"] = wide_df[run_cols].mean(axis=1)
    wide_df["variance"] = wide_df[run_cols].var(axis=1, ddof=args.var_ddof)

    pair_order_df = pd.DataFrame(
        [(display_name(m1), display_name(m2)) for m1, m2 in model_pairs],
        columns=["M1", "M2"],
    )
    summary_df = pair_order_df.merge(wide_df, on=["M1", "M2"], how="left")

    summary_output_path = output_dir / "PI_LIME_jaccard_summary_mean_variance.csv"
    summary_df.to_csv(summary_output_path, index=False)
    print("\nSaved summary table:", summary_output_path)

    rounded_df = summary_df.copy()
    for col in run_cols + ["mean"]:
        rounded_df[col] = rounded_df[col].round(3)
    rounded_df["variance"] = rounded_df["variance"].round(6)

    rounded_output_path = output_dir / "PI_LIME_jaccard_summary_mean_variance_rounded.csv"
    rounded_df.to_csv(rounded_output_path, index=False)
    print("Saved rounded summary table:", rounded_output_path)

    if not args.no_latex:
        latex_code = generate_latex_table(summary_df, run_cols, args.k)
        latex_output_path = output_dir / "PI_LIME_jaccard_summary_latex.txt"
        latex_output_path.write_text(latex_code, encoding="utf-8")
        print("Saved LaTeX table:", latex_output_path)

    print("\nFinal rounded PI LIME Jaccard table:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()
