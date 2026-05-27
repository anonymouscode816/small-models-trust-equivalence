#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create pairwise Jaccard similarity tables from NLI LIME explanation CSV files.

Anonymized submission-ready script:
- no hard-coded usernames, private paths, URLs, or keys;
- all input/output locations are supplied by CLI arguments;
- reads the neutral output layout produced by run_bert_nli_lime_all_5_models_fast.py.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


METHOD = "LIME"
METHOD_LOWER = "lime"
CLASS_NAMES = ["contradiction", "neutral", "entailment"]
ID_TO_LABEL = {0: "contradiction", 1: "neutral", 2: "entailment"}

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "bert_base": {"display_name": "BERT-base"},
    "distilbert": {"display_name": "DistilBERT"},
    "bert_medium": {"display_name": "BERT-Medium"},
    "bert_mini": {"display_name": "BERT-Mini"},
    "bert_tiny": {"display_name": "BERT-Tiny"},
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
    ("bert_mini", "bert_tiny"),
]

SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]", "<s>", "</s>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Create NLI {METHOD} Jaccard similarity tables from model/run CSV files."
    )
    parser.add_argument("--input_root", type=str, default=f"./outputs/nli_{METHOD_LOWER}",
                        help=f"Root directory containing NLI {METHOD} output folders for each model.")
    parser.add_argument("--output_dir", type=str, default=f"./outputs/nli_{METHOD_LOWER}_jaccard_tables",
                        help="Directory where Jaccard tables will be saved.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_KEYS,
                        help=f"Model keys to include. Choices: {', '.join(DEFAULT_MODEL_KEYS)}")
    parser.add_argument("--num_runs", type=int, default=3,
                        help=f"Number of repeated {METHOD} runs to aggregate.")
    parser.add_argument("--start_run", type=int, default=1,
                        help="First run index. Usually 1.")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-K tokens used for Jaccard similarity.")
    parser.add_argument("--label_source", choices=["true", "predicted"], default="true",
                        help="Choose explanation class from the gold label or predicted label.")
    parser.add_argument("--var_ddof", type=int, default=1,
                        help="Delta degrees of freedom for variance across runs. Use 1 for sample variance.")
    parser.add_argument("--keep_only_word_tokens", action="store_true",
                        help="Ignore tokens without word characters.")
    parser.add_argument("--keep_empty_token", action="store_true",
                        help="Keep [EMPTY] tokens if present.")
    parser.add_argument("--keep_special_tokens", action="store_true",
                        help="Keep tokens such as [SEP] if present.")
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


def display_name(model_key: str) -> str:
    return MODEL_SPECS[model_key]["display_name"]


def get_csv_path(input_root: Path, model_key: str, run_id: int) -> Path:
    """Support the new separate-runner layout and the older combined-runner layout."""
    candidates = [
        input_root / model_key / f"run_{run_id:02d}" / f"{model_key}_nli_{METHOD_LOWER}_run_{run_id:02d}.csv",
        input_root / model_key / f"run_{run_id}" / f"{model_key}_nli_{METHOD_LOWER}_run_{run_id}.csv",
        input_root / METHOD_LOWER / model_key / f"run_{run_id:02d}" / f"{model_key}_{METHOD_LOWER}_run_{run_id:02d}.csv",
        input_root / METHOD_LOWER / model_key / f"run_{run_id}" / f"{model_key}_{METHOD_LOWER}_run_{run_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def clean_old_numpy_strings(text: object) -> str:
    text = str(text)
    text = re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", text)
    text = re.sub(r'np\.str_\("([^"]*)"\)', r'"\1"', text)
    text = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", text)
    text = re.sub(r"np\.int\d*\(([^)]*)\)", r"\1", text)
    return text


def is_valid_token(
    token: object,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
    remove_special_tokens: bool,
) -> bool:
    token = str(token).strip()
    if token == "":
        return False
    if remove_empty_token and token == "[EMPTY]":
        return False
    if remove_special_tokens and token in SPECIAL_TOKENS:
        return False
    if keep_only_word_tokens and re.search(r"\w", token) is None:
        return False
    return True


def flatten_items(obj):
    if isinstance(obj, tuple):
        yield obj
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (str, int, float, np.integer, np.floating)):
            yield tuple(obj)
        else:
            for item in obj:
                yield from flatten_items(item)
    else:
        return


def parse_explanation_cell(
    cell: object,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
    remove_special_tokens: bool,
) -> List[Tuple[str, float]]:
    """Parse explanation cells containing tuples like (token, score) or (token, score, base_value)."""
    if isinstance(cell, (list, tuple)):
        parsed = cell
    else:
        text = clean_old_numpy_strings(cell)
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []

    output: List[Tuple[str, float]] = []
    for item in flatten_items(parsed):
        if len(item) < 2:
            continue
        token = str(item[0]).strip()
        if not is_valid_token(token, keep_only_word_tokens, remove_empty_token, remove_special_tokens):
            continue
        try:
            score = abs(float(item[1]))
        except Exception:
            score = 0.0
        output.append((token, score))

    output.sort(key=lambda pair: pair[1], reverse=True)
    return output


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in CLASS_NAMES:
        return text
    try:
        return ID_TO_LABEL[int(float(text))]
    except Exception:
        return text


def class_column_name(label: str, columns: Sequence[str]) -> str:
    label = normalize_label(label)
    if label == "contradiction":
        if "contradiction" in columns:
            return "contradiction"
        if "contrdiction" in columns:
            return "contrdiction"
    if label in columns:
        return label
    raise ValueError(f"Could not find explanation column for label '{label}'. Columns: {list(columns)}")


def get_top_k_words_from_row(
    row: pd.Series,
    k: int,
    label_source: str,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
    remove_special_tokens: bool,
) -> Set[str]:
    label_value = row["predicted_out"] if label_source == "predicted" else row["true_out"]
    selected_col = class_column_name(label_value, row.index)
    parsed = parse_explanation_cell(
        row[selected_col],
        keep_only_word_tokens=keep_only_word_tokens,
        remove_empty_token=remove_empty_token,
        remove_special_tokens=remove_special_tokens,
    )

    words: List[str] = []
    for token, _score in parsed:
        token = str(token).strip()
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
    label_source: str,
    keep_only_word_tokens: bool,
    remove_empty_token: bool,
    remove_special_tokens: bool,
) -> pd.DataFrame:
    csv_path = get_csv_path(input_root, model_key, run_id)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["true_out", "neutral", "entailment"]
    missing = [col for col in required_cols if col not in df.columns]
    if "contradiction" not in df.columns and "contrdiction" not in df.columns:
        missing.append("contradiction")
    if label_source == "predicted" and "predicted_out" not in df.columns:
        missing.append("predicted_out")
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    df["top_words_selected"] = df.apply(
        lambda row: get_top_k_words_from_row(
            row=row,
            k=k,
            label_source=label_source,
            keep_only_word_tokens=keep_only_word_tokens,
            remove_empty_token=remove_empty_token,
            remove_special_tokens=remove_special_tokens,
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
    n = min(len(df1), len(df2))
    if n == 0:
        return float("nan")
    scores = [
        jaccard_similarity(df1.iloc[i]["top_words_selected"], df2.iloc[i]["top_words_selected"])
        for i in range(n)
    ]
    return float(np.mean(scores))


def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_")


def generate_latex_table(df: pd.DataFrame, run_cols: List[str], k: int, label_source: str) -> str:
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
        rf"\caption{{NLI interpretability alignment between models using {METHOD}-based "
        rf"Jaccard coefficient with $K={k}$. The selected class is based on the "
        rf"{label_source} label. Mean and variance are computed across repeated runs.}}"
    )
    lines.append(rf"\label{{tab:nli_{METHOD_LOWER}_jaccard}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = validate_model_keys(args.models)
    model_pairs = selected_model_pairs(model_keys)
    remove_empty_token = not args.keep_empty_token
    remove_special_tokens = not args.keep_special_tokens

    if not model_pairs:
        raise ValueError("No valid model pairs are available for the selected model set.")

    all_run_tables: List[pd.DataFrame] = []

    for run_id in range(args.start_run, args.start_run + args.num_runs):
        print(f"\nProcessing {METHOD} run {run_id}")
        model_dfs: Dict[str, pd.DataFrame] = {}

        for model_key in model_keys:
            try:
                print(f"Loading {display_name(model_key)}, run {run_id}")
                model_dfs[model_key] = load_model_run_dataframe(
                    input_root=input_root,
                    model_key=model_key,
                    run_id=run_id,
                    k=args.k,
                    label_source=args.label_source,
                    keep_only_word_tokens=args.keep_only_word_tokens,
                    remove_empty_token=remove_empty_token,
                    remove_special_tokens=remove_special_tokens,
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
                f"NLI_{METHOD}_Jaccard": score,
                "run_id": run_id,
            })
            print(f"Run {run_id}: {display_name(m1)} vs {display_name(m2)} = {score:.4f}")

        if not run_rows:
            print(f"No rows created for run {run_id}; skipping.")
            continue

        run_df = pd.DataFrame(run_rows)
        run_output_path = output_dir / f"NLI_{METHOD}_jaccard_run_{run_id}.csv"
        run_df.to_csv(run_output_path, index=False)
        print(f"Saved run table: {run_output_path}")
        all_run_tables.append(run_df)

    if not all_run_tables:
        raise RuntimeError("No Jaccard tables were generated. Check input paths and run IDs.")

    score_col = f"NLI_{METHOD}_Jaccard"
    combined_df = pd.concat(all_run_tables, ignore_index=True)
    combined_output_path = output_dir / f"NLI_{METHOD}_jaccard_all_runs_long.csv"
    combined_df.to_csv(combined_output_path, index=False)

    wide_df = combined_df.pivot_table(
        index=["M1", "M2"],
        columns="run_id",
        values=score_col,
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

    summary_output_path = output_dir / f"NLI_{METHOD}_jaccard_summary_mean_variance.csv"
    summary_df.to_csv(summary_output_path, index=False)
    print("\nSaved summary table:", summary_output_path)

    rounded_df = summary_df.copy()
    for col in run_cols + ["mean"]:
        rounded_df[col] = rounded_df[col].round(3)
    rounded_df["variance"] = rounded_df["variance"].round(6)

    rounded_output_path = output_dir / f"NLI_{METHOD}_jaccard_summary_mean_variance_rounded.csv"
    rounded_df.to_csv(rounded_output_path, index=False)
    print("Saved rounded summary table:", rounded_output_path)

    if not args.no_latex:
        latex_code = generate_latex_table(summary_df, run_cols, args.k, args.label_source)
        latex_output_path = output_dir / f"NLI_{METHOD}_jaccard_summary_latex.txt"
        latex_output_path.write_text(latex_code, encoding="utf-8")
        print("Saved LaTeX table:", latex_output_path)

    print(f"\nFinal rounded NLI {METHOD} Jaccard table:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()
