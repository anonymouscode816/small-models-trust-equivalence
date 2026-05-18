import os
import re
import ast
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT_ROOT = "./shap_outputs_all_5_models"
OUTPUT_DIR = "./pi_shap_jaccard_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_RUNS = 3
K = 10

# Variance across 3 runs:
# ddof=1 gives sample variance.
# Use ddof=0 if you want population variance.
VAR_DDOF = 1

# If False, punctuation tokens such as "/", "?" are allowed.
# If True, only word-like tokens are used.
KEEP_ONLY_WORD_TOKENS = False

# Remove SHAP placeholder token.
REMOVE_EMPTY_TOKEN = True

# Use top-K unique words.
# Recommended for Jaccard because Jaccard is set-based.
TOP_K_UNIQUE = True


# ============================================================
# Model file information
# ============================================================

MODEL_INFO = {
    "BERT-base": {
        "folder": "bert-base",
        "file_stem": "bert-base"
    },
    "Distil-BERT": {
        "folder": "distil-BERT",
        "file_stem": "distil-BERT"
    },
    "BERT-Medium": {
        "folder": "BERT-medium",
        "file_stem": "BERT-medium"
    },
    "BERT-Mini": {
        "folder": "BERT-mini",
        "file_stem": "BERT-mini"
    },
    "BERT-Tiny": {
        "folder": "BERT-tiny",
        "file_stem": "BERT-tiny"
    }
}


# Same pair order as your table
MODEL_PAIRS = [
    ("BERT-base", "Distil-BERT"),
    ("BERT-base", "BERT-Medium"),
    ("BERT-base", "BERT-Mini"),
    ("BERT-base", "BERT-Tiny"),

    ("Distil-BERT", "BERT-Medium"),
    ("Distil-BERT", "BERT-Mini"),
    ("Distil-BERT", "BERT-Tiny"),

    ("BERT-Medium", "BERT-Mini"),
    ("BERT-Medium", "BERT-Tiny"),

    ("BERT-Tiny", "BERT-Mini")
]


# ============================================================
# Helper functions
# ============================================================

def get_csv_path(model_display_name, run_id):
    info = MODEL_INFO[model_display_name]

    return os.path.join(
        INPUT_ROOT,
        info["folder"],
        f"{info['file_stem']}_QQP_shap_run_{run_id}.csv"
    )


def clean_old_numpy_strings(text):
    """
    Makes parser robust even if some older files contain np.str_('word')
    or np.float32(...).
    """
    text = str(text)

    text = re.sub(r"np\.str_\('([^']*)'\)", r"'\1'", text)
    text = re.sub(r'np\.str_\("([^"]*)"\)', r'"\1"', text)

    text = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", text)
    text = re.sub(r"np\.int\d*\(([^)]*)\)", r"\1", text)

    return text


def is_valid_token(token):
    token = str(token).strip()

    if token == "":
        return False

    if REMOVE_EMPTY_TOKEN and token == "[EMPTY]":
        return False

    if KEEP_ONLY_WORD_TOKENS:
        # Keeps tokens containing letters/numbers/underscore.
        # Removes pure punctuation such as "/", "?", ",".
        if re.search(r"\w", token) is None:
            return False

    return True


def parse_shap_word_list(cell):
    """
    Input cell example:
        [('pay', 0.0017, 0.9932), ('debit', 0.0017, 0.9932)]

    Returns:
        [('pay', 0.0017, 0.9932), ...]
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


def get_top_k_words_from_row(row, k=10):
    """
    Gold-label-based selection:
        true_out = 1 => duplicate => use d_words
        true_out = 0 => non-duplicate => use nd_words
    """
    true_label = int(row["true_out"])

    if true_label == 1:
        selected_col = "d_words"
    else:
        selected_col = "nd_words"

    items = parse_shap_word_list(row[selected_col])

    words = []

    for token, score, base_value in items:
        token = str(token).strip()

        if not is_valid_token(token):
            continue

        if TOP_K_UNIQUE:
            if token not in words:
                words.append(token)
        else:
            words.append(token)

        if len(words) >= k:
            break

    return set(words)


def load_model_run_dataframe(model_display_name, run_id):
    csv_path = get_csv_path(model_display_name, run_id)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["true_out", "d_words", "nd_words"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {csv_path}")

    df["top_words_gold"] = df.apply(
        lambda row: get_top_k_words_from_row(row, K),
        axis=1
    )

    return df


def jaccard_similarity(set_a, set_b):
    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a | set_b

    if len(union) == 0:
        return 0.0

    intersection = set_a & set_b

    return len(intersection) / len(union)


def compute_pairwise_jaccard(df1, df2):
    """
    Computes mean Jaccard similarity over all test samples.
    """
    n = min(len(df1), len(df2))

    scores = []

    for i in range(n):
        words_1 = df1.iloc[i]["top_words_gold"]
        words_2 = df2.iloc[i]["top_words_gold"]

        score = jaccard_similarity(words_1, words_2)
        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# Main computation
# ============================================================

all_run_tables = []

for run_id in range(1, NUM_RUNS + 1):
    print(f"\nProcessing SHAP run {run_id}/{NUM_RUNS}")

    model_dfs = {}

    for model_name in MODEL_INFO.keys():
        print(f"Loading {model_name}, run {run_id}")
        model_dfs[model_name] = load_model_run_dataframe(model_name, run_id)

    run_rows = []

    for m1, m2 in MODEL_PAIRS:
        score = compute_pairwise_jaccard(
            model_dfs[m1],
            model_dfs[m2]
        )

        run_rows.append({
            "M1": m1,
            "M2": m2,
            "PI_SHAP_Jaccard": score
        })

        print(f"Run {run_id}: {m1} vs {m2} = {score:.4f}")

    run_df = pd.DataFrame(run_rows)
    run_df["run_id"] = run_id

    run_output_path = os.path.join(
        OUTPUT_DIR,
        f"PI_SHAP_jaccard_run_{run_id}.csv"
    )

    run_df.to_csv(run_output_path, index=False)
    print(f"Saved run table: {run_output_path}")

    all_run_tables.append(run_df)


# ============================================================
# Combine runs and compute mean/variance
# ============================================================

combined_df = pd.concat(all_run_tables, ignore_index=True)

wide_df = combined_df.pivot_table(
    index=["M1", "M2"],
    columns="run_id",
    values="PI_SHAP_Jaccard"
).reset_index()

wide_df.columns.name = None

# Rename run columns
rename_map = {}
for run_id in range(1, NUM_RUNS + 1):
    rename_map[run_id] = f"run_{run_id}"

wide_df = wide_df.rename(columns=rename_map)

run_cols = [f"run_{i}" for i in range(1, NUM_RUNS + 1)]

wide_df["mean"] = wide_df[run_cols].mean(axis=1)
wide_df["variance"] = wide_df[run_cols].var(axis=1, ddof=VAR_DDOF)

# Restore table order
pair_order_df = pd.DataFrame(MODEL_PAIRS, columns=["M1", "M2"])
summary_df = pair_order_df.merge(wide_df, on=["M1", "M2"], how="left")

summary_output_path = os.path.join(
    OUTPUT_DIR,
    "PI_SHAP_jaccard_summary_mean_variance.csv"
)

summary_df.to_csv(summary_output_path, index=False)

print("\nSaved summary table:")
print(summary_output_path)


# ============================================================
# Also save a rounded version for paper table
# ============================================================

rounded_df = summary_df.copy()

for col in run_cols + ["mean"]:
    rounded_df[col] = rounded_df[col].round(3)

rounded_df["variance"] = rounded_df["variance"].round(6)

rounded_output_path = os.path.join(
    OUTPUT_DIR,
    "PI_SHAP_jaccard_summary_mean_variance_rounded.csv"
)

rounded_df.to_csv(rounded_output_path, index=False)

print("Saved rounded summary table:")
print(rounded_output_path)


# ============================================================
# Generate LaTeX table similar to your screenshot
# ============================================================

def latex_escape(text):
    text = str(text)
    text = text.replace("_", r"\_")
    return text


def generate_latex_table(df):
    lines = []

    lines.append(r"\begin{table}[hbt]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{$M_1$} & \textbf{$M_2$} & "
        r"\textbf{Run 1} & \textbf{Run 2} & \textbf{Run 3} & "
        r"\textbf{Mean} & \textbf{Variance} \\"
    )
    lines.append(r"\hline")

    group_counts = df.groupby("M1").size().to_dict()
    used_m1 = set()

    for _, row in df.iterrows():
        m1 = row["M1"]
        m2 = row["M2"]

        if m1 not in used_m1:
            m1_cell = rf"\multirow{{{group_counts[m1]}}}{{*}}{{\textbf{{{latex_escape(m1)}}}}}"
            used_m1.add(m1)
        else:
            m1_cell = ""

        line = (
            f"{m1_cell} & {latex_escape(m2)} & "
            f"{row['run_1']:.3f} & "
            f"{row['run_2']:.3f} & "
            f"{row['run_3']:.3f} & "
            f"{row['mean']:.3f} & "
            f"{row['variance']:.6f} \\\\"
        )

        lines.append(line)

        # Add hline after each M1 block
        current_m1_rows = df[df["M1"] == m1]
        if row.name == current_m1_rows.index[-1]:
            lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{PI interpretability alignment between models using "
        r"SHAP-based Jaccard coefficient with $K=10$. The mean and variance "
        r"are computed across three SHAP runs.}"
    )
    lines.append(r"\label{tab:pi_shap_jaccard}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


latex_code = generate_latex_table(summary_df)

latex_output_path = os.path.join(
    OUTPUT_DIR,
    "PI_SHAP_jaccard_summary_latex.txt"
)

with open(latex_output_path, "w") as f:
    f.write(latex_code)

print("Saved LaTeX table:")
print(latex_output_path)


# ============================================================
# Print final rounded table
# ============================================================

print("\nFinal rounded PI SHAP Jaccard table:")
print(rounded_df.to_string(index=False))
