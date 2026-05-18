import os
import time
import gc
import random
import warnings
import re

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import cuda
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel

import shap


# ============================================================
# Global configuration
# ============================================================

INPUT_CSV = "../questions.csv"

OUTPUT_DIR = "./shap_outputs_all_5_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES_TO_RUN = 15000
NUM_RUNS = 3

NUM_LABELS = 2
CLASS_NAMES = ["not duplicate", "duplicate"]

# Speed-related settings
MAX_LEN = 128
PRED_BATCH_SIZE = 512

# SHAP settings
# This is the minimum value. The script will automatically increase
# max_evals for long samples when SHAP requires it.
SHAP_MIN_EVALS = 100
SHAP_BATCH_SIZE = 512

SAVE_EVERY = 100

# Same regex used by the SHAP text masker
SHAP_TOKEN_PATTERN = r"\w+|[^\w\s]"


# ============================================================
# Models to run
# ============================================================

MODEL_LIST = [
    {
        "model_name": "bert-base",
        "model_path": "../QQP_model_bert_base"
    },
    {
        "model_name": "distil-BERT",
        "model_path": "../QQP_MODEL_distil"
    },
    {
        "model_name": "BERT-medium",
        "model_path": "../QQP_MODEL_medium"
    },
    {
        "model_name": "BERT-mini",
        "model_path": "../QQP_MODEL_mini"
    },
    {
        "model_name": "BERT-tiny",
        "model_path": "../QQP_MODEL_tiny"
    }
]


# ============================================================
# GPU setup
# ============================================================

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


# ============================================================
# Loss function
# ============================================================

class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        loss = loss.mean()
        return loss


# ============================================================
# Universal BERT / DistilBERT model class
# ============================================================

class MainModel(PreTrainedModel):
    def __init__(self, config, loss_fn=None):
        super(MainModel, self).__init__(config)

        self.num_labels = NUM_LABELS
        self.loss_fn = loss_fn
        self.config = config

        config.output_hidden_states = True

        # Keep the module name as self.bert because your saved custom
        # MainModel checkpoints are expected to contain keys such as:
        # bert.embeddings..., bert.encoder..., classifier...
        self.bert = AutoModel.from_config(config)

        if hasattr(config, "hidden_size"):
            classifier_input_size = config.hidden_size
        elif hasattr(config, "dim"):
            classifier_input_size = config.dim
        else:
            raise ValueError("Could not find hidden size in config.")

        self.classifier = nn.Linear(classifier_input_size, self.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        device=None
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # DistilBERT does not accept token_type_ids.
        if self.config.model_type != "distilbert" and token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        output = self.bert(**model_inputs)

        cls_output = output.last_hidden_state[:, 0, :]
        classifier_out = self.classifier(cls_output)
        main_prob = F.softmax(classifier_out, dim=1)

        if labels is not None:
            main_gold_prob = torch.gather(main_prob, 1, labels)
            loss_main = self.loss_fn.forward(main_gold_prob)
            return loss_main, main_prob

        return main_prob


# ============================================================
# Load QQP data once
# ============================================================

print("\nLoading QQP data...")

df = pd.read_csv(INPUT_CSV)
df = df.head(NUM_SAMPLES_TO_RUN)
df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)

texts = [
    str(q1).strip() + "  " + str(q2).strip()
    for q1, q2 in zip(df["question1"], df["question2"])
]

# Your uploaded SHAP code used lowercase texts.
lowercase_texts = [text.lower() for text in texts]

print("Total samples after dropna:", len(lowercase_texts))
print("First sample:", lowercase_texts[0])


# ============================================================
# Helper functions
# ============================================================

def split_question_pair(text):
    text = str(text)

    if "  " in text:
        q1, q2 = text.split("  ", 1)
    else:
        q1, q2 = text, ""

    return q1.strip(), q2.strip()


def get_class_name(prob_array):
    predicted_class_idx = int(np.argmax(prob_array))
    return CLASS_NAMES[predicted_class_idx]


def clean_token(token):
    token = str(token).strip()

    if token == "":
        token = "[EMPTY]"

    return token


def sort_list(x):
    return sorted(x, key=lambda item: item[1], reverse=True)


def list_to_clean_string(items):
    clean_items = []

    for token, shap_value, base_value in items:
        clean_items.append(
            (
                str(token),
                float(shap_value),
                float(base_value)
            )
        )

    return str(clean_items)


def save_checkpoint(results, path):
    temp_df = pd.DataFrame(results)
    temp_df.to_csv(path, index=False)


def get_required_shap_max_evals(text):
    """
    SHAP permutation explainer requires:
        max_evals >= 2 * num_features + 1

    We estimate num_features using the same regex tokenizer used
    by shap.maskers.Text.
    """
    num_features = len(re.findall(SHAP_TOKEN_PATTERN, str(text)))
    required = 2 * num_features + 1

    return max(SHAP_MIN_EVALS, required)


def parse_required_max_evals_from_error(error_message):
    """
    Handles error messages like:
        max_evals=100 is too low ...
        it must be at least 2 * num_features + 1 = 139!
    """
    error_message = str(error_message)

    match = re.search(r"=\s*(\d+)\s*!", error_message)
    if match:
        return int(match.group(1))

    match = re.search(r"at least\s+(\d+)", error_message)
    if match:
        return int(match.group(1))

    return None


def extract_shap_words(shap_values):
    """
    Converts SHAP output into:
        d_words  = SHAP values for duplicate class, class index 1
        nd_words = SHAP values for non-duplicate class, class index 0

    Each element:
        (token, shap_value, base_value)
    """

    tokens = shap_values.data[0]
    values = shap_values.values[0]
    base_values = shap_values.base_values[0]

    d_arr = []
    nd_arr = []

    values = np.array(values)
    base_values = np.array(base_values)

    # Expected shape:
    # values: [num_tokens, 2]
    if values.ndim == 2 and values.shape[-1] == 2:
        for token, token_values in zip(tokens, values):
            token = clean_token(token)

            nd_value = round(float(token_values[0]), 4)
            d_value = round(float(token_values[1]), 4)

            nd_base = round(float(base_values[0]), 4)
            d_base = round(float(base_values[1]), 4)

            nd_arr.append((token, nd_value, nd_base))
            d_arr.append((token, d_value, d_base))

    else:
        raise ValueError(
            f"Unexpected SHAP values shape: {values.shape}. "
            "Expected [num_tokens, 2]."
        )

    d_arr = sort_list(d_arr)
    nd_arr = sort_list(nd_arr)

    return d_arr, nd_arr


# ============================================================
# Run one model
# ============================================================

def run_for_one_model(model_name, model_path):
    print("\n" + "#" * 80)
    print(f"Starting model: {model_name}")
    print(f"Model path: {model_path}")
    print("#" * 80)

    model_output_dir = os.path.join(OUTPUT_DIR, model_name.replace(" ", "_"))
    os.makedirs(model_output_dir, exist_ok=True)

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True
    )

    config = AutoConfig.from_pretrained(model_path)

    print("Model type:", config.model_type)

    if hasattr(config, "hidden_size"):
        print("Hidden size:", config.hidden_size)

    if hasattr(config, "dim"):
        print("Hidden dim:", config.dim)

    if hasattr(config, "num_hidden_layers"):
        print("Number of layers:", config.num_hidden_layers)

    if hasattr(config, "n_layers"):
        print("Number of layers:", config.n_layers)

    model = MainModel.from_pretrained(
        model_path,
        config=config,
        loss_fn=LossFunction()
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    def predict_f(texts_old, batch_size=PRED_BATCH_SIZE):
        """
        Function used by SHAP.
        Returns numpy array:
            [number_of_examples, number_of_classes]
        """
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
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device, non_blocking=True)
                for key, value in encoded.items()
            }

            with torch.inference_mode():
                probs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids", None),
                    labels=None,
                    device=device
                )

            all_probs.append(probs.detach().cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    # Same text masker style as your uploaded SHAP code.
    masker = shap.maskers.Text(tokenizer=SHAP_TOKEN_PATTERN)

    def explain_one_text_with_dynamic_max_evals(explainer, text):
        """
        Uses dynamic max_evals.
        If SHAP still raises a max_evals error, parse the required value
        from the error message and retry once.
        """
        sample_max_evals = get_required_shap_max_evals(text)

        try:
            return explainer(
                [text],
                max_evals=sample_max_evals,
                batch_size=SHAP_BATCH_SIZE
            )

        except ValueError as e:
            message = str(e)

            if "max_evals" in message and "too low" in message:
                required = parse_required_max_evals_from_error(message)

                if required is None:
                    required = sample_max_evals * 2

                required = max(required, sample_max_evals + 1)

                print(
                    f"\nRetrying SHAP with larger max_evals. "
                    f"Old: {sample_max_evals}, New: {required}\n"
                )

                return explainer(
                    [text],
                    max_evals=required,
                    batch_size=SHAP_BATCH_SIZE
                )

            raise e

    def run_shap_experiment(run_id):
        print("\n" + "=" * 70)
        print(f"Model: {model_name} | SHAP run {run_id}/{NUM_RUNS}")
        print("=" * 70)

        output_csv = os.path.join(
            model_output_dir,
            f"{model_name}_QQP_shap_run_{run_id}.csv"
        )

        checkpoint_csv = os.path.join(
            model_output_dir,
            f"{model_name}_QQP_shap_run_{run_id}_checkpoint.csv"
        )

        run_seed = 42 + run_id

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
            seed=run_seed
        )

        results = []
        begin = time.time()

        print("MAX_LEN:", MAX_LEN)
        print("PRED_BATCH_SIZE:", PRED_BATCH_SIZE)
        print("SHAP_MIN_EVALS:", SHAP_MIN_EVALS)
        print("SHAP_BATCH_SIZE:", SHAP_BATCH_SIZE)
        print("Run seed:", run_seed)
        print("Output CSV:", output_csv)
        print("Checkpoint CSV:", checkpoint_csv)
        print("\n")

        for idx, item in enumerate(tqdm(lowercase_texts, total=len(lowercase_texts))):

            # ------------------------------------------------
            # Main model prediction
            # ------------------------------------------------
            predicted_prob = predict_f([item])[0]

            predicted_prob_list = [
                float(predicted_prob[0]),
                float(predicted_prob[1])
            ]

            predicted_label = get_class_name(predicted_prob)

            # ------------------------------------------------
            # SHAP explanation with dynamic max_evals
            # ------------------------------------------------
            shap_values = explain_one_text_with_dynamic_max_evals(
                explainer,
                item
            )

            d_arr, nd_arr = extract_shap_words(shap_values)

            row = {
                "model_name": model_name,
                "run_id": run_id,
                "sent1": str(df.loc[idx, "question1"]),
                "sent2": str(df.loc[idx, "question2"]),
                "true_out": int(df.loc[idx, "is_duplicate"]),
                "d_words": list_to_clean_string(d_arr),
                "nd_words": list_to_clean_string(nd_arr),
                "predicted_out": predicted_label,
                "predicted_prob": str(predicted_prob_list)
            }

            results.append(row)

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------
            if (idx + 1) % SAVE_EVERY == 0:
                save_checkpoint(results, checkpoint_csv)

                elapsed = time.time() - begin
                print(
                    f"\nModel {model_name}, run {run_id}: "
                    f"checkpoint saved at sample {idx + 1}. "
                    f"Elapsed time: {elapsed:.2f} sec\n"
                )

                if device == "cuda":
                    torch.cuda.empty_cache()

                gc.collect()

        final_result = pd.DataFrame(results)
        final_result.to_csv(output_csv, index=False)

        end = time.time()

        print("\nFinished")
        print("Model:", model_name)
        print("Run:", run_id)
        print(f"Runtime: {end - begin:.2f} seconds")
        print("Final output saved to:", output_csv)

        return output_csv

    model_output_files = []

    for run_id in range(1, NUM_RUNS + 1):
        output_file = run_shap_experiment(run_id)
        model_output_files.append(output_file)

        if device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    # Free model before loading the next model
    del model
    del tokenizer
    del masker

    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return model_output_files


# ============================================================
# Main execution
# ============================================================

all_output_files = []

total_begin = time.time()

for model_info in MODEL_LIST:
    files = run_for_one_model(
        model_name=model_info["model_name"],
        model_path=model_info["model_path"]
    )

    all_output_files.extend(files)

total_end = time.time()

print("\n" + "=" * 80)
print("All models and all SHAP runs completed.")
print("=" * 80)

print("\nGenerated output files:")

for file_path in all_output_files:
    print(file_path)

print(f"\nTotal runtime: {total_end - total_begin:.2f} seconds")
