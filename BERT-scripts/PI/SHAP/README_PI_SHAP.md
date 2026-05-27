# PI SHAP Analysis and Jaccard Similarity

This folder contains anonymized scripts for SHAP-based interpretability analysis on the Paraphrase Identification (PI) task.

The workflow has two steps:

1. `run_qqp_shap_all_5_models_fast_anonymized.py`  
   Runs SHAP on the fine-tuned PI models and saves the most important words for each input pair.

2. `create_pi_shap_jaccard_table_anonymized.py`  
   Reads the SHAP output files and computes Jaccard similarity between pairs of large and small models.

No private paths, usernames, machine-specific directories, private links, or keys are required. All file locations are passed using command-line arguments.

---

## 1. Supported Models

The scripts support the following five models:

| Model Key | Display Name | Default Fine-tuned Checkpoint Folder |
|---|---|---|
| `bert_base` | BERT-base | `./outputs/pi_models/bert_base` |
| `distilbert` | Distil-BERT | `./outputs/pi_models/distilbert` |
| `bert_medium` | BERT-Medium | `./outputs/pi_models/bert_medium` |
| `bert_mini` | BERT-Mini | `./outputs/pi_models/bert_mini` |
| `bert_tiny` | BERT-Tiny | `./outputs/pi_models/bert_tiny` |

The default folder structure is compatible with the combined PI fine-tuning script:

```text
outputs/pi_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

---

## 2. Required Input Data

The SHAP script expects a CSV file containing sentence/question pairs and a binary PI label.

Default column names:

| Column | Meaning |
|---|---|
| `question1` | First sentence/question |
| `question2` | Second sentence/question |
| `is_duplicate` | Gold label, where `1` means duplicate/paraphrase and `0` means non-duplicate/non-paraphrase |

Example input file:

```text
data/questions.csv
```

Recommended anonymized project structure:

```text
project/
├── data/
│   └── questions.csv
├── outputs/
│   └── pi_models/
│       ├── bert_base/
│       ├── distilbert/
│       ├── bert_medium/
│       ├── bert_mini/
│       └── bert_tiny/
├── run_qqp_shap_all_5_models_fast_anonymized.py
└── create_pi_shap_jaccard_table_anonymized.py
```

Avoid public documentation with absolute machine-specific paths. Use relative paths instead:

```text
./data/questions.csv
./outputs/pi_models
./outputs/pi_shap
```

---

## 3. Install Requirements

Install the required packages:

```bash
pip install torch transformers pandas numpy tqdm shap safetensors scikit-learn
```

If your environment already has PyTorch and Transformers installed, only install the missing packages.

---

## 4. Check Available Models

To list the model keys supported by the SHAP script:

```bash
python3 run_qqp_shap_all_5_models_fast_anonymized.py --list_models
```

To list the model keys supported by the Jaccard script:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py --list_models
```

---

## 5. Run SHAP for All Five Models

Run the SHAP extraction script using the default checkpoint folder structure:

```bash
python3 run_qqp_shap_all_5_models_fast_anonymized.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_shap \
  --checkpoint_format auto \
  --num_samples 15000 \
  --num_runs 3 \
  --max_len 128 \
  --pred_batch_size 512 \
  --shap_min_evals 100 \
  --shap_batch_size 512 \
  --save_every 100
```

This creates SHAP output files such as:

```text
outputs/pi_shap/
├── BERT-base/
│   ├── BERT-base_QQP_shap_run_1.csv
│   ├── BERT-base_QQP_shap_run_2.csv
│   └── BERT-base_QQP_shap_run_3.csv
├── Distil-BERT/
├── BERT-Medium/
├── BERT-Mini/
└── BERT-Tiny/
```

Each output CSV contains columns such as:

```text
model_name, run_id, sent1, sent2, true_out, d_words, nd_words, predicted_out, predicted_prob
```

where:

- `d_words` stores words important for the duplicate/paraphrase class.
- `nd_words` stores words important for the non-duplicate/non-paraphrase class.
- `true_out` is used later to choose the gold-label-based SHAP word list.

---

## 6. Run SHAP for Selected Models Only

To run only BERT-base, BERT-Mini, and BERT-Tiny:

```bash
python3 run_qqp_shap_all_5_models_fast_anonymized.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_shap \
  --models bert_base bert_mini bert_tiny \
  --checkpoint_format auto \
  --num_samples 15000 \
  --num_runs 3
```

---

## 7. If Your Fine-tuned Model Folders Have Different Names

Create a JSON file, for example `model_paths.json`:

```json
{
  "bert_base": "./outputs/pi_models/bert_base",
  "distilbert": "./outputs/pi_models/distilbert",
  "bert_medium": "./outputs/pi_models/bert_medium",
  "bert_mini": "./outputs/pi_models/bert_mini",
  "bert_tiny": "./outputs/pi_models/bert_tiny"
}
```

Then run:

```bash
python3 run_qqp_shap_all_5_models_fast_anonymized.py \
  --input_csv ./data/questions.csv \
  --model_paths_json ./model_paths.json \
  --output_dir ./outputs/pi_shap \
  --checkpoint_format auto \
  --num_samples 15000 \
  --num_runs 3
```

Do not include private local paths in `model_paths.json` if the file will be submitted for review.

---

## 8. Checkpoint Format Option

The SHAP script supports three checkpoint formats:

| Option | Use Case |
|---|---|
| `auto` | Recommended default. The script tries to infer the format. |
| `sequence_classification` | Use this for models saved with Hugging Face `AutoModelForSequenceClassification`. |
| `custom` | Use this for older checkpoints saved with a custom `MainModel` class. |

For models produced by the combined PI fine-tuning script, use:

```bash
--checkpoint_format auto
```

or explicitly:

```bash
--checkpoint_format sequence_classification
```

For older custom checkpoints, use:

```bash
--checkpoint_format custom
```

---

## 9. Compute SHAP Jaccard Similarity

After SHAP extraction is complete, run:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --num_runs 3 \
  --k 10
```

This computes the Jaccard similarity of the top-`K` SHAP words between different model pairs.

Default model pairs:

```text
BERT-base    vs Distil-BERT
BERT-base    vs BERT-Medium
BERT-base    vs BERT-Mini
BERT-base    vs BERT-Tiny
Distil-BERT  vs BERT-Medium
Distil-BERT  vs BERT-Mini
Distil-BERT  vs BERT-Tiny
BERT-Medium  vs BERT-Mini
BERT-Medium  vs BERT-Tiny
BERT-Tiny    vs BERT-Mini
```

---

## 10. Jaccard Output Files

The Jaccard script saves:

```text
outputs/pi_shap_jaccard_tables/
├── PI_SHAP_jaccard_run_1.csv
├── PI_SHAP_jaccard_run_2.csv
├── PI_SHAP_jaccard_run_3.csv
├── PI_SHAP_jaccard_summary_mean_variance.csv
├── PI_SHAP_jaccard_summary_mean_variance_rounded.csv
└── PI_SHAP_jaccard_summary_latex.txt
```

The rounded CSV is usually the most convenient file for directly preparing a paper table.

The LaTeX table file can be copied into a manuscript after checking formatting.

---

## 11. Run Jaccard for All Pairwise Combinations

To compute all pairwise combinations among the selected models:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --pair_mode all \
  --num_runs 3 \
  --k 10
```

---

## 12. Run Jaccard for Selected Models

Example using only BERT-base, BERT-Mini, and BERT-Tiny:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --models bert_base bert_mini bert_tiny \
  --pair_mode all \
  --num_runs 3 \
  --k 10
```

---

## 13. Token Filtering Options

By default, punctuation tokens are allowed and the SHAP placeholder token `[EMPTY]` is removed.

To keep only word-like tokens:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --keep_only_word_tokens \
  --num_runs 3 \
  --k 10
```

To keep the `[EMPTY]` token:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --keep_empty_token \
  --num_runs 3 \
  --k 10
```

---

## 14. Complete Example Workflow

Fine-tuned PI models are assumed to be available in:

```text
./outputs/pi_models
```

Input data is assumed to be available at:

```text
./data/questions.csv
```

Run SHAP:

```bash
python3 run_qqp_shap_all_5_models_fast_anonymized.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_shap \
  --checkpoint_format auto \
  --num_samples 15000 \
  --num_runs 3
```

Compute Jaccard similarity:

```bash
python3 create_pi_shap_jaccard_table_anonymized.py \
  --input_root ./outputs/pi_shap \
  --output_dir ./outputs/pi_shap_jaccard_tables \
  --num_runs 3 \
  --k 10
```
