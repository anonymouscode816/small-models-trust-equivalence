# PI LIME Explanation and Jaccard Similarity Scripts

This folder contains anonymized scripts for running LIME-based interpretability analysis on Paraphrase Identification (PI) models and computing pairwise Jaccard similarity between the important words identified for different model pairs.

The scripts avoid hard-coded usernames, machine-specific directories, private repository links, cloud links, and private keys. All paths are supplied through command-line arguments.

## Files

| File | Purpose |
|---|---|
| `run_qqp_lime_all_5_models_fast.py` | Runs LIME explanations for all five PI models or a selected subset. |
| `create_pi_lime_jaccard_table.py` | Computes pairwise Jaccard similarity from the LIME output CSV files. |

## Supported Models

| Model key | Model name |
|---|---|
| `bert_base` | BERT-base |
| `distilbert` | DistilBERT |
| `bert_medium` | BERT-Medium |
| `bert_mini` | BERT-Mini |
| `bert_tiny` | BERT-Tiny |

The script expects fine-tuned checkpoints to be saved in separate folders under one checkpoint root directory.

Example checkpoint layout:

```text
outputs/pi_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

## Requirements

Install the required packages:

```bash
pip install torch transformers pandas numpy tqdm lime safetensors
```

If your environment already has `torch` installed with the correct CUDA version, do not reinstall it blindly. Install only the missing packages.

## Input Data Format

By default, the LIME script expects a CSV file with the following columns:

```text
question1, question2, is_duplicate
```

Example layout:

```text
data/
└── questions.csv
```

The label column should contain binary labels:

```text
0 = not duplicate
1 = duplicate
```

To use different column names, pass:

```bash
--question1_col <column_name_for_first_sentence>
--question2_col <column_name_for_second_sentence>
--label_col <label_column_name>
```

## 1. List Available Models

```bash
python3 run_qqp_lime_all_5_models_fast.py --list_models
```

## 2. Run LIME for All Five Models

```bash
python3 run_qqp_lime_all_5_models_fast.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_lime \
  --num_samples 15000 \
  --num_runs 3 \
  --lime_num_samples 100 \
  --lime_num_features 20 \
  --pred_batch_size 512
```

This will create one folder per model:

```text
outputs/pi_lime/
├── BERT-base/
├── Distil-BERT/
├── BERT-Medium/
├── BERT-Mini/
└── BERT-Tiny/
```

Each folder will contain files such as:

```text
BERT-base_QQP_lime_run_1.csv
BERT-base_QQP_lime_run_2.csv
BERT-base_QQP_lime_run_3.csv
```

## 3. Run LIME for Selected Models

Example: run only BERT-base, BERT-Mini, and BERT-Tiny.

```bash
python3 run_qqp_lime_all_5_models_fast.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_lime \
  --models bert_base bert_mini bert_tiny \
  --num_samples 15000 \
  --num_runs 3
```

## 4. Use a JSON File for Checkpoint Paths

If your checkpoint folders are not under a single root directory, create a JSON file such as:

```json
{
  "bert_base": "./checkpoints/pi/bert_base",
  "distilbert": "./checkpoints/pi/distilbert",
  "bert_medium": "./checkpoints/pi/bert_medium",
  "bert_mini": "./checkpoints/pi/bert_mini",
  "bert_tiny": "./checkpoints/pi/bert_tiny"
}
```

Then run:

```bash
python3 run_qqp_lime_all_5_models_fast.py \
  --input_csv ./data/questions.csv \
  --model_paths_json ./model_paths.json \
  --output_dir ./outputs/pi_lime \
  --num_samples 15000 \
  --num_runs 3
```

Use relative paths in the JSON file for anonymous submission.

## 5. Checkpoint Format

For models saved using Hugging Face `AutoModelForSequenceClassification`, use the default:

```bash
--checkpoint_format auto
```

For older custom checkpoints, use:

```bash
--checkpoint_format custom
```

For standard Hugging Face sequence-classification checkpoints, use:

```bash
--checkpoint_format sequence_classification
```

The default `auto` option tries to infer the correct format.

## 6. Compute LIME Jaccard Similarity

After generating LIME outputs for all required models and runs, compute the pairwise Jaccard tables:

```bash
python3 create_pi_lime_jaccard_table.py \
  --input_root ./outputs/pi_lime \
  --output_dir ./outputs/pi_lime_jaccard_tables \
  --num_runs 3 \
  --k 10
```

This creates:

```text
outputs/pi_lime_jaccard_tables/
├── PI_LIME_jaccard_run_1.csv
├── PI_LIME_jaccard_run_2.csv
├── PI_LIME_jaccard_run_3.csv
├── PI_LIME_jaccard_all_runs_long.csv
├── PI_LIME_jaccard_summary_mean_variance.csv
├── PI_LIME_jaccard_summary_mean_variance_rounded.csv
└── PI_LIME_jaccard_summary_latex.txt
```

## 7. Compute Jaccard for Selected Models

Example:

```bash
python3 create_pi_lime_jaccard_table.py \
  --input_root ./outputs/pi_lime \
  --output_dir ./outputs/pi_lime_jaccard_tables \
  --models bert_base bert_mini bert_tiny \
  --num_runs 3 \
  --k 10
```

Only model pairs available within the selected set will be computed.

## 8. Meaning of `K`

The argument `--k 10` means that the top 10 gold-label-relevant LIME words are used for each example.

For each sample:

- if `true_out = 1`, the script uses `d_words`, i.e., words supporting the duplicate class;
- if `true_out = 0`, the script uses `nd_words`, i.e., words supporting the non-duplicate class.

Then the average Jaccard similarity is computed across aligned samples for each model pair.

## 9. Anonymous Submission Notes

For anonymous or double-blind review:

- Use relative paths such as `./data/questions.csv` and `./outputs/pi_lime`.
- Do not include usernames in paths.
- Do not include institution-specific directory names.
- Do not include private GitHub, Google Drive, Dropbox, or server links.
- Do not include API keys, access tokens, or Hugging Face tokens.
- Keep checkpoint folders and output folders neutral.

Recommended command style:

```bash
python3 run_qqp_lime_all_5_models_fast.py \
  --input_csv ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_lime
```
