# Fine-tuning Transformer Models for Paraphrase Identification

This repository contains a combined script for fine-tuning and testing multiple transformer-based models on the Paraphrase Identification (PI) task.

The script can be used with QQP, PAWS, or a custom sentence-pair dataset with binary labels.

---

## Supported Models

| Model | Model Key | Hugging Face Path |
|---|---|---|
| BERT-base | `bert_base` | `bert-base-uncased` |
| DistilBERT | `distilbert` | `distilbert/distilbert-base-uncased` |
| BERT-Medium | `bert_medium` | `google/bert_uncased_L-8_H-512_A-8` |
| BERT-Mini | `bert_mini` | `google/bert_uncased_L-4_H-256_A-4` |
| BERT-Tiny | `bert_tiny` | `google/bert_uncased_L-2_H-128_A-2` |

---

## Requirements

Install the required packages:

```bash
pip install torch transformers pandas numpy tqdm
```

If a `requirements.txt` file is provided, you can also use:

```bash
pip install -r requirements.txt
```

---

## Script Name

Use the following combined script:

```text
finetune_pi_5models.py
```

This script supports both training and testing.

---

## Option 1: Run with Existing Pickle Files

Use this option if your dataset has already been preprocessed into pickle files similar to the original BERT-base PI code.

Expected directory structure:

```text
dataset/
├── train.pkl
├── val.pkl
└── test.pkl
```

Run all five models:

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

This will fine-tune and test:

- BERT-base
- DistilBERT
- BERT-Medium
- BERT-Mini
- BERT-Tiny

---

## Option 2: Run with Raw QQP Files

Use this option if you have QQP-style CSV files.

Expected columns:

```text
question1, question2, is_duplicate
```

Example directory structure:

```text
QQP/
├── train.csv
├── val.csv
└── test.csv
```

Run all five models:

```bash
python3 finetune_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --data_dir ./QQP \
  --output_dir ./outputs/pi_models \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

If your file names are different, pass them explicitly:

```bash
python3 finetune_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --train_file ./QQP/train.csv \
  --val_file ./QQP/val.csv \
  --test_file ./QQP/test.csv \
  --output_dir ./outputs/pi_models \
  --mode train_test
```

---

## Option 3: Run with Raw PAWS Files

Use this option if you have PAWS-style TSV files.

Expected columns:

```text
sentence1, sentence2, label
```

Example directory structure:

```text
PAWS/
├── train.tsv
├── dev.tsv
└── test.tsv
```

Run all five models:

```bash
python3 finetune_pi_5models.py \
  --input_format raw \
  --dataset_type paws \
  --data_dir ./PAWS \
  --output_dir ./outputs/pi_models \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

---

## Run a Specific Model

Use the `--models` argument to run only selected models.

For example, run only BERT-base:

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --models bert_base \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

Run only BERT-Mini:

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --models bert_mini \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

Run multiple selected models:

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --models bert_base bert_mini bert_tiny \
  --mode train_test \
  --epochs 10 \
  --batch_size 32
```

---

## List Available Models

To print the supported model keys:

```bash
python3 finetune_pi_5models.py --list_models
```

Expected keys:

```text
bert_base
distilbert
bert_medium
bert_mini
bert_tiny
```

---

## Training Only

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --mode train \
  --epochs 10 \
  --batch_size 32
```

---

## Testing Only

Use this after the models have already been fine-tuned and saved.

```bash
python3 finetune_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --output_dir ./outputs/pi_models \
  --mode test \
  --batch_size 32
```

---

## Output Directory

After training and testing, the output directory will contain separate folders for each model:

```text
outputs/pi_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
├── bert_tiny/
├── predictions/
└── combined_results.json
```

Each model folder contains the saved fine-tuned model, tokenizer, and metric files.

Prediction files are saved in:

```text
outputs/pi_models/predictions/
```

Example prediction files:

```text
test_pred_bert_base.txt
test_prob_bert_base.txt
test_labels_bert_base.txt
```

---

## Custom Dataset Columns

For a custom CSV or TSV dataset, use:

```bash
python3 finetune_pi_5models.py \
  --input_format raw \
  --dataset_type custom \
  --train_file ./data/train.csv \
  --val_file ./data/val.csv \
  --test_file ./data/test.csv \
  --sentence1_col sentence_a \
  --sentence2_col sentence_b \
  --label_col label \
  --output_dir ./outputs/pi_models \
  --mode train_test
```

The label column must contain binary labels:

```text
0 = not paraphrase
1 = paraphrase
```

---
