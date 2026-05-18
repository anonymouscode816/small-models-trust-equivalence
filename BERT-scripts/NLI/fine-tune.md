# Fine-tuning Transformer Models on SNLI for Natural Language Inference

This repository contains a combined script for fine-tuning and testing multiple transformer-based models on the SNLI dataset for the Natural Language Inference (NLI) task.

The script supports five models in a single workflow:

| Model | Hugging Face Model Path |
|---|---|
| BERT-base | `bert-base-uncased` |
| DistilBERT | `distilbert/distilbert-base-uncased` |
| BERT-Medium | `google/bert_uncased_L-8_H-512_A-8` |
| BERT-Mini | `google/bert_uncased_L-4_H-256_A-4` |
| BERT-Tiny | `google/bert_uncased_L-2_H-128_A-2` |

---

## 1. Requirements

Install the required Python packages before running the script.

```bash
pip install torch transformers scikit-learn numpy tqdm
```

---

## 2. Dataset Preparation

Download and place the SNLI dataset files inside a local dataset directory.

The expected dataset structure is:

```text
snli_1.0/
├── snli_1.0_train.txt
├── snli_1.0_dev.txt
└── snli_1.0_test.txt
```

The script expects the dataset directory to be passed using the `--data_dir` argument.

Example:

```bash
--data_dir ./snli_1.0
```

Avoid using absolute system-specific paths in public code or documentation.

Recommended:

```text
./snli_1.0
./outputs/snli_nli_models
```

Not recommended:

```text
/home/username/project/...
/Users/username/Desktop/...
```

---

## 3. List Available Models

To check all supported models, run:

```bash
python3 finetune_snli_5models.py --list_models
```

Expected model keys:

```text
bert_base
distilbert
bert_medium
bert_mini
bert_tiny
```

---

## 4. Fine-tune and Test All Models

To fine-tune and test all five models in one run:

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

This will train and evaluate:

- BERT-base
- DistilBERT
- BERT-Medium
- BERT-Mini
- BERT-Tiny

Each model will be saved in a separate output folder.

---

## 5. Fine-tune and Test a Specific Model

Use the `--models` argument to run only selected models.

### BERT-base

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models bert_base \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

### DistilBERT

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models distilbert \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

### BERT-Medium

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models bert_medium \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

### BERT-Mini

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models bert_mini \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

### BERT-Tiny

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models bert_tiny \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

---

## 6. Run Multiple Selected Models

You can also run more than one selected model by passing multiple model keys.

Example:

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --models bert_base bert_mini bert_tiny \
  --mode train_test \
  --epochs 5 \
  --batch_size 32
```

---

## 7. Testing Only

If the models are already fine-tuned and saved, testing can be performed using:

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --mode test
```

---

## 8. Training Only

To fine-tune the models without running the final test step:

```bash
python3 finetune_snli_5models.py \
  --data_dir ./snli_1.0 \
  --output_dir ./outputs/snli_nli_models \
  --mode train \
  --epochs 5 \
  --batch_size 32
```

---

## 9. Output Directory

After training and testing, the output directory will contain model-specific folders such as:

```text
outputs/snli_nli_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

Each folder may contain:

- Fine-tuned model weights
- Tokenizer files
- Training logs
- Evaluation results

---
