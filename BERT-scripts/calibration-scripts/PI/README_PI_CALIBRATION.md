# PI Model Calibration for Five BERT-family Models

This folder contains an anonymized script for running calibration analysis on five transformer models for the Paraphrase Identification (PI) task, such as QQP or PAWS.

The script evaluates model confidence and saves calibration metrics, reliability diagrams, probability files, and a combined summary table.

## Files

```text
calibrate_pi_5models.py
README_PI_CALIBRATION.md
```

## Supported models

| Model key | Hugging Face model |
|---|---|
| `bert_base` | `bert-base-uncased` |
| `distilbert` | `distilbert/distilbert-base-uncased` |
| `bert_medium` | `google/bert_uncased_L-8_H-512_A-8` |
| `bert_mini` | `google/bert_uncased_L-4_H-256_A-4` |
| `bert_tiny` | `google/bert_uncased_L-2_H-128_A-2` |

## Requirements

Install the required packages:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib tqdm
```

## Expected checkpoint layout

The default checkpoint root is:

```text
./outputs/pi_models
```

The script expects fine-tuned models to be saved as Hugging Face compatible checkpoints:

```text
outputs/pi_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

Each model folder should contain files such as:

```text
config.json
pytorch_model.bin or model.safetensors
tokenizer.json / vocab.txt
```

This layout matches the output of the combined PI fine-tuning script.

## Dataset format

### QQP format

For QQP-style data, the CSV file should contain:

```text
question1, question2, is_duplicate
```

Example path:

```text
./data/questions.csv
```

### PAWS format

For PAWS-style data, the TSV file should contain:

```text
sentence1, sentence2, label
```

Example path:

```text
./data/paws_test.tsv
```

### Custom format

For a custom CSV/TSV file, pass the column names explicitly:

```bash
--sentence1_column sentence1 \
--sentence2_column sentence2 \
--label_column label
```

Labels must be binary: `0`/`1`, where `0` means not paraphrase and `1` means paraphrase.

## List available models

```bash
python3 calibrate_pi_5models.py --list_models
```

## Run calibration for all five models on QQP

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --test_file ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models all \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile \
  --save_probabilities
```

## Run calibration for one model

Example for BERT-base:

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --test_file ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models bert_base \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile \
  --save_probabilities
```

Example for BERT-Mini:

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --test_file ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models bert_mini \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile
```

## Run calibration for selected models

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type qqp \
  --test_file ./data/questions.csv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models bert_base bert_mini bert_tiny \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile
```

## Run calibration on PAWS

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type paws \
  --test_file ./data/paws_test.tsv \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration_paws \
  --models all \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile
```

## Run calibration with a custom CSV file

```bash
python3 calibrate_pi_5models.py \
  --input_format raw \
  --dataset_type custom \
  --test_file ./data/test.csv \
  --sentence1_column sentence1 \
  --sentence2_column sentence2 \
  --label_column label \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models all
```

## Optional pickle input

If the evaluation split is stored as `test.pkl`, use:

```bash
python3 calibrate_pi_5models.py \
  --input_format pickle \
  --data_dir ./dataset \
  --checkpoint_root ./outputs/pi_models \
  --output_dir ./outputs/pi_calibration \
  --models all
```

Raw CSV/TSV input is recommended for reproducibility and model-tokenizer compatibility.

## Output files

For each model, the script creates:

```text
outputs/pi_calibration/<model_key>/
├── calibration_metrics.json
├── calibration_bins.csv
├── prediction_probabilities.csv
└── reliability_diagram.png
```

At the top level, it also creates:

```text
outputs/pi_calibration/
├── summary.json
├── summary.csv
└── calibration_comparison.png
```

## Metrics saved

The script saves:

- Accuracy
- Expected Calibration Error, ECE
- Maximum Calibration Error, MCE
- Brier score
- Log loss
- Bin-wise accuracy
- Bin-wise confidence
- Reliability diagrams

## Notes for anonymized submission

The script and README use only relative paths, for example:

```text
./data/questions.csv
./outputs/pi_models
./outputs/pi_calibration
```

Avoid adding absolute paths such as:

```text
/home/username/...
/Users/username/...
/private/server/path/...
```

Also avoid including private GitHub links, Google Drive links, usernames, institutional cluster paths, or API/private keys in public or review-submission files.
