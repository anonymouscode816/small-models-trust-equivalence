# NLI Model Calibration for Five BERT-family Models

This document explains how to run calibration analysis for five fine-tuned NLI models on the SNLI test set.

The Python script is anonymized and uses command-line arguments instead of hard-coded private paths, usernames, server paths, or repository links.

## Supported models

| Model key | Hugging Face base model |
|---|---|
| `bert_base` | `bert-base-uncased` |
| `distilbert` | `distilbert/distilbert-base-uncased` |
| `bert_medium` | `google/bert_uncased_L-8_H-512_A-8` |
| `bert_mini` | `google/bert_uncased_L-4_H-256_A-4` |
| `bert_tiny` | `google/bert_uncased_L-2_H-128_A-2` |

## Files

```text
calibrate_nli_5models.py
README_NLI_CALIBRATION.md
```

## Requirements

Install the required Python packages:

```bash
pip install torch transformers scikit-learn numpy matplotlib tqdm
```

## Expected dataset layout

Place the SNLI dataset in a relative directory such as:

```text
snli_1.0/
├── snli_1.0_train.txt
├── snli_1.0_dev.txt
└── snli_1.0_test.txt
```

The calibration script uses the test file by default:

```text
./snli_1.0/snli_1.0_test.txt
```

## Expected checkpoint layout

By default, the script expects fine-tuned checkpoints in this structure:

```text
outputs/snli_nli_models/
├── bert_base/
│   └── checkpoint_best/
├── distilbert/
│   └── checkpoint_best/
├── bert_medium/
│   └── checkpoint_best/
├── bert_mini/
│   └── checkpoint_best/
└── bert_tiny/
    └── checkpoint_best/
```

Each `checkpoint_best` directory should contain:

```text
pytorch_model.bin
metadata.json
config/tokenizer files
```

The `metadata.json` file is recommended because it stores the original Hugging Face model identifier and maximum sequence length. If it is not present, the script falls back to the default model registry.

## List available models

```bash
python3 calibrate_nli_5models.py --list_models
```

## Run calibration for all five models

```bash
python3 calibrate_nli_5models.py \
  --test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_calibration \
  --models all \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile \
  --save_probabilities
```

## Run calibration for a single model

Example for BERT-base:

```bash
python3 calibrate_nli_5models.py \
  --test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_calibration \
  --models bert_base \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile \
  --save_probabilities
```

Example for BERT-Mini:

```bash
python3 calibrate_nli_5models.py \
  --test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_calibration \
  --models bert_mini \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile
```

## Run selected models only

```bash
python3 calibrate_nli_5models.py \
  --test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_calibration \
  --models bert_base bert_mini bert_tiny \
  --batch_size 512 \
  --num_bins 10 \
  --binning quantile
```

## Use explicit checkpoint paths

If your checkpoint folders do not follow the default structure, pass explicit checkpoint paths using model-specific arguments.

Example:

```bash
python3 calibrate_nli_5models.py \
  --test_file ./snli_1.0/snli_1.0_test.txt \
  --output_dir ./outputs/nli_calibration \
  --models bert_base distilbert \
  --checkpoint_path_bert_base ./checkpoints/nli/bert_base/checkpoint_best \
  --checkpoint_path_distilbert ./checkpoints/nli/distilbert/checkpoint_best
```

## Output files

For each model, the script creates a separate folder:

```text
outputs/nli_calibration/
├── bert_base/
│   ├── calibration_metrics.json
│   ├── calibration_bins.csv
│   ├── prediction_probabilities.csv
│   └── reliability_diagram.png
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

The top-level output directory also contains:

```text
summary.json
summary.csv
```

## Reported metrics

The script reports:

- Accuracy
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Multiclass Brier score
- Log loss
- Inference time
- Per-bin accuracy and confidence
- Reliability diagram

## Binning options

Two binning strategies are supported:

```bash
--binning quantile
```

or

```bash
--binning fixed
```

`quantile` binning creates bins with approximately similar numbers of examples. `fixed` binning uses equal-width confidence intervals.
