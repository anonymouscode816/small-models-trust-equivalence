# NLI SHAP Explanation and Jaccard Similarity Workflow

This folder contains anonymized scripts for running SHAP explanations on five BERT-family models fine-tuned for the SNLI/NLI task and for computing pairwise Jaccard similarity between the important tokens selected by different models.

No private usernames, local machine paths, private repository links, or keys are hard-coded in the scripts. All dataset, checkpoint, and output locations are passed through command-line arguments.

---

## 1. Supported Models

| Model key | Model | Hugging Face base model |
|---|---|---|
| `bert_base` | BERT-base | `bert-base-uncased` |
| `distilbert` | DistilBERT | `distilbert/distilbert-base-uncased` |
| `bert_medium` | BERT-Medium | `google/bert_uncased_L-8_H-512_A-8` |
| `bert_mini` | BERT-Mini | `google/bert_uncased_L-4_H-256_A-4` |
| `bert_tiny` | BERT-Tiny | `google/bert_uncased_L-2_H-128_A-2` |

---

## 2. Install Requirements

```bash
pip install torch transformers pandas numpy tqdm shap scikit-learn
```

If your checkpoints are saved as `.safetensors`, also install:

```bash
pip install safetensors
```

---

## 3. Expected Dataset Layout

Place the SNLI test file in a neutral relative directory, for example:

```text
snli_1.0/
└── snli_1.0_test.txt
```

The script also supports a simple three-column tab-separated format:

```text
label<TAB>sentence1<TAB>sentence2
```

Valid labels are:

```text
contradiction
neutral
entailment
```

---

## 4. Expected Checkpoint Layout

By default, the script expects one checkpoint folder per model under `./outputs/snli_nli_models`:

```text
outputs/snli_nli_models/
├── bert_base/
├── distilbert/
├── bert_medium/
├── bert_mini/
└── bert_tiny/
```

Each checkpoint folder should contain the saved model file, tokenizer files, and preferably `metadata.json` from the fine-tuning script.

Use neutral relative paths in public documentation:

```text
./snli_1.0/snli_1.0_test.txt
./outputs/snli_nli_models
./outputs/nli_shap
```

Do not include private absolute paths such as:

```text
/home/username/...
/Users/username/...
```

---

## 5. Run SHAP for All Five Models

```bash
python3 run_bert_nli_shap_all_5_models_fast.py \
  --snli_test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_shap \
  --num_runs 3 \
  --max_len 256 \
  --predict_batch_size 256 \
  --shap_max_evals 500
```

Use `--shap_max_evals 0` to let SHAP choose the default number of evaluations.

The output will be saved as:

```text
outputs/nli_shap/
├── bert_base/run_01/bert_base_nli_shap_run_01.csv
├── distilbert/run_01/distilbert_nli_shap_run_01.csv
├── bert_medium/run_01/bert_medium_nli_shap_run_01.csv
├── bert_mini/run_01/bert_mini_nli_shap_run_01.csv
└── bert_tiny/run_01/bert_tiny_nli_shap_run_01.csv
```

For a quick debug run, use a small number of samples:

```bash
python3 run_bert_nli_shap_all_5_models_fast.py \
  --snli_test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_shap_debug \
  --num_runs 1 \
  --limit_samples 20 \
  --shap_max_evals 100
```

---

## 6. Run SHAP for Selected Models

Example: run only BERT-base, BERT-Mini, and BERT-Tiny.

```bash
python3 run_bert_nli_shap_all_5_models_fast.py \
  --snli_test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_root ./outputs/snli_nli_models \
  --output_dir ./outputs/nli_shap \
  --models bert_base bert_mini bert_tiny \
  --num_runs 3 \
  --shap_max_evals 500
```

---

## 7. Optional: Use a Checkpoint Map

If your checkpoint folders do not follow the default names, create a JSON file such as `nli_checkpoint_map.json`:

```json
{
  "bert_base": "./checkpoints/nli/bert_base",
  "distilbert": "./checkpoints/nli/distilbert",
  "bert_medium": "./checkpoints/nli/bert_medium",
  "bert_mini": "./checkpoints/nli/bert_mini",
  "bert_tiny": "./checkpoints/nli/bert_tiny"
}
```

Then run:

```bash
python3 run_bert_nli_shap_all_5_models_fast.py \
  --snli_test_file ./snli_1.0/snli_1.0_test.txt \
  --checkpoint_map ./nli_checkpoint_map.json \
  --output_dir ./outputs/nli_shap \
  --num_runs 3 \
  --shap_max_evals 500
```

---

## 8. Compute SHAP-Based Jaccard Similarity Matrix

After generating the SHAP CSV files, compute pairwise Jaccard similarity using the top-K important tokens.

```bash
python3 create_nli_shap_jaccard_table.py \
  --input_root ./outputs/nli_shap \
  --output_dir ./outputs/nli_shap_jaccard_tables \
  --num_runs 3 \
  --k 10 \
  --label_source true
```

The default `--label_source true` selects the explanation column corresponding to the gold NLI label. To use the predicted class instead, run:

```bash
python3 create_nli_shap_jaccard_table.py \
  --input_root ./outputs/nli_shap \
  --output_dir ./outputs/nli_shap_jaccard_tables_predicted \
  --num_runs 3 \
  --k 10 \
  --label_source predicted
```

---

## 9. Jaccard Outputs

The Jaccard script saves:

```text
outputs/nli_shap_jaccard_tables/
├── NLI_SHAP_jaccard_run_1.csv
├── NLI_SHAP_jaccard_run_2.csv
├── NLI_SHAP_jaccard_run_3.csv
├── NLI_SHAP_jaccard_all_runs_long.csv
├── NLI_SHAP_jaccard_summary_mean_variance.csv
├── NLI_SHAP_jaccard_summary_mean_variance_rounded.csv
└── NLI_SHAP_jaccard_summary_latex.txt
```

---

## 10. Files

| File | Purpose |
|---|---|
| `run_bert_nli_shap_all_5_models_fast.py` | Runs SHAP explanations for all or selected NLI models |
| `create_nli_shap_jaccard_table.py` | Computes pairwise Jaccard similarity from SHAP outputs |
| `README_NLI_SHAP.md` | Instructions for running the SHAP workflow |
