# llm-highamp-benchmark (Protein Embeddings + Logistic Regression)
HighAMP-Bench: Benchmarking LLMs for Highly Active Antimicrobial Peptide

Benchmark multiple **protein language models (PLMs)** on a **binary classification** dataset by:
1) extracting fixed-length sequence embeddings, then  
2) training a **logistic regression (LR)** linear probe on top.

This repo is intended for **fair horizontal comparison** across ~10 models (e.g., **ESM-C**, **ESM2**, **ANKH**, **Mistral-Prot**, etc.). Even if some models underperform stronger task-specific baselines, the comparison is still useful for **model selection** and follow-up optimization.

---

## Repo layout

```text
.
├─ plm_lr_benchmark.py          # main benchmark script (embeddings -> LR -> metrics -> JSON)
├─ summarize_results.py         # aggregate results/*.json -> results/summary.(csv|xlsx)
├─ data.csv                     # (not included) your dataset
├─ cache/                       # cached embeddings (*.npy)
└─ results/                     # per-model JSON outputs + summary tables
````

---

## Dataset

### Expected input (CSV)

Your CSV must contain at least:

| column     | type | description                    |
| ---------- | ---- | ------------------------------ |
| `sequence` | str  | amino acid sequence            |
| `label` | int  | label (1=positive, 0=negative) |

Example:

```csv
sequence,positive
AAAKAA...,1
```

### Preprocessing in code

Before tokenization, sequences are normalized:

* `strip()` and `upper()`
* map rare amino acids: `U, Z, O, B -> X`

---

## Method

### 1) Embedding extraction (per model)

For each model, sequences are tokenized and fed through the model in inference mode (`torch.no_grad()`).

**Pooling (default):**

* mean pooling over valid (non-padding) tokens
* special tokens can be excluded via `special_tokens_mask` (when reliable)

**Model-specific routing (as implemented):**

* **ESM-C / ESM3**: uses `esm` SDK route (not pure Transformers)
* **ANKH**: encoder-only **T5EncoderModel** route (recommended FP32)
* **DPLM**: prefer `byprot` official route if installed; otherwise fallback to Transformers
* **Mistral-Prot**: often uses **max pooling** (model card example), but FP16 max-pool masking needs care

**Caching:**
Embeddings are cached per model:

* `cache/<safe_model_tag>.npy`

So repeated runs don’t recompute embeddings.

---

### 2) Train/test split

Uses stratified split:

* `train_test_split(..., test_size=0.2, seed=42, stratify=y)`

Split indices are saved to:

* `results/split_seed{seed}_test{test_size}.npz`

---

### 3) Linear probe classifier

A scikit-learn pipeline:

* `SimpleImputer(strategy="constant", fill_value=0.0)`
* `MinMaxScaler()`
* `LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")`

Decision threshold:

* `y_pred = (y_prob >= 0.5)`

---

### 4) Metrics

Computed on the **test split**:

* ACC
* Balanced Accuracy (BACC)
* Sensitivity / Recall (Sn)
* Specificity (Sp)
* Matthews Correlation Coefficient (MCC)
* AUROC (AUC)
* Average Precision (AP)

---

## Models

You can pass any Hugging Face model id via `--model_id`. The script also contains model-specific handling for:

### Transformers family

* `facebook/esm2_*` (ESM2)
* `Synthyra/ANKH_base` (ANKH, encoder-only T5)
* `RaphaelMourad/Mistral-Prot-v1-134M` (Mistral-Prot)
* `Rostlab/prot_bert`, `Rostlab/prot_t5_*` (often needs space-separated AAs)
* others as long as `AutoTokenizer` + `AutoModel` works

### ESM SDK family (`esm` library)

* `EvolutionaryScale/esmc-300m-2024-12`
* `EvolutionaryScale/esmc-600m-2024-12`
* `EvolutionaryScale/esm3-sm-open-v1` *(may require license / gated access)*

---

## Installation

### Option A (venv + pip)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip

# Core
pip install torch numpy pandas scikit-learn tqdm openpyxl

# Hugging Face
pip install transformers huggingface_hub

# Optional: ESM SDK (needed for ESM-C/ESM3 routes)
pip install esm

# Optional: accelerate (helps with some HF loading patterns)
pip install accelerate
```

### Option B (conda)

```bash
conda create -n plm-bench python=3.10 -y
conda activate plm-bench
pip install -U pip
pip install torch numpy pandas scikit-learn tqdm openpyxl transformers huggingface_hub
pip install esm accelerate
```

---

## Reproducibility

### Record your environment

Run:

```bash
python - << 'PY'
import platform, torch
import transformers, sklearn, pandas, numpy
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
PY
```

Freeze exact deps:

```bash
pip freeze > requirements_freeze.txt
```

### Hugging Face login (for gated/private models)

```bash
hf auth login
```

---

## Run the benchmark

### Single model

```bash
python -u plm_lr_benchmark.py ^
  --data_csv data.csv ^
  --seq_col sequence ^
  --label_col positive ^
  --model_id Synthyra/ANKH_base ^
  --batch_size 1 ^
  --max_length 256
```

### Another model (example: ESM2)

```bash
python -u plm_lr_benchmark.py ^
  --data_csv data.csv ^
  --seq_col sequence ^
  --label_col positive ^
  --model_id facebook/esm2_t33_650M_UR50D ^
  --batch_size 2 ^
  --max_length 256
```

> Tips:
>
> * Increase `--max_length` if your sequences are long (avoid truncation).
> * Start with a small `--batch_size` to avoid OOM, then scale up.

---

## Outputs

Each run writes:

* `cache/<model_tag>.npy` — cached embeddings
* `results/split_seed{seed}_test{test_size}.npz` — split indices
* `results/<model_tag>.json` — metrics and metadata

Example JSON:

```json
{
  "model_id": "facebook/esm2_t33_650M_UR50D",
  "n": 4546,
  "dim": 1280,
  "metrics": {
    "ACC": 0.55,
    "BACC": 0.54,
    "Sn": 0.60,
    "Sp": 0.48,
    "MCC": 0.08,
    "AUC": 0.55,
    "AP": 0.65
  },
  "split": {"method": "train_test_split", "test_size": 0.2, "seed": 42},
  "note": "..."
}
```

---

## Summarize results

Aggregate all `results/*.json` into a single table:

```bash
python summarize_results.py
```

Outputs:

* `results/summary.csv`
* `results/summary.xlsx`

Sorted by: `MCC` ↓, then `AP` ↓, then `AUC` ↓.

---

## Troubleshooting

### TensorFlow oneDNN message

If you see:
`To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0`

Windows PowerShell (current session):

```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
python -u plm_lr_benchmark.py ...
```

### FP16 overflow in masked max pooling (e.g., Mistral-Prot)

If you see errors like:
`value cannot be converted to type at::Half without overflow`

Fix: cast to float32 for the masked-fill / pooling step, then cast back if needed.

### Weird “collapsed” embeddings (many duplicate rows)

Sanity checks you already print are good:

* `X mean std over dims`
* `Unique rows`
  If uniqueness is very low, it’s usually tokenizer/pooling/precision.

---

## References (model cards / docs)

* ANKH_base: [https://huggingface.co/Synthyra/ANKH_base](https://huggingface.co/Synthyra/ANKH_base)
* Mistral-Prot-v1-134M: [https://huggingface.co/RaphaelMourad/Mistral-Prot-v1-134M](https://huggingface.co/RaphaelMourad/Mistral-Prot-v1-134M)
* ESM repo (repr-layers/include notes): [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)
* Hugging Face Transformers install: [https://huggingface.co/docs/transformers/en/installation](https://huggingface.co/docs/transformers/en/installation)
* Hugging Face Hub CLI (hf auth login): [https://huggingface.co/docs/huggingface_hub/main/guides/cli](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
* PyTorch `no_grad`: [https://docs.pytorch.org/docs/stable/generated/torch.no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad)
* scikit-learn `train_test_split`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* scikit-learn `LogisticRegression`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---
