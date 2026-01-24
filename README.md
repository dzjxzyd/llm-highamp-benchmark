# llm-highamp-benchmark
HighAMP-Bench: Benchmarking LLMs for Highly Active Antimicrobial Peptide

# Protein Language Model Benchmark (Embeddings + Logistic Regression)

Benchmark multiple protein language models (PLMs) on a binary classification dataset by:
1) extracting fixed-length sequence embeddings, then
2) training a Logistic Regression (LR) classifier on top.

This repo is designed for **fast, apples-to-apples horizontal comparison** across ~10 PLMs (e.g., ESM-C, ESM2, ANKH, Mistral-Prot, etc.). Even if some models underperform a stronger task-specific baseline (e.g., AMP Classifier), the comparison is still valuable for model selection and future improvements.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Method](#method)
- [Models Supported](#models-supported)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Run a Benchmark](#run-a-benchmark)
- [Outputs](#outputs)
- [Summarize Results](#summarize-results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

### What this repo does
For each PLM:
- Load model & tokenizer
- Convert protein sequences -> embeddings (cached to disk)
- Split data into train/test with stratification
- Fit a scikit-learn LR classifier on embeddings
- Report metrics (ACC, BACC, Sn, Sp, MCC, AUC, AP)
- Save results as JSON for later aggregation

### Why LR?
LR is simple, strong enough as a linear probe, easy to reproduce, and makes the embedding quality differences easy to interpret.

---

## Dataset

### Input format (CSV)
A single CSV file is expected (e.g., `data.csv`) with at least:

| column      | type | description |
|-------------|------|-------------|
| `sequence`  | str  | amino acid sequence (protein/peptide) |
| `positive`  | int  | binary label: 1=positive, 0=negative |

Example row:
```csv
sequence,positive
AAAKAA...,1

Sequence cleaning

Before tokenization, sequences are normalized:

strip & uppercase

map rare amino acids U, Z, O, B -> X

Note: You should ensure the sequences are valid for your chosen tokenizer/model (some models expect extra structure tokens, see Models Supported
).

Dataset statistics (fill in if you plan to publish)

Total samples: [TODO]

Positive rate: [TODO]

Sequence length: min/median/95%/max [TODO]

Source / license: [TODO]

Method
1) Embedding extraction

By default:

Tokenize with Hugging Face tokenizer (or esm SDK for ESM-C/ESM3)

Compute per-sequence embedding by pooling per-token hidden states:

Mean pooling over non-padding tokens (default for most models)

Optional max pooling for models where the model card suggests it (e.g., Mistral-Prot)

Implementation notes:

Embeddings are cached under cache/<safe_model_id>.npy

Embedding functions run under torch.no_grad() for efficiency

2) Train/test split

Stratified train_test_split

Default: test_size=0.2, seed=42

A split file is saved so the same split can be reused.

3) Classifier

scikit-learn Pipeline:

SimpleImputer(strategy="constant", fill_value=0.0)

MinMaxScaler()

LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")

Threshold:

y_pred = (y_prob >= 0.5)

4) Metrics

Reported on test split:

ACC

Balanced Accuracy (BACC)

Sensitivity / Recall (Sn)

Specificity (Sp)

Matthews Correlation Coefficient (MCC)

AUROC (AUC)

Average Precision (AP)

Models Supported

You can pass any Hugging Face model id via --model_id, but this repo includes special handling for several families:

Transformers-based PLMs (Hugging Face)

facebook/esm2_t33_650M_UR50D (ESM2)

Synthyra/ANKH_base (ANKH, encoder-only T5)

RaphaelMourad/Mistral-Prot-v1-134M (Mistral-Prot)

yarongef/DistilProtBert

Rostlab/prot_bert / Rostlab/prot_t5_* (space-separated tokenization)

ESM SDK-based models (esm library)

EvolutionaryScale/esmc-300m-2024-12

EvolutionaryScale/esmc-600m-2024-12

EvolutionaryScale/esm3-sm-open-v1 (may require license acceptance / gated access)

Model-specific caveats

ANKH: encoder-only; load with T5EncoderModel; recommended tensor type F32 (more stable).

Mistral-Prot: model card expects embedding hidden size 256; requires Transformers ≥ 4.34.0.

ESM: ESM authors explicitly warn not to use BOS embedding for pretrained models; prefer mean/per_tok pooling.

Installation
Option A: pip (recommended)
Create an environment:

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1


Install dependencies:

pip install -U pip
pip install torch
pip install transformers scikit-learn pandas numpy tqdm openpyxl
pip install huggingface_hub


Optional extras:

# For ESM-C / ESM3 (if you use those routes)
pip install esm

# For faster downloads / better device placement (optional)
pip install accelerate

Option B: conda
conda create -n plm-bench python=3.10 -y
conda activate plm-bench
pip install torch transformers scikit-learn pandas numpy tqdm openpyxl huggingface_hub

Reproducibility
1) Record environment

Run and save output in your paper/notes:

python -c "import platform, torch, transformers, sklearn; \
print('python', platform.python_version()); \
print('torch', torch.__version__); \
print('cuda', torch.version.cuda); \
print('transformers', transformers.__version__); \
print('sklearn', sklearn.__version__)"


Also save:

pip freeze > requirements_freeze.txt

2) Hugging Face login (for gated/private models)
hf auth login
# or legacy:
huggingface-cli login

3) Cache & offline mode

Default cache is ~/.cache/huggingface/hub (Windows: C:\Users\<you>\.cache\huggingface\hub)

Change cache:

# Linux/Mac
export HF_HUB_CACHE=/path/to/hf_cache
# Windows PowerShell
setx HF_HUB_CACHE "D:\hf_cache"


Offline mode (requires models already cached):

# Linux/Mac
export HF_HUB_OFFLINE=1
# Windows PowerShell
setx HF_HUB_OFFLINE 1

4) Determinism notes

Train/test split controlled by --seed

GPU floating point and some ops may still introduce minor nondeterminism.

For strict determinism, consider running embeddings on CPU (slower) and pin versions.

Run a Benchmark

Basic usage:

python -u plm_lr_benchmark.py \
  --data_csv data.csv \
  --seq_col sequence \
  --label_col positive \
  --model_id facebook/esm2_t33_650M_UR50D \
  --batch_size 2 \
  --max_length 256


Recommended (GPU):

Start with smaller batch_size and increase until VRAM is stable.

Increase --max_length if your sequences are long (avoid heavy truncation).

Run multiple models (example):

python -u plm_lr_benchmark.py --data_csv data.csv --seq_col sequence --label_col positive --model_id Synthyra/ANKH_base --batch_size 1 --max_length 256
python -u plm_lr_benchmark.py --data_csv data.csv --seq_col sequence --label_col positive --model_id facebook/esm2_t33_650M_UR50D --batch_size 2 --max_length 512
python -u plm_lr_benchmark.py --data_csv data.csv --seq_col sequence --label_col positive --model_id EvolutionaryScale/esmc-300m-2024-12 --batch_size 1 --max_length 256

Outputs

After each run, files are written to:

cache/<model_tag>.npy — cached embeddings

results/split_seed{seed}_test{test_size}.npz — indices for reproducible split

results/<model_tag>.json — metrics and metadata

Example JSON structure:

{
  "model_id": "facebook/esm2_t33_650M_UR50D",
  "n": 4546,
  "dim": 1280,
  "metrics": {"ACC":0.55,"BACC":0.54,"Sn":0.60,"Sp":0.48,"MCC":0.08,"AUC":0.55,"AP":0.65},
  "split": {"method":"train_test_split","test_size":0.2,"seed":42},
  "note": "..."
}

Summarize Results

Aggregate all results/*.json into a single table:

python summarize_results.py


Outputs:

results/summary.csv

results/summary.xlsx

By default, rows are sorted by MCC, then AP, then AUC.

Troubleshooting
“oneDNN custom operations are on …”

This comes from TensorFlow logs (some environments import TF indirectly). To suppress:

# Windows PowerShell (current session)
$env:TF_ENABLE_ONEDNN_OPTS="0"
python -u plm_lr_benchmark.py ...

Mistral-Prot overflow with FP16 max pooling

Use BF16/FP32 for pooling or cast to float32 before masked fill.

ESM2 performance seems unexpectedly low

Common causes:

Heavy truncation by --max_length

Pooling choice (mean vs pooler_output vs layer selection)

Dataset mismatch (peptides vs full proteins)
