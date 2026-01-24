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
