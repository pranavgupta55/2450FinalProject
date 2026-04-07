# Project Architecture

This repository is organized as a lightweight experimentation stack for predicting significant absolute alpha events.

## Goal

Predict whether a firm's `T+5` return deviates from the S&P 500 by more than `1%` in absolute value.

## Current Modeling Setup

The current structured dataset comes from `data/*/weekly_event_dataset.csv`. Each row is a `(ticker, week)` event record that combines:

- price and volume features
- SEC 8-K coverage counts
- Finnhub news counts
- Yahoo Finance RSS fallback counts
- future alpha label

## Split Strategy

To avoid leakage, the project uses a chronological split by `week_start`.

- Earlier weeks go to train
- Later weeks go to test
- No week appears in both splits
- All preprocessing statistics are fit only on the training portion

This is safer than a random row split because the target is based on future returns.

## Directory Layout

- `data/`: raw and generated datasets
- `docs/`: design notes and architecture plans
- `scripts/`: entry points for preparing splits and training models
- `src/alpha_signal/`: shared project code
- `dashboard/`: experiment visualization app
- `artifacts/`: generated train/test splits, trained models, metrics, and predictions

## Shared Code Modules

- `src/alpha_signal/data/dataset.py`
  Loads `weekly_event_dataset` and engineers structured features.
- `src/alpha_signal/data/splitting.py`
  Builds deterministic chronological train/test splits and saves metadata.
- `src/alpha_signal/features/tabular.py`
  Fits train-only imputations and categorical encodings for tabular models.
- `src/alpha_signal/models/training.py`
  Runs a common train/evaluate/save workflow for model experiments.
- `src/alpha_signal/evaluation/metrics.py`
  Computes F1, precision, recall, ROC-AUC, PR-AUC, and confusion counts.

## Model Roadmap

### Implemented Now

- `Random Forest`
  Baseline tree ensemble over structured weekly features
- `XGBoost`
  Stronger boosted-tree baseline over the same structured features
  Note: on macOS, the local runtime may require `libomp` for XGBoost to load.

### Planned Next

- `Logistic Regression`
  Technical-only baseline for comparison against tree models
- `FinBERT + Similarity Features`
  Event embeddings and nearest-neighbor signal features
- `Multimodal Attention Model`
  Joint numerical + embedding model

## Dashboard Plan

The dashboard will read from `artifacts/experiments/` and show:

- dataset split metadata
- per-model metrics
- prediction samples
- feature importances
- side-by-side experiment comparisons

The placeholder dashboard is already wired to this artifact layout.

## Recommended Workflow

1. Generate or refresh a dataset in `data/<dataset_name>/`
2. Run `scripts/prepare_train_test_split.py`
3. Train one or more models from `scripts/`
4. Review metrics in `artifacts/experiments/`
5. Open `dashboard/app.py` with Streamlit
