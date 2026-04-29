# 2450FinalProject

Alpha prediction pipeline for CIS 2450 with a standalone Next.js dashboard for browsing saved experiment artifacts, mock-trading results, and cross-run comparisons.

## Current Workflow

1. Download data with `data/data_downloader.py`
2. Prepare a chronological train/test split from `weekly_event_dataset.csv`
3. Train one or more models:
   - `random_baseline`
   - `random_forest`
   - `xgboost`
   - `finbert_multimodal_attention`
   - `sp500_buy_hold`
4. Save predictions, alpha scores, metrics, and mock-trading artifacts to `artifacts/`
5. Load experiment outputs into the dashboard

## What The Models Use

- `Random Forest` and `XGBoost`
  Structured weekly features plus an auxiliary alpha regressor for mock trading
- `FinBERT Multimodal Attention`
  FinBERT text embeddings over SEC filing text and weekly news text, fused with tabular market/event/time features using self-attention
- `Random Baseline`
  Uniform random classification probabilities plus random alpha scores for a floor comparison

## Useful Commands

Install Python dependencies:

```bash
source 2450venv/bin/activate
pip install -r requirements.txt
```

Install dashboard dependencies:

```bash
cd dashboard
npm install
```

Prepare a train/test split from the small test dataset:

```bash
python scripts/prepare_train_test_split.py --input-dir data/test_ds --dataset-name test_ds
```

Train the random-guess baseline:

```bash
python scripts/train_random_baseline.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Train a Random Forest baseline with alpha-score backtesting:

```bash
python scripts/train_random_forest.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Train an XGBoost baseline with alpha-score backtesting:

```bash
python scripts/train_xgboost.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Train the FinBERT multimodal attention model:

```bash
python scripts/train_multimodal_attention.py --split-dir artifacts/splits/test_ds --dataset-name test_ds --epochs 3 --batch-size 8
```

Build the S&P 500 buy-and-hold benchmark over the test window:

```bash
python scripts/build_sp500_benchmark.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Use `--local-files-only` if the FinBERT model is already cached locally. The first run may need network access to download `ProsusAI/finbert`.

Note: on macOS, XGBoost may need `brew install libomp` before the first run.

Launch the dashboard in development:

```bash
cd dashboard
npm run dev
```

The dashboard reads existing artifacts from `artifacts/experiments/` and does not modify the ML pipeline.

## Artifact Layout

Each experiment run is stored under:

```text
artifacts/experiments/<model_name>/<dataset_name>/
```

Important files per run:

- `metrics.json`
- `metadata.json`
- `predictions.csv`
- `feature_importance.csv`
- `trading_summary.json`
- `trade_log.csv`
- `model.pkl` or `model.pt`

More detailed design notes in `docs/PROJECT_ARCHITECTURE.md`.
