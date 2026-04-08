# 2450FinalProject

Alpha prediction pipeline for CIS 2450 with a standalone Next.js dashboard for browsing saved experiment artifacts.

## Current Workflow

1. Download data with `data/data_downloader.py`
2. Prepare a chronological train/test split from `weekly_event_dataset.csv`
3. Train baseline tabular models on structured features
4. Save metrics and predictions to `artifacts/`
5. Load experiment outputs into a dashboard

## Useful Commands

Install Python dependencies:

```bash
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

Train a Random Forest baseline:

```bash
python scripts/train_random_forest.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Train an XGBoost baseline:

```bash
python scripts/train_xgboost.py --split-dir artifacts/splits/test_ds --dataset-name test_ds
```

Note: on macOS, XGBoost may need `brew install libomp` before the first run.

Launch the dashboard in development:

```bash
cd dashboard
npm run dev
```

The dashboard reads existing artifacts from `artifacts/experiments/` and does not modify the ML pipeline.

More detailed design notes live in `docs/PROJECT_ARCHITECTURE.md`.
