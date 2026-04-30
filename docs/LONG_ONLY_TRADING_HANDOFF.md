# Long-Only Trading Handoff

This branch keeps the dashboard and strategy-analysis layer ready for externally trained model artifacts. It does not require local retraining.

## Artifact Contract

Place experiment artifacts under:

- `artifacts/experiments/random_forest/<dataset_name>/`
- `artifacts/experiments/xgboost/<dataset_name>/`
- `artifacts/experiments/sp500_buy_hold/<dataset_name>/`
- `artifacts/experiments/finbert_multimodal_attention/<dataset_name>/`
- `artifacts/experiments/finbert/<dataset_name>/`

FinBERT can use either `finbert_multimodal_attention` or `finbert`. If both exist for the same dataset, the loader and strategy builder use `finbert_multimodal_attention` first and skip the duplicate alias.

Each model experiment needs `predictions.csv` for strategy analysis. The dashboard additionally reads `metrics.json`, `metadata.json`, `trade_log.csv`, `trading_summary.json`, and optionally `feature_importance.csv`.

## Long-Only Semantics

Current model outputs are treated as positive-event ranking signals:

- High model score means eligible long candidate.
- Low or negative score means flat, not short.
- Model strategy outputs must report `trading_mode: long_only`, `signal_regime: positive_only`, and `portfolio_type: long_only`.
- `short_trades` should remain `0`.
- The S&P 500 buy-and-hold benchmark remains unchanged.

## Rebuild Strategy Analysis

After merging new experiment artifacts, rerun:

```bash
python scripts/build_strategy_analysis.py \
  --dataset-name sp500_500_2yr \
  --split-dir artifacts/splits/sp500_500_2yr
```

For the smaller quarterly demo dataset:

```bash
python scripts/build_strategy_analysis.py \
  --dataset-name sp500_500_2yr_quarterly \
  --split-dir artifacts/splits/sp500_500_2yr_quarterly
```

The script is availability-driven. Missing FinBERT or benchmark artifacts are reported in the JSON `skipped` block without breaking RF, XGBoost, or random strategy generation.

## Dashboard Behavior

The dashboard discovers strategy bundles from `artifacts/strategy_analysis`. It renders whatever is present and omits missing strategies. Once a real FinBERT artifact appears under either supported artifact root and strategy analysis is rerun, the FinBERT line appears automatically without UI changes.
