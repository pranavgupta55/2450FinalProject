from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import (
    DEFAULT_ALPHA_TRADE_OBJECTIVE,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MIN_TRADES_FOR_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_THRESHOLD_METRIC,
    DEFAULT_VALIDATION_RATIO,
    PORTFOLIO_TYPE_LONG_ONLY,
    SIGNAL_REGIME_POSITIVE_ONLY,
    TRADING_MODE_LONG_ONLY,
)
from src.alpha_signal.data.dataset import ensure_target_columns
from src.alpha_signal.data.splitting import (
    compute_label_audit,
    load_split_artifacts,
    time_based_train_validation_split,
)
from src.alpha_signal.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)
from src.alpha_signal.evaluation.selection import (
    select_alpha_trade_threshold,
    select_classification_threshold,
)
from src.alpha_signal.evaluation.trading import simulate_alpha_trading
from src.alpha_signal.models.training import save_experiment_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a random-guess baseline.")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-metric", type=str, default=DEFAULT_THRESHOLD_METRIC)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--capital-per-trade", type=float, default=10_000.0)
    parser.add_argument("--alpha-trade-threshold", type=float, default=None)
    parser.add_argument("--alpha-trade-objective", type=str, default=DEFAULT_ALPHA_TRADE_OBJECTIVE)
    parser.add_argument("--min-trades-for-threshold", type=int, default=DEFAULT_MIN_TRADES_FOR_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df, test_df, split_metadata = load_split_artifacts(args.split_dir)
    train_df = ensure_target_columns(train_df)
    test_df = ensure_target_columns(test_df)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "random_baseline" / args.dataset_name
    )

    rng = np.random.default_rng(args.random_state)
    inner_train_df, val_df, validation_metadata = time_based_train_validation_split(
        train_df,
        validation_ratio=args.validation_ratio,
    )
    y_val = val_df[args.label_column].astype(int).to_numpy()
    y_test = test_df[args.label_column].astype(int).to_numpy()
    alpha_train = pd.to_numeric(train_df["future_alpha_5d"], errors="coerce").dropna()
    alpha_val = pd.to_numeric(val_df["future_alpha_5d"], errors="coerce").to_numpy(dtype=float)
    alpha_test = pd.to_numeric(test_df["future_alpha_5d"], errors="coerce").to_numpy(dtype=float)

    alpha_mean = float(alpha_train.mean()) if len(alpha_train) else 0.0
    alpha_std = float(alpha_train.std(ddof=0)) if len(alpha_train) else 0.01
    if alpha_std == 0:
        alpha_std = 0.01
    val_probability = rng.random(len(val_df))
    val_alpha_score = rng.normal(loc=alpha_mean, scale=alpha_std, size=len(val_df))
    threshold_info = (
        {
            "threshold": float(args.threshold),
            "metric": args.threshold_metric,
            "metric_value": None,
            "metrics": compute_binary_classification_metrics(y_true=y_val, y_score=val_probability, threshold=float(args.threshold)),
        }
        if args.threshold is not None
        else select_classification_threshold(y_true=y_val, y_score=val_probability, metric=args.threshold_metric)
    )
    alpha_threshold_info = (
        {
            "threshold": float(args.alpha_trade_threshold),
            "objective": args.alpha_trade_objective,
            "objective_value": None,
            "summary": simulate_alpha_trading(
                pd.DataFrame({"future_alpha_5d": alpha_val, "predicted_alpha_score": val_alpha_score}),
                capital_per_trade=args.capital_per_trade,
                alpha_threshold=float(args.alpha_trade_threshold),
            )[1],
        }
        if args.alpha_trade_threshold is not None
        else select_alpha_trade_threshold(
            realized_alpha=alpha_val,
            predicted_alpha_score=val_alpha_score,
            capital_per_trade=args.capital_per_trade,
            objective=args.alpha_trade_objective,
            min_trades=args.min_trades_for_threshold,
        )
    )

    predicted_probability = rng.random(len(test_df))
    predicted_alpha_score = rng.normal(loc=alpha_mean, scale=alpha_std, size=len(test_df))

    metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_score=predicted_probability,
        threshold=float(threshold_info["threshold"]),
    )
    metrics.update(compute_regression_metrics(alpha_test, predicted_alpha_score))

    predictions = test_df[
        ["ticker", "week_start", "last_date", "future_alpha_5d", args.label_column]
    ].copy()
    predictions["predicted_probability"] = predicted_probability
    predictions["predicted_label"] = (
        predictions["predicted_probability"] >= float(threshold_info["threshold"])
    ).astype(int)
    predictions["predicted_signal"] = predictions["predicted_label"]
    predictions["predicted_alpha_score"] = predicted_alpha_score
    predictions["predicted_direction"] = (predictions["predicted_alpha_score"] >= 0).astype(int)

    trade_log_df, trading_summary = simulate_alpha_trading(
        predictions,
        capital_per_trade=args.capital_per_trade,
        alpha_threshold=float(alpha_threshold_info["threshold"]),
    )
    label_audit = compute_label_audit(
        full_df=pd.concat([train_df, test_df], ignore_index=True),
        train_df=train_df,
        test_df=test_df,
        label_column=args.label_column,
    )

    metadata = {
        "model_name": "random_baseline",
        "dataset_name": args.dataset_name,
        "label_column": args.label_column,
        "trading_mode": TRADING_MODE_LONG_ONLY,
        "signal_regime": SIGNAL_REGIME_POSITIVE_ONLY,
        "portfolio_type": PORTFOLIO_TYPE_LONG_ONLY,
        "random_state": args.random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        **label_audit,
        "capital_per_trade": args.capital_per_trade,
        "threshold": float(threshold_info["threshold"]),
        "threshold_metric": args.threshold_metric,
        "threshold_selection": threshold_info,
        "validation_ratio": args.validation_ratio,
        "validation_split": validation_metadata,
        "alpha_trade_threshold": float(alpha_threshold_info["threshold"]),
        "alpha_trade_objective": args.alpha_trade_objective,
        "alpha_trade_threshold_selection": alpha_threshold_info,
        "baseline_strategy": "uniform_probability_and_gaussian_alpha_score",
        "split_metadata": split_metadata,
    }

    feature_importance_df = pd.DataFrame(
        [{"feature": "random_noise", "importance": 1.0}]
    )
    output_path = save_experiment_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        metadata=metadata,
        predictions=predictions,
        feature_importance_df=feature_importance_df,
        trade_log_df=trade_log_df,
        trading_summary=trading_summary,
    )

    with (output_path / "model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "random_state": args.random_state,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
            },
            handle,
        )

    print(f"Saved random baseline artifacts to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
