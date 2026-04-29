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

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR, DEFAULT_LABEL_COLUMN, DEFAULT_RANDOM_STATE
from src.alpha_signal.data.splitting import load_split_artifacts
from src.alpha_signal.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)
from src.alpha_signal.evaluation.trading import simulate_alpha_trading
from src.alpha_signal.models.training import save_experiment_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a random-guess baseline.")
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--capital-per-trade", type=float, default=10_000.0)
    parser.add_argument("--alpha-trade-threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df, test_df, split_metadata = load_split_artifacts(args.split_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "random_baseline" / args.dataset_name
    )

    rng = np.random.default_rng(args.random_state)
    y_test = test_df[DEFAULT_LABEL_COLUMN].astype(int).to_numpy()
    alpha_train = pd.to_numeric(train_df["future_alpha_5d"], errors="coerce").dropna()
    alpha_test = pd.to_numeric(test_df["future_alpha_5d"], errors="coerce").to_numpy(dtype=float)

    predicted_probability = rng.random(len(test_df))
    alpha_mean = float(alpha_train.mean()) if len(alpha_train) else 0.0
    alpha_std = float(alpha_train.std(ddof=0)) if len(alpha_train) else 0.01
    if alpha_std == 0:
        alpha_std = 0.01
    predicted_alpha_score = rng.normal(loc=alpha_mean, scale=alpha_std, size=len(test_df))

    metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_score=predicted_probability,
        threshold=args.threshold,
    )
    metrics.update(compute_regression_metrics(alpha_test, predicted_alpha_score))

    predictions = test_df[
        ["ticker", "week_start", "last_date", "future_alpha_5d", DEFAULT_LABEL_COLUMN]
    ].copy()
    predictions["predicted_probability"] = predicted_probability
    predictions["predicted_label"] = (predictions["predicted_probability"] >= args.threshold).astype(int)
    predictions["predicted_alpha_score"] = predicted_alpha_score
    predictions["predicted_direction"] = (predictions["predicted_alpha_score"] >= 0).astype(int)

    trade_log_df, trading_summary = simulate_alpha_trading(
        predictions,
        capital_per_trade=args.capital_per_trade,
        alpha_threshold=args.alpha_trade_threshold,
    )

    metadata = {
        "model_name": "random_baseline",
        "dataset_name": args.dataset_name,
        "label_column": DEFAULT_LABEL_COLUMN,
        "random_state": args.random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "capital_per_trade": args.capital_per_trade,
        "alpha_trade_threshold": args.alpha_trade_threshold,
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
