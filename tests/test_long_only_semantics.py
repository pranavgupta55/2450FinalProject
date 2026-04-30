from __future__ import annotations

import pandas as pd

from src.alpha_signal.data.splitting import compute_label_audit
from src.alpha_signal.evaluation.strategy_analysis import build_cross_sectional_strategy
from src.alpha_signal.evaluation.trading import simulate_alpha_trading


def test_row_wise_trading_never_shorts_negative_scores():
    predictions = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "week_start": pd.to_datetime(["2026-01-05"] * 3),
            "future_alpha_5d": [-0.02, 0.05, 0.01],
            "predicted_alpha_score": [0.20, -0.90, 0.05],
        }
    )

    trade_log, summary = simulate_alpha_trading(predictions, alpha_threshold=0.10)

    assert trade_log["trade_position"].tolist() == [1, 0, 0]
    assert trade_log["trade_side"].tolist() == ["long", "flat", "flat"]
    assert summary["short_trades"] == 0
    assert summary["portfolio_type"] == "long_only"


def test_ranked_strategy_selects_top_k_longs_only():
    predictions = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "week_start": pd.to_datetime(["2026-01-05"] * 3),
            "future_alpha_5d": [0.01, -0.02, 0.03],
            "predicted_alpha_score": [0.20, 0.90, 0.50],
            "predicted_probability": [0.4, 0.9, 0.7],
        }
    )

    artifacts = build_cross_sectional_strategy(
        predictions_df=predictions,
        strategy_name="test_strategy",
        source_model_name="test_model",
        dataset_name="toy",
        requested_k_long=2,
        requested_k_short=2,
        adaptive=True,
        selection_mode="ranked",
    )

    assert artifacts.trade_log["ticker"].tolist() == ["B", "C"]
    assert artifacts.trade_log["trade_position"].tolist() == [1, 1]
    assert artifacts.trade_log["trade_side"].tolist() == ["long", "long"]
    assert artifacts.trading_summary["short_trades"] == 0
    assert artifacts.metadata["portfolio_construction"] == "top_k_long_only"
    assert artifacts.metadata["requested_k_short"] == 0


def test_label_audit_emits_split_counts_and_positive_rates():
    full_df = pd.DataFrame({"label_abs_alpha_gt_1pct": [1, 0, 1, 0]})
    train_df = full_df.iloc[:3].copy()
    test_df = full_df.iloc[3:].copy()

    audit = compute_label_audit(
        full_df=full_df,
        train_df=train_df,
        test_df=test_df,
        label_column="label_abs_alpha_gt_1pct",
    )

    assert audit["full_class_counts"] == {"0": 2, "1": 2}
    assert audit["train_class_counts"] == {"0": 1, "1": 2}
    assert audit["test_class_counts"] == {"0": 1, "1": 0}
    assert audit["full_positive_rate"] == 0.5
    assert audit["train_positive_rate"] == 2 / 3
    assert audit["test_positive_rate"] == 0.0
