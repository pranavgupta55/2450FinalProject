from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.alpha_signal.config import DEFAULT_ALPHA_TRADE_OBJECTIVE, DEFAULT_THRESHOLD_METRIC
from src.alpha_signal.evaluation.metrics import compute_binary_classification_metrics
from src.alpha_signal.evaluation.trading import simulate_alpha_trading


def _safe_metric_value(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if np.isnan(numeric):
        return float("-inf")
    return numeric


def select_classification_threshold(
    y_true,
    y_score,
    *,
    metric: str = DEFAULT_THRESHOLD_METRIC,
    candidate_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 37)

    best_threshold = 0.5
    best_metrics = compute_binary_classification_metrics(y_true=y_true, y_score=y_score, threshold=best_threshold)
    best_value = _safe_metric_value(best_metrics.get(metric))

    for threshold in candidate_thresholds:
        metrics = compute_binary_classification_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(threshold),
        )
        metric_value = _safe_metric_value(metrics.get(metric))
        if metric_value > best_value or (
            metric_value == best_value and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = metric_value

    return {
        "threshold": float(best_threshold),
        "metric": metric,
        "metric_value": None if best_value == float("-inf") else float(best_value),
        "metrics": best_metrics,
    }


def select_alpha_trade_threshold(
    realized_alpha,
    predicted_alpha_score,
    *,
    capital_per_trade: float,
    objective: str = DEFAULT_ALPHA_TRADE_OBJECTIVE,
    min_trades: int = 10,
) -> dict[str, Any]:
    realized_alpha = np.asarray(realized_alpha, dtype=float)
    predicted_alpha_score = np.asarray(predicted_alpha_score, dtype=float)

    valid_mask = np.isfinite(realized_alpha) & np.isfinite(predicted_alpha_score)
    if not valid_mask.any():
        return {
            "threshold": 0.0,
            "objective": objective,
            "objective_value": None,
            "summary": {"trades_executed": 0},
        }

    frame = pd.DataFrame(
        {
            "future_alpha_5d": realized_alpha[valid_mask],
            "predicted_alpha_score": predicted_alpha_score[valid_mask],
        }
    )

    long_candidate_scores = frame.loc[
        frame["predicted_alpha_score"] >= 0,
        "predicted_alpha_score",
    ].to_numpy(dtype=float)
    if len(long_candidate_scores) == 0:
        long_candidate_scores = frame["predicted_alpha_score"].to_numpy(dtype=float)

    quantile_candidates = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    candidates = sorted(
        {float(np.quantile(long_candidate_scores, quantile)) for quantile in quantile_candidates}
    )
    if 0.0 not in candidates:
        candidates.insert(0, 0.0)

    best_threshold = 0.0
    _, best_summary = simulate_alpha_trading(
        frame,
        capital_per_trade=capital_per_trade,
        alpha_threshold=best_threshold,
    )
    best_value = _safe_metric_value(best_summary.get(objective))

    for threshold in candidates:
        _, summary = simulate_alpha_trading(
            frame,
            capital_per_trade=capital_per_trade,
            alpha_threshold=float(threshold),
        )
        if summary["trades_executed"] < min_trades:
            continue
        metric_value = _safe_metric_value(summary.get(objective))
        if metric_value > best_value or (
            metric_value == best_value and float(threshold) > best_threshold
        ):
            best_threshold = float(threshold)
            best_summary = summary
            best_value = metric_value

    return {
        "threshold": float(best_threshold),
        "objective": objective,
        "objective_value": None if best_value == float("-inf") else float(best_value),
        "summary": best_summary,
    }
