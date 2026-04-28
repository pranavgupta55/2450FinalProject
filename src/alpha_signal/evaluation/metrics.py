from __future__ import annotations

from typing import Any

import numpy as np


def compute_binary_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["average_precision"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    )
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        return {
            "alpha_mae": None,
            "alpha_rmse": None,
            "alpha_r2": None,
            "alpha_direction_accuracy": None,
        }

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    direction_true = (y_true >= 0).astype(int)
    direction_pred = (y_pred >= 0).astype(int)

    return {
        "alpha_mae": float(mean_absolute_error(y_true, y_pred)),
        "alpha_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "alpha_r2": float(r2_score(y_true, y_pred)),
        "alpha_direction_accuracy": float((direction_true == direction_pred).mean()),
    }
