from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.alpha_signal.config import DEFAULT_LABEL_COLUMN, DEFAULT_RANDOM_STATE
from src.alpha_signal.data.dataset import get_feature_spec
from src.alpha_signal.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_regression_metrics,
)
from src.alpha_signal.evaluation.trading import simulate_alpha_trading
from src.alpha_signal.features.tabular import (
    fit_tabular_transform,
    transform_tabular_dataset,
)
from src.alpha_signal.utils.io import ensure_dir, write_dataframe, write_json


def prepare_training_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> dict[str, Any]:
    feature_spec = get_feature_spec(train_df)
    transform = fit_tabular_transform(
        train_df=train_df,
        numeric_columns=feature_spec["numeric"],
        categorical_columns=feature_spec["categorical"],
    )

    X_train = transform_tabular_dataset(train_df, transform)
    X_test = transform_tabular_dataset(test_df, transform)
    y_train = train_df[label_column].astype(int).to_numpy()
    y_test = test_df[label_column].astype(int).to_numpy()
    alpha_train = pd.to_numeric(train_df["future_alpha_5d"], errors="coerce")
    alpha_test = pd.to_numeric(test_df["future_alpha_5d"], errors="coerce")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "alpha_train": alpha_train.to_numpy(dtype=float),
        "alpha_test": alpha_test.to_numpy(dtype=float),
        "transform": transform,
        "feature_columns": transform.feature_columns,
    }


def maybe_apply_smote(
    X_train: pd.DataFrame,
    y_train,
    use_smote: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    if not use_smote:
        return X_train, y_train, None

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise SystemExit(
            "imbalanced-learn is required for SMOTE. Run `pip install -r requirements.txt`."
        ) from exc

    sampler = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    metadata = {
        "before_counts": pd.Series(y_train).value_counts().sort_index().to_dict(),
        "after_counts": pd.Series(y_resampled).value_counts().sort_index().to_dict(),
    }
    return X_resampled, y_resampled, metadata


def estimate_alpha_scores_from_probability(
    y_score,
    train_df: pd.DataFrame,
) -> np.ndarray:
    alpha_train = pd.to_numeric(train_df["future_alpha_5d"], errors="coerce").dropna()
    mean_abs_alpha = float(alpha_train.abs().mean()) if len(alpha_train) else 0.01
    centered = np.asarray(y_score, dtype=float) - 0.5
    return centered * 2.0 * mean_abs_alpha


def fit_alpha_regressor(alpha_regressor, X_train: pd.DataFrame, train_df: pd.DataFrame):
    if alpha_regressor is None:
        return None, None

    alpha_train = pd.to_numeric(train_df["future_alpha_5d"], errors="coerce")
    valid_mask = alpha_train.notna()
    if not valid_mask.any():
        return None, None

    alpha_regressor.fit(X_train.loc[valid_mask], alpha_train.loc[valid_mask].to_numpy(dtype=float))
    return alpha_regressor, valid_mask


def save_experiment_artifacts(
    output_dir: str | Path,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    predictions: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    trade_log_df: pd.DataFrame | None = None,
    trading_summary: dict[str, Any] | None = None,
) -> Path:
    output_path = ensure_dir(output_dir)
    write_dataframe(predictions, output_path / "predictions.csv")
    write_dataframe(feature_importance_df, output_path / "feature_importance.csv")
    write_json(output_path / "metrics.json", metrics)
    write_json(output_path / "metadata.json", metadata)
    if trade_log_df is not None:
        write_dataframe(trade_log_df, output_path / "trade_log.csv")
    if trading_summary is not None:
        write_json(output_path / "trading_summary.json", trading_summary)
    return output_path


def train_and_evaluate_model(
    model,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    dataset_name: str,
    label_column: str = DEFAULT_LABEL_COLUMN,
    threshold: float = 0.5,
    use_smote: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
    extra_metadata: dict[str, Any] | None = None,
    alpha_regressor=None,
    capital_per_trade: float = 10_000.0,
    alpha_trade_threshold: float = 0.0,
) -> dict[str, Any]:
    prepared = prepare_training_matrices(train_df, test_df, label_column=label_column)
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    alpha_train = prepared["alpha_train"]
    alpha_test = prepared["alpha_test"]

    X_train_fit, y_train_fit, smote_metadata = maybe_apply_smote(
        X_train=X_train,
        y_train=y_train,
        use_smote=use_smote,
        random_state=random_state,
    )

    model.fit(X_train_fit, y_train_fit)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test)

    metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_score=y_score,
        threshold=threshold,
    )

    fitted_alpha_regressor, alpha_valid_mask = fit_alpha_regressor(
        alpha_regressor=alpha_regressor,
        X_train=X_train,
        train_df=train_df,
    )
    if fitted_alpha_regressor is not None:
        predicted_alpha_score = np.asarray(fitted_alpha_regressor.predict(X_test), dtype=float)
        regression_metrics = compute_regression_metrics(alpha_test, predicted_alpha_score)
    else:
        predicted_alpha_score = estimate_alpha_scores_from_probability(y_score, train_df)
        regression_metrics = compute_regression_metrics(alpha_test, predicted_alpha_score)

    metrics.update(regression_metrics)

    predictions = test_df[
        ["ticker", "week_start", "last_date", "future_alpha_5d", label_column]
    ].copy()
    predictions["predicted_probability"] = y_score
    predictions["predicted_label"] = (predictions["predicted_probability"] >= threshold).astype(int)
    predictions["predicted_alpha_score"] = predicted_alpha_score
    predictions["predicted_direction"] = (predictions["predicted_alpha_score"] >= 0).astype(int)

    feature_importance_df = build_feature_importance_frame(model, prepared["feature_columns"])
    trade_log_df, trading_summary = simulate_alpha_trading(
        predictions,
        capital_per_trade=capital_per_trade,
        alpha_threshold=alpha_trade_threshold,
    )

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "label_column": label_column,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "test_positive_rate": float(y_test.mean()) if len(y_test) else 0.0,
        "use_smote": use_smote,
        "random_state": random_state,
        "feature_columns": prepared["feature_columns"],
        "transform": prepared["transform"].to_dict(),
        "alpha_trade_threshold": alpha_trade_threshold,
        "capital_per_trade": capital_per_trade,
    }
    if smote_metadata is not None:
        metadata["smote"] = smote_metadata
    if fitted_alpha_regressor is not None:
        metadata["alpha_regressor"] = fitted_alpha_regressor.__class__.__name__
        metadata["alpha_regressor_train_rows"] = int(alpha_valid_mask.sum()) if alpha_valid_mask is not None else 0
    if extra_metadata:
        metadata.update(extra_metadata)

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
                "classifier": model,
                "alpha_regressor": fitted_alpha_regressor,
            },
            handle,
        )

    return {
        "metrics": metrics,
        "metadata": metadata,
        "predictions": predictions,
        "feature_importance": feature_importance_df,
        "trade_log": trade_log_df,
        "trading_summary": trading_summary,
    }


def build_feature_importance_frame(model, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )
    return feature_importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
