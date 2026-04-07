from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha_signal.config import DEFAULT_LABEL_COLUMN, DEFAULT_RANDOM_STATE
from src.alpha_signal.data.dataset import get_feature_spec
from src.alpha_signal.evaluation.metrics import compute_binary_classification_metrics
from src.alpha_signal.features.tabular import (
    TabularTransform,
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

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
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
) -> dict[str, Any]:
    prepared = prepare_training_matrices(train_df, test_df, label_column=label_column)
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

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

    predictions = test_df[
        ["ticker", "week_start", "last_date", "future_alpha_5d", label_column]
    ].copy()
    predictions["predicted_probability"] = y_score
    predictions["predicted_label"] = (predictions["predicted_probability"] >= threshold).astype(int)

    output_path = ensure_dir(output_dir)
    feature_importance_df = build_feature_importance_frame(model, prepared["feature_columns"])

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
    }
    if smote_metadata is not None:
        metadata["smote"] = smote_metadata
    if extra_metadata:
        metadata.update(extra_metadata)

    with (output_path / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    write_dataframe(predictions, output_path / "predictions.csv")
    write_dataframe(feature_importance_df, output_path / "feature_importance.csv")
    write_json(output_path / "metrics.json", metrics)
    write_json(output_path / "metadata.json", metadata)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "predictions": predictions,
        "feature_importance": feature_importance_df,
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
