from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha_signal.config import DEFAULT_LABEL_COLUMN, DEFAULT_TEST_RATIO, DEFAULT_TIME_COLUMN
from src.alpha_signal.utils.io import ensure_dir, read_json, write_dataframe, write_json


def time_based_train_test_split(
    df: pd.DataFrame,
    time_column: str = DEFAULT_TIME_COLUMN,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    work = df.sort_values([time_column, "ticker"]).reset_index(drop=True)
    unique_periods = pd.Index(sorted(work[time_column].dropna().unique()))

    if len(unique_periods) < 2:
        raise ValueError("Need at least two distinct time periods to create a train/test split.")

    requested_test_periods = ceil(len(unique_periods) * test_ratio)
    test_periods = min(max(1, requested_test_periods), len(unique_periods) - 1)
    split_index = len(unique_periods) - test_periods

    train_periods = unique_periods[:split_index]
    test_periods_index = unique_periods[split_index:]

    train_df = work[work[time_column].isin(train_periods)].copy()
    test_df = work[work[time_column].isin(test_periods_index)].copy()

    metadata = {
        "time_column": time_column,
        "test_ratio": test_ratio,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_periods": int(len(train_periods)),
        "test_periods": int(len(test_periods_index)),
        "train_start": str(train_periods.min()),
        "train_end": str(train_periods.max()),
        "test_start": str(test_periods_index.min()),
        "test_end": str(test_periods_index.max()),
    }
    return train_df, test_df, metadata


def time_based_train_validation_split(
    df: pd.DataFrame,
    time_column: str = DEFAULT_TIME_COLUMN,
    validation_ratio: float = DEFAULT_TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if validation_ratio <= 0:
        metadata = {
            "time_column": time_column,
            "validation_ratio": validation_ratio,
            "train_rows": int(len(df)),
            "validation_rows": 0,
        }
        return df.copy(), df.iloc[0:0].copy(), metadata

    train_df, val_df, split_metadata = time_based_train_test_split(
        df=df,
        time_column=time_column,
        test_ratio=validation_ratio,
    )
    metadata = {
        "time_column": split_metadata["time_column"],
        "validation_ratio": split_metadata["test_ratio"],
        "train_rows": split_metadata["train_rows"],
        "validation_rows": split_metadata["test_rows"],
        "train_periods": split_metadata["train_periods"],
        "validation_periods": split_metadata["test_periods"],
        "train_start": split_metadata["train_start"],
        "train_end": split_metadata["train_end"],
        "validation_start": split_metadata["test_start"],
        "validation_end": split_metadata["test_end"],
    }
    return train_df, val_df, metadata


def _class_counts(df: pd.DataFrame, label_column: str) -> dict[str, int]:
    if label_column not in df.columns or df.empty:
        return {"0": 0, "1": 0}

    labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)
    counts = labels.value_counts().to_dict()
    return {
        "0": int(counts.get(0, 0)),
        "1": int(counts.get(1, 0)),
    }


def _positive_rate(df: pd.DataFrame, label_column: str) -> float:
    if label_column not in df.columns or df.empty:
        return 0.0

    labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)
    return float(labels.mean()) if len(labels) else 0.0


def compute_label_audit(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> dict[str, Any]:
    """Return stable class-balance metadata for model and split artifacts."""

    audit: dict[str, Any] = {
        "label_column": label_column,
        "full_class_counts": _class_counts(full_df, label_column),
        "full_positive_rate": _positive_rate(full_df, label_column),
    }

    if train_df is not None:
        audit.update(
            {
                "train_class_counts": _class_counts(train_df, label_column),
                "train_positive_rate": _positive_rate(train_df, label_column),
            }
        )

    if test_df is not None:
        audit.update(
            {
                "test_class_counts": _class_counts(test_df, label_column),
                "test_positive_rate": _positive_rate(test_df, label_column),
            }
        )

    return audit


def save_split_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    resolved = ensure_dir(output_dir)

    combined = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )

    write_dataframe(train_df, resolved / "train.csv")
    write_dataframe(test_df, resolved / "test.csv")
    write_dataframe(combined, resolved / "full_with_split.csv")
    write_json(resolved / "split_metadata.json", metadata)


def load_split_artifacts(split_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved = Path(split_dir)
    train_df = pd.read_csv(
        resolved / "train.csv",
        parse_dates=["week_start", "last_date"],
        low_memory=False,
    )
    test_df = pd.read_csv(
        resolved / "test.csv",
        parse_dates=["week_start", "last_date"],
        low_memory=False,
    )
    metadata = read_json(resolved / "split_metadata.json")
    return train_df, test_df, metadata
