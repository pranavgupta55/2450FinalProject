from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpha_signal.config import DEFAULT_TEST_RATIO, DEFAULT_TIME_COLUMN
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
    train_df = pd.read_csv(resolved / "train.csv", parse_dates=["week_start", "last_date"])
    test_df = pd.read_csv(resolved / "test.csv", parse_dates=["week_start", "last_date"])
    metadata = read_json(resolved / "split_metadata.json")
    return train_df, test_df, metadata
