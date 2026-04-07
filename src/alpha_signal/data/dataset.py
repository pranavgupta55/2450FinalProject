from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.alpha_signal.config import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_TIME_COLUMN,
)


def load_weekly_event_dataset(input_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(input_dir)
    csv_path = dataset_dir / "weekly_event_dataset.csv"
    parquet_path = dataset_dir / "weekly_event_dataset.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Could not find weekly_event_dataset.csv or weekly_event_dataset.parquet in {dataset_dir}"
    )


def _safe_relative_gap(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    ratio = np.where((den.notna()) & (den != 0), num / den - 1.0, np.nan)
    return pd.Series(ratio, index=numerator.index, dtype=float)


def build_structured_modeling_dataset(
    raw_df: pd.DataFrame,
    label_column: str = DEFAULT_LABEL_COLUMN,
    time_column: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    work = raw_df.copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    work["last_date"] = pd.to_datetime(work["last_date"], errors="coerce")

    work = work.dropna(subset=["ticker", time_column, label_column]).copy()
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    work = work.dropna(subset=[label_column]).copy()
    work[label_column] = work[label_column].astype(int)

    for col in ["sec_event_count", "finnhub_event_count", "yahoo_event_count"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    numeric_seed_columns = [
        "close",
        "volume",
        "vol_ma_5",
        "price_ma_5",
        "price_ma_20",
        "volatility_20",
        "future_alpha_5d",
    ]
    for col in numeric_seed_columns:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work["has_sec_filing"] = (work["sec_event_count"] > 0).astype(int)
    work["has_finnhub_news"] = (work["finnhub_event_count"] > 0).astype(int)
    work["has_yahoo_news"] = (work["yahoo_event_count"] > 0).astype(int)

    work["price_vs_ma_5"] = _safe_relative_gap(work["close"], work["price_ma_5"])
    work["price_vs_ma_20"] = _safe_relative_gap(work["close"], work["price_ma_20"])
    work["volume_vs_ma_5"] = _safe_relative_gap(work["volume"], work["vol_ma_5"])
    work["week_of_year"] = work[time_column].dt.isocalendar().week.astype(int)
    work["month"] = work[time_column].dt.month.astype(int)
    work["days_from_start"] = (
        work[time_column] - work[time_column].min()
    ).dt.days.astype(int)

    work = work.sort_values([time_column, "ticker"]).reset_index(drop=True)
    return work


def get_feature_spec(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_columns = [col for col in DEFAULT_NUMERIC_FEATURES if col in df.columns]
    categorical_columns = [col for col in DEFAULT_CATEGORICAL_FEATURES if col in df.columns]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found in the dataset.")

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }
