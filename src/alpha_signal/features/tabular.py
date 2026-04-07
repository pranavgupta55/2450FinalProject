from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class TabularTransform:
    numeric_columns: list[str]
    categorical_columns: list[str]
    fill_values: dict[str, float]
    dummy_columns: list[str]
    feature_columns: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TabularTransform":
        return cls(**payload)


def _build_dummy_frame(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    if not categorical_columns:
        return pd.DataFrame(index=df.index)

    cat_frame = df[categorical_columns].copy().fillna("MISSING").astype(str)
    return pd.get_dummies(cat_frame, prefix=categorical_columns)


def fit_tabular_transform(
    train_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> TabularTransform:
    numeric_frame = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = {}
    for column in numeric_columns:
        median = numeric_frame[column].median()
        fill_values[column] = 0.0 if pd.isna(median) else float(median)

    dummy_frame = _build_dummy_frame(train_df, categorical_columns)
    feature_columns = numeric_columns + dummy_frame.columns.tolist()

    return TabularTransform(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
        dummy_columns=dummy_frame.columns.tolist(),
        feature_columns=feature_columns,
    )


def transform_tabular_dataset(df: pd.DataFrame, transform: TabularTransform) -> pd.DataFrame:
    numeric_frame = (
        df[transform.numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(transform.fill_values)
    )

    dummy_frame = _build_dummy_frame(df, transform.categorical_columns).reindex(
        columns=transform.dummy_columns,
        fill_value=0,
    )

    output = pd.concat(
        [
            numeric_frame.reset_index(drop=True),
            dummy_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    return output.reindex(columns=transform.feature_columns, fill_value=0).astype(float)
