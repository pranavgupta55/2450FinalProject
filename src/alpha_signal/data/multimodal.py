from __future__ import annotations

from pathlib import Path
import html
import re

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _limit_text(value: str, max_chars: int) -> str:
    return value[:max_chars].strip()


def _week_start(series: pd.Series) -> pd.Series:
    # The weekly dataset stores `week_start` on the Tuesday that opens each W-MON period.
    return pd.to_datetime(series, errors="coerce").dt.to_period("W-MON").apply(lambda p: p.start_time)


def _load_optional_csv(dataset_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = dataset_dir / f"{stem}.csv"
    parquet_path = dataset_dir / f"{stem}.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def _join_text_columns(frame: pd.DataFrame, columns: list[str], max_chars: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)

    joined = []
    for _, row in frame[columns].fillna("").iterrows():
        parts = [_clean_text(row[column]) for column in columns]
        parts = [part for part in parts if part]
        joined.append(_limit_text(" ".join(parts), max_chars))
    return pd.Series(joined, index=frame.index, dtype=str)


def build_weekly_text_bundle(
    dataset_dir: str | Path,
    max_sec_chars: int = 4000,
    max_news_chars: int = 2000,
    max_combined_chars: int = 6000,
) -> pd.DataFrame:
    resolved = Path(dataset_dir)
    weekly_path = resolved / "weekly_event_dataset.csv"
    if weekly_path.exists():
        weekly_df = pd.read_csv(weekly_path)
    else:
        weekly_df = pd.read_parquet(resolved / "weekly_event_dataset.parquet")

    bundle = weekly_df[["ticker", "week_start"]].copy()
    bundle["week_start"] = pd.to_datetime(bundle["week_start"], errors="coerce")

    sec_df = _load_optional_csv(resolved, "sec_filings_text")
    if not sec_df.empty:
        sec_df = sec_df.copy()
        sec_df["week_start"] = _week_start(sec_df["filing_date"])
        sec_df["sec_text_clean"] = _join_text_columns(sec_df, ["filing_text"], max_sec_chars)
        sec_weekly = (
            sec_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(sec_text=("sec_text_clean", lambda values: _limit_text(" ".join(v for v in values if v), max_sec_chars)))
        )
        bundle = bundle.merge(sec_weekly, on=["ticker", "week_start"], how="left")
    else:
        bundle["sec_text"] = ""

    finnhub_df = _load_optional_csv(resolved, "finnhub_news")
    if not finnhub_df.empty:
        finnhub_df = finnhub_df.copy()
        finnhub_df["week_start"] = _week_start(finnhub_df["date"])
        finnhub_df["news_text_clean"] = _join_text_columns(
            finnhub_df,
            ["headline", "summary", "source", "category"],
            max_news_chars,
        )
        finnhub_weekly = (
            finnhub_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                finnhub_text=(
                    "news_text_clean",
                    lambda values: _limit_text(" ".join(v for v in values if v), max_news_chars),
                )
            )
        )
        bundle = bundle.merge(finnhub_weekly, on=["ticker", "week_start"], how="left")
    else:
        bundle["finnhub_text"] = ""

    yahoo_df = _load_optional_csv(resolved, "yahoo_news")
    if not yahoo_df.empty:
        yahoo_df = yahoo_df.copy()
        yahoo_df["week_start"] = _week_start(yahoo_df["date"])
        yahoo_df["news_text_clean"] = _join_text_columns(
            yahoo_df,
            ["headline", "summary", "source"],
            max_news_chars,
        )
        yahoo_weekly = (
            yahoo_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                yahoo_text=(
                    "news_text_clean",
                    lambda values: _limit_text(" ".join(v for v in values if v), max_news_chars),
                )
            )
        )
        bundle = bundle.merge(yahoo_weekly, on=["ticker", "week_start"], how="left")
    else:
        bundle["yahoo_text"] = ""

    for column in ["sec_text", "finnhub_text", "yahoo_text"]:
        if column not in bundle.columns:
            bundle[column] = ""
        bundle[column] = bundle[column].fillna("").map(_clean_text)

    combined_text = []
    for _, row in bundle.iterrows():
        parts = []
        if row["sec_text"]:
            parts.append(f"SEC filing: {row['sec_text']}")
        if row["finnhub_text"]:
            parts.append(f"Finnhub news: {row['finnhub_text']}")
        if row["yahoo_text"]:
            parts.append(f"Yahoo news: {row['yahoo_text']}")
        combined_text.append(_limit_text(" ".join(parts), max_combined_chars))

    bundle["combined_text"] = combined_text
    bundle["has_text"] = (bundle["combined_text"].str.len() > 0).astype(int)
    return bundle


def attach_weekly_text_bundle(
    split_df: pd.DataFrame,
    dataset_dir: str | Path,
) -> pd.DataFrame:
    text_bundle = build_weekly_text_bundle(dataset_dir)
    work = split_df.copy()
    work["week_start"] = pd.to_datetime(work["week_start"], errors="coerce")
    merged = work.merge(text_bundle, on=["ticker", "week_start"], how="left")
    for column in ["sec_text", "finnhub_text", "yahoo_text", "combined_text"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna("")
    if "has_text" in merged.columns:
        merged["has_text"] = merged["has_text"].fillna(0).astype(int)
    return merged
