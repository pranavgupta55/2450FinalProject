from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.data.dataset import load_weekly_event_dataset


REQUIRED_NUMERIC_COLUMNS = [
    "close",
    "volume",
    "vol_ma_5",
    "price_ma_5",
    "price_ma_20",
    "volatility_20",
    "sec_event_count",
    "sec_filing_text_count",
    "finnhub_event_count",
    "yahoo_event_count",
    "has_text",
    "text_source_count",
    "event_text_char_count",
]

REQUIRED_TEXT_COLUMNS = [
    "sec_filing_text",
    "finnhub_news_text",
    "yahoo_news_text",
    "combined_event_text",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate that weekly_event_dataset has numerical and text columns."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    try:
        df = load_weekly_event_dataset(input_dir)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    required_columns = REQUIRED_NUMERIC_COLUMNS + REQUIRED_TEXT_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print("Dataset is missing required multimodal columns:")
        for col in missing_columns:
            print(f"- {col}")
        raise SystemExit(1)

    text_lengths = df["combined_event_text"].fillna("").astype(str).str.len()
    rows_with_text = int((text_lengths > 0).sum())
    total_text_chars = int(text_lengths.sum())

    if rows_with_text == 0 or total_text_chars == 0:
        print("Dataset has the text columns, but combined_event_text is empty.")
        raise SystemExit(1)

    numeric_frame = df[REQUIRED_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_non_null = int(numeric_frame.notna().sum().sum())
    if numeric_non_null == 0:
        print("Dataset has the numeric columns, but they are empty or non-numeric.")
        raise SystemExit(1)

    print(f"Rows: {len(df)}")
    print(f"Rows with text: {rows_with_text}")
    print(f"Combined text characters: {total_text_chars}")
    print("Dataset has the expected numerical and text columns.")


if __name__ == "__main__":
    main()
