from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"

DEFAULT_LABEL_COLUMN = "label_abs_alpha_gt_1pct"
DEFAULT_TIME_COLUMN = "week_start"
DEFAULT_TEST_RATIO = 0.20
DEFAULT_RANDOM_STATE = 42

IDENTIFIER_COLUMNS = ["ticker", "week_start", "last_date"]
TARGET_COLUMNS = ["future_alpha_5d", DEFAULT_LABEL_COLUMN]

BASE_NUMERIC_FEATURES = [
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
]

DERIVED_NUMERIC_FEATURES = [
    "has_sec_filing",
    "has_finnhub_news",
    "has_yahoo_news",
    "has_text",
    "text_source_count",
    "event_text_char_count",
    "price_vs_ma_5",
    "price_vs_ma_20",
    "volume_vs_ma_5",
    "week_of_year",
    "month",
    "days_from_start",
]

DEFAULT_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
DEFAULT_CATEGORICAL_FEATURES = ["ticker"]

TEXT_PLACEHOLDER_COLUMNS = [
    "sec_latest_filing_url",
    "sec_filing_text",
    "finnhub_headline_sample",
    "finnhub_summary_sample",
    "finnhub_news_text",
    "yahoo_headline_sample",
    "yahoo_summary_sample",
    "yahoo_news_text",
    "combined_event_text",
]
