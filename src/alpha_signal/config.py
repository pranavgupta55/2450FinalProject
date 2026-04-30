from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"

SIGNED_DIRECTION_LABEL_COLUMN = "label_alpha_positive"
ABS_ALPHA_LABEL_COLUMN = "label_abs_alpha_gt_1pct"
DEFAULT_LABEL_COLUMN = ABS_ALPHA_LABEL_COLUMN

TRADING_MODE_LONG_ONLY = "long_only"
SIGNAL_REGIME_POSITIVE_ONLY = "positive_only"
PORTFOLIO_TYPE_LONG_ONLY = "long_only"
DEFAULT_TIME_COLUMN = "week_start"
DEFAULT_TEST_RATIO = 0.20
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_RANDOM_STATE = 42
DEFAULT_INCLUDE_TICKER = False
DEFAULT_THRESHOLD_METRIC = "f1"
DEFAULT_CHECKPOINT_METRIC = "average_precision"
DEFAULT_ALPHA_TRADE_OBJECTIVE = "return_on_traded_capital"
DEFAULT_MIN_TRADES_FOR_THRESHOLD = 10

IDENTIFIER_COLUMNS = ["ticker", "week_start", "last_date"]
TARGET_COLUMNS = ["future_alpha_5d", SIGNED_DIRECTION_LABEL_COLUMN, ABS_ALPHA_LABEL_COLUMN]

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
    "has_text",
    "text_source_count",
    "event_text_char_count",
]

DERIVED_NUMERIC_FEATURES = [
    "has_sec_filing",
    "has_finnhub_news",
    "has_yahoo_news",
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
    "finnhub_headline_sample",
    "yahoo_headline_sample",
]
