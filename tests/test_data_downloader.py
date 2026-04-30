from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = REPO_ROOT / "data" / "data_downloader.py"

spec = importlib.util.spec_from_file_location("data_downloader", DOWNLOADER_PATH)
data_downloader = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_downloader)


class WeeklyEventDatasetTests(unittest.TestCase):
    def test_weekly_dataset_includes_text_aggregates(self) -> None:
        price_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ticker": ["AAPL", "AAPL"],
                "Adj Close": [190.0, 192.0],
                "Volume": [1000, 1200],
                "vol_ma_5": [950.0, 1000.0],
                "price_ma_5": [188.0, 189.0],
                "price_ma_20": [185.0, 186.0],
                "volatility_20": [0.01, 0.02],
                "future_alpha_5d": [0.02, 0.03],
                "label_abs_alpha_gt_1pct": [1, 1],
            }
        )
        sec_meta_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "filing_date": ["2024-01-03"],
                "accession_number": ["0000320193-24-000001"],
                "filing_url": ["https://www.sec.gov/example"],
            }
        )
        sec_text_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "filing_date": ["2024-01-03"],
                "accession_number": ["0000320193-24-000001"],
                "filing_url": ["https://www.sec.gov/example"],
                "filing_text": ["<html><body>Material 8-K filing text.</body></html>"],
            }
        )
        finnhub_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["2024-01-03"],
                "headline": ["Apple announces product update"],
                "summary": ["Management discussed demand and margins."],
                "source": ["Finnhub"],
                "url": ["https://finnhub.example/aapl"],
            }
        )
        yahoo_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": ["Wed, 03 Jan 2024 14:00:00 GMT"],
                "headline": ["Analysts review Apple"],
                "summary": ["Yahoo summary with <b>HTML</b> markup."],
                "source": ["Yahoo Finance RSS"],
                "url": ["https://finance.yahoo.example/aapl"],
            }
        )

        weekly_df = data_downloader.build_weekly_event_dataset(
            tickers=["AAPL"],
            price_df=price_df,
            sec_meta_df=sec_meta_df,
            sec_text_df=sec_text_df,
            finnhub_df=finnhub_df,
            yahoo_df=yahoo_df,
        )

        self.assertEqual(len(weekly_df), 1)
        row = weekly_df.iloc[0]
        self.assertEqual(row["sec_event_count"], 1)
        self.assertEqual(row["sec_filing_text_count"], 1)
        self.assertEqual(row["finnhub_event_count"], 1)
        self.assertEqual(row["yahoo_event_count"], 1)
        self.assertEqual(row["has_text"], 1)
        self.assertEqual(row["text_source_count"], 3)
        self.assertGreater(row["event_text_char_count"], 0)
        self.assertIn("Material 8-K filing text.", row["sec_filing_text"])
        self.assertIn("Management discussed demand and margins.", row["finnhub_news_text"])
        self.assertIn("Yahoo summary with HTML markup.", row["yahoo_news_text"])
        self.assertIn("Apple announces product update", row["combined_event_text"])


if __name__ == "__main__":
    unittest.main()
