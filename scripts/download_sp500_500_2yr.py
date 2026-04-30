from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_downloader import FINNHUB_KEY, SEC_USER_AGENT_DEFAULT, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the full S&P 500 universe over the last 2 years."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/sp500_500_2yr",
        help="Output directory for generated CSV/Parquet files.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=500,
        help="Cap on how many S&P 500 tickers to process.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="How many years of history to download.",
    )
    parser.add_argument(
        "--finnhub-key",
        type=str,
        default=FINNHUB_KEY,
        help="Finnhub API key used for company news.",
    )
    parser.add_argument(
        "--sec-user-agent",
        type=str,
        default=SEC_USER_AGENT_DEFAULT,
        help="User-Agent header used for SEC requests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_args = SimpleNamespace(
        outdir=args.outdir,
        years=args.years,
        max_tickers=args.max_tickers,
        tickers=None,
        finnhub_key=args.finnhub_key,
        sec_user_agent=args.sec_user_agent,
    )
    run_pipeline(pipeline_args)


if __name__ == "__main__":
    main()
