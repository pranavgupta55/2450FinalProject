from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = REPO_ROOT / "data" / "sp500_500_2yr_quarterly"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a smaller 2-year / 500-stock S&P 500 dataset aggregated to "
            "quarterly event rows. The output remains compatible with the existing "
            "training stack through weekly_event_dataset.* aliases."
        )
    )
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--max-tickers", type=int, default=500)
    parser.add_argument("--price-workers", type=int, default=16)
    parser.add_argument("--sec-workers", type=int, default=32)
    parser.add_argument(
        "--news-workers",
        type=int,
        default=16,
        help="Lower this to 1 if Yahoo RSS starts returning 429 rate-limit warnings.",
    )
    parser.add_argument("--max-sec-text-chars", type=int, default=8_000)
    parser.add_argument(
        "--max-sec-filings-per-ticker",
        type=int,
        default=4,
        help=(
            "Date-filtered cap on SEC 8-K text downloads per ticker. Default 4 "
            "keeps the quarterly dataset much faster than fetching every 8-K."
        ),
    )
    parser.add_argument("--max-quarterly-text-chars", type=int, default=50_000)
    parser.add_argument("--finnhub-key", type=str, default=None)
    parser.add_argument("--sec-user-agent", type=str, default=None)
    parser.add_argument(
        "--no-top-by-market-cap",
        action="store_true",
        help="Keep Wikipedia S&P 500 order instead of selecting/ranking by market cap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    downloader = REPO_ROOT / "data" / "data_downloader.py"
    command = [
        sys.executable,
        str(downloader),
        "--outdir",
        args.outdir,
        "--years",
        str(args.years),
        "--max-tickers",
        str(args.max_tickers),
        "--aggregation-period",
        "quarterly",
        "--price-workers",
        str(args.price_workers),
        "--sec-workers",
        str(args.sec_workers),
        "--news-workers",
        str(args.news_workers),
        "--max-sec-text-chars",
        str(args.max_sec_text_chars),
        "--max-sec-filings-per-ticker",
        str(args.max_sec_filings_per_ticker),
        "--max-weekly-text-chars",
        str(args.max_quarterly_text_chars),
    ]

    if not args.no_top_by_market_cap:
        command.append("--top-by-market-cap")
    if args.finnhub_key:
        command.extend(["--finnhub-key", args.finnhub_key])
    if args.sec_user_agent:
        command.extend(["--sec-user-agent", args.sec_user_agent])

    print("Running quarterly downloader:")
    print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    print("\nDone. Main outputs:")
    print(f"  {Path(args.outdir) / 'quarterly_event_dataset.csv'}")
    print(f"  {Path(args.outdir) / 'weekly_event_dataset.csv'}  # compatibility alias")
    print(f"  {Path(args.outdir) / 'coverage_overall.json'}")
    print("\nExpected modeling rows: roughly 500 stocks x 8 quarters = about 4,000 rows.")
    print(
        "SEC text is intentionally capped at "
        f"{args.max_sec_filings_per_ticker} filing(s) per ticker and "
        f"{args.max_sec_text_chars} characters per filing."
    )


if __name__ == "__main__":
    main()
