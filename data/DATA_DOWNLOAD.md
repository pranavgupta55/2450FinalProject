# Data Downloading

The main download script lives at `data/data_downloader.py`.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

If you are using the project virtual environment:

```bash
source 2450venv/bin/activate
```

## Main Run

Run the full downloader with a bounded ticker count:

```bash
python data/data_downloader.py --years 5 --outdir data/raw_data --max-tickers 500
```

## Small Test Run

Run a small reproducible test dataset with four tickers:

```bash
python data/data_downloader.py --years 1 --tickers AAPL,MSFT,NVDA,AMZN --max-tickers 4 --outdir data/test_ds
```

## Useful Arguments

- `--outdir`: output directory for generated files
- `--years`: number of years of historical data to pull
- `--max-tickers`: cap on how many tickers are processed
- `--tickers`: comma-separated ticker list for targeted runs
- `--finnhub-key`: Finnhub API key for company news
- `--sec-user-agent`: user-agent string sent to SEC endpoints

## Outputs

The downloader writes both CSV and Parquet files when possible:

- `prices_raw`
- `price_features`
- `sec_filings_meta`
- `sec_filings_text`
- `finnhub_news`
- `yahoo_news`
- `weekly_event_dataset`
- `coverage_summary_by_ticker`
- `coverage_overall.json`

## Notes

- The script now shows progress bars for each major stage.
- `data/raw_data/` and `data/test_ds/` are ignored by git because they are generated datasets.
