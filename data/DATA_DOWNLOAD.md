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
export FINNHUB_API_KEY="your_finnhub_api_key"

python data/data_downloader.py \
  --years 5 \
  --outdir data/raw_data \
  --max-tickers 500 \
  --finnhub-key "$FINNHUB_API_KEY" \
  --sec-user-agent "Your Name your.email@example.com"
```

## Top 50 Five-Year Run

Run a faster 50-stock, 5-year dataset with parallel fetching:

```bash
export FINNHUB_API_KEY="your_finnhub_api_key"

python data/data_downloader.py \
  --years 5 \
  --top-by-market-cap \
  --max-tickers 50 \
  --outdir data/top50_5yr \
  --finnhub-key "$FINNHUB_API_KEY" \
  --sec-user-agent "Your Name your.email@example.com" \
  --price-workers 10 \
  --sec-workers 4 \
  --news-workers 6
```

## Small Test Run

Run a small reproducible test dataset with four tickers:

```bash
export FINNHUB_API_KEY="your_finnhub_api_key"

python data/data_downloader.py \
  --years 1 \
  --tickers AAPL,MSFT,NVDA,AMZN \
  --max-tickers 4 \
  --outdir data/test_ds \
  --finnhub-key "$FINNHUB_API_KEY" \
  --sec-user-agent "Your Name your.email@example.com"
```

## Useful Arguments

- `--outdir`: output directory for generated files
- `--years`: number of years of historical data to pull
- `--max-tickers`: cap on how many tickers are processed
- `--tickers`: comma-separated ticker list for targeted runs
- `--top-by-market-cap`: rank the S&P 500 universe by live yfinance market cap before applying `--max-tickers`
- `--finnhub-key`: Finnhub API key for company news
- `--sec-user-agent`: user-agent string sent to SEC endpoints
- `--max-sec-text-chars`: maximum cleaned characters saved per SEC filing
- `--max-weekly-text-chars`: maximum characters saved in each weekly aggregate text field
- `--price-workers`: parallel yfinance ticker downloads
- `--sec-workers`: parallel SEC metadata and SEC filing-text downloads
- `--news-workers`: parallel Finnhub and Yahoo RSS downloads

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

`weekly_event_dataset` now contains both numerical columns and weekly text aggregates:

- `sec_filing_text`: cleaned SEC 8-K filing text grouped by ticker-week
- `finnhub_news_text`: Finnhub headline and summary text grouped by ticker-week
- `yahoo_news_text`: Yahoo Finance RSS headline and summary text grouped by ticker-week
- `combined_event_text`: SEC, Finnhub, and Yahoo text joined into one weekly text field
- `has_text`, `text_source_count`, and `event_text_char_count`: numeric text-coverage features

If `weekly_event_dataset.csv` is missing these columns, regenerate the dataset with one of the commands above.

You can validate an existing generated dataset before rerunning a long download:

```bash
python scripts/validate_multimodal_dataset.py --input-dir data/test_ds
python scripts/validate_multimodal_dataset.py --input-dir data/raw_data
```

After validation, rebuild the split and train FinBERT:

```bash
python scripts/prepare_train_test_split.py --input-dir data/test_ds --dataset-name test_ds

python scripts/train_finbert_multimodal.py \
  --split-dir artifacts/splits/test_ds \
  --dataset-name test_ds \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --unfreeze-finbert-layers 12 \
  --max-text-chunks 0
```

## Notes

- The script now shows progress bars for each major stage.
- `data/raw_data/` and `data/test_ds/` are ignored by git because they are generated datasets.
