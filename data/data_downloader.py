import os
import time
import json
import argparse
from datetime import datetime, timedelta
from io import StringIO
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import numpy as np
import yfinance as yf


FINNHUB_KEY = "d7anmf9r01qtpbh969ngd7anmf9r01qtpbh969o0"
SEC_USER_AGENT_DEFAULT = "Rishabh Tole rtole@seas.upenn.edu"

REQUEST_TIMEOUT = 30
SLEEP_SEC = 0.25


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_date_range(years: int) -> tuple[str, str]:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def safe_request(url: str, method: str = "GET", **kwargs) -> requests.Response | None:
    try:
        r = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        if r.status_code == 200:
            return r
        print(f"[WARN] Request failed {r.status_code}: {url}")
        return None
    except Exception as e:
        print(f"[WARN] Request error for {url}: {e}")
        return None


def normalize_ticker_for_yfinance(ticker: str) -> str:
    return ticker.replace(".", "-")


def normalize_ticker_for_sec_lookup(ticker: str) -> str:
    return ticker.replace("-", ".")


def get_sp500_table() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = safe_request(url, headers=headers)
    if response is None:
        raise RuntimeError("Failed to fetch S&P 500 company list from Wikipedia")

    tables = pd.read_html(StringIO(response.text))
    df = tables[0].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


def get_sp500_tickers() -> list[str]:
    df = get_sp500_table()
    return df["Symbol"].tolist()


def build_ticker_to_cik_map() -> dict[str, str]:
    headers = {"User-Agent": SEC_USER_AGENT_DEFAULT}
    url = "https://www.sec.gov/files/company_tickers.json"
    r = safe_request(url, headers=headers)
    if r is None:
        return {}

    data = r.json()
    mapping = {}
    for _, item in data.items():
        ticker = str(item.get("ticker", "")).upper().strip()
        cik = str(item.get("cik_str", "")).strip()
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    return mapping


def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    print("[INFO] Downloading price data...")
    frames = []

    for raw_ticker in tickers:
        ticker = normalize_ticker_for_yfinance(raw_ticker)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if df.empty:
                print(f"[WARN] No price data for {raw_ticker}")
                continue

            df = df.reset_index()

            # Flatten MultiIndex columns if yfinance returns them
            flattened_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    flattened_cols.append(col[0] if col[0] else col[-1])
                else:
                    flattened_cols.append(col)
            df.columns = flattened_cols

            required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"[WARN] Missing expected columns for {raw_ticker}: {missing}")
                continue

            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]

            df["ticker"] = raw_ticker
            df["return_1d"] = df["Adj Close"].pct_change()
            df["return_5d"] = df["Adj Close"].pct_change(5)
            df["vol_ma_5"] = df["Volume"].rolling(5).mean()
            df["price_ma_5"] = df["Adj Close"].rolling(5).mean()
            df["price_ma_20"] = df["Adj Close"].rolling(20).mean()
            df["volatility_20"] = df["return_1d"].rolling(20).std()

            frames.append(df)
            time.sleep(SLEEP_SEC)
        except Exception as e:
            print(f"[WARN] Error downloading {raw_ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out


def compute_alpha_and_labels(price_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    print("[INFO] Computing alpha and labels...")
    sp = yf.download(
        "^GSPC",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if sp.empty:
        raise RuntimeError("Failed to download ^GSPC data.")

    sp = sp.reset_index()

    flattened_cols = []
    for col in sp.columns:
        if isinstance(col, tuple):
            flattened_cols.append(col[0] if col[0] else col[-1])
        else:
            flattened_cols.append(col)
    sp.columns = flattened_cols

    if "Adj Close" not in sp.columns:
        sp["Adj Close"] = sp["Close"]

    sp["Date"] = pd.to_datetime(sp["Date"]).dt.normalize()
    sp["sp_return_1d"] = sp["Adj Close"].pct_change()
    sp["sp_return_5d"] = sp["Adj Close"].pct_change(5)

    merged = price_df.merge(
        sp[["Date", "sp_return_1d", "sp_return_5d"]],
        on="Date",
        how="left",
    )

    merged["alpha_1d"] = merged["return_1d"] - merged["sp_return_1d"]
    merged["alpha_5d"] = merged["return_5d"] - merged["sp_return_5d"]

    merged["future_stock_return_5d"] = (
        merged.groupby("ticker")["Adj Close"].shift(-5) / merged["Adj Close"] - 1
    )
    merged["future_sp_return_5d"] = merged["sp_return_5d"].shift(-5)
    merged["future_alpha_5d"] = (
        merged["future_stock_return_5d"] - merged["future_sp_return_5d"]
    )

    merged["label_abs_alpha_gt_1pct"] = (
        merged["future_alpha_5d"].abs() > 0.01
    ).astype("Int64")

    return merged


def fetch_sec_8k_filings_for_ticker(
    ticker: str,
    cik_map: dict[str, str],
    user_agent: str,
) -> list[dict]:
    sec_ticker = normalize_ticker_for_sec_lookup(ticker).upper()
    cik = cik_map.get(sec_ticker)
    if not cik:
        print(f"[WARN] No CIK found for {ticker}")
        return []

    headers = {"User-Agent": user_agent}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = safe_request(url, headers=headers)
    if r is None:
        return []

    try:
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        rows = []
        for form, filing_date, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
            if form != "8-K":
                continue

            accession_nodashes = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_nodashes}/{primary_doc}"
            )

            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "filing_date": filing_date,
                    "form": form,
                    "accession_number": accession,
                    "primary_document": primary_doc,
                    "filing_url": filing_url,
                }
            )
        return rows
    except Exception as e:
        print(f"[WARN] Failed parsing SEC data for {ticker}: {e}")
        return []


def fetch_sec_filing_text(filing_url: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    r = safe_request(filing_url, headers=headers)
    if r is None:
        return ""
    text = r.text
    if not text:
        return ""
    cleaned = " ".join(text.split())
    return cleaned[:20000]


def fetch_finnhub_news(ticker: str, start: str, end: str, api_key: str) -> list[dict]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start,
        "to": end,
        "token": api_key,
    }

    r = safe_request(url, params=params)
    if r is None:
        return []

    try:
        data = r.json()
        if not isinstance(data, list):
            return []

        rows = []
        for item in data:
            ts = item.get("datetime")
            date_str = None
            if ts:
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            rows.append(
                {
                    "ticker": ticker,
                    "date": date_str,
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "category": item.get("category", ""),
                    "image": item.get("image", ""),
                }
            )
        return rows
    except Exception as e:
        print(f"[WARN] Failed parsing Finnhub news for {ticker}: {e}")
        return []


def fetch_yahoo_rss_news(ticker: str) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    r = safe_request(url)
    if r is None:
        return []

    try:
        root = ET.fromstring(r.text)
        rows = []
        for item in root.findall(".//item"):
            pub_date = item.findtext("pubDate", default="")
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            description = item.findtext("description", default="")
            rows.append(
                {
                    "ticker": ticker,
                    "date": pub_date,
                    "headline": title,
                    "summary": description,
                    "source": "Yahoo Finance RSS",
                    "url": link,
                }
            )
        return rows
    except Exception as e:
        print(f"[WARN] Failed parsing Yahoo RSS for {ticker}: {e}")
        return []


def build_weekly_event_dataset(
    tickers: list[str],
    price_df: pd.DataFrame,
    sec_meta_df: pd.DataFrame,
    finnhub_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
) -> pd.DataFrame:
    print("[INFO] Building weekly event dataset...")

    if price_df.empty:
        return pd.DataFrame()

    work = price_df.copy()
    work["week_start"] = work["Date"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    numeric_weekly = (
        work.sort_values(["ticker", "Date"])
        .groupby(["ticker", "week_start"], as_index=False)
        .agg(
            last_date=("Date", "max"),
            close=("Adj Close", "last"),
            volume=("Volume", "last"),
            vol_ma_5=("vol_ma_5", "last"),
            price_ma_5=("price_ma_5", "last"),
            price_ma_20=("price_ma_20", "last"),
            volatility_20=("volatility_20", "last"),
            future_alpha_5d=("future_alpha_5d", "last"),
            label_abs_alpha_gt_1pct=("label_abs_alpha_gt_1pct", "last"),
        )
    )

    if not sec_meta_df.empty:
        sec_meta_df = sec_meta_df.copy()
        sec_meta_df["filing_date"] = pd.to_datetime(sec_meta_df["filing_date"], errors="coerce")
        sec_meta_df["week_start"] = sec_meta_df["filing_date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        sec_weekly = (
            sec_meta_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                sec_event_count=("accession_number", "count"),
                sec_latest_filing_url=("filing_url", "last"),
            )
        )
    else:
        sec_weekly = pd.DataFrame(columns=["ticker", "week_start", "sec_event_count", "sec_latest_filing_url"])

    if not finnhub_df.empty:
        finnhub_df = finnhub_df.copy()
        finnhub_df["date"] = pd.to_datetime(finnhub_df["date"], errors="coerce")
        finnhub_df["week_start"] = finnhub_df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        finnhub_weekly = (
            finnhub_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                finnhub_event_count=("headline", "count"),
                finnhub_headline_sample=("headline", "last"),
            )
        )
    else:
        finnhub_weekly = pd.DataFrame(columns=["ticker", "week_start", "finnhub_event_count", "finnhub_headline_sample"])

    if not yahoo_df.empty:
        yahoo_df = yahoo_df.copy()
        yahoo_df["date"] = pd.to_datetime(yahoo_df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        yahoo_df["week_start"] = yahoo_df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        yahoo_weekly = (
            yahoo_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                yahoo_event_count=("headline", "count"),
                yahoo_headline_sample=("headline", "last"),
            )
        )
    else:
        yahoo_weekly = pd.DataFrame(columns=["ticker", "week_start", "yahoo_event_count", "yahoo_headline_sample"])

    out = numeric_weekly.merge(sec_weekly, on=["ticker", "week_start"], how="left")
    out = out.merge(finnhub_weekly, on=["ticker", "week_start"], how="left")
    out = out.merge(yahoo_weekly, on=["ticker", "week_start"], how="left")

    for col in ["sec_event_count", "finnhub_event_count", "yahoo_event_count"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    return out


def build_coverage_summary(
    tickers: list[str],
    price_df: pd.DataFrame,
    sec_meta_df: pd.DataFrame,
    finnhub_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "ticker": ticker,
                "price_rows": int((price_df["ticker"] == ticker).sum()) if not price_df.empty else 0,
                "sec_events": int((sec_meta_df["ticker"] == ticker).sum()) if not sec_meta_df.empty else 0,
                "finnhub_news": int((finnhub_df["ticker"] == ticker).sum()) if not finnhub_df.empty else 0,
                "yahoo_news": int((yahoo_df["ticker"] == ticker).sum()) if not yahoo_df.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def save_df(df: pd.DataFrame, base_path_no_ext: str) -> None:
    csv_path = f"{base_path_no_ext}.csv"
    df.to_csv(csv_path, index=False)

    try:
        parquet_path = f"{base_path_no_ext}.parquet"
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[WARN] Could not save parquet for {base_path_no_ext}: {e}")


def run_pipeline(args) -> None:
    ensure_dir(args.outdir)

    start, end = get_date_range(args.years)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_sp500_tickers()

    if args.max_tickers:
        tickers = tickers[: args.max_tickers]

    print(f"[INFO] Using {len(tickers)} tickers from {start} to {end}")

    price_df = fetch_price_data(tickers, start, end)
    if price_df.empty:
        raise RuntimeError("No price data downloaded.")

    price_features_df = compute_alpha_and_labels(price_df, start, end)
    save_df(price_df, os.path.join(args.outdir, "prices_raw"))
    save_df(price_features_df, os.path.join(args.outdir, "price_features"))

    cik_map = build_ticker_to_cik_map()

    print("[INFO] Fetching SEC filings metadata...")
    sec_meta_rows = []
    for ticker in tickers:
        sec_meta_rows.extend(fetch_sec_8k_filings_for_ticker(ticker, cik_map, args.sec_user_agent))
        time.sleep(SLEEP_SEC)

    sec_meta_df = pd.DataFrame(sec_meta_rows)
    if sec_meta_df.empty:
        sec_meta_df = pd.DataFrame(
            columns=["ticker", "cik", "filing_date", "form", "accession_number", "primary_document", "filing_url"]
        )
    save_df(sec_meta_df, os.path.join(args.outdir, "sec_filings_meta"))

    print("[INFO] Fetching SEC filing text...")
    sec_text_rows = []
    if not sec_meta_df.empty:
        for _, row in sec_meta_df.iterrows():
            filing_text = fetch_sec_filing_text(row["filing_url"], args.sec_user_agent)
            sec_text_rows.append(
                {
                    "ticker": row["ticker"],
                    "filing_date": row["filing_date"],
                    "accession_number": row["accession_number"],
                    "filing_url": row["filing_url"],
                    "filing_text": filing_text,
                }
            )
            time.sleep(SLEEP_SEC)

    sec_text_df = pd.DataFrame(sec_text_rows)
    if sec_text_df.empty:
        sec_text_df = pd.DataFrame(
            columns=["ticker", "filing_date", "accession_number", "filing_url", "filing_text"]
        )
    save_df(sec_text_df, os.path.join(args.outdir, "sec_filings_text"))

    print("[INFO] Fetching Finnhub news...")
    finnhub_rows = []
    if args.finnhub_key:
        for ticker in tickers:
            finnhub_rows.extend(fetch_finnhub_news(ticker, start, end, args.finnhub_key))
            time.sleep(SLEEP_SEC)

    finnhub_df = pd.DataFrame(finnhub_rows)
    if finnhub_df.empty:
        finnhub_df = pd.DataFrame(
            columns=["ticker", "date", "headline", "summary", "source", "url", "category", "image"]
        )
    save_df(finnhub_df, os.path.join(args.outdir, "finnhub_news"))

    print("[INFO] Fetching Yahoo RSS fallback news...")
    yahoo_rows = []
    for ticker in tickers:
        yahoo_rows.extend(fetch_yahoo_rss_news(ticker))
        time.sleep(SLEEP_SEC)

    yahoo_df = pd.DataFrame(yahoo_rows)
    if yahoo_df.empty:
        yahoo_df = pd.DataFrame(
            columns=["ticker", "date", "headline", "summary", "source", "url"]
        )
    save_df(yahoo_df, os.path.join(args.outdir, "yahoo_news"))

    weekly_event_df = build_weekly_event_dataset(
        tickers=tickers,
        price_df=price_features_df,
        sec_meta_df=sec_meta_df,
        finnhub_df=finnhub_df,
        yahoo_df=yahoo_df,
    )
    save_df(weekly_event_df, os.path.join(args.outdir, "weekly_event_dataset"))

    coverage_summary_df = build_coverage_summary(
        tickers=tickers,
        price_df=price_df,
        sec_meta_df=sec_meta_df,
        finnhub_df=finnhub_df,
        yahoo_df=yahoo_df,
    )
    save_df(coverage_summary_df, os.path.join(args.outdir, "coverage_summary_by_ticker"))

    coverage_overall = {
        "date_generated": datetime.now().isoformat(),
        "num_tickers": len(tickers),
        "date_range": {"start": start, "end": end},
        "prices_raw_rows": int(len(price_df)),
        "price_feature_rows": int(len(price_features_df)),
        "sec_filings_meta_rows": int(len(sec_meta_df)),
        "sec_filings_text_rows": int(len(sec_text_df)),
        "finnhub_news_rows": int(len(finnhub_df)),
        "yahoo_news_rows": int(len(yahoo_df)),
        "weekly_event_rows": int(len(weekly_event_df)),
    }

    with open(os.path.join(args.outdir, "coverage_overall.json"), "w") as f:
        json.dump(coverage_overall, f, indent=2)

    print("\n[INFO] Done.")
    print(json.dumps(coverage_overall, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock, SEC, and news data for project coverage testing.")

    parser.add_argument("--outdir", type=str, default="data_dump")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--max-tickers", type=int, default=25)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--finnhub-key", type=str, default=FINNHUB_KEY)
    parser.add_argument("--sec-user-agent", type=str, default=SEC_USER_AGENT_DEFAULT)

    args = parser.parse_args()
    run_pipeline(args)
