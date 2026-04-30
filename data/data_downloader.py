import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from io import StringIO
import xml.etree.ElementTree as ET

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.auto import tqdm


FINNHUB_KEY = "d7anmf9r01qtpbh969ngd7anmf9r01qtpbh969o0"
SEC_USER_AGENT_DEFAULT = "Rishabh Tole rtole@seas.upenn.edu"

REQUEST_TIMEOUT = 30
SLEEP_SEC = 0.25
PIPELINE_STEP_COUNT = 8
DEFAULT_MAX_SEC_TEXT_CHARS = 20000
DEFAULT_MAX_WEEKLY_TEXT_CHARS = 50000
DEFAULT_PRICE_WORKERS = 8
DEFAULT_SEC_WORKERS = 4
DEFAULT_NEWS_WORKERS = 8
MAX_REQUEST_RETRIES = 3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
TEXT_SEPARATOR = "\n\n---\n\n"


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def progress_bar(iterable, total: int | None = None, desc: str = ""):
    return tqdm(iterable, total=total, desc=desc, leave=False)


def flatten_rows(chunks: list[list[dict]]) -> list[dict]:
    return [row for chunk in chunks for row in chunk]


def run_threaded(items, worker, desc: str, max_workers: int):
    items = list(items)
    if not items:
        return []

    if max_workers <= 1:
        results = []
        for item in progress_bar(items, total=len(items), desc=desc):
            try:
                results.append(worker(item))
            except Exception as exc:
                print(f"[WARN] {desc} failed for {item}: {exc}")
        return results

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in progress_bar(as_completed(futures), total=len(futures), desc=desc):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[WARN] {desc} failed for {item}: {exc}")
    return results


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    raw = str(value)
    if not raw.strip():
        return ""

    extracted = raw
    if "<" in raw and ">" in raw:
        parser = HTMLTextExtractor()
        try:
            parser.feed(raw)
            parser.close()
            extracted = " ".join(parser.parts) or raw
        except Exception:
            extracted = raw

    return " ".join(unescape(extracted).split())


def truncate_text(text: str, max_chars: int | None) -> str:
    cleaned = clean_text(text)
    if max_chars is None or max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def join_text_values(values, max_chars: int | None = DEFAULT_MAX_WEEKLY_TEXT_CHARS) -> str:
    parts = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        parts.append(text)
        seen.add(text)

    return truncate_text(TEXT_SEPARATOR.join(parts), max_chars)


def format_news_text(row: pd.Series) -> str:
    source = clean_text(row.get("source", ""))
    headline = clean_text(row.get("headline", ""))
    summary = clean_text(row.get("summary", ""))
    pieces = [piece for piece in [headline, summary] if piece]
    if not pieces:
        return ""

    body = " ".join(pieces)
    if source:
        return f"{source}: {body}"
    return body


def get_date_range(years: int) -> tuple[str, str]:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def safe_request(url: str, method: str = "GET", **kwargs) -> requests.Response | None:
    for attempt in range(MAX_REQUEST_RETRIES + 1):
        try:
            r = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            if r.status_code == 200:
                return r

            should_retry = r.status_code in RETRY_STATUS_CODES and attempt < MAX_REQUEST_RETRIES
            if should_retry:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 0.5 * (2**attempt)
                time.sleep(delay)
                continue

            print(f"[WARN] Request failed {r.status_code}: {url}")
            return None
        except Exception as e:
            if attempt < MAX_REQUEST_RETRIES:
                time.sleep(0.5 * (2**attempt))
                continue
            print(f"[WARN] Request error for {url}: {e}")
            return None

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


def fetch_market_cap_for_ticker(raw_ticker: str) -> dict:
    ticker = normalize_ticker_for_yfinance(raw_ticker)
    try:
        ticker_obj = yf.Ticker(ticker)
        market_cap = None

        try:
            fast_info = ticker_obj.fast_info
            if isinstance(fast_info, dict):
                market_cap = fast_info.get("market_cap") or fast_info.get("marketCap")
            else:
                market_cap = getattr(fast_info, "market_cap", None)
                if market_cap is None:
                    try:
                        market_cap = fast_info["market_cap"]
                    except Exception:
                        market_cap = fast_info["marketCap"]
        except Exception:
            market_cap = None

        if not market_cap:
            try:
                market_cap = ticker_obj.info.get("marketCap")
            except Exception:
                market_cap = None

        return {
            "ticker": raw_ticker,
            "market_cap": float(market_cap or 0),
        }
    except Exception as e:
        print(f"[WARN] Could not fetch market cap for {raw_ticker}: {e}")
        return {"ticker": raw_ticker, "market_cap": 0.0}


def sort_tickers_by_market_cap(tickers: list[str], max_workers: int) -> list[str]:
    print(f"[INFO] Ranking S&P 500 tickers by market cap with {max_workers} worker(s)...")
    rows = run_threaded(
        tickers,
        fetch_market_cap_for_ticker,
        "Market caps",
        max_workers,
    )
    if not any(row["market_cap"] > 0 for row in rows):
        print("[WARN] Market cap lookup failed for all tickers; keeping original S&P 500 order.")
        return tickers

    ranked = sorted(rows, key=lambda row: row["market_cap"], reverse=True)
    return [row["ticker"] for row in ranked]


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


def fetch_price_data_for_ticker(raw_ticker: str, start: str, end: str) -> pd.DataFrame:
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
            return pd.DataFrame()

        df = df.reset_index()

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
            return pd.DataFrame()

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        df["ticker"] = raw_ticker
        df["return_1d"] = df["Adj Close"].pct_change()
        df["return_5d"] = df["Adj Close"].pct_change(5)
        df["vol_ma_5"] = df["Volume"].rolling(5).mean()
        df["price_ma_5"] = df["Adj Close"].rolling(5).mean()
        df["price_ma_20"] = df["Adj Close"].rolling(20).mean()
        df["volatility_20"] = df["return_1d"].rolling(20).std()
        return df
    except Exception as e:
        print(f"[WARN] Error downloading {raw_ticker}: {e}")
        return pd.DataFrame()


def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
    max_workers: int = DEFAULT_PRICE_WORKERS,
) -> pd.DataFrame:
    print(f"[INFO] Downloading price data with {max_workers} worker(s)...")
    frames = [
        frame
        for frame in run_threaded(
            tickers,
            lambda raw_ticker: fetch_price_data_for_ticker(raw_ticker, start, end),
            "Price data",
            max_workers,
        )
        if not frame.empty
    ]

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


def fetch_sec_filing_text(
    filing_url: str,
    user_agent: str,
    max_chars: int = DEFAULT_MAX_SEC_TEXT_CHARS,
) -> str:
    headers = {"User-Agent": user_agent}
    r = safe_request(filing_url, headers=headers)
    if r is None:
        return ""
    return truncate_text(r.text, max_chars)


def fetch_sec_filing_text_row(
    row: dict,
    user_agent: str,
    max_chars: int = DEFAULT_MAX_SEC_TEXT_CHARS,
) -> dict:
    filing_text = fetch_sec_filing_text(
        row["filing_url"],
        user_agent,
        max_chars=max_chars,
    )
    return {
        "ticker": row["ticker"],
        "filing_date": row["filing_date"],
        "accession_number": row["accession_number"],
        "filing_url": row["filing_url"],
        "filing_text": filing_text,
    }


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
    sec_text_df: pd.DataFrame,
    finnhub_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
    max_weekly_text_chars: int = DEFAULT_MAX_WEEKLY_TEXT_CHARS,
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

    if not sec_text_df.empty:
        sec_text_df = sec_text_df.copy()
        sec_text_df["filing_date"] = pd.to_datetime(sec_text_df["filing_date"], errors="coerce")
        sec_text_df["week_start"] = sec_text_df["filing_date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        sec_text_df["filing_text"] = sec_text_df["filing_text"].map(clean_text)
        sec_text_df = sec_text_df.dropna(subset=["week_start"])
        sec_text_df = sec_text_df[sec_text_df["filing_text"].str.len() > 0]
        if not sec_text_df.empty:
            sec_text_weekly = (
                sec_text_df.groupby(["ticker", "week_start"], as_index=False)
                .agg(
                    sec_filing_text_count=("filing_text", "count"),
                    sec_filing_text=(
                        "filing_text",
                        lambda values: join_text_values(values, max_weekly_text_chars),
                    ),
                )
            )
        else:
            sec_text_weekly = pd.DataFrame(
                columns=["ticker", "week_start", "sec_filing_text_count", "sec_filing_text"]
            )
    else:
        sec_text_weekly = pd.DataFrame(
            columns=["ticker", "week_start", "sec_filing_text_count", "sec_filing_text"]
        )

    if not finnhub_df.empty:
        finnhub_df = finnhub_df.copy()
        finnhub_df["date"] = pd.to_datetime(finnhub_df["date"], errors="coerce")
        finnhub_df["week_start"] = finnhub_df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        finnhub_df["news_text"] = finnhub_df.apply(format_news_text, axis=1)
        finnhub_df["_event_row"] = 1
        finnhub_weekly = (
            finnhub_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                finnhub_event_count=("_event_row", "sum"),
                finnhub_headline_sample=("headline", "last"),
                finnhub_summary_sample=("summary", "last"),
                finnhub_news_text=(
                    "news_text",
                    lambda values: join_text_values(values, max_weekly_text_chars),
                ),
            )
        )
    else:
        finnhub_weekly = pd.DataFrame(
            columns=[
                "ticker",
                "week_start",
                "finnhub_event_count",
                "finnhub_headline_sample",
                "finnhub_summary_sample",
                "finnhub_news_text",
            ]
        )

    if not yahoo_df.empty:
        yahoo_df = yahoo_df.copy()
        yahoo_df["date"] = pd.to_datetime(yahoo_df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        yahoo_df["week_start"] = yahoo_df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        yahoo_df["news_text"] = yahoo_df.apply(format_news_text, axis=1)
        yahoo_df["_event_row"] = 1
        yahoo_weekly = (
            yahoo_df.groupby(["ticker", "week_start"], as_index=False)
            .agg(
                yahoo_event_count=("_event_row", "sum"),
                yahoo_headline_sample=("headline", "last"),
                yahoo_summary_sample=("summary", "last"),
                yahoo_news_text=(
                    "news_text",
                    lambda values: join_text_values(values, max_weekly_text_chars),
                ),
            )
        )
    else:
        yahoo_weekly = pd.DataFrame(
            columns=[
                "ticker",
                "week_start",
                "yahoo_event_count",
                "yahoo_headline_sample",
                "yahoo_summary_sample",
                "yahoo_news_text",
            ]
        )

    out = numeric_weekly.merge(sec_weekly, on=["ticker", "week_start"], how="left")
    out = out.merge(sec_text_weekly, on=["ticker", "week_start"], how="left")
    out = out.merge(finnhub_weekly, on=["ticker", "week_start"], how="left")
    out = out.merge(yahoo_weekly, on=["ticker", "week_start"], how="left")

    for col in ["sec_event_count", "sec_filing_text_count", "finnhub_event_count", "yahoo_event_count"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    text_columns = ["sec_filing_text", "finnhub_news_text", "yahoo_news_text"]
    sample_text_columns = [
        "sec_latest_filing_url",
        "finnhub_headline_sample",
        "finnhub_summary_sample",
        "yahoo_headline_sample",
        "yahoo_summary_sample",
    ]
    for col in text_columns + sample_text_columns:
        if col in out.columns:
            out[col] = out[col].fillna("").map(clean_text)

    out["combined_event_text"] = out[text_columns].apply(
        lambda row: join_text_values(row.tolist(), max_weekly_text_chars),
        axis=1,
    )
    out["has_text"] = (out["combined_event_text"].str.len() > 0).astype(int)
    out["text_source_count"] = out[text_columns].apply(
        lambda row: sum(1 for value in row if clean_text(value)),
        axis=1,
    )
    out["event_text_char_count"] = out["combined_event_text"].str.len().astype(int)

    return out


def build_coverage_summary(
    tickers: list[str],
    price_df: pd.DataFrame,
    sec_meta_df: pd.DataFrame,
    finnhub_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ticker in progress_bar(tickers, total=len(tickers), desc="Coverage summary"):
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
    step_progress = tqdm(total=PIPELINE_STEP_COUNT, desc="Pipeline steps")

    start, end = get_date_range(args.years)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_sp500_tickers()

    if args.top_by_market_cap:
        tickers = sort_tickers_by_market_cap(tickers, args.price_workers)

    if args.max_tickers:
        tickers = tickers[: args.max_tickers]

    print(f"[INFO] Using {len(tickers)} tickers from {start} to {end}")

    price_df = fetch_price_data(
        tickers,
        start,
        end,
        max_workers=args.price_workers,
    )
    step_progress.update(1)
    if price_df.empty:
        step_progress.close()
        raise RuntimeError("No price data downloaded.")

    price_features_df = compute_alpha_and_labels(price_df, start, end)
    save_df(price_df, os.path.join(args.outdir, "prices_raw"))
    save_df(price_features_df, os.path.join(args.outdir, "price_features"))
    step_progress.update(1)

    cik_map = build_ticker_to_cik_map()
    step_progress.update(1)

    print(f"[INFO] Fetching SEC filings metadata with {args.sec_workers} worker(s)...")
    sec_meta_rows = flatten_rows(
        run_threaded(
            tickers,
            lambda ticker: fetch_sec_8k_filings_for_ticker(
                ticker,
                cik_map,
                args.sec_user_agent,
            ),
            "SEC metadata",
            args.sec_workers,
        )
    )

    sec_meta_df = pd.DataFrame(sec_meta_rows)
    if sec_meta_df.empty:
        sec_meta_df = pd.DataFrame(
            columns=["ticker", "cik", "filing_date", "form", "accession_number", "primary_document", "filing_url"]
        )
    save_df(sec_meta_df, os.path.join(args.outdir, "sec_filings_meta"))
    step_progress.update(1)

    print(f"[INFO] Fetching SEC filing text with {args.sec_workers} worker(s)...")
    sec_text_rows = (
        run_threaded(
            sec_meta_df.to_dict("records"),
            lambda row: fetch_sec_filing_text_row(
                row,
                args.sec_user_agent,
                max_chars=args.max_sec_text_chars,
            ),
            "SEC filing text",
            args.sec_workers,
        )
        if not sec_meta_df.empty
        else []
    )

    sec_text_df = pd.DataFrame(sec_text_rows)
    if sec_text_df.empty:
        sec_text_df = pd.DataFrame(
            columns=["ticker", "filing_date", "accession_number", "filing_url", "filing_text"]
        )
    save_df(sec_text_df, os.path.join(args.outdir, "sec_filings_text"))
    step_progress.update(1)

    print(f"[INFO] Fetching Finnhub news with {args.news_workers} worker(s)...")
    finnhub_rows = (
        flatten_rows(
            run_threaded(
                tickers,
                lambda ticker: fetch_finnhub_news(ticker, start, end, args.finnhub_key),
                "Finnhub news",
                args.news_workers,
            )
        )
        if args.finnhub_key
        else []
    )

    finnhub_df = pd.DataFrame(finnhub_rows)
    if finnhub_df.empty:
        finnhub_df = pd.DataFrame(
            columns=["ticker", "date", "headline", "summary", "source", "url", "category", "image"]
        )
    save_df(finnhub_df, os.path.join(args.outdir, "finnhub_news"))
    step_progress.update(1)

    print(f"[INFO] Fetching Yahoo RSS fallback news with {args.news_workers} worker(s)...")
    yahoo_rows = flatten_rows(
        run_threaded(
            tickers,
            fetch_yahoo_rss_news,
            "Yahoo RSS",
            args.news_workers,
        )
    )

    yahoo_df = pd.DataFrame(yahoo_rows)
    if yahoo_df.empty:
        yahoo_df = pd.DataFrame(
            columns=["ticker", "date", "headline", "summary", "source", "url"]
        )
    save_df(yahoo_df, os.path.join(args.outdir, "yahoo_news"))
    step_progress.update(1)

    weekly_event_df = build_weekly_event_dataset(
        tickers=tickers,
        price_df=price_features_df,
        sec_meta_df=sec_meta_df,
        sec_text_df=sec_text_df,
        finnhub_df=finnhub_df,
        yahoo_df=yahoo_df,
        max_weekly_text_chars=args.max_weekly_text_chars,
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
    step_progress.update(1)
    step_progress.close()

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
        "weekly_event_rows_with_text": (
            int(weekly_event_df["has_text"].sum()) if "has_text" in weekly_event_df else 0
        ),
        "weekly_event_text_characters": int(weekly_event_df["event_text_char_count"].sum())
        if "event_text_char_count" in weekly_event_df
        else 0,
        "max_sec_text_chars": args.max_sec_text_chars,
        "max_weekly_text_chars": args.max_weekly_text_chars,
        "price_workers": args.price_workers,
        "sec_workers": args.sec_workers,
        "news_workers": args.news_workers,
        "top_by_market_cap": args.top_by_market_cap,
    }

    with open(os.path.join(args.outdir, "coverage_overall.json"), "w") as f:
        json.dump(coverage_overall, f, indent=2)

    print("\n[INFO] Done.")
    print(json.dumps(coverage_overall, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download stock, SEC, and news data for project coverage testing."
    )

    parser.add_argument("--outdir", type=str, default="data_dump")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--max-tickers", type=int, default=25)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--finnhub-key", type=str, default=FINNHUB_KEY)
    parser.add_argument("--sec-user-agent", type=str, default=SEC_USER_AGENT_DEFAULT)
    parser.add_argument("--top-by-market-cap", action="store_true")
    parser.add_argument("--max-sec-text-chars", type=int, default=DEFAULT_MAX_SEC_TEXT_CHARS)
    parser.add_argument("--max-weekly-text-chars", type=int, default=DEFAULT_MAX_WEEKLY_TEXT_CHARS)
    parser.add_argument("--price-workers", type=int, default=DEFAULT_PRICE_WORKERS)
    parser.add_argument("--sec-workers", type=int, default=DEFAULT_SEC_WORKERS)
    parser.add_argument("--news-workers", type=int, default=DEFAULT_NEWS_WORKERS)

    args = parser.parse_args()
    run_pipeline(args)
