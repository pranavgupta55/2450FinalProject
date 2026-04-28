from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR
from src.alpha_signal.data.splitting import load_split_artifacts
from src.alpha_signal.models.training import save_experiment_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an S&P 500 buy-and-hold benchmark artifact bundle."
    )
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="default_dataset")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument(
        "--benchmark-start-date",
        type=str,
        default=None,
        help="Override the buy date. Defaults to the split test_start date.",
    )
    parser.add_argument(
        "--benchmark-end-date",
        type=str,
        default=None,
        help="Override the sell date. Defaults to the latest available benchmark date in price_features.",
    )
    return parser.parse_args()


def load_price_features(dataset_dir: str | Path) -> pd.DataFrame:
    resolved = Path(dataset_dir)
    csv_path = resolved / "price_features.csv"
    parquet_path = resolved / "price_features.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["Date"])
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Could not find price_features.csv or price_features.parquet in {resolved}")


def build_benchmark_trade_log(
    split_metadata: dict,
    input_dir: str | Path,
    initial_capital: float,
    benchmark_start_date: str | None = None,
    benchmark_end_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    price_features = load_price_features(input_dir)
    benchmark_df = (
        price_features[["Date", "sp_return_1d", "sp_return_5d"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    requested_start = pd.to_datetime(benchmark_start_date or split_metadata["test_start"])
    requested_end = pd.to_datetime(benchmark_end_date) if benchmark_end_date else None

    available_end = benchmark_df["Date"].max()
    effective_end = min(requested_end, available_end) if requested_end is not None else available_end

    benchmark_window = benchmark_df[
        (benchmark_df["Date"] >= requested_start) & (benchmark_df["Date"] <= effective_end)
    ].copy()
    if benchmark_window.empty:
        raise ValueError("No S&P 500 benchmark rows were available in the requested window.")

    benchmark_window["sp_return_1d"] = pd.to_numeric(
        benchmark_window["sp_return_1d"],
        errors="coerce",
    ).fillna(0.0)
    benchmark_window["cumulative_return"] = (1.0 + benchmark_window["sp_return_1d"]).cumprod() - 1.0
    benchmark_window["position"] = 1
    benchmark_window["capital_deployed"] = float(initial_capital)
    benchmark_window["daily_pnl_dollars"] = (
        initial_capital * benchmark_window["sp_return_1d"]
    )
    benchmark_window["cumulative_pnl_dollars"] = (
        initial_capital * benchmark_window["cumulative_return"]
    )
    benchmark_window["trade_side"] = "long"
    benchmark_window["trade_taken"] = 1
    benchmark_window["ticker"] = "^GSPC"

    total_return = float(benchmark_window["cumulative_return"].iloc[-1])
    net_pnl = float(benchmark_window["cumulative_pnl_dollars"].iloc[-1])
    start_date = pd.Timestamp(benchmark_window["Date"].iloc[0])
    end_date = pd.Timestamp(benchmark_window["Date"].iloc[-1])
    holding_days = int((end_date - start_date).days)

    summary = {
        "capital_per_trade": float(initial_capital),
        "alpha_threshold": None,
        "trades_executed": 1,
        "long_trades": 1,
        "short_trades": 0,
        "flat_rows": 0,
        "hit_rate": 1.0 if net_pnl >= 0 else 0.0,
        "gross_profit_dollars": max(net_pnl, 0.0),
        "gross_loss_dollars": min(net_pnl, 0.0),
        "net_pnl_dollars": net_pnl,
        "average_pnl_dollars": net_pnl,
        "max_cumulative_pnl_dollars": float(benchmark_window["cumulative_pnl_dollars"].max()),
        "min_cumulative_pnl_dollars": float(benchmark_window["cumulative_pnl_dollars"].min()),
        "return_on_traded_capital": total_return,
        "benchmark_symbol": "^GSPC",
        "benchmark_start_date": start_date.isoformat(),
        "benchmark_end_date": end_date.isoformat(),
        "holding_days": holding_days,
    }
    return benchmark_window, summary


def main():
    args = parse_args()
    _, test_df, split_metadata = load_split_artifacts(args.split_dir)
    input_dir = args.input_dir or split_metadata.get("input_dir")
    if not input_dir:
        raise SystemExit("Could not determine the dataset input directory. Pass --input-dir explicitly.")

    trade_log_df, trading_summary = build_benchmark_trade_log(
        split_metadata=split_metadata,
        input_dir=input_dir,
        initial_capital=args.initial_capital,
        benchmark_start_date=args.benchmark_start_date,
        benchmark_end_date=args.benchmark_end_date,
    )

    predictions = pd.DataFrame(
        [
            {
                "ticker": "^GSPC",
                "start_date": trading_summary["benchmark_start_date"],
                "end_date": trading_summary["benchmark_end_date"],
                "holding_days": trading_summary["holding_days"],
                "benchmark_return": trading_summary["return_on_traded_capital"],
                "benchmark_net_pnl_dollars": trading_summary["net_pnl_dollars"],
                "benchmark_type": "buy_and_hold",
            }
        ]
    )

    metrics = {
        "benchmark_total_return": trading_summary["return_on_traded_capital"],
        "benchmark_net_pnl_dollars": trading_summary["net_pnl_dollars"],
        "benchmark_holding_days": trading_summary["holding_days"],
    }

    metadata = {
        "model_name": "sp500_buy_hold",
        "dataset_name": args.dataset_name,
        "benchmark_symbol": "^GSPC",
        "benchmark_strategy": "buy_then_hold_until_latest_available_date",
        "initial_capital": float(args.initial_capital),
        "split_metadata": split_metadata,
        "input_dir": str(Path(input_dir).resolve()),
        "test_rows": int(len(test_df)),
        "benchmark_start_date": trading_summary["benchmark_start_date"],
        "benchmark_end_date": trading_summary["benchmark_end_date"],
    }

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ARTIFACT_DIR / "experiments" / "sp500_buy_hold" / args.dataset_name
    )
    save_experiment_artifacts(
        output_dir=output_dir,
        metrics=metrics,
        metadata=metadata,
        predictions=predictions,
        feature_importance_df=pd.DataFrame(columns=["feature", "importance"]),
        trade_log_df=trade_log_df,
        trading_summary=trading_summary,
    )

    print(f"Saved S&P 500 buy-and-hold artifacts to: {output_dir}")
    print(json.dumps(trading_summary, indent=2))


if __name__ == "__main__":
    main()
