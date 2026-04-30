from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.alpha_signal.config import DEFAULT_ARTIFACT_DIR, DEFAULT_RANDOM_STATE
from src.alpha_signal.utils.io import ensure_dir, write_dataframe, write_json


@dataclass(frozen=True)
class StrategyArtifacts:
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    predictions: pd.DataFrame
    trade_log: pd.DataFrame
    trading_summary: dict[str, Any]


def _coerce_prediction_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "week_start", "future_alpha_5d"]
    missing = [column for column in required if column not in predictions_df.columns]
    if missing:
        raise ValueError(f"Strategy analysis is missing required columns: {missing}")

    work = predictions_df.copy()
    work["week_start"] = pd.to_datetime(work["week_start"], errors="coerce")
    work["future_alpha_5d"] = pd.to_numeric(work["future_alpha_5d"], errors="coerce")
    if "predicted_alpha_score" in work.columns:
        work["predicted_alpha_score"] = pd.to_numeric(
            work["predicted_alpha_score"],
            errors="coerce",
        )
    else:
        work["predicted_alpha_score"] = 0.0
    if "predicted_probability" in work.columns:
        work["predicted_probability"] = pd.to_numeric(
            work["predicted_probability"],
            errors="coerce",
        )
    return work.dropna(subset=["week_start"]).reset_index(drop=True)


def _resolve_position_size(
    universe_size: int,
    requested_long: int,
    requested_short: int,
    adaptive: bool,
) -> tuple[int, int]:
    if universe_size < 2:
        return 0, 0

    if not adaptive and universe_size < requested_long + requested_short:
        raise ValueError(
            f"Requested {requested_long + requested_short} positions but only "
            f"{universe_size} names were available for this rebalance."
        )

    max_symmetric = universe_size // 2
    effective_long = min(requested_long, max_symmetric)
    effective_short = min(requested_short, max_symmetric)

    if not adaptive and (effective_long != requested_long or effective_short != requested_short):
        raise ValueError(
            "The requested basket size could not be satisfied exactly for every rebalance period."
        )

    return effective_long, effective_short


def _build_week_positions(
    week_df: pd.DataFrame,
    requested_long: int,
    requested_short: int,
    selection_mode: str,
    adaptive: bool,
    rng: np.random.Generator | None,
) -> pd.DataFrame:
    universe = week_df.dropna(subset=["future_alpha_5d"]).copy()
    if universe.empty:
        return pd.DataFrame()

    effective_long, effective_short = _resolve_position_size(
        universe_size=len(universe),
        requested_long=requested_long,
        requested_short=requested_short,
        adaptive=adaptive,
    )
    if effective_long == 0 or effective_short == 0:
        return pd.DataFrame()

    if selection_mode == "random":
        if rng is None:
            raise ValueError("Random strategy selection requires an RNG.")
        sampled = universe.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        long_positions = sampled.head(effective_long).copy()
        short_positions = sampled.iloc[effective_long : effective_long + effective_short].copy()
    else:
        ranked = universe.sort_values(
            ["predicted_alpha_score", "ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)
        long_positions = ranked.head(effective_long).copy()
        short_positions = ranked.tail(effective_short).copy()

    long_weight = 1.0 / effective_long if effective_long else 0.0
    short_weight = 1.0 / effective_short if effective_short else 0.0

    long_positions["trade_position"] = 1
    long_positions["position_weight"] = long_weight
    short_positions["trade_position"] = -1
    short_positions["position_weight"] = short_weight

    selected = pd.concat([long_positions, short_positions], ignore_index=True)
    selected["realized_alpha"] = selected["future_alpha_5d"].fillna(0.0)
    selected["period_return_contribution"] = (
        selected["trade_position"] * selected["position_weight"] * selected["realized_alpha"]
    )
    selected["trade_taken"] = 1
    selected["requested_k_long"] = requested_long
    selected["requested_k_short"] = requested_short
    selected["effective_k_long"] = effective_long
    selected["effective_k_short"] = effective_short
    selected["trade_side"] = np.where(selected["trade_position"] > 0, "long", "short")
    selected["trade_correct_direction"] = (
        np.sign(selected["realized_alpha"]) == np.sign(selected["trade_position"])
    ).astype(int)
    return selected


def _compute_drawdown(portfolio_values: list[float]) -> float:
    if not portfolio_values:
        return 0.0

    running_peak = portfolio_values[0]
    max_drawdown = 0.0
    for value in portfolio_values:
        running_peak = max(running_peak, value)
        if running_peak > 0:
            max_drawdown = min(max_drawdown, value / running_peak - 1.0)
    return float(max_drawdown)


def _summarize_strategy_trade_log(
    trade_log_df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if trade_log_df.empty:
        empty_metrics = {
            "strategy_total_return": 0.0,
            "strategy_ending_value": float(initial_capital),
            "strategy_weeks_traded": 0,
            "strategy_max_drawdown": 0.0,
            "strategy_hit_rate": 0.0,
            "strategy_volatility": 0.0,
        }
        empty_summary = {
            "strategy_name": strategy_name,
            "initial_capital": float(initial_capital),
            "ending_portfolio_value": float(initial_capital),
            "total_return": 0.0,
            "weeks_traded": 0,
            "long_trades": 0,
            "short_trades": 0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "average_weekly_return": 0.0,
        }
        return empty_metrics, empty_summary, pd.DataFrame(columns=["week_start", "period_return"])

    weekly_returns = (
        trade_log_df.groupby("week_start", as_index=False)["period_return_contribution"].sum()
        .rename(columns={"period_return_contribution": "period_return"})
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    portfolio_values = [float(initial_capital)]
    for period_return in weekly_returns["period_return"].to_numpy(dtype=float):
        portfolio_values.append(portfolio_values[-1] * (1.0 + period_return))

    ending_value = float(portfolio_values[-1])
    total_return = float(ending_value / initial_capital - 1.0) if initial_capital else 0.0
    max_drawdown = _compute_drawdown(portfolio_values)
    weekly_return_values = weekly_returns["period_return"].to_numpy(dtype=float)
    volatility = float(weekly_return_values.std(ddof=0)) if len(weekly_return_values) else 0.0
    hit_rate = float((weekly_return_values > 0).mean()) if len(weekly_return_values) else 0.0

    summary = {
        "strategy_name": strategy_name,
        "initial_capital": float(initial_capital),
        "ending_portfolio_value": ending_value,
        "total_return": total_return,
        "weeks_traded": int(len(weekly_returns)),
        "long_trades": int((trade_log_df["trade_position"] > 0).sum()),
        "short_trades": int((trade_log_df["trade_position"] < 0).sum()),
        "hit_rate": hit_rate,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "average_weekly_return": float(weekly_return_values.mean()) if len(weekly_return_values) else 0.0,
        "requested_k_long": int(trade_log_df["requested_k_long"].max()),
        "requested_k_short": int(trade_log_df["requested_k_short"].max()),
        "effective_k_long_min": int(trade_log_df["effective_k_long"].min()),
        "effective_k_long_max": int(trade_log_df["effective_k_long"].max()),
        "effective_k_short_min": int(trade_log_df["effective_k_short"].min()),
        "effective_k_short_max": int(trade_log_df["effective_k_short"].max()),
    }
    metrics = {
        "strategy_total_return": total_return,
        "strategy_ending_value": ending_value,
        "strategy_weeks_traded": int(len(weekly_returns)),
        "strategy_max_drawdown": max_drawdown,
        "strategy_hit_rate": hit_rate,
        "strategy_volatility": volatility,
    }
    return metrics, summary, weekly_returns


def build_cross_sectional_strategy(
    predictions_df: pd.DataFrame,
    strategy_name: str,
    source_model_name: str | None,
    dataset_name: str,
    requested_k_long: int = 5,
    requested_k_short: int = 5,
    initial_capital: float = 100.0,
    random_state: int | None = None,
    adaptive: bool = False,
    selection_mode: str = "ranked",
    source_artifact_path: str | None = None,
) -> StrategyArtifacts:
    work = _coerce_prediction_frame(predictions_df)
    rng = np.random.default_rng(random_state) if selection_mode == "random" else None

    rebalance_rows: list[pd.DataFrame] = []
    for _, week_df in work.groupby("week_start", sort=True):
        selected = _build_week_positions(
            week_df=week_df,
            requested_long=requested_k_long,
            requested_short=requested_k_short,
            selection_mode=selection_mode,
            adaptive=adaptive,
            rng=rng,
        )
        if selected.empty:
            continue
        rebalance_rows.append(selected)

    trade_log_df = (
        pd.concat(rebalance_rows, ignore_index=True)
        if rebalance_rows
        else pd.DataFrame(
            columns=[
                "ticker",
                "week_start",
                "future_alpha_5d",
                "predicted_alpha_score",
                "predicted_probability",
                "trade_position",
                "position_weight",
                "realized_alpha",
                "period_return_contribution",
                "trade_taken",
                "requested_k_long",
                "requested_k_short",
                "effective_k_long",
                "effective_k_short",
                "trade_side",
                "trade_correct_direction",
            ]
        )
    )
    metrics, summary, weekly_returns = _summarize_strategy_trade_log(
        trade_log_df=trade_log_df,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
    )

    metadata = {
        "strategy_name": strategy_name,
        "source_model_name": source_model_name,
        "dataset_name": dataset_name,
        "rebalance_frequency": "weekly",
        "portfolio_construction": "top_bottom_k_long_short"
        if selection_mode == "ranked"
        else "random_long_short",
        "requested_k_long": int(requested_k_long),
        "requested_k_short": int(requested_k_short),
        "initial_capital": float(initial_capital),
        "random_state": int(random_state) if random_state is not None else None,
        "adaptive_position_sizing": bool(adaptive),
        "source_artifact_path": source_artifact_path,
        "weeks_with_positions": int(len(weekly_returns)),
    }
    predictions_preview = weekly_returns.copy()
    predictions_preview["strategy_name"] = strategy_name
    predictions_preview["portfolio_value"] = [
        float(initial_capital * np.prod(1.0 + weekly_returns["period_return"].iloc[: index + 1]))
        for index in range(len(weekly_returns))
    ]

    return StrategyArtifacts(
        metrics=metrics,
        metadata=metadata,
        predictions=predictions_preview,
        trade_log=trade_log_df,
        trading_summary=summary,
    )


def build_buy_hold_strategy_from_trade_log(
    trade_log_df: pd.DataFrame,
    dataset_name: str,
    initial_capital: float,
    source_artifact_path: str | None = None,
) -> StrategyArtifacts:
    work = trade_log_df.copy()
    if "Date" in work.columns and "week_start" not in work.columns:
        work["week_start"] = pd.to_datetime(work["Date"], errors="coerce")
    else:
        work["week_start"] = pd.to_datetime(work["week_start"], errors="coerce")

    if "sp_return_1d" in work.columns:
        work["realized_alpha"] = pd.to_numeric(work["sp_return_1d"], errors="coerce").fillna(0.0)
    elif "daily_return" in work.columns:
        work["realized_alpha"] = pd.to_numeric(work["daily_return"], errors="coerce").fillna(0.0)
    else:
        work["realized_alpha"] = pd.to_numeric(work["market_return"], errors="coerce").fillna(0.0)

    work["trade_position"] = pd.to_numeric(work.get("position", 1), errors="coerce").fillna(1).astype(int)
    work["position_weight"] = 1.0
    work["period_return_contribution"] = work["trade_position"] * work["realized_alpha"]
    work["trade_taken"] = 1
    work["requested_k_long"] = 1
    work["requested_k_short"] = 0
    work["effective_k_long"] = 1
    work["effective_k_short"] = 0
    work["trade_side"] = "long"
    work["trade_correct_direction"] = (work["realized_alpha"] >= 0).astype(int)
    work["ticker"] = work.get("ticker", "^GSPC")
    work["predicted_alpha_score"] = pd.NA

    metrics, summary, daily_returns = _summarize_strategy_trade_log(
        trade_log_df=work,
        strategy_name="sp500_buy_hold",
        initial_capital=initial_capital,
    )
    metadata = {
        "strategy_name": "sp500_buy_hold",
        "source_model_name": None,
        "dataset_name": dataset_name,
        "rebalance_frequency": "daily",
        "portfolio_construction": "buy_and_hold",
        "requested_k_long": 1,
        "requested_k_short": 0,
        "initial_capital": float(initial_capital),
        "random_state": None,
        "adaptive_position_sizing": False,
        "source_artifact_path": source_artifact_path,
        "periods_with_positions": int(len(daily_returns)),
    }
    predictions_preview = daily_returns.copy()
    predictions_preview["strategy_name"] = "sp500_buy_hold"
    predictions_preview["portfolio_value"] = [
        float(initial_capital * np.prod(1.0 + daily_returns["period_return"].iloc[: index + 1]))
        for index in range(len(daily_returns))
    ]

    return StrategyArtifacts(
        metrics=metrics,
        metadata=metadata,
        predictions=predictions_preview,
        trade_log=work,
        trading_summary=summary,
    )


def save_strategy_artifacts(
    output_dir: str | Path,
    artifacts: StrategyArtifacts,
) -> Path:
    output_path = ensure_dir(output_dir)
    write_dataframe(artifacts.predictions, output_path / "predictions.csv")
    write_dataframe(artifacts.trade_log, output_path / "trade_log.csv")
    write_json(output_path / "metrics.json", artifacts.metrics)
    write_json(output_path / "metadata.json", artifacts.metadata)
    write_json(output_path / "trading_summary.json", artifacts.trading_summary)
    return output_path


def get_strategy_analysis_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root else DEFAULT_ARTIFACT_DIR / "strategy_analysis"


def build_strategy_output_dir(
    strategy_name: str,
    dataset_name: str,
    output_root: str | Path | None = None,
) -> Path:
    return get_strategy_analysis_root(output_root) / strategy_name / dataset_name


def get_default_random_state(random_state: int | None = None) -> int:
    return DEFAULT_RANDOM_STATE if random_state is None else random_state
