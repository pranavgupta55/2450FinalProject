from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simulate_alpha_trading(
    predictions_df: pd.DataFrame,
    capital_per_trade: float = 10_000.0,
    alpha_threshold: float = 0.0,
    alpha_column: str = "predicted_alpha_score",
    realized_alpha_column: str = "future_alpha_5d",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = predictions_df.copy()
    if alpha_column not in work.columns:
        raise ValueError(f"Missing required alpha score column: {alpha_column}")

    work[alpha_column] = pd.to_numeric(work[alpha_column], errors="coerce").fillna(0.0)
    work[realized_alpha_column] = pd.to_numeric(
        work.get(realized_alpha_column),
        errors="coerce",
    )

    has_realized_alpha = work[realized_alpha_column].notna()
    score = work[alpha_column].to_numpy(dtype=float)
    position = np.where(
        score >= alpha_threshold,
        1,
        np.where(score <= -alpha_threshold, -1, 0),
    )
    position = np.where(has_realized_alpha.to_numpy(), position, 0)

    realized_alpha = work[realized_alpha_column].fillna(0.0).to_numpy(dtype=float)
    pnl = capital_per_trade * position * realized_alpha
    predicted_edge = capital_per_trade * score

    work["trade_position"] = position
    work["trade_side"] = np.where(position > 0, "long", np.where(position < 0, "short", "flat"))
    work["predicted_edge_dollars"] = predicted_edge
    work["realized_alpha"] = realized_alpha
    work["pnl_dollars"] = pnl
    work["cumulative_pnl_dollars"] = np.cumsum(pnl)
    work["trade_taken"] = (position != 0).astype(int)
    work["trade_correct_direction"] = np.where(
        position == 0,
        0,
        (np.sign(realized_alpha) == np.sign(position)).astype(int),
    )

    traded = work[work["trade_taken"] == 1].copy()
    gross_profit = float(traded.loc[traded["pnl_dollars"] > 0, "pnl_dollars"].sum())
    gross_loss = float(traded.loc[traded["pnl_dollars"] < 0, "pnl_dollars"].sum())
    net_pnl = float(traded["pnl_dollars"].sum())
    trades_executed = int(len(traded))
    hit_rate = float(traded["trade_correct_direction"].mean()) if trades_executed else 0.0

    summary = {
        "capital_per_trade": float(capital_per_trade),
        "alpha_threshold": float(alpha_threshold),
        "trades_executed": trades_executed,
        "long_trades": int((traded["trade_position"] > 0).sum()),
        "short_trades": int((traded["trade_position"] < 0).sum()),
        "flat_rows": int((work["trade_position"] == 0).sum()),
        "hit_rate": hit_rate,
        "gross_profit_dollars": gross_profit,
        "gross_loss_dollars": gross_loss,
        "net_pnl_dollars": net_pnl,
        "average_pnl_dollars": float(net_pnl / trades_executed) if trades_executed else 0.0,
        "max_cumulative_pnl_dollars": float(work["cumulative_pnl_dollars"].max()) if len(work) else 0.0,
        "min_cumulative_pnl_dollars": float(work["cumulative_pnl_dollars"].min()) if len(work) else 0.0,
        "return_on_traded_capital": float(net_pnl / (capital_per_trade * trades_executed))
        if trades_executed
        else 0.0,
    }
    return work, summary
