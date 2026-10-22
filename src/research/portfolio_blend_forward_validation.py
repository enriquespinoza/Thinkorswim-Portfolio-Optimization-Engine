from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR
from src.research.portfolio_blend_analysis import (
    blend_weights,
)
from src.research.portfolio_comparison import (
    max_drawdown,
    portfolio_returns,
)


def forward_price_returns(
    prices: pd.DataFrame,
    *,
    symbols: Iterable[str],
    freeze_session: str,
    evaluation_session: str,
) -> pd.DataFrame:
    required = sorted({str(symbol).strip().upper() for symbol in symbols})
    missing = [
        symbol
        for symbol in required
        if symbol not in prices.columns
    ]
    if missing:
        raise ValueError(
            f"forward price matrix missing symbols: {missing}"
        )

    frame = prices.loc[:, required].copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index = (
        pd.DatetimeIndex(frame.index)
        .tz_localize(None)
        .normalize()
    )
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    freeze = pd.Timestamp(freeze_session)
    evaluation = pd.Timestamp(evaluation_session)
    if evaluation <= freeze:
        raise ValueError(
            "evaluation_session must be after freeze_session"
        )

    base_rows = frame.loc[frame.index <= freeze]
    if base_rows.empty:
        raise ValueError(
            "price matrix has no synchronized base session "
            "at or before freeze_session"
        )

    forward_rows = frame.loc[
        (frame.index > freeze)
        & (frame.index <= evaluation)
    ]
    if forward_rows.empty:
        raise ValueError(
            "price matrix has no forward sessions after freeze_session"
        )

    base = base_rows.tail(1)
    selected = pd.concat([base, forward_rows], axis=0)
    returns = selected.pct_change(fill_method=None).iloc[1:]

    if returns.isna().any().any():
        raise ValueError("forward returns contain missing values")

    return returns


def realized_volatility(series: pd.Series) -> float | None:
    if len(series) < 2:
        return None
    value = float(
        series.std(ddof=1)
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    return value


def cumulative_return(series: pd.Series) -> float:
    return float((1.0 + series).prod() - 1.0)


def build_forward_blend_table(
    *,
    returns: pd.DataFrame,
    personal_weights: pd.Series,
    optimizer_weights: pd.Series,
    optimizer_blend_weights: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty:
        raise ValueError("forward returns cannot be empty")

    personal_returns = portfolio_returns(
        returns,
        personal_weights,
    )

    daily = pd.DataFrame(index=returns.index)
    rows: list[dict[str, float | int | None]] = []

    for raw_alpha in optimizer_blend_weights:
        alpha = float(raw_alpha)
        weights = blend_weights(
            personal_weights,
            optimizer_weights,
            optimizer_weight=alpha,
        )
        series = portfolio_returns(
            returns,
            weights,
        )

        label = f"optimizer_{int(round(alpha * 100)):03d}"
        daily[label] = series

        row: dict[str, float | int | None] = {
            "optimizer_blend_weight": alpha,
            "personal_blend_weight": 1.0 - alpha,
            "observed_sessions": int(len(series)),
            "cumulative_return": cumulative_return(series),
            "realized_annualized_volatility": realized_volatility(series),
            "max_drawdown": max_drawdown(series),
            "best_day": float(series.max()),
            "worst_day": float(series.min()),
            "positive_session_rate": float((series > 0).mean()),
            "cumulative_excess_return_vs_personal": (
                cumulative_return(series)
                - cumulative_return(personal_returns)
            ),
        }
        rows.append(row)

    daily.index.name = "session"
    return pd.DataFrame(rows), daily
