from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR
from src.research.portfolio_comparison import (
    annualized_volatility,
    concentration,
    effective_positions,
    max_drawdown,
    normalize_weights,
    portfolio_returns,
    risk_contribution,
    tracking_error,
)


def validate_blend_weights(values: Iterable[float]) -> list[float]:
    weights = sorted({float(value) for value in values})
    if not weights:
        raise ValueError("at least one optimizer blend weight is required")
    if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("optimizer blend weights must be finite values in [0, 1]")
    return weights


def blend_weights(
    personal_weights: pd.Series,
    optimizer_weights: pd.Series,
    *,
    optimizer_weight: float,
) -> pd.Series:
    alpha = float(optimizer_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("optimizer_weight must be in [0, 1]")

    symbols = sorted(
        set(personal_weights.index)
        | set(optimizer_weights.index)
    )
    personal = (
        normalize_weights(personal_weights)
        .reindex(symbols)
        .fillna(0.0)
    )
    optimizer = (
        normalize_weights(optimizer_weights)
        .reindex(symbols)
        .fillna(0.0)
    )

    blended = (1.0 - alpha) * personal + alpha * optimizer
    return normalize_weights(blended)


def annualized_geometric_return(series: pd.Series) -> float:
    if series.empty:
        raise ValueError("return series cannot be empty")

    wealth = float((1.0 + series).prod())
    if wealth <= 0.0:
        return -1.0

    years = len(series) / float(TRADING_DAYS_PER_YEAR)
    return float(wealth ** (1.0 / years) - 1.0)


def sharpe_ratio(
    series: pd.Series,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    volatility = annualized_volatility(series)
    if volatility <= 0.0:
        return 0.0

    annualized_mean = float(series.mean()) * TRADING_DAYS_PER_YEAR
    return float((annualized_mean - risk_free_rate) / volatility)


def sortino_ratio(
    series: pd.Series,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    daily_target = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = series - daily_target
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside)))
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if downside_deviation <= 0.0:
        return 0.0

    annualized_excess = float(excess.mean()) * TRADING_DAYS_PER_YEAR
    return float(annualized_excess / downside_deviation)


def worst_rolling_return(
    series: pd.Series,
    *,
    window: int,
) -> float:
    if window < 1:
        raise ValueError("rolling window must be positive")
    if len(series) < window:
        raise ValueError(
            f"need at least {window} returns for rolling analysis"
        )

    rolling = (
        (1.0 + series)
        .rolling(window)
        .apply(np.prod, raw=True)
        - 1.0
    )
    return float(rolling.min())


def turnover_from_personal(
    personal_weights: pd.Series,
    blended_weights: pd.Series,
) -> float:
    symbols = sorted(
        set(personal_weights.index)
        | set(blended_weights.index)
    )
    personal = (
        normalize_weights(personal_weights)
        .reindex(symbols)
        .fillna(0.0)
    )
    blended = (
        normalize_weights(blended_weights)
        .reindex(symbols)
        .fillna(0.0)
    )
    return float(0.5 * (personal - blended).abs().sum())


def build_blend_table(
    *,
    returns: pd.DataFrame,
    personal_weights: pd.Series,
    optimizer_weights: pd.Series,
    optimizer_blend_weights: Iterable[float],
    rolling_windows: tuple[int, ...] = (20, 60),
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    blend_levels = validate_blend_weights(optimizer_blend_weights)

    personal_returns = portfolio_returns(
        returns,
        personal_weights,
    )
    optimizer_returns = portfolio_returns(
        returns,
        optimizer_weights,
    )

    personal_volatility = annualized_volatility(personal_returns)
    personal_drawdown = max_drawdown(personal_returns)
    personal_concentration = concentration(personal_weights)
    personal_annualized_return = annualized_geometric_return(
        personal_returns
    )

    rows: list[dict[str, float | bool]] = []

    for alpha in blend_levels:
        weights = blend_weights(
            personal_weights,
            optimizer_weights,
            optimizer_weight=alpha,
        )
        blended_returns = portfolio_returns(
            returns,
            weights,
        )

        volatility = annualized_volatility(blended_returns)
        drawdown = max_drawdown(blended_returns)
        hhi = concentration(weights)
        annualized_return = annualized_geometric_return(
            blended_returns
        )
        contributions = risk_contribution(
            returns,
            weights,
        )

        row: dict[str, float | bool] = {
            "optimizer_blend_weight": alpha,
            "personal_blend_weight": 1.0 - alpha,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "max_drawdown": drawdown,
            "sharpe_ratio_zero_rf": sharpe_ratio(
                blended_returns,
                risk_free_rate=risk_free_rate,
            ),
            "sortino_ratio_zero_rf": sortino_ratio(
                blended_returns,
                risk_free_rate=risk_free_rate,
            ),
            "concentration": hhi,
            "effective_positions": effective_positions(weights),
            "tracking_error_vs_personal": tracking_error(
                blended_returns,
                personal_returns,
            ),
            "tracking_error_vs_optimizer": tracking_error(
                blended_returns,
                optimizer_returns,
            ),
            "turnover_from_personal": turnover_from_personal(
                personal_weights,
                weights,
            ),
            "largest_absolute_risk_contribution": float(
                contributions.abs().max()
            ),
            "annualized_return_change_vs_personal": (
                annualized_return - personal_annualized_return
            ),
            "volatility_change_vs_personal": (
                volatility - personal_volatility
            ),
            "max_drawdown_change_vs_personal": (
                drawdown - personal_drawdown
            ),
            "concentration_change_vs_personal": (
                hhi - personal_concentration
            ),
            "lower_volatility_than_personal": (
                volatility < personal_volatility
            ),
            "shallower_drawdown_than_personal": (
                drawdown > personal_drawdown
            ),
            "lower_concentration_than_personal": (
                hhi < personal_concentration
            ),
        }

        for window in rolling_windows:
            row[f"worst_rolling_{window}d_return"] = (
                worst_rolling_return(
                    blended_returns,
                    window=window,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)
