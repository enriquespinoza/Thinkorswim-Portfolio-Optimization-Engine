from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.personal_portfolio_benchmark import (
    BENCHMARK_BUILDER_VERSION,
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
)
from config.settings import TRADING_DAYS_PER_YEAR


def normalize_weights(weights: pd.Series) -> pd.Series:
    values = pd.to_numeric(weights, errors="coerce").astype(float)
    if values.isna().any() or (~np.isfinite(values.to_numpy())).any():
        raise ValueError("weights must be finite")
    if (values < 0).any():
        raise ValueError("weights cannot be negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return values / total


def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    w = normalize_weights(weights).reindex(returns.columns).fillna(0.0)
    return returns.mul(w, axis=1).sum(axis=1)


def annualized_volatility(series: pd.Series) -> float:
    return float(series.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(series: pd.Series) -> float:
    wealth = (1.0 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def concentration(weights: pd.Series) -> float:
    w = normalize_weights(weights)
    return float(np.square(w.to_numpy()).sum())


def effective_positions(weights: pd.Series) -> float:
    return float(1.0 / concentration(weights))


def compare_weights(
    personal_weights: pd.Series,
    optimizer_weights: pd.Series,
) -> dict[str, float]:
    symbols = sorted(set(personal_weights.index) | set(optimizer_weights.index))
    personal = normalize_weights(personal_weights).reindex(symbols).fillna(0.0)
    optimizer = normalize_weights(optimizer_weights).reindex(symbols).fillna(0.0)
    difference = personal - optimizer
    l1 = float(difference.abs().sum())
    overlap = float(np.minimum(personal.to_numpy(), optimizer.to_numpy()).sum())
    return {
        "portfolio_overlap": overlap,
        "weight_difference": l1,
        "turnover": 0.5 * l1,
    }


def tracking_error(
    personal_returns: pd.Series,
    optimizer_returns: pd.Series,
) -> float:
    aligned = pd.concat(
        [personal_returns.rename("personal"), optimizer_returns.rename("optimizer")],
        axis=1,
        join="inner",
    ).dropna()
    active = aligned["personal"] - aligned["optimizer"]
    return annualized_volatility(active)


def risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    w = normalize_weights(weights).reindex(returns.columns).fillna(0.0)
    covariance = returns.cov() * TRADING_DAYS_PER_YEAR
    vector = w.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    variance = float(vector.T @ sigma @ vector)

    if variance <= 0:
        return pd.Series(0.0, index=returns.columns, name="risk_contribution")

    contribution = vector * (sigma @ vector) / variance
    return pd.Series(
        contribution,
        index=returns.columns,
        name="risk_contribution",
    )


def build_summary(
    *,
    as_of_session: str,
    personal_weights: pd.Series,
    optimizer_weights: pd.Series,
    returns: pd.DataFrame,
    benchmark_symbol: str = "SPY",
) -> dict[str, float | str | bool]:
    personal = portfolio_returns(returns, personal_weights)
    optimizer = portfolio_returns(returns, optimizer_weights)

    equal_names = [symbol for symbol in optimizer_weights.index if symbol in returns.columns]
    equal_weights = pd.Series(
        1.0 / len(equal_names),
        index=equal_names,
    )
    equal = portfolio_returns(returns, equal_weights)

    benchmark_symbol = benchmark_symbol.upper().strip()
    if benchmark_symbol not in returns.columns:
        raise ValueError(f"benchmark symbol missing from returns: {benchmark_symbol}")

    weight_metrics = compare_weights(personal_weights, optimizer_weights)

    return {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_builder_version": BENCHMARK_BUILDER_VERSION,
        "as_of_session": as_of_session,
        "personal_portfolio_volatility": annualized_volatility(personal),
        "personal_max_drawdown": max_drawdown(personal),
        "personal_concentration": concentration(personal_weights),
        "personal_effective_positions": effective_positions(personal_weights),
        "optimizer_portfolio_volatility": annualized_volatility(optimizer),
        "optimizer_max_drawdown": max_drawdown(optimizer),
        "optimizer_concentration": concentration(optimizer_weights),
        "optimizer_effective_positions": effective_positions(optimizer_weights),
        "equal_weight_volatility": annualized_volatility(equal),
        "spy_volatility": annualized_volatility(returns[benchmark_symbol]),
        "tracking_error": tracking_error(personal, optimizer),
        **weight_metrics,
        "read_only_research": True,
        "order_submission_permitted": False,
    }
