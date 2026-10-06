from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Compound annual growth rate from periodic returns.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    growth = (
        1.0 + returns
    ).prod()

    if growth <= 0:
        return np.nan

    return float(
        growth
        ** (
            periods_per_year
            / len(returns)
        )
        - 1.0
    )


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized standard deviation of periodic returns.
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return float(
        returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    Conventional annualized Sharpe ratio using periodic
    excess-return mean and standard deviation.
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    periodic_risk_free = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = (
        returns
        - periodic_risk_free
    )

    volatility = (
        excess_returns.std(
            ddof=1
        )
    )

    if (
        volatility <= 0
        or not np.isfinite(volatility)
    ):
        return np.nan

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized Sortino ratio.

    Downside deviation includes all periods, with positive
    excess returns contributing zero downside.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    periodic_risk_free = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = (
        returns
        - periodic_risk_free
    )

    downside = np.minimum(
        excess_returns.to_numpy(),
        0.0,
    )

    downside_deviation = float(
        np.sqrt(
            np.mean(
                downside ** 2
            )
        )
        * np.sqrt(
            periods_per_year
        )
    )

    if (
        downside_deviation <= 0
        or not np.isfinite(
            downside_deviation
        )
    ):
        return np.nan

    annualized_excess_return = float(
        excess_returns.mean()
        * periods_per_year
    )

    return (
        annualized_excess_return
        / downside_deviation
    )


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum drawdown relative to an initial
    portfolio value of 1.0.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    wealth = np.concatenate(
        [
            [1.0],
            (
                1.0 + returns.to_numpy()
            ).cumprod(),
        ]
    )

    running_peak = np.maximum.accumulate(
        wealth
    )

    drawdowns = (
        wealth
        / running_peak
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def summarize_return_series(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Calculate common performance metrics for one periodic
    return series.
    """
    returns = returns.dropna()

    if returns.empty:
        return {
            "periods": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
        }

    return {
        "periods":
            len(returns),

        "total_return":
            float(
                (1.0 + returns).prod()
                - 1.0
            ),

        "annualized_return":
            calculate_annualized_return(
                returns,
                periods_per_year,
            ),

        "annualized_volatility":
            calculate_annualized_volatility(
                returns,
                periods_per_year,
            ),

        "sharpe":
            calculate_sharpe_ratio(
                returns,
                risk_free_rate,
                periods_per_year,
            ),

        "sortino":
            calculate_sortino_ratio(
                returns,
                risk_free_rate,
                periods_per_year,
            ),

        "max_drawdown":
            calculate_max_drawdown(
                returns
            ),
    }


def summarize_backtest(
    period_results: pd.DataFrame,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Summarize walk-forward performance by model.
    """
    required = {
        "model",
        "period_end",
        "net_return",
        "turnover",
    }

    missing = (
        required
        - set(period_results.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    records = []

    for (
        model,
        group,
    ) in period_results.groupby(
        "model"
    ):

        group = group.sort_values(
            "period_end"
        )

        metrics = (
            summarize_return_series(
                group["net_return"],
                periods_per_year=
                    periods_per_year,
                risk_free_rate=
                    risk_free_rate,
            )
        )

        records.append(
            {
                "model":
                    model,

                **metrics,

                "average_turnover":
                    group[
                        "turnover"
                    ].mean(),
            }
        )

    return (
        pd.DataFrame(records)
        .set_index("model")
        .sort_values(
            "sharpe",
            ascending=False,
        )
    )
