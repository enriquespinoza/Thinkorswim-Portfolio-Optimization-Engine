from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a price DataFrame.

    Requirements:
    - Datetime-like index
    - At least one asset column
    - Positive prices
    - Sorted chronologically

    Parameters
    ----------
    prices:
        DataFrame of asset prices indexed by date.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset column.")

    cleaned = prices.copy()

    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()

    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    if (cleaned <= 0).any().any():
        raise ValueError("prices must contain only positive values.")

    cleaned = cleaned.dropna(how="all")

    if cleaned.empty:
        raise ValueError("prices contain no usable observations.")

    return cleaned


def calculate_daily_returns(
    prices: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Calculate simple daily percentage returns.

    Formula
    -------
    r_t = P_t / P_(t-1) - 1
    """
    prices = validate_prices(prices)

    returns = prices.pct_change(fill_method=None)

    if dropna:
        returns = returns.dropna(how="any")

    return returns


def calculate_log_returns(
    prices: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Calculate continuously compounded/log returns.

    Formula
    -------
    r_t = ln(P_t / P_(t-1))
    """
    prices = validate_prices(prices)

    log_returns = np.log(prices / prices.shift(1))

    if dropna:
        log_returns = log_returns.dropna(how="any")

    return log_returns


def calculate_cumulative_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cumulative simple returns.

    Formula
    -------
    cumulative_t = product(1 + r_i) - 1
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    return (1.0 + returns).cumprod() - 1.0


def calculate_annualized_return(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate geometric annualized return for each asset.

    This is preferable to simply multiplying average daily return
    by 252 because it accounts for compounding.

    Formula
    -------
    annualized_return =
        product(1 + daily_returns)^(trading_days / observations) - 1
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if trading_days <= 0:
        raise ValueError("trading_days must be greater than zero.")

    growth = (1.0 + returns).prod()
    observations = returns.count()

    annualized = growth.pow(trading_days / observations) - 1.0

    return annualized


def calculate_annualized_volatility(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate annualized volatility for each asset.

    Formula
    -------
    annualized_volatility =
        standard_deviation(daily_returns) * sqrt(trading_days)
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if trading_days <= 0:
        raise ValueError("trading_days must be greater than zero.")

    return returns.std(ddof=1) * np.sqrt(trading_days)


def build_return_matrix(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the aligned daily return matrix used by the optimization engine.

    Rows represent trading dates.
    Columns represent assets.

    Any row missing a return for one or more assets is removed so the
    covariance and optimization engines operate on synchronized data.
    """
    returns = calculate_daily_returns(
        prices=prices,
        dropna=False,
    )

    returns = returns.dropna(how="any")

    if returns.empty:
        raise ValueError(
            "No synchronized return observations remain after removing "
            "missing values."
        )

    return returns


def summarize_returns(
    prices: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Produce a summary table of return statistics.

    Returns
    -------
    pd.DataFrame
        One row per asset containing:
        - total_return
        - annualized_return
        - annualized_volatility
    """
    returns = build_return_matrix(prices)

    cumulative = calculate_cumulative_returns(returns)

    summary = pd.DataFrame(
        {
            "total_return": cumulative.iloc[-1],
            "annualized_return": calculate_annualized_return(
                returns,
                trading_days=trading_days,
            ),
            "annualized_volatility": calculate_annualized_volatility(
                returns,
                trading_days=trading_days,
            ),
        }
    )

    return summary.sort_index()
