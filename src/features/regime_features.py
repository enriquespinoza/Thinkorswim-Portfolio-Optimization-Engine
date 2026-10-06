from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR


def validate_daily_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a daily return matrix used for regime features.
    """
    if not isinstance(
        returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    if not isinstance(
        returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "returns must use a DatetimeIndex."
        )

    cleaned = (
        returns
        .sort_index()
        .copy()
    )

    cleaned = cleaned.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if cleaned.isna().any().any():
        raise ValueError(
            "returns contain missing or non-numeric values."
        )

    if not np.isfinite(
        cleaned.to_numpy()
    ).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    return cleaned


def calculate_trailing_return(
    returns: pd.DataFrame,
    lookback: int,
) -> pd.Series:
    """
    Calculate compounded return over the trailing window.
    """
    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than zero."
        )

    if len(returns) < lookback:
        raise ValueError(
            f"At least {lookback} observations are required."
        )

    window = returns.tail(
        lookback
    )

    result = (
        (1.0 + window)
        .prod()
        - 1.0
    )

    return result


def calculate_trailing_volatility(
    returns: pd.DataFrame,
    lookback: int,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate annualized realized volatility over a trailing window.
    """
    if lookback <= 1:
        raise ValueError(
            "lookback must be greater than 1."
        )

    if len(returns) < lookback:
        raise ValueError(
            f"At least {lookback} observations are required."
        )

    return (
        returns
        .tail(lookback)
        .std(ddof=1)
        * np.sqrt(trading_days)
    )


def calculate_current_drawdown(
    returns: pd.DataFrame,
    lookback: int,
) -> pd.Series:
    """
    Calculate each asset's current drawdown from its highest
    cumulative wealth level within the trailing window.
    """
    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than zero."
        )

    if len(returns) < lookback:
        raise ValueError(
            f"At least {lookback} observations are required."
        )

    window = returns.tail(
        lookback
    )

    wealth = (
        1.0 + window
    ).cumprod()

    peak = wealth.cummax()

    drawdown = (
        wealth
        / peak
        - 1.0
    )

    return drawdown.iloc[-1]


def calculate_average_pairwise_correlation(
    returns: pd.DataFrame,
    lookback: int,
) -> float:
    """
    Average off-diagonal correlation across assets.
    """
    if lookback <= 1:
        raise ValueError(
            "lookback must be greater than 1."
        )

    if len(returns) < lookback:
        raise ValueError(
            f"At least {lookback} observations are required."
        )

    correlation = (
        returns
        .tail(lookback)
        .corr()
        .to_numpy()
    )

    number_of_assets = (
        correlation.shape[0]
    )

    if number_of_assets < 2:
        return 0.0

    mask = ~np.eye(
        number_of_assets,
        dtype=bool,
    )

    return float(
        correlation[mask].mean()
    )


def calculate_pair_correlation(
    returns: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    lookback: int,
) -> float:
    """
    Calculate trailing correlation between two assets.
    """
    if asset_a not in returns.columns:
        raise ValueError(
            f"Unknown asset: {asset_a}"
        )

    if asset_b not in returns.columns:
        raise ValueError(
            f"Unknown asset: {asset_b}"
        )

    if len(returns) < lookback:
        raise ValueError(
            f"At least {lookback} observations are required."
        )

    return float(
        returns[
            [
                asset_a,
                asset_b,
            ]
        ]
        .tail(lookback)
        .corr()
        .iloc[0, 1]
    )


def build_regime_features_for_date(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    return_windows: tuple[int, ...] = (
        21,
        63,
        126,
    ),
    volatility_windows: tuple[int, ...] = (
        21,
        63,
    ),
    drawdown_window: int = 126,
    correlation_window: int = 63,
) -> pd.Series:
    """
    Build market-state features using only information available
    through as_of_date.

    No observations after as_of_date are used.
    """
    returns = validate_daily_returns(
        returns
    )

    as_of_date = pd.Timestamp(
        as_of_date
    )

    history = returns.loc[
        returns.index <= as_of_date
    ]

    required_history = max(
        max(return_windows),
        max(volatility_windows),
        drawdown_window,
        correlation_window,
    )

    if len(history) < required_history:
        raise ValueError(
            "Insufficient history to calculate regime features. "
            f"Need at least {required_history} observations."
        )

    features: dict[
        str,
        float,
    ] = {}

    trailing_returns = {}

    for window in return_windows:
        values = (
            calculate_trailing_return(
                history,
                lookback=window,
            )
        )

        trailing_returns[
            window
        ] = values

        for asset, value in values.items():
            features[
                f"{asset}_return_{window}d"
            ] = float(value)

        features[
            f"cross_asset_return_dispersion_{window}d"
        ] = float(
            values.std(
                ddof=0
            )
        )

    for window in volatility_windows:
        values = (
            calculate_trailing_volatility(
                history,
                lookback=window,
            )
        )

        for asset, value in values.items():
            features[
                f"{asset}_vol_{window}d"
            ] = float(value)

        features[
            f"cross_asset_vol_dispersion_{window}d"
        ] = float(
            values.std(
                ddof=0
            )
        )

    drawdowns = (
        calculate_current_drawdown(
            history,
            lookback=
                drawdown_window,
        )
    )

    for asset, value in drawdowns.items():
        features[
            f"{asset}_drawdown_{drawdown_window}d"
        ] = float(value)

    features[
        f"average_pairwise_corr_{correlation_window}d"
    ] = (
        calculate_average_pairwise_correlation(
            history,
            lookback=
                correlation_window,
        )
    )

    pair_definitions = [
        (
            "SPY",
            "TLT",
            "equity_bond_corr",
        ),
        (
            "SPY",
            "GLD",
            "equity_gold_corr",
        ),
        (
            "QQQ",
            "TLT",
            "growth_bond_corr",
        ),
        (
            "QQQ",
            "SCHD",
            "growth_dividend_corr",
        ),
    ]

    for (
        asset_a,
        asset_b,
        name,
    ) in pair_definitions:

        if (
            asset_a in history.columns
            and asset_b in history.columns
        ):
            features[
                f"{name}_{correlation_window}d"
            ] = (
                calculate_pair_correlation(
                    history,
                    asset_a=
                        asset_a,
                    asset_b=
                        asset_b,
                    lookback=
                        correlation_window,
                )
            )

    if 63 in trailing_returns:
        ret63 = trailing_returns[
            63
        ]

        if {
            "SPY",
            "TLT",
        }.issubset(
            ret63.index
        ):
            features[
                "equity_bond_momentum_spread_63d"
            ] = float(
                ret63["SPY"]
                - ret63["TLT"]
            )

        if {
            "QQQ",
            "SCHD",
        }.issubset(
            ret63.index
        ):
            features[
                "growth_dividend_momentum_spread_63d"
            ] = float(
                ret63["QQQ"]
                - ret63["SCHD"]
            )

        if {
            "GLD",
            "TLT",
        }.issubset(
            ret63.index
        ):
            features[
                "gold_bond_momentum_spread_63d"
            ] = float(
                ret63["GLD"]
                - ret63["TLT"]
            )

    feature_series = pd.Series(
        features,
        dtype=float,
        name=as_of_date,
    )

    return feature_series


def build_regime_feature_matrix(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | list,
) -> pd.DataFrame:
    """
    Build one feature row for every rebalance date.

    Each row uses only market data available on or before that
    rebalance date.
    """
    returns = validate_daily_returns(
        returns
    )

    rows = []

    for date in pd.to_datetime(
        rebalance_dates
    ):
        try:
            features = (
                build_regime_features_for_date(
                    returns=returns,
                    as_of_date=date,
                )
            )

        except ValueError:
            continue

        features.name = date

        rows.append(
            features
        )

    if not rows:
        raise ValueError(
            "No regime feature rows could be generated."
        )

    feature_matrix = pd.DataFrame(
        rows
    )

    feature_matrix.index.name = (
        "rebalance_date"
    )

    return (
        feature_matrix
        .sort_index()
    )
