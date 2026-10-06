from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR
from src.risk.covariance import validate_returns


@dataclass(frozen=True)
class ExpectedReturnResult:
    """
    Collection of expected-return estimates.

    These are statistical baselines derived from historical returns.
    They are not guarantees of future performance.
    """

    geometric: pd.Series
    arithmetic: pd.Series
    exponentially_weighted: pd.Series
    blended: pd.Series
    robust: pd.Series
    estimator_dispersion: pd.Series


def calculate_geometric_expected_return(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate geometric annualized historical return.

    Formula
    -------
    mu_geo =
        product(1 + r_t)^(trading_days / N) - 1

    This captures realized compounding over the historical sample.
    """
    returns = validate_returns(
        returns
    )

    if trading_days <= 0:
        raise ValueError(
            "trading_days must be greater than zero."
        )

    growth = (
        1.0 + returns
    ).prod()

    observations = (
        returns.count()
    )

    annualized = (
        growth.pow(
            trading_days / observations
        )
        - 1.0
    )

    annualized.name = (
        "geometric_expected_return"
    )

    return annualized


def calculate_arithmetic_expected_return(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate annualized arithmetic mean return.

    Formula
    -------
    mu_arithmetic =
        mean(daily_return) * trading_days

    This estimator is commonly used with mean-variance models because
    expected portfolio return is linear in portfolio weights.
    """
    returns = validate_returns(
        returns
    )

    if trading_days <= 0:
        raise ValueError(
            "trading_days must be greater than zero."
        )

    annualized = (
        returns.mean()
        * trading_days
    )

    annualized.name = (
        "arithmetic_expected_return"
    )

    return annualized


def calculate_exponentially_weighted_expected_return(
    returns: pd.DataFrame,
    span: int = 126,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Calculate an exponentially weighted annualized expected return.

    More recent observations receive greater influence than older
    observations.

    Parameters
    ----------
    returns:
        Daily return matrix.

    span:
        Effective EMA span measured in observations.

        For daily data:
            63  ~= one quarter
            126 ~= six months
            252 ~= one year

    trading_days:
        Annualization factor.

    Returns
    -------
    pd.Series
        Annualized exponentially weighted return estimate.
    """
    returns = validate_returns(
        returns
    )

    if span <= 1:
        raise ValueError(
            "span must be greater than 1."
        )

    if trading_days <= 0:
        raise ValueError(
            "trading_days must be greater than zero."
        )

    ewma_daily = (
        returns
        .ewm(
            span=span,
            adjust=False,
        )
        .mean()
        .iloc[-1]
    )

    annualized = (
        ewma_daily
        * trading_days
    )

    annualized.name = (
        "exponentially_weighted_expected_return"
    )

    return annualized


def calculate_blended_expected_return(
    geometric: pd.Series,
    arithmetic: pd.Series,
    exponentially_weighted: pd.Series,
    geometric_weight: float = 1 / 3,
    arithmetic_weight: float = 1 / 3,
    exponentially_weighted_weight: float = 1 / 3,
) -> pd.Series:
    """
    Blend multiple expected-return estimators.

    The default uses equal estimator weights.

    Parameters
    ----------
    geometric:
        Geometric annualized estimate.

    arithmetic:
        Arithmetic annualized estimate.

    exponentially_weighted:
        EWMA annualized estimate.

    geometric_weight:
        Influence assigned to geometric return.

    arithmetic_weight:
        Influence assigned to arithmetic return.

    exponentially_weighted_weight:
        Influence assigned to EWMA return.

    Returns
    -------
    pd.Series
        Blended annualized expected return.
    """
    estimators = [
        geometric,
        arithmetic,
        exponentially_weighted,
    ]

    for estimator in estimators:
        if not isinstance(
            estimator,
            pd.Series,
        ):
            raise TypeError(
                "All expected-return estimates "
                "must be pandas Series."
            )

    if not (
        geometric.index.equals(
            arithmetic.index
        )
        and geometric.index.equals(
            exponentially_weighted.index
        )
    ):
        raise ValueError(
            "Expected-return estimator indexes must match."
        )

    weights = np.array(
        [
            geometric_weight,
            arithmetic_weight,
            exponentially_weighted_weight,
        ],
        dtype=float,
    )

    if not np.isfinite(
        weights
    ).all():
        raise ValueError(
            "Estimator weights must be finite."
        )

    if (
        weights < 0
    ).any():
        raise ValueError(
            "Estimator weights cannot be negative."
        )

    if np.isclose(
        weights.sum(),
        0.0,
    ):
        raise ValueError(
            "Estimator weights cannot sum to zero."
        )

    weights = (
        weights
        / weights.sum()
    )

    blended = (
        geometric
        * weights[0]
        + arithmetic
        * weights[1]
        + exponentially_weighted
        * weights[2]
    )

    blended.name = (
        "blended_expected_return"
    )

    return blended


def calculate_estimator_dispersion(
    geometric: pd.Series,
    arithmetic: pd.Series,
    exponentially_weighted: pd.Series,
) -> pd.Series:
    """
    Measure disagreement among expected-return estimators.

    Higher dispersion means the historical estimators disagree more
    about the asset's expected return.
    """
    estimates = pd.concat(
        [
            geometric,
            arithmetic,
            exponentially_weighted,
        ],
        axis=1,
    )

    estimates.columns = [
        "geometric",
        "arithmetic",
        "exponentially_weighted",
    ]

    dispersion = (
        estimates.std(
            axis=1,
            ddof=0,
        )
    )

    dispersion.name = (
        "expected_return_dispersion"
    )

    return dispersion


def estimate_expected_returns(
    returns: pd.DataFrame,
    ewma_span: int = 126,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    geometric_weight: float = 1 / 3,
    arithmetic_weight: float = 1 / 3,
    exponentially_weighted_weight: float = 1 / 3,
) -> ExpectedReturnResult:
    """
    Calculate the complete baseline expected-return model.

    Returns
    -------
    ExpectedReturnResult
        Geometric, arithmetic, EWMA, blended and robust expected
        returns, plus estimator disagreement.
    """
    returns = validate_returns(
        returns
    )

    geometric = (
        calculate_geometric_expected_return(
            returns,
            trading_days=trading_days,
        )
    )

    arithmetic = (
        calculate_arithmetic_expected_return(
            returns,
            trading_days=trading_days,
        )
    )

    exponentially_weighted = (
        calculate_exponentially_weighted_expected_return(
            returns,
            span=ewma_span,
            trading_days=trading_days,
        )
    )

    blended = (
        calculate_blended_expected_return(
            geometric=geometric,
            arithmetic=arithmetic,
            exponentially_weighted=
                exponentially_weighted,
            geometric_weight=
                geometric_weight,
            arithmetic_weight=
                arithmetic_weight,
            exponentially_weighted_weight=
                exponentially_weighted_weight,
        )
    )

    robust = (
        calculate_robust_expected_return(
            geometric=geometric,
            arithmetic=arithmetic,
            exponentially_weighted=
                exponentially_weighted,
        )
    )

    dispersion = (
        calculate_estimator_dispersion(
            geometric=geometric,
            arithmetic=arithmetic,
            exponentially_weighted=
                exponentially_weighted,
        )
    )

    return ExpectedReturnResult(
        geometric=geometric,
        arithmetic=arithmetic,
        exponentially_weighted=
            exponentially_weighted,
        blended=blended,
        robust=robust,
        estimator_dispersion=
            dispersion,
    )


def expected_return_table(
    returns: pd.DataFrame,
    ewma_span: int = 126,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Produce a comparison table for all expected-return estimators.
    """
    result = estimate_expected_returns(
        returns=returns,
        ewma_span=ewma_span,
        trading_days=trading_days,
    )

    return pd.DataFrame(
        {
            "geometric":
                result.geometric,

            "arithmetic":
                result.arithmetic,

            "ewma":
                result.exponentially_weighted,

            "blended":
                result.blended,

            "robust":
                result.robust,

            "estimator_dispersion":
                result.estimator_dispersion,
        }
    )

def calculate_robust_expected_return(
    geometric: pd.Series,
    arithmetic: pd.Series,
    exponentially_weighted: pd.Series,
    strategic_weight: float = 0.80,
    tactical_weight: float = 0.20,
) -> pd.Series:
    """
    Calculate a conservative strategic/tactical expected return.

    Strategic component
    -------------------
    Average of geometric and arithmetic long-run estimates.

    Tactical component
    ------------------
    Exponentially weighted recent return estimate.

    Default allocation
    ------------------
    80% strategic
    20% tactical

    This prevents recent return regimes from dominating the
    expected-return vector used by portfolio optimizers.
    """
    if not (
        geometric.index.equals(arithmetic.index)
        and geometric.index.equals(
            exponentially_weighted.index
        )
    ):
        raise ValueError(
            "Expected-return estimator indexes must match."
        )

    if strategic_weight < 0 or tactical_weight < 0:
        raise ValueError(
            "Expected-return weights cannot be negative."
        )

    total_weight = (
        strategic_weight
        + tactical_weight
    )

    if np.isclose(
        total_weight,
        0.0,
    ):
        raise ValueError(
            "Expected-return weights cannot sum to zero."
        )

    strategic_weight /= total_weight
    tactical_weight /= total_weight

    strategic = (
        geometric
        + arithmetic
    ) / 2.0

    robust = (
        strategic
        * strategic_weight
        + exponentially_weighted
        * tactical_weight
    )

    robust.name = (
        "robust_expected_return"
    )

    return robust

def test_robust_expected_return():
    from src.forecasts.expected_returns import (
        calculate_robust_expected_return,
    )

    geometric = pd.Series(
        {"AAA": 0.10}
    )

    arithmetic = pd.Series(
        {"AAA": 0.12}
    )

    ewma = pd.Series(
        {"AAA": 0.30}
    )

    result = (
        calculate_robust_expected_return(
            geometric,
            arithmetic,
            ewma,
            strategic_weight=0.80,
            tactical_weight=0.20,
        )
    )

    strategic = (
        0.10 + 0.12
    ) / 2

    expected = (
        strategic * 0.80
        + 0.30 * 0.20
    )

    assert np.isclose(
        result["AAA"],
        expected,
    )