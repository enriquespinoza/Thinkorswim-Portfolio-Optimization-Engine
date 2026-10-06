from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk.portfolio_risk import (
    calculate_portfolio_volatility,
    validate_covariance,
)


@dataclass(frozen=True)
class MaximumSharpeResult:
    """
    Result returned by the maximum-Sharpe optimizer.
    """

    weights: pd.Series
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    iterations: int
    objective_value: float


def validate_expected_returns(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Validate and align expected returns with covariance assets.
    """
    covariance = validate_covariance(
        covariance
    )

    if not isinstance(
        expected_returns,
        pd.Series,
    ):
        raise TypeError(
            "expected_returns must be a pandas Series."
        )

    if expected_returns.empty:
        raise ValueError(
            "expected_returns cannot be empty."
        )

    missing = (
        set(covariance.columns)
        - set(expected_returns.index)
    )

    extra = (
        set(expected_returns.index)
        - set(covariance.columns)
    )

    if missing:
        raise ValueError(
            f"expected_returns are missing assets: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"expected_returns contain unknown assets: "
            f"{sorted(extra)}"
        )

    aligned = (
        expected_returns
        .reindex(
            covariance.columns
        )
    )

    aligned = pd.to_numeric(
        aligned,
        errors="coerce",
    )

    if aligned.isna().any():
        raise ValueError(
            "expected_returns contain missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        aligned.to_numpy()
    ).all():
        raise ValueError(
            "expected_returns contain non-finite values."
        )

    return aligned.astype(float)


def _validate_bounds(
    number_of_assets: int,
    min_weight: float,
    max_weight: float,
) -> None:
    """
    Validate long-only portfolio constraints.
    """
    if number_of_assets <= 0:
        raise ValueError(
            "number_of_assets must be greater than zero."
        )

    if min_weight < 0:
        raise ValueError(
            "min_weight cannot be negative for "
            "the long-only optimizer."
        )

    if max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed 1.0 for "
            "the long-only optimizer."
        )

    if min_weight > max_weight:
        raise ValueError(
            "min_weight cannot be greater than max_weight."
        )

    if (
        number_of_assets
        * min_weight
        > 1.0
    ):
        raise ValueError(
            "min_weight is too large to create "
            "a fully invested portfolio."
        )

    if (
        number_of_assets
        * max_weight
        < 1.0
    ):
        raise ValueError(
            "max_weight is too small to create "
            "a fully invested portfolio."
        )


def _build_initial_weights(
    number_of_assets: int,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """
    Construct a feasible initial portfolio.
    """
    equal_weight = (
        1.0 / number_of_assets
    )

    if (
        min_weight
        <= equal_weight
        <= max_weight
    ):
        return np.full(
            number_of_assets,
            equal_weight,
            dtype=float,
        )

    weights = np.full(
        number_of_assets,
        min_weight,
        dtype=float,
    )

    remaining = (
        1.0
        - weights.sum()
    )

    for index in range(
        number_of_assets
    ):
        if remaining <= 1e-12:
            break

        available = (
            max_weight
            - weights[index]
        )

        addition = min(
            available,
            remaining,
        )

        weights[index] += addition
        remaining -= addition

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Unable to construct feasible initial weights."
        )

    return weights


def calculate_portfolio_expected_return(
    weights: pd.Series,
    expected_returns: pd.Series,
) -> float:
    """
    Calculate expected portfolio return.

    Formula
    -------
    expected_return = weights.T @ expected_returns
    """
    if not isinstance(
        weights,
        pd.Series,
    ):
        raise TypeError(
            "weights must be a pandas Series."
        )

    if not isinstance(
        expected_returns,
        pd.Series,
    ):
        raise TypeError(
            "expected_returns must be a pandas Series."
        )

    if set(
        weights.index
    ) != set(
        expected_returns.index
    ):
        raise ValueError(
            "weights and expected_returns must "
            "contain the same assets."
        )

    aligned_returns = (
        expected_returns.reindex(
            weights.index
        )
    )

    return float(
        weights.to_numpy()
        @ aligned_returns.to_numpy()
    )


def negative_sharpe_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Negative Sharpe ratio used by scipy.optimize.minimize.

    Maximizing Sharpe is equivalent to minimizing negative Sharpe.
    """
    portfolio_return = float(
        weights
        @ expected_returns
    )

    portfolio_variance = float(
        weights.T
        @ covariance
        @ weights
    )

    if portfolio_variance <= 0:
        return 1e12

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    sharpe = (
        portfolio_return
        - risk_free_rate
    ) / portfolio_volatility

    return float(
        -sharpe
    )


def optimize_maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 2000,
) -> MaximumSharpeResult:
    """
    Solve for the long-only maximum-Sharpe portfolio.

    Parameters
    ----------
    expected_returns:
        Annualized expected return for each asset.

    covariance:
        Annualized covariance matrix.

    risk_free_rate:
        Annualized risk-free rate expressed as a decimal.

    min_weight:
        Minimum allowable asset weight.

    max_weight:
        Maximum allowable asset weight.

    tolerance:
        Numerical optimization tolerance.

    max_iterations:
        Maximum SLSQP iterations.
    """
    covariance = validate_covariance(
        covariance
    )

    expected_returns = (
        validate_expected_returns(
            expected_returns,
            covariance,
        )
    )

    number_of_assets = len(
        covariance.columns
    )

    _validate_bounds(
        number_of_assets=
            number_of_assets,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if not np.isfinite(
        risk_free_rate
    ):
        raise ValueError(
            "risk_free_rate must be finite."
        )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be greater than zero."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    initial_weights = (
        _build_initial_weights(
            number_of_assets=
                number_of_assets,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    )

    bounds = [
        (
            min_weight,
            max_weight,
        )
        for _ in range(
            number_of_assets
        )
    ]

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights)
                - 1.0
            ),
        }
    ]

    result = minimize(
        fun=negative_sharpe_objective,
        x0=initial_weights,
        args=(
            expected_returns.to_numpy(),
            covariance.to_numpy(),
            risk_free_rate,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tolerance,
        options={
            "maxiter":
                max_iterations,
            "ftol":
                tolerance,
            "disp":
                False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed: "
            f"{result.message}"
        )

    optimized_weights = pd.Series(
        result.x,
        index=covariance.columns,
        name="weight",
    )

    optimized_weights[
        np.abs(
            optimized_weights
        ) < 1e-12
    ] = 0.0

    optimized_weights = (
        optimized_weights
        / optimized_weights.sum()
    )

    portfolio_return = (
        calculate_portfolio_expected_return(
            optimized_weights,
            expected_returns,
        )
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            optimized_weights,
            covariance,
        )
    )

    sharpe_ratio = (
        portfolio_return
        - risk_free_rate
    ) / portfolio_volatility

    return MaximumSharpeResult(
        weights=optimized_weights,
        expected_return=
            portfolio_return,
        portfolio_volatility=
            portfolio_volatility,
        sharpe_ratio=float(
            sharpe_ratio
        ),
        success=bool(
            result.success
        ),
        message=str(
            result.message
        ),
        iterations=int(
            result.nit
        ),
        objective_value=float(
            result.fun
        ),
    )
