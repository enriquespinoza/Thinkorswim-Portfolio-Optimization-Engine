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
class MinimumVarianceResult:
    """
    Result returned by the minimum-variance optimizer.
    """

    weights: pd.Series
    portfolio_volatility: float
    success: bool
    message: str
    iterations: int
    objective_value: float


def _validate_bounds(
    number_of_assets: int,
    min_weight: float,
    max_weight: float,
) -> None:
    """
    Validate portfolio weight constraints.
    """
    if number_of_assets <= 0:
        raise ValueError(
            "number_of_assets must be greater than zero."
        )

    if min_weight > max_weight:
        raise ValueError(
            "min_weight cannot be greater than max_weight."
        )

    if min_weight < 0:
        raise ValueError(
            "min_weight cannot be negative for the long-only optimizer."
        )

    if max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed 1.0 for the long-only optimizer."
        )

    minimum_possible_total = (
        number_of_assets * min_weight
    )

    maximum_possible_total = (
        number_of_assets * max_weight
    )

    if minimum_possible_total > 1.0:
        raise ValueError(
            "min_weight is too large to create a portfolio "
            "whose weights sum to 1."
        )

    if maximum_possible_total < 1.0:
        raise ValueError(
            "max_weight is too small to create a portfolio "
            "whose weights sum to 1."
        )


def _build_initial_weights(
    assets: pd.Index,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """
    Construct a feasible starting portfolio.

    The default starting point is equal weight.
    """
    number_of_assets = len(assets)

    equal_weight = 1.0 / number_of_assets

    if (
        equal_weight >= min_weight
        and equal_weight <= max_weight
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

    remaining = 1.0 - weights.sum()

    for index in range(number_of_assets):
        if remaining <= 0:
            break

        available_capacity = (
            max_weight - weights[index]
        )

        allocation = min(
            available_capacity,
            remaining,
        )

        weights[index] += allocation
        remaining -= allocation

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Unable to construct feasible initial weights."
        )

    return weights


def portfolio_variance_objective(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Objective function minimized by the optimizer.

    Formula
    -------
    variance = w.T @ covariance @ w
    """
    return float(
        weights.T
        @ covariance
        @ weights
    )


def optimize_minimum_variance(
    covariance: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> MinimumVarianceResult:
    """
    Solve for the long-only minimum-variance portfolio.

    Parameters
    ----------
    covariance:
        Annualized covariance matrix.

    min_weight:
        Minimum allowable weight for each asset.

    max_weight:
        Maximum allowable weight for each asset.

    tolerance:
        Optimization convergence tolerance.

    max_iterations:
        Maximum number of optimization iterations.

    Returns
    -------
    MinimumVarianceResult
        Optimized weights and diagnostics.
    """
    covariance = validate_covariance(
        covariance
    )

    assets = covariance.columns

    number_of_assets = len(assets)

    _validate_bounds(
        number_of_assets=number_of_assets,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if tolerance <= 0:
        raise ValueError(
            "tolerance must be greater than zero."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    covariance_array = covariance.to_numpy(
        dtype=float
    )

    initial_weights = _build_initial_weights(
        assets=assets,
        min_weight=min_weight,
        max_weight=max_weight,
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
                np.sum(weights) - 1.0
            ),
        }
    ]

    result = minimize(
        fun=portfolio_variance_objective,
        x0=initial_weights,
        args=(covariance_array,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tolerance,
        options={
            "maxiter": max_iterations,
            "ftol": tolerance,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-variance optimization failed: "
            f"{result.message}"
        )

    optimized_weights = pd.Series(
        result.x,
        index=assets,
        name="weight",
    )

    # Remove tiny floating-point artifacts.
    optimized_weights[
        np.abs(optimized_weights) < 1e-12
    ] = 0.0

    optimized_weights = (
        optimized_weights
        / optimized_weights.sum()
    )

    volatility = (
        calculate_portfolio_volatility(
            optimized_weights,
            covariance,
        )
    )

    return MinimumVarianceResult(
        weights=optimized_weights,
        portfolio_volatility=volatility,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        objective_value=float(result.fun),
    )


def build_equal_weight_portfolio(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Construct an equal-weight portfolio for comparison.
    """
    covariance = validate_covariance(
        covariance
    )

    number_of_assets = len(
        covariance.columns
    )

    return pd.Series(
        1.0 / number_of_assets,
        index=covariance.columns,
        name="weight",
    )


def compare_minimum_variance_to_equal_weight(
    covariance: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Compare the optimized minimum-variance portfolio
    against an equal-weight benchmark.
    """
    covariance = validate_covariance(
        covariance
    )

    equal_weights = (
        build_equal_weight_portfolio(
            covariance
        )
    )

    optimized = optimize_minimum_variance(
        covariance=covariance,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    equal_volatility = (
        calculate_portfolio_volatility(
            equal_weights,
            covariance,
        )
    )

    comparison = pd.DataFrame(
        {
            "equal_weight": equal_weights,
            "minimum_variance":
                optimized.weights,
        }
    )

    comparison.loc[
        "portfolio_volatility"
    ] = [
        equal_volatility,
        optimized.portfolio_volatility,
    ]

    return comparison
