from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk.portfolio_risk import (
    calculate_percentage_risk_contribution,
    calculate_portfolio_volatility,
    risk_concentration_index,
    validate_covariance,
)


@dataclass(frozen=True)
class RiskParityResult:
    """
    Result returned by the equal-risk-contribution optimizer.
    """

    weights: pd.Series
    portfolio_volatility: float
    percentage_risk_contribution: pd.Series
    risk_concentration_index: float
    target_risk_contribution: float
    risk_contribution_error: float
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
    Validate long-only portfolio bounds.
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

    if number_of_assets * min_weight > 1.0:
        raise ValueError(
            "min_weight is too large to create a "
            "portfolio whose weights sum to 1."
        )

    if number_of_assets * max_weight < 1.0:
        raise ValueError(
            "max_weight is too small to create a "
            "portfolio whose weights sum to 1."
        )


def _build_initial_weights(
    number_of_assets: int,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """
    Construct a feasible initial portfolio.

    Equal weighting is used whenever it satisfies the bounds.
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

    remaining = 1.0 - weights.sum()

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


def percentage_risk_contributions_array(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Calculate percentage risk contribution directly from arrays.

    Parameters
    ----------
    weights:
        Portfolio weight vector.

    covariance:
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Percentage contribution to total portfolio volatility.
    """
    portfolio_variance = float(
        weights.T
        @ covariance
        @ weights
    )

    if portfolio_variance <= 0:
        raise ValueError(
            "Portfolio variance must be positive."
        )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    marginal_risk = (
        covariance
        @ weights
    ) / portfolio_volatility

    component_risk = (
        weights
        * marginal_risk
    )

    return (
        component_risk
        / portfolio_volatility
    )


def risk_parity_objective(
    weights: np.ndarray,
    covariance: np.ndarray,
    target_risk: np.ndarray,
) -> float:
    """
    Minimize squared deviations from target risk contributions.

    For equal risk parity:

        target_risk_i = 1 / N
    """
    contributions = (
        percentage_risk_contributions_array(
            weights,
            covariance,
        )
    )

    differences = (
        contributions
        - target_risk
    )

    return float(
        np.sum(
            differences ** 2
        )
    )


def optimize_risk_parity(
    covariance: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    tolerance: float = 1e-12,
    max_iterations: int = 2000,
) -> RiskParityResult:
    """
    Solve for a long-only equal-risk-contribution portfolio.

    The objective attempts to make each asset contribute an equal
    percentage of total portfolio volatility.

    Parameters
    ----------
    covariance:
        Annualized covariance matrix.

    min_weight:
        Minimum allowable portfolio weight.

    max_weight:
        Maximum allowable portfolio weight.

    tolerance:
        Numerical convergence tolerance.

    max_iterations:
        Maximum SLSQP iterations.

    Returns
    -------
    RiskParityResult
        Optimized weights and risk diagnostics.
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

    covariance_array = (
        covariance.to_numpy(
            dtype=float
        )
    )

    initial_weights = (
        _build_initial_weights(
            number_of_assets=number_of_assets,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    )

    target_risk = np.full(
        number_of_assets,
        1.0 / number_of_assets,
        dtype=float,
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
        fun=risk_parity_objective,
        x0=initial_weights,
        args=(
            covariance_array,
            target_risk,
        ),
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
            "Risk-parity optimization failed: "
            f"{result.message}"
        )

    optimized_weights = pd.Series(
        result.x,
        index=assets,
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

    percentage_risk = (
        calculate_percentage_risk_contribution(
            optimized_weights,
            covariance,
        )
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            optimized_weights,
            covariance,
        )
    )

    concentration = (
        risk_concentration_index(
            optimized_weights,
            covariance,
        )
    )

    target = (
        1.0 / number_of_assets
    )

    risk_error = float(
        np.sqrt(
            np.mean(
                (
                    percentage_risk.to_numpy()
                    - target
                ) ** 2
            )
        )
    )

    return RiskParityResult(
        weights=optimized_weights,
        portfolio_volatility=portfolio_volatility,
        percentage_risk_contribution=percentage_risk,
        risk_concentration_index=concentration,
        target_risk_contribution=target,
        risk_contribution_error=risk_error,
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
