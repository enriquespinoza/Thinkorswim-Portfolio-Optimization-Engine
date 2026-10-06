from __future__ import annotations

import numpy as np
import pandas as pd


def validate_covariance(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a covariance matrix.

    Parameters
    ----------
    covariance:
        Square covariance matrix with matching index and columns.

    Returns
    -------
    pd.DataFrame
        Clean numeric covariance matrix.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(
            "covariance must be a pandas DataFrame."
        )

    if covariance.empty:
        raise ValueError(
            "covariance cannot be empty."
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance matrix must be square."
        )

    if list(covariance.index) != list(covariance.columns):
        raise ValueError(
            "covariance index and columns must match "
            "and use the same order."
        )

    cleaned = covariance.copy()

    cleaned = cleaned.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if cleaned.isna().any().any():
        raise ValueError(
            "covariance contains missing or "
            "non-numeric values."
        )

    values = cleaned.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "covariance contains non-finite values."
        )

    if not np.allclose(
        values,
        values.T,
        atol=1e-10,
    ):
        raise ValueError(
            "covariance matrix must be symmetric."
        )

    return cleaned


def align_weights(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Validate and align portfolio weights to covariance assets.

    Weight values do not have to sum to 1 here because this risk
    module may later be used for leveraged, hedged, or partially
    invested portfolios.

    Parameters
    ----------
    weights:
        Portfolio weights indexed by asset symbol.
    covariance:
        Covariance matrix.

    Returns
    -------
    pd.Series
        Weights aligned to covariance asset order.
    """
    covariance = validate_covariance(
        covariance
    )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "weights must be a pandas Series."
        )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    missing = set(covariance.columns) - set(
        weights.index
    )

    extra = set(weights.index) - set(
        covariance.columns
    )

    if missing:
        raise ValueError(
            f"weights are missing assets: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"weights contain unknown assets: "
            f"{sorted(extra)}"
        )

    aligned = weights.reindex(
        covariance.columns
    )

    aligned = pd.to_numeric(
        aligned,
        errors="coerce",
    )

    if aligned.isna().any():
        raise ValueError(
            "weights contain missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        aligned.to_numpy()
    ).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    return aligned.astype(float)


def calculate_portfolio_variance(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate portfolio variance.

    Formula
    -------
    variance = w.T @ covariance @ w
    """
    covariance = validate_covariance(
        covariance
    )

    weights = align_weights(
        weights,
        covariance,
    )

    w = weights.to_numpy()
    sigma = covariance.to_numpy()

    variance = float(
        w.T @ sigma @ w
    )

    # Protect against tiny negative values caused by
    # floating-point precision.
    if variance < 0 and np.isclose(
        variance,
        0.0,
        atol=1e-12,
    ):
        variance = 0.0

    if variance < 0:
        raise ValueError(
            "portfolio variance is negative. "
            "Check the covariance matrix."
        )

    return variance


def calculate_portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate portfolio volatility.

    Formula
    -------
    volatility = sqrt(w.T @ covariance @ w)
    """
    variance = calculate_portfolio_variance(
        weights,
        covariance,
    )

    return float(
        np.sqrt(variance)
    )


def calculate_marginal_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate marginal contribution to portfolio volatility.

    Marginal risk contribution measures how portfolio volatility
    changes for a small change in an asset's weight.

    Formula
    -------
    MRC_i = (covariance @ weights)_i / portfolio_volatility
    """
    covariance = validate_covariance(
        covariance
    )

    weights = align_weights(
        weights,
        covariance,
    )

    volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    if np.isclose(
        volatility,
        0.0,
    ):
        raise ValueError(
            "marginal risk contribution is undefined "
            "for a zero-volatility portfolio."
        )

    marginal = (
        covariance.to_numpy()
        @ weights.to_numpy()
    ) / volatility

    return pd.Series(
        marginal,
        index=covariance.columns,
        name="marginal_risk_contribution",
    )


def calculate_component_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate each asset's contribution to total portfolio volatility.

    Formula
    -------
    CRC_i = weight_i * MRC_i

    The component risk contributions sum to total portfolio
    volatility.
    """
    covariance = validate_covariance(
        covariance
    )

    weights = align_weights(
        weights,
        covariance,
    )

    marginal = (
        calculate_marginal_risk_contribution(
            weights,
            covariance,
        )
    )

    component = weights * marginal

    component.name = (
        "component_risk_contribution"
    )

    return component


def calculate_percentage_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate each asset's percentage contribution to portfolio risk.

    Formula
    -------
    PRC_i = component_risk_i / portfolio_volatility

    For a standard long-only portfolio these normally sum to 1.
    Hedged portfolios can contain negative risk contributions.
    """
    covariance = validate_covariance(
        covariance
    )

    weights = align_weights(
        weights,
        covariance,
    )

    volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    if np.isclose(
        volatility,
        0.0,
    ):
        raise ValueError(
            "percentage risk contribution is undefined "
            "for a zero-volatility portfolio."
        )

    component = (
        calculate_component_risk_contribution(
            weights,
            covariance,
        )
    )

    percentage = component / volatility

    percentage.name = (
        "percentage_risk_contribution"
    )

    return percentage


def risk_contribution_table(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a portfolio risk decomposition table.

    Returns
    -------
    pd.DataFrame
        Columns:
        - weight
        - asset_volatility
        - marginal_risk_contribution
        - component_risk_contribution
        - percentage_risk_contribution
    """
    covariance = validate_covariance(
        covariance
    )

    weights = align_weights(
        weights,
        covariance,
    )

    marginal = (
        calculate_marginal_risk_contribution(
            weights,
            covariance,
        )
    )

    component = (
        calculate_component_risk_contribution(
            weights,
            covariance,
        )
    )

    percentage = (
        calculate_percentage_risk_contribution(
            weights,
            covariance,
        )
    )

    asset_volatility = pd.Series(
        np.sqrt(
            np.diag(
                covariance.to_numpy()
            )
        ),
        index=covariance.columns,
        name="asset_volatility",
    )

    table = pd.DataFrame(
        {
            "weight": weights,
            "asset_volatility":
                asset_volatility,
            "marginal_risk_contribution":
                marginal,
            "component_risk_contribution":
                component,
            "percentage_risk_contribution":
                percentage,
        }
    )

    return table


def risk_concentration_index(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate a Herfindahl-style risk concentration index.

    Uses squared percentage risk contributions.

    Interpretation
    --------------
    Lower values indicate risk is distributed more evenly.
    Higher values indicate portfolio risk is concentrated in
    fewer assets.

    For N assets with perfectly equal risk contribution:

        minimum approximately = 1 / N
    """
    contributions = (
        calculate_percentage_risk_contribution(
            weights,
            covariance,
        )
    )

    return float(
        np.sum(
            np.square(
                contributions.to_numpy()
            )
        )
    )
def effective_number_of_risk_positions(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate the effective number of equally important
    portfolio risk contributors.

    Formula
    -------
    effective_positions = 1 / sum(PRC_i ** 2)

    Interpretation
    --------------
    A value near N indicates broadly distributed portfolio risk.
    A value near 1 indicates highly concentrated portfolio risk.
    """
    concentration = risk_concentration_index(
        weights,
        covariance,
    )

    if concentration <= 0:
        raise ValueError(
            "Risk concentration must be positive."
        )

    return float(
        1.0 / concentration
    )