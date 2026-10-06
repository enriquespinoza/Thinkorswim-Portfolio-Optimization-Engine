from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config.settings import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class CovarianceComparison:
    """
    Diagnostics comparing sample and Ledoit-Wolf covariance estimates.
    """

    sample_condition_number: float
    ledoit_wolf_condition_number: float
    sample_min_eigenvalue: float
    ledoit_wolf_min_eigenvalue: float
    sample_max_eigenvalue: float
    ledoit_wolf_max_eigenvalue: float
    shrinkage: float
    frobenius_difference: float


def validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a return matrix before covariance estimation.

    Parameters
    ----------
    returns:
        DataFrame with rows as observations and columns as assets.

    Returns
    -------
    pd.DataFrame
        Numeric, synchronized return observations.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if returns.shape[1] < 1:
        raise ValueError("returns must contain at least one asset.")

    cleaned = returns.copy()

    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(how="any")

    if cleaned.empty:
        raise ValueError(
            "No complete return observations remain after cleaning."
        )

    if len(cleaned) < 2:
        raise ValueError(
            "At least two return observations are required."
        )

    return cleaned


def calculate_sample_covariance(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Calculate the annualized sample covariance matrix.

    Parameters
    ----------
    returns:
        Daily return matrix.
    trading_days:
        Number of trading periods used for annualization.

    Returns
    -------
    pd.DataFrame
        Annualized sample covariance matrix.
    """
    if trading_days <= 0:
        raise ValueError("trading_days must be greater than zero.")

    clean_returns = validate_returns(returns)

    covariance = clean_returns.cov() * trading_days

    return covariance


def calculate_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.DataFrame, float]:
    """
    Calculate annualized Ledoit-Wolf shrinkage covariance.

    Ledoit-Wolf shrinks the empirical covariance matrix toward a
    structured target to improve numerical stability and reduce
    estimation error.

    Parameters
    ----------
    returns:
        Daily return matrix.
    trading_days:
        Number of trading periods used for annualization.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Annualized covariance matrix and shrinkage coefficient.
    """
    if trading_days <= 0:
        raise ValueError("trading_days must be greater than zero.")

    clean_returns = validate_returns(returns)

    estimator = LedoitWolf(
        assume_centered=False,
        store_precision=False,
    )

    estimator.fit(clean_returns)

    covariance = estimator.covariance_ * trading_days

    covariance_df = pd.DataFrame(
        covariance,
        index=clean_returns.columns,
        columns=clean_returns.columns,
    )

    return covariance_df, float(estimator.shrinkage_)


def covariance_to_correlation(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a covariance matrix to a correlation matrix.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if covariance.empty:
        raise ValueError("covariance cannot be empty.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square.")

    variances = np.diag(covariance.to_numpy())

    if np.any(variances <= 0):
        raise ValueError(
            "covariance matrix must have positive diagonal variances."
        )

    standard_deviations = np.sqrt(variances)

    denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = covariance.to_numpy() / denominator

    correlation = np.clip(
        correlation,
        -1.0,
        1.0,
    )

    return pd.DataFrame(
        correlation,
        index=covariance.index,
        columns=covariance.columns,
    )


def calculate_condition_number(
    covariance: pd.DataFrame,
) -> float:
    """
    Calculate the matrix condition number.

    Lower values generally indicate a numerically more stable matrix.
    Very large values indicate near-singularity and potential optimizer
    instability.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square.")

    return float(
        np.linalg.cond(
            covariance.to_numpy(),
        )
    )


def covariance_eigenvalues(
    covariance: pd.DataFrame,
) -> np.ndarray:
    """
    Return covariance eigenvalues in ascending order.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square.")

    values = np.linalg.eigvalsh(
        covariance.to_numpy(),
    )

    return np.sort(values)


def compare_covariance_estimators(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> CovarianceComparison:
    """
    Compare sample covariance against Ledoit-Wolf covariance.

    Diagnostics include:
    - condition number
    - minimum eigenvalue
    - maximum eigenvalue
    - Ledoit-Wolf shrinkage coefficient
    - Frobenius distance between the matrices
    """
    sample_cov = calculate_sample_covariance(
        returns,
        trading_days=trading_days,
    )

    lw_cov, shrinkage = calculate_ledoit_wolf_covariance(
        returns,
        trading_days=trading_days,
    )

    sample_eigenvalues = covariance_eigenvalues(
        sample_cov,
    )

    lw_eigenvalues = covariance_eigenvalues(
        lw_cov,
    )

    difference = (
        sample_cov.to_numpy()
        - lw_cov.to_numpy()
    )

    frobenius_difference = float(
        np.linalg.norm(
            difference,
            ord="fro",
        )
    )

    return CovarianceComparison(
        sample_condition_number=calculate_condition_number(
            sample_cov
        ),
        ledoit_wolf_condition_number=calculate_condition_number(
            lw_cov
        ),
        sample_min_eigenvalue=float(
            sample_eigenvalues.min()
        ),
        ledoit_wolf_min_eigenvalue=float(
            lw_eigenvalues.min()
        ),
        sample_max_eigenvalue=float(
            sample_eigenvalues.max()
        ),
        ledoit_wolf_max_eigenvalue=float(
            lw_eigenvalues.max()
        ),
        shrinkage=shrinkage,
        frobenius_difference=frobenius_difference,
    )


def covariance_comparison_table(
    returns: pd.DataFrame,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Return human-readable covariance diagnostics.
    """
    comparison = compare_covariance_estimators(
        returns,
        trading_days=trading_days,
    )

    return pd.DataFrame(
        {
            "sample_covariance": {
                "condition_number":
                    comparison.sample_condition_number,
                "min_eigenvalue":
                    comparison.sample_min_eigenvalue,
                "max_eigenvalue":
                    comparison.sample_max_eigenvalue,
                "shrinkage":
                    0.0,
            },
            "ledoit_wolf": {
                "condition_number":
                    comparison.ledoit_wolf_condition_number,
                "min_eigenvalue":
                    comparison.ledoit_wolf_min_eigenvalue,
                "max_eigenvalue":
                    comparison.ledoit_wolf_max_eigenvalue,
                "shrinkage":
                    comparison.shrinkage,
            },
        }
    )
