from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    leaves_list,
    linkage,
)
from scipy.spatial.distance import squareform

from src.risk.covariance import (
    covariance_to_correlation,
)
from src.risk.portfolio_risk import (
    calculate_percentage_risk_contribution,
    calculate_portfolio_volatility,
    risk_concentration_index,
    validate_covariance,
)


@dataclass(frozen=True)
class HRPResult:
    """
    Result returned by the Hierarchical Risk Parity allocator.
    """

    weights: pd.Series
    portfolio_volatility: float
    percentage_risk_contribution: pd.Series
    risk_concentration_index: float
    ordered_assets: tuple[str, ...]
    linkage_matrix: np.ndarray | None
    success: bool
    message: str


def correlation_to_distance(
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a correlation matrix into a distance matrix.

    Formula
    -------
    d(i, j) = sqrt((1 - rho(i, j)) / 2)

    Properties
    ----------
    correlation = 1  -> distance = 0
    correlation = 0  -> distance ~= 0.7071
    correlation = -1 -> distance = 1
    """
    if not isinstance(
        correlation,
        pd.DataFrame,
    ):
        raise TypeError(
            "correlation must be a pandas DataFrame."
        )

    if correlation.empty:
        raise ValueError(
            "correlation cannot be empty."
        )

    if (
        correlation.shape[0]
        != correlation.shape[1]
    ):
        raise ValueError(
            "correlation matrix must be square."
        )

    if list(
        correlation.index
    ) != list(
        correlation.columns
    ):
        raise ValueError(
            "correlation index and columns must match."
        )

    values = correlation.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "correlation contains non-finite values."
        )

    if not np.allclose(
        values,
        values.T,
        atol=1e-10,
    ):
        raise ValueError(
            "correlation matrix must be symmetric."
        )

    values = np.clip(
        values,
        -1.0,
        1.0,
    )

    distance = np.sqrt(
        np.maximum(
            (1.0 - values) / 2.0,
            0.0,
        )
    )

    # Force exact zero diagonal for squareform.
    np.fill_diagonal(
        distance,
        0.0,
    )

    return pd.DataFrame(
        distance,
        index=correlation.index,
        columns=correlation.columns,
    )


def hierarchical_cluster_order(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
) -> tuple[
    tuple[str, ...],
    np.ndarray | None,
]:
    """
    Determine the hierarchical ordering of portfolio assets.

    Parameters
    ----------
    covariance:
        Covariance matrix.

    linkage_method:
        SciPy hierarchical clustering linkage method.

    Returns
    -------
    tuple
        Ordered asset names and linkage matrix.
    """
    covariance = validate_covariance(
        covariance
    )

    assets = covariance.columns

    if len(assets) == 1:
        return (
            tuple(assets),
            None,
        )

    correlation = (
        covariance_to_correlation(
            covariance
        )
    )

    distance = (
        correlation_to_distance(
            correlation
        )
    )

    condensed_distance = squareform(
        distance.to_numpy(),
        checks=False,
    )

    linkage_matrix = linkage(
        condensed_distance,
        method=linkage_method,
    )

    leaf_order = leaves_list(
        linkage_matrix
    )

    ordered_assets = tuple(
        assets[index]
        for index in leaf_order
    )

    return (
        ordered_assets,
        linkage_matrix,
    )


def inverse_variance_weights(
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Calculate inverse-variance portfolio weights.

    Used internally by HRP when estimating cluster variance.
    """
    covariance = validate_covariance(
        covariance
    )

    variances = pd.Series(
        np.diag(
            covariance.to_numpy()
        ),
        index=covariance.columns,
        dtype=float,
    )

    if (
        variances <= 0
    ).any():
        raise ValueError(
            "All covariance diagonal values "
            "must be positive."
        )

    inverse_variances = (
        1.0 / variances
    )

    weights = (
        inverse_variances
        / inverse_variances.sum()
    )

    weights.name = (
        "inverse_variance_weight"
    )

    return weights


def calculate_cluster_variance(
    covariance: pd.DataFrame,
    cluster_assets: list[str]
    | tuple[str, ...],
) -> float:
    """
    Calculate the variance of a cluster using
    inverse-variance weights within the cluster.
    """
    covariance = validate_covariance(
        covariance
    )

    if not cluster_assets:
        raise ValueError(
            "cluster_assets cannot be empty."
        )

    unknown_assets = (
        set(cluster_assets)
        - set(covariance.columns)
    )

    if unknown_assets:
        raise ValueError(
            f"Unknown cluster assets: "
            f"{sorted(unknown_assets)}"
        )

    sub_covariance = covariance.loc[
        cluster_assets,
        cluster_assets,
    ]

    weights = (
        inverse_variance_weights(
            sub_covariance
        )
    )

    weight_array = (
        weights.to_numpy()
    )

    covariance_array = (
        sub_covariance.to_numpy()
    )

    variance = float(
        weight_array.T
        @ covariance_array
        @ weight_array
    )

    if variance <= 0:
        raise ValueError(
            "Cluster variance must be positive."
        )

    return variance


def recursive_bisection(
    covariance: pd.DataFrame,
    ordered_assets: tuple[str, ...]
    | list[str],
) -> pd.Series:
    """
    Allocate portfolio weights through recursive bisection.

    Each cluster is split into two subclusters. Capital is allocated
    inversely to cluster variance, so the lower-risk cluster receives
    more capital.
    """
    covariance = validate_covariance(
        covariance
    )

    ordered_assets = list(
        ordered_assets
    )

    if not ordered_assets:
        raise ValueError(
            "ordered_assets cannot be empty."
        )

    if set(
        ordered_assets
    ) != set(
        covariance.columns
    ):
        raise ValueError(
            "ordered_assets must contain exactly "
            "the covariance assets."
        )

    weights = pd.Series(
        1.0,
        index=ordered_assets,
        dtype=float,
    )

    clusters: list[list[str]] = [
        ordered_assets
    ]

    while clusters:
        next_clusters: list[
            list[str]
        ] = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            split = (
                len(cluster) // 2
            )

            left_cluster = (
                cluster[:split]
            )

            right_cluster = (
                cluster[split:]
            )

            left_variance = (
                calculate_cluster_variance(
                    covariance,
                    left_cluster,
                )
            )

            right_variance = (
                calculate_cluster_variance(
                    covariance,
                    right_cluster,
                )
            )

            total_variance = (
                left_variance
                + right_variance
            )

            if total_variance <= 0:
                raise ValueError(
                    "Combined cluster variance "
                    "must be positive."
                )

            # Lower variance cluster receives more capital.
            left_allocation = (
                1.0
                - left_variance
                / total_variance
            )

            right_allocation = (
                1.0
                - left_allocation
            )

            weights.loc[
                left_cluster
            ] *= left_allocation

            weights.loc[
                right_cluster
            ] *= right_allocation

            if len(
                left_cluster
            ) > 1:
                next_clusters.append(
                    left_cluster
                )

            if len(
                right_cluster
            ) > 1:
                next_clusters.append(
                    right_cluster
                )

        clusters = (
            next_clusters
        )

    weights = (
        weights
        / weights.sum()
    )

    weights = weights.reindex(
        covariance.columns
    )

    weights.name = "weight"

    return weights


def optimize_hrp(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
) -> HRPResult:
    """
    Construct a Hierarchical Risk Parity portfolio.

    HRP workflow
    ------------
    1. Convert covariance to correlation.
    2. Convert correlation to distance.
    3. Perform hierarchical clustering.
    4. Reorder assets by cluster structure.
    5. Recursively allocate capital based on cluster variance.

    Parameters
    ----------
    covariance:
        Annualized covariance matrix.

    linkage_method:
        Hierarchical clustering method.

    Returns
    -------
    HRPResult
        HRP weights and diagnostics.
    """
    covariance = validate_covariance(
        covariance
    )

    ordered_assets, linkage_matrix = (
        hierarchical_cluster_order(
            covariance=covariance,
            linkage_method=linkage_method,
        )
    )

    if len(
        covariance.columns
    ) == 1:
        weights = pd.Series(
            [1.0],
            index=covariance.columns,
            name="weight",
        )

    else:
        weights = recursive_bisection(
            covariance=covariance,
            ordered_assets=ordered_assets,
        )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            weights,
            covariance,
        )
    )

    percentage_risk = (
        calculate_percentage_risk_contribution(
            weights,
            covariance,
        )
    )

    concentration = (
        risk_concentration_index(
            weights,
            covariance,
        )
    )

    return HRPResult(
        weights=weights,
        portfolio_volatility=portfolio_volatility,
        percentage_risk_contribution=percentage_risk,
        risk_concentration_index=concentration,
        ordered_assets=ordered_assets,
        linkage_matrix=linkage_matrix,
        success=True,
        message=(
            "HRP allocation completed successfully."
        ),
    )
