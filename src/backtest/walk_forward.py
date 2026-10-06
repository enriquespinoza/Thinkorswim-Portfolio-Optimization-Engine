from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ensemble.consensus import (
    build_consensus_portfolio,
)
from src.forecasts.expected_returns import (
    estimate_expected_returns,
)
from src.optimizers.hrp import (
    optimize_hrp,
)
from src.optimizers.maximum_sharpe import (
    optimize_maximum_sharpe,
)
from src.optimizers.minimum_variance import (
    build_equal_weight_portfolio,
    optimize_minimum_variance,
)
from src.optimizers.risk_parity import (
    optimize_risk_parity,
)
from src.risk.covariance import (
    calculate_ledoit_wolf_covariance,
)


@dataclass(frozen=True)
class WalkForwardResult:
    """
    Complete walk-forward backtest output.
    """

    period_results: pd.DataFrame
    weights: pd.DataFrame


def generate_month_end_dates(
    returns: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Return the final trading date of each month.
    """
    if not isinstance(
        returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "returns must use a DatetimeIndex."
        )

    dates = (
        returns
        .groupby(
            returns.index.to_period("M")
        )
        .tail(1)
        .index
    )

    return pd.DatetimeIndex(
        dates
    )


def calculate_period_asset_returns(
    returns: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Calculate compounded asset returns after the rebalance date
    through the next rebalance date.

    The start date itself is excluded to prevent using the
    rebalance day's return after weights are determined.
    """
    period = returns.loc[
        (
            returns.index > start_date
        )
        & (
            returns.index <= end_date
        )
    ]

    if period.empty:
        raise ValueError(
            "No returns available for the requested holding period."
        )

    return (
        (1.0 + period)
        .prod()
        - 1.0
    )


def calculate_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series | None,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Between two fully invested portfolios:

        turnover = 0.5 * sum(abs(w_new - w_old))

    Initial investment is treated as 100% turnover.
    """
    if previous_weights is None:
        return 1.0

    previous_weights = (
        previous_weights
        .reindex(
            current_weights.index
        )
        .fillna(0.0)
    )

    return float(
        0.5
        * np.abs(
            current_weights
            - previous_weights
        ).sum()
    )


def _build_model_weights(
    training_returns: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    ewma_span: int,
) -> dict[str, pd.Series]:
    """
    Estimate portfolio weights using only the supplied
    historical training window.
    """
    covariance, _ = (
        calculate_ledoit_wolf_covariance(
            training_returns
        )
    )

    forecasts = (
        estimate_expected_returns(
            training_returns,
            ewma_span=ewma_span,
        )
    )

    equal_weight = (
        build_equal_weight_portfolio(
            covariance
        )
    )

    minimum_variance = (
        optimize_minimum_variance(
            covariance=covariance,
            max_weight=max_weight,
        )
    )

    risk_parity = (
        optimize_risk_parity(
            covariance=covariance,
            max_weight=max_weight,
        )
    )

    hrp = optimize_hrp(
        covariance=covariance
    )

    risk_model_weights = pd.DataFrame(
        {
            "minimum_variance":
                minimum_variance.weights,
            "risk_parity":
                risk_parity.weights,
            "hrp":
                hrp.weights,
        }
    )

    consensus = (
        build_consensus_portfolio(
            risk_model_weights
        )
    )

    maximum_sharpe = (
        optimize_maximum_sharpe(
            expected_returns=
                forecasts.robust,
            covariance=covariance,
            risk_free_rate=
                risk_free_rate,
            max_weight=max_weight,
        )
    )

    return {
        "equal_weight":
            equal_weight,

        "minimum_variance":
            minimum_variance.weights,

        "risk_parity":
            risk_parity.weights,

        "hrp":
            hrp.weights,

        "risk_consensus":
            consensus.consensus_weights,

        "maximum_sharpe":
            maximum_sharpe.weights,
    }


def run_walk_forward(
    returns: pd.DataFrame,
    min_train_observations: int = 756,
    max_weight: float = 0.40,
    risk_free_rate: float = 0.0,
    ewma_span: int = 126,
    transaction_cost_bps: float = 0.0,
) -> WalkForwardResult:
    """
    Run an expanding-window monthly walk-forward backtest.

    Parameters
    ----------
    returns:
        Daily synchronized asset returns.

    min_train_observations:
        Minimum history required before generating the first
        portfolio. 756 trading days is approximately three years.

    max_weight:
        Maximum weight used by constrained optimizers.

    risk_free_rate:
        Annualized risk-free rate used by maximum Sharpe.

    ewma_span:
        EWMA span used by the expected-return model.

    transaction_cost_bps:
        Transaction cost applied to portfolio turnover.

        Example:
            5 = five basis points per unit of turnover.

    Returns
    -------
    WalkForwardResult
        Period performance and historical portfolio weights.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    if min_train_observations < 2:
        raise ValueError(
            "min_train_observations must be at least 2."
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )

    returns = (
        returns
        .sort_index()
        .copy()
    )

    month_end_dates = (
        generate_month_end_dates(
            returns
        )
    )

    period_records = []
    weight_records = []

    previous_drifted_weights: dict[
        str,
        pd.Series,
    ] = {}

    for index in range(
        len(month_end_dates) - 1
    ):
        rebalance_date = (
            month_end_dates[index]
        )

        next_rebalance_date = (
            month_end_dates[
                index + 1
            ]
        )

        training_returns = (
            returns.loc[
                returns.index
                <= rebalance_date
            ]
        )

        if (
            len(training_returns)
            < min_train_observations
        ):
            continue

        model_weights = (
            _build_model_weights(
                training_returns=
                    training_returns,
                max_weight=
                    max_weight,
                risk_free_rate=
                    risk_free_rate,
                ewma_span=
                    ewma_span,
            )
        )

        realized_asset_returns = (
            calculate_period_asset_returns(
                returns=returns,
                start_date=
                    rebalance_date,
                end_date=
                    next_rebalance_date,
            )
        )

        for (
            model_name,
            weights,
        ) in model_weights.items():

            weights = (
                weights
                .reindex(
                    returns.columns
                )
                .fillna(0.0)
            )

            gross_return = float(
                weights
                @ realized_asset_returns
            )

            turnover = (
                calculate_turnover(
                    current_weights=
                        weights,
                    previous_weights=
                        previous_drifted_weights.get(
                            model_name
                        ),
                )
            )

            transaction_cost = (
                turnover
                * transaction_cost_bps
                / 10000.0
            )

            net_return = (
                gross_return
                - transaction_cost
            )

            drifted_weights = (
                calculate_drifted_weights(
                    starting_weights=weights,
                    asset_returns=
                    realized_asset_returns,
                )
            )

            period_records.append(
                {
                    "model":
                        model_name,
                    "rebalance_date":
                        rebalance_date,
                    "period_end":
                        next_rebalance_date,
                    "gross_return":
                        gross_return,
                    "turnover":
                        turnover,
                    "transaction_cost":
                        transaction_cost,
                    "net_return":
                        net_return,
                }
            )

            for asset in weights.index:
                weight_records.append(
                    {
                        "model":
                            model_name,
                        "rebalance_date":
                            rebalance_date,
                        "asset":
                            asset,
                        "weight":
                            weights[
                                asset
                            ],
                    }
                )

            previous_drifted_weights[
                model_name
            ] = (
                drifted_weights.copy()
            )
            

    period_results = pd.DataFrame(
        period_records
    )

    weights = pd.DataFrame(
        weight_records
    )

    if period_results.empty:
        raise ValueError(
            "Walk-forward backtest produced no test periods. "
            "Check available history and min_train_observations."
        )

    return WalkForwardResult(
        period_results=
            period_results,
        weights=weights,
    )
def calculate_drifted_weights(
    starting_weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    """
    Calculate portfolio weights after asset prices move but
    before the next rebalance.

    Formula
    -------
    ending_value_i =
        starting_weight_i * (1 + asset_return_i)

    drifted_weight_i =
        ending_value_i / sum(ending_values)
    """
    asset_returns = asset_returns.reindex(
        starting_weights.index
    )

    if asset_returns.isna().any():
        raise ValueError(
            "asset_returns are missing one or more "
            "portfolio assets."
        )

    ending_values = (
        starting_weights
        * (
            1.0
            + asset_returns
        )
    )

    total_value = (
        ending_values.sum()
    )

    if total_value <= 0:
        raise ValueError(
            "Portfolio ending value must be positive."
        )

    drifted_weights = (
        ending_values
        / total_value
    )

    drifted_weights.name = (
        "drifted_weight"
    )

    return drifted_weights