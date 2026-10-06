from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.walk_forward import (
    calculate_drifted_weights,
    calculate_period_asset_returns,
    calculate_turnover,
)


def validate_weight_history(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate long-format historical target weights.

    Required columns:
        model
        rebalance_date
        asset
        weight
    """
    required = {
        "model",
        "rebalance_date",
        "asset",
        "weight",
    }

    missing = (
        required
        - set(weights.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    if weights.empty:
        raise ValueError(
            "weights cannot be empty."
        )

    cleaned = weights.copy()

    cleaned[
        "rebalance_date"
    ] = pd.to_datetime(
        cleaned[
            "rebalance_date"
        ]
    )

    cleaned[
        "weight"
    ] = pd.to_numeric(
        cleaned[
            "weight"
        ],
        errors="coerce",
    )

    if cleaned[
        "weight"
    ].isna().any():
        raise ValueError(
            "weights contain missing or non-numeric values."
        )

    if not np.isfinite(
        cleaned[
            "weight"
        ].to_numpy()
    ).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    return cleaned


def build_blended_weight_history(
    weights: pd.DataFrame,
    model_a: str,
    model_b: str,
    model_b_weight: float,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """
    Create historical target weights that blend two portfolio
    models.

    blended =
        (1 - model_b_weight) * model_a
        + model_b_weight * model_b

    Example
    -------
    model_a = equal_weight
    model_b = maximum_sharpe
    model_b_weight = 0.50

    produces a 50/50 blend.
    """
    weights = validate_weight_history(
        weights
    )

    if not (
        0.0
        <= model_b_weight
        <= 1.0
    ):
        raise ValueError(
            "model_b_weight must be between 0 and 1."
        )

    available_models = set(
        weights[
            "model"
        ].unique()
    )

    if model_a not in available_models:
        raise ValueError(
            f"Unknown model: {model_a}"
        )

    if model_b not in available_models:
        raise ValueError(
            f"Unknown model: {model_b}"
        )

    if strategy_name is None:
        strategy_name = (
            f"blend_{model_b_weight:.2f}"
        )

    subset = weights[
        weights[
            "model"
        ].isin(
            [
                model_a,
                model_b,
            ]
        )
    ]

    pivot = (
        subset.pivot(
            index=[
                "rebalance_date",
                "asset",
            ],
            columns="model",
            values="weight",
        )
        .sort_index()
    )

    if (
        pivot[
            [
                model_a,
                model_b,
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "One or more rebalance dates are missing "
            "weights for a blend model."
        )

    pivot[
        "weight"
    ] = (
        (
            1.0
            - model_b_weight
        )
        * pivot[
            model_a
        ]
        + model_b_weight
        * pivot[
            model_b
        ]
    )

    blended = (
        pivot[
            [
                "weight",
            ]
        ]
        .reset_index()
    )

    blended[
        "model"
    ] = strategy_name

    return blended[
        [
            "model",
            "rebalance_date",
            "asset",
            "weight",
        ]
    ]


def simulate_weight_history(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    period_map: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Simulate a strategy from historical target weights.

    Turnover is measured against drifted end-of-period weights,
    not the previous target weights.
    """
    target_weights = (
        validate_weight_history(
            target_weights
        )
    )

    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )

    required_period_columns = {
        "rebalance_date",
        "period_end",
    }

    missing = (
        required_period_columns
        - set(period_map.columns)
    )

    if missing:
        raise ValueError(
            f"Missing period columns: "
            f"{sorted(missing)}"
        )

    periods = (
        period_map[
            [
                "rebalance_date",
                "period_end",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "rebalance_date"
        )
        .copy()
    )

    periods[
        "rebalance_date"
    ] = pd.to_datetime(
        periods[
            "rebalance_date"
        ]
    )

    periods[
        "period_end"
    ] = pd.to_datetime(
        periods[
            "period_end"
        ]
    )

    strategy_names = (
        target_weights[
            "model"
        ]
        .unique()
    )

    if len(
        strategy_names
    ) != 1:
        raise ValueError(
            "target_weights must contain exactly one strategy."
        )

    strategy_name = (
        strategy_names[0]
    )

    previous_drifted_weights = None

    records = []

    for row in (
        periods.itertuples(
            index=False
        )
    ):
        rebalance_date = (
            row.rebalance_date
        )

        period_end = (
            row.period_end
        )

        date_weights = (
            target_weights[
                target_weights[
                    "rebalance_date"
                ]
                == rebalance_date
            ]
            .set_index(
                "asset"
            )[
                "weight"
            ]
            .reindex(
                returns.columns
            )
            .fillna(0.0)
        )

        if date_weights.empty:
            continue

        total_weight = (
            date_weights.sum()
        )

        if not np.isclose(
            total_weight,
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Weights on {rebalance_date.date()} "
                f"sum to {total_weight:.8f}."
            )

        realized_asset_returns = (
            calculate_period_asset_returns(
                returns=returns,
                start_date=
                    rebalance_date,
                end_date=
                    period_end,
            )
        )

        gross_return = float(
            date_weights
            @ realized_asset_returns
        )

        turnover = (
            calculate_turnover(
                current_weights=
                    date_weights,
                previous_weights=
                    previous_drifted_weights,
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

        records.append(
            {
                "model":
                    strategy_name,

                "rebalance_date":
                    rebalance_date,

                "period_end":
                    period_end,

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

        previous_drifted_weights = (
            calculate_drifted_weights(
                starting_weights=
                    date_weights,
                asset_returns=
                    realized_asset_returns,
            )
        )

    return pd.DataFrame(
        records
    )
