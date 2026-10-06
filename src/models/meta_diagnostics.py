from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    summarize_return_series,
)


def prediction_information_coefficients(
    predictions: pd.DataFrame,
    candidate_models: list[str],
) -> pd.DataFrame:
    """
    Measure association between predicted and realized
    excess returns.

    Pearson:
        Linear relationship.

    Spearman:
        Rank relationship.
    """
    records = []

    for model in candidate_models:
        predicted = predictions[
            f"{model}_predicted_excess"
        ]

        realized = predictions[
            f"{model}_realized_excess"
        ]

        records.append(
            {
                "model":
                    model,

                "pearson_ic":
                    predicted.corr(
                        realized,
                        method="pearson",
                    ),

                "spearman_ic":
                    predicted.corr(
                        realized,
                        method="spearman",
                    ),
            }
        )

    return (
        pd.DataFrame(records)
        .set_index("model")
    )


def prediction_calibration_table(
    predictions: pd.DataFrame,
    candidate_models: list[str],
    bins: int = 4,
) -> pd.DataFrame:
    """
    Sort predictions into quantile buckets and examine
    realized performance.

    A useful forecasting model should generally show higher
    realized excess returns in higher predicted-return buckets.
    """
    if bins < 2:
        raise ValueError(
            "bins must be at least 2."
        )

    records = []

    for model in candidate_models:
        predicted = predictions[
            f"{model}_predicted_excess"
        ]

        realized = predictions[
            f"{model}_realized_excess"
        ]

        # Rank first so qcut remains stable if predictions
        # contain duplicate values.
        ranked = predicted.rank(
            method="first"
        )

        bucket = pd.qcut(
            ranked,
            q=bins,
            labels=False,
        ) + 1

        frame = pd.DataFrame(
            {
                "predicted":
                    predicted,

                "realized":
                    realized,

                "bucket":
                    bucket,
            }
        )

        for (
            bucket_number,
            group,
        ) in frame.groupby(
            "bucket"
        ):
            records.append(
                {
                    "model":
                        model,

                    "bucket":
                        int(
                            bucket_number
                        ),

                    "observations":
                        len(group),

                    "mean_predicted_excess":
                        group[
                            "predicted"
                        ].mean(),

                    "mean_realized_excess":
                        group[
                            "realized"
                        ].mean(),

                    "positive_realized_rate":
                        (
                            group[
                                "realized"
                            ]
                            > 0
                        ).mean(),
                }
            )

    return pd.DataFrame(
        records
    )


def sign_diagnostics(
    predictions: pd.DataFrame,
    candidate_models: list[str],
) -> pd.DataFrame:
    """
    Examine realized performance conditional on the model
    predicting positive or negative excess return.
    """
    records = []

    for model in candidate_models:
        predicted = predictions[
            f"{model}_predicted_excess"
        ]

        realized = predictions[
            f"{model}_realized_excess"
        ]

        positive_signal = (
            predicted > 0
        )

        negative_signal = (
            ~positive_signal
        )

        for (
            signal_name,
            mask,
        ) in [
            (
                "predicted_positive",
                positive_signal,
            ),
            (
                "predicted_nonpositive",
                negative_signal,
            ),
        ]:
            subset = realized[
                mask
            ]

            if subset.empty:
                continue

            records.append(
                {
                    "model":
                        model,

                    "signal":
                        signal_name,

                    "observations":
                        len(subset),

                    "mean_realized_excess":
                        subset.mean(),

                    "median_realized_excess":
                        subset.median(),

                    "positive_realized_rate":
                        (
                            subset > 0
                        ).mean(),
                }
            )

    return pd.DataFrame(
        records
    )


def strategy_benchmark_table(
    predictions: pd.DataFrame,
    candidate_models: list[str],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Compare the meta-selector with static model choices over
    exactly the same out-of-sample prediction window.

    The meta-selector remains a proxy because switching costs
    between different optimizer portfolios are not yet modeled.
    """
    equal_returns = predictions[
        "equal_weight_forward_return"
    ]

    strategies: dict[
        str,
        pd.Series,
    ] = {
        "equal_weight":
            equal_returns,

        "meta_selector_proxy":
            predictions[
                "selected_forward_return"
            ],
    }

    model_forward_returns = {}

    for model in candidate_models:
        model_returns = (
            equal_returns
            + predictions[
                f"{model}_realized_excess"
            ]
        )

        model_forward_returns[
            model
        ] = model_returns

        strategies[
            model
        ] = model_returns

    comparison = pd.DataFrame(
        {
            "equal_weight":
                equal_returns,

            **model_forward_returns,
        }
    )

    strategies[
        "oracle_best"
    ] = comparison.max(
        axis=1
    )

    records = []

    for (
        name,
        returns,
    ) in strategies.items():

        metrics = (
            summarize_return_series(
                returns,
                periods_per_year=
                    periods_per_year,
                risk_free_rate=0.0,
            )
        )

        records.append(
            {
                "strategy":
                    name,

                **metrics,
            }
        )

    return (
        pd.DataFrame(records)
        .set_index("strategy")
        .sort_values(
            "sharpe",
            ascending=False,
        )
    )


def selection_confusion_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare selected model with the model that actually
    performed best during the next period.
    """
    return pd.crosstab(
        predictions[
            "actual_best_model"
        ],
        predictions[
            "selected_model"
        ],
        rownames=[
            "actual_best"
        ],
        colnames=[
            "selected"
        ],
        margins=True,
    )
