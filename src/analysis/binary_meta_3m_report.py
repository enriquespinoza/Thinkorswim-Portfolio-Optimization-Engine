from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import (
    FEATURE_DATA_DIR,
    RAW_DATA_DIR,
)

from src.analysis.macro_meta_3m_report import (
    MARKET_FEATURES,
)

from src.backtest.walk_forward import (
    run_walk_forward,
)

from src.features.horizon_labels import (
    build_forward_horizon_labels,
)

from src.portfolio.returns import (
    build_return_matrix,
)


HORIZON_PERIODS = 3
MIN_TRAINING_LABELS = 36
RIDGE_ALPHA = 10.0
THRESHOLD = 0.0


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA
                ),
            ),
        ]
    )


def safe_corr(
    actual: pd.Series,
    predicted: pd.Series,
    method: str,
) -> float:
    if (
        actual.nunique() <= 1
        or predicted.nunique() <= 1
    ):
        return np.nan

    return float(
        actual.corr(
            predicted,
            method=method,
        )
    )


def run_binary_predictions(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purged expanding-window binary allocation gate.

    Predict:

        Maximum-Sharpe 3-month return
        minus
        Equal-Weight 3-month return

    Decision:

        prediction > 0:
            Maximum Sharpe

        prediction <= 0:
            Equal Weight
    """
    dataset = (
        dataset
        .sort_index()
        .copy()
    )

    target = (
        "maximum_sharpe"
        "_excess_vs_equal_weight"
    )

    records = []

    for current_date in dataset.index:
        training = (
            dataset[
                dataset[
                    "horizon_end"
                ]
                < current_date
            ]
        )

        if len(
            training
        ) < MIN_TRAINING_LABELS:
            continue

        current = (
            dataset.loc[
                [
                    current_date
                ]
            ]
        )

        estimator = (
            build_pipeline()
        )

        estimator.fit(
            training[
                MARKET_FEATURES
            ],
            training[
                target
            ],
        )

        predicted_excess = float(
            estimator.predict(
                current[
                    MARKET_FEATURES
                ]
            )[0]
        )

        actual_excess = float(
            current[
                target
            ].iloc[0]
        )

        if (
            predicted_excess
            > THRESHOLD
        ):
            selected_model = (
                "maximum_sharpe"
            )
        else:
            selected_model = (
                "equal_weight"
            )

        if selected_model == "maximum_sharpe":
            selected_excess = (
                actual_excess
            )
        else:
            selected_excess = (
                0.0
            )

        records.append(
            {
                "rebalance_date":
                    current_date,

                "training_labels":
                    len(
                        training
                    ),

                "predicted_excess":
                    predicted_excess,

                "actual_excess":
                    actual_excess,

                "selected_model":
                    selected_model,

                "selected_excess":
                    selected_excess,
            }
        )

    return pd.DataFrame(
        records
    )


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    stored = pd.read_csv(
        FEATURE_DATA_DIR
        / "meta_learning_dataset_with_macro.csv",
        index_col=0,
        parse_dates=True,
    )

    prices = pd.read_csv(
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = (
        build_return_matrix(
            prices
        )
        .loc[
            :"2026-09-30"
        ]
    )

    backtest = (
        run_walk_forward(
            returns=returns,
            min_train_observations=756,
            max_weight=0.40,
            risk_free_rate=0.0,
            ewma_span=126,
            transaction_cost_bps=5.0,
        )
    )

    labels = (
        build_forward_horizon_labels(
            backtest.period_results,
            horizon_periods=
                HORIZON_PERIODS,
        )
    )

    dataset = (
        stored[
            MARKET_FEATURES
        ]
        .join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    predictions = (
        run_binary_predictions(
            dataset
        )
    )

    actual = (
        predictions[
            "actual_excess"
        ]
    )

    predicted = (
        predictions[
            "predicted_excess"
        ]
    )

    active = (
        predictions[
            "selected_model"
        ]
        == "maximum_sharpe"
    )

    correct_direction = (
        np.sign(
            actual
        )
        == np.sign(
            predicted
        )
    )

    print()
    print("=" * 115)
    print(
        "BINARY 3-MONTH EW / MAXIMUM-SHARPE GATE"
    )
    print("=" * 115)

    print()
    print(
        "Observations:",
        len(
            predictions
        ),
    )

    print(
        "First prediction:",
        predictions[
            "rebalance_date"
        ].min(),
    )

    print(
        "Last prediction:",
        predictions[
            "rebalance_date"
        ].max(),
    )

    print()
    print(
        "Pearson IC:",
        round(
            safe_corr(
                actual,
                predicted,
                "pearson",
            ),
            3,
        ),
    )

    print(
        "Spearman IC:",
        round(
            safe_corr(
                actual,
                predicted,
                "spearman",
            ),
            3,
        ),
    )

    print(
        "MAE (%):",
        round(
            mean_absolute_error(
                actual,
                predicted,
            )
            * 100,
            3,
        ),
    )

    print(
        "RMSE (%):",
        round(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
            * 100,
            3,
        ),
    )

    print(
        "Directional accuracy (%):",
        round(
            correct_direction.mean()
            * 100,
            1,
        ),
    )

    print()
    print(
        "Maximum-Sharpe selections:",
        int(
            active.sum()
        ),
    )

    print(
        "Maximum-Sharpe selection rate (%):",
        round(
            active.mean()
            * 100,
            1,
        ),
    )

    if active.any():
        print(
            "Active win rate (%):",
            round(
                (
                    actual[
                        active
                    ]
                    > 0
                ).mean()
                * 100,
                1,
            ),
        )

        print(
            "Mean realized excess when active (%):",
            round(
                actual[
                    active
                ].mean()
                * 100,
                3,
            ),
        )

    print(
        "Mean selected excess (%):",
        round(
            predictions[
                "selected_excess"
            ].mean()
            * 100,
            3,
        ),
    )

    print()
    print("=" * 115)
    print(
        "PREDICTIONS"
    )
    print("=" * 115)

    display = (
        predictions.copy()
    )

    display[
        "predicted_excess"
    ] *= 100

    display[
        "actual_excess"
    ] *= 100

    display[
        "selected_excess"
    ] *= 100

    print(
        display.round(
            {
                "predicted_excess": 3,
                "actual_excess": 3,
                "selected_excess": 3,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
