from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import FEATURE_DATA_DIR
from src.backtest.metrics import summarize_return_series


INITIAL_TRAIN_PERIODS = 36
RIDGE_ALPHA = 10.0
SELECTION_THRESHOLD = 0.0


MARKET_FEATURES = [
    "SPY_return_21d",
    "SPY_return_63d",
    "QQQ_return_63d",
    "SCHD_return_63d",
    "TLT_return_63d",
    "GLD_return_63d",
    "SPY_vol_21d",
    "TLT_vol_21d",
    "average_pairwise_corr_63d",
    "equity_bond_corr_63d",
    "cross_asset_return_dispersion_63d",
    "cross_asset_vol_dispersion_63d",
]


MACRO_CORE_V1 = [
    "curve_10y_2y_bp",
    "curve_30y_10y_bp",
    "policy_10y_ff_bp",
    "yield_10y_change_63d_bp",
    "yield_30y_change_63d_bp",
    "real_yield_10y_level",
    "breakeven_10y_change_63d_bp",
    "yield_10y_change_std_21d_bp",
    "unemployment_change_126d_pp",
    "initial_claims_change_63d_pct",
]


CANDIDATE_MODELS = [
    "maximum_sharpe",
    "minimum_variance",
    "hrp",
    "risk_parity",
    "risk_consensus",
]


FEATURE_SETS = {
    "market_only": (
        MARKET_FEATURES
    ),

    "macro_only": (
        MACRO_CORE_V1
    ),

    "market_plus_macro": (
        MARKET_FEATURES
        + MACRO_CORE_V1
    ),
}


def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    """
    Confirm that all required features and targets exist.
    """
    required_columns = set()

    for features in (
        FEATURE_SETS.values()
    ):
        required_columns.update(
            features
        )

    for model in (
        CANDIDATE_MODELS
    ):
        required_columns.add(
            f"{model}_excess_vs_equal_weight"
        )

        required_columns.add(
            f"{model}_forward_return"
        )

    required_columns.add(
        "equal_weight_forward_return"
    )

    required_columns.add(
        "best_model"
    )

    missing = (
        required_columns
        - set(
            dataset.columns
        )
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(
        dataset
    ) <= INITIAL_TRAIN_PERIODS:
        raise ValueError(
            "Dataset does not contain enough "
            "observations for expanding-window testing."
        )


def build_ridge_pipeline() -> Pipeline:
    """
    Standardized Ridge regression.

    Hyperparameters are intentionally fixed across all
    feature sets.
    """
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=
                        RIDGE_ALPHA
                ),
            ),
        ]
    )


def safe_correlation(
    actual: pd.Series,
    predicted: pd.Series,
    method: str,
) -> float:
    """
    Correlation helper that returns NaN for degenerate inputs.
    """
    if (
        actual.nunique()
        <= 1
        or predicted.nunique()
        <= 1
    ):
        return np.nan

    return float(
        actual.corr(
            predicted,
            method=method,
        )
    )


def run_expanding_predictions(
    dataset: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
) -> pd.DataFrame:
    """
    Generate true expanding-window predictions.

    At each date t:

        train = observations strictly before t
        predict = observation at t

    No future observation enters training.
    """
    records = []

    dataset = (
        dataset
        .sort_index()
        .copy()
    )

    for position in range(
        INITIAL_TRAIN_PERIODS,
        len(dataset),
    ):
        training = (
            dataset.iloc[
                :position
            ]
        )

        current = (
            dataset.iloc[
                [position]
            ]
        )

        date = (
            dataset.index[
                position
            ]
        )

        predicted_excess = {}

        for model in (
            CANDIDATE_MODELS
        ):
            target_column = (
                f"{model}"
                "_excess_vs_equal_weight"
            )

            estimator = (
                build_ridge_pipeline()
            )

            estimator.fit(
                training[
                    features
                ],
                training[
                    target_column
                ],
            )

            prediction = float(
                estimator.predict(
                    current[
                        features
                    ]
                )[0]
            )

            predicted_excess[
                model
            ] = prediction

        best_candidate = max(
            predicted_excess,
            key=
                predicted_excess.get,
        )

        best_prediction = (
            predicted_excess[
                best_candidate
            ]
        )

        if (
            best_prediction
            > SELECTION_THRESHOLD
        ):
            selected_model = (
                best_candidate
            )
        else:
            selected_model = (
                "equal_weight"
            )

        actual_best_model = (
            current[
                "best_model"
            ].iloc[0]
        )

        selected_forward_return = (
            current[
                f"{selected_model}"
                "_forward_return"
            ].iloc[0]
        )

        equal_weight_forward_return = (
            current[
                "equal_weight_forward_return"
            ].iloc[0]
        )

        record = {
            "feature_set":
                feature_set_name,

            "rebalance_date":
                date,

            "selected_model":
                selected_model,

            "actual_best_model":
                actual_best_model,

            "selected_forward_return":
                float(
                    selected_forward_return
                ),

            "equal_weight_forward_return":
                float(
                    equal_weight_forward_return
                ),

            "selected_excess_vs_equal_weight":
                float(
                    selected_forward_return
                    - equal_weight_forward_return
                ),
        }

        for model in (
            CANDIDATE_MODELS
        ):
            record[
                f"{model}_predicted_excess"
            ] = (
                predicted_excess[
                    model
                ]
            )

            record[
                f"{model}_actual_excess"
            ] = float(
                current[
                    f"{model}"
                    "_excess_vs_equal_weight"
                ].iloc[
                    0
                ]
            )

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    result[
        "rebalance_date"
    ] = pd.to_datetime(
        result[
            "rebalance_date"
        ]
    )

    return result


def build_prediction_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate each candidate model's excess-return prediction.
    """
    records = []

    for (
        feature_set,
        group,
    ) in predictions.groupby(
        "feature_set"
    ):
        for model in (
            CANDIDATE_MODELS
        ):
            actual = (
                group[
                    f"{model}"
                    "_actual_excess"
                ]
            )

            predicted = (
                group[
                    f"{model}"
                    "_predicted_excess"
                ]
            )

            directional_accuracy = (
                np.sign(
                    actual
                )
                == np.sign(
                    predicted
                )
            ).mean()

            predicted_positive = (
                predicted
                > 0
            )

            if (
                predicted_positive.any()
            ):
                realized_when_positive = (
                    actual[
                        predicted_positive
                    ].mean()
                )

                win_rate_when_positive = (
                    actual[
                        predicted_positive
                    ]
                    > 0
                ).mean()

            else:
                realized_when_positive = (
                    np.nan
                )

                win_rate_when_positive = (
                    np.nan
                )

            records.append(
                {
                    "feature_set":
                        feature_set,

                    "model":
                        model,

                    "mae":
                        mean_absolute_error(
                            actual,
                            predicted,
                        ),

                    "rmse":
                        np.sqrt(
                            mean_squared_error(
                                actual,
                                predicted,
                            )
                        ),

                    "pearson_ic":
                        safe_correlation(
                            actual,
                            predicted,
                            method=
                                "pearson",
                        ),

                    "spearman_ic":
                        safe_correlation(
                            actual,
                            predicted,
                            method=
                                "spearman",
                        ),

                    "directional_accuracy":
                        directional_accuracy,

                    "predicted_positive_count":
                        int(
                            predicted_positive.sum()
                        ),

                    "mean_realized_excess_when_positive":
                        realized_when_positive,

                    "positive_prediction_win_rate":
                        win_rate_when_positive,
                }
            )

    return pd.DataFrame(
        records
    )


def build_selector_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize strategy-selection performance for each
    information set.
    """
    records = []

    for (
        feature_set,
        group,
    ) in predictions.groupby(
        "feature_set"
    ):
        group = (
            group
            .sort_values(
                "rebalance_date"
            )
        )

        selected_returns = (
            group[
                "selected_forward_return"
            ]
        )

        equal_weight_returns = (
            group[
                "equal_weight_forward_return"
            ]
        )

        selected_metrics = (
            summarize_return_series(
                selected_returns,
                periods_per_year=12,
                risk_free_rate=0.0,
            )
        )

        equal_weight_metrics = (
            summarize_return_series(
                equal_weight_returns,
                periods_per_year=12,
                risk_free_rate=0.0,
            )
        )

        selection_accuracy = (
            group[
                "selected_model"
            ]
            == group[
                "actual_best_model"
            ]
        ).mean()

        active = (
            group[
                "selected_model"
            ]
            != "equal_weight"
        )

        positive_excess = (
            group[
                "selected_excess_vs_equal_weight"
            ]
            > 0
        )

        if active.any():
            active_win_rate = (
                positive_excess[
                    active
                ].mean()
            )

            active_mean_excess = (
                group.loc[
                    active,
                    "selected_excess_vs_equal_weight",
                ]
                .mean()
            )

        else:
            active_win_rate = (
                np.nan
            )

            active_mean_excess = (
                np.nan
            )

        records.append(
            {
                "feature_set":
                    feature_set,

                "feature_count":
                    len(
                        FEATURE_SETS[
                            feature_set
                        ]
                    ),

                "periods":
                    len(
                        group
                    ),

                "selection_accuracy":
                    selection_accuracy,

                "active_rate":
                    active.mean(),

                "active_win_rate":
                    active_win_rate,

                "active_mean_excess":
                    active_mean_excess,

                "average_selected_excess":
                    group[
                        "selected_excess_vs_equal_weight"
                    ].mean(),

                "total_return":
                    selected_metrics[
                        "total_return"
                    ],

                "annualized_return":
                    selected_metrics[
                        "annualized_return"
                    ],

                "annualized_volatility":
                    selected_metrics[
                        "annualized_volatility"
                    ],

                "sharpe":
                    selected_metrics[
                        "sharpe"
                    ],

                "sortino":
                    selected_metrics[
                        "sortino"
                    ],

                "max_drawdown":
                    selected_metrics[
                        "max_drawdown"
                    ],

                "equal_weight_total_return":
                    equal_weight_metrics[
                        "total_return"
                    ],

                "equal_weight_sharpe":
                    equal_weight_metrics[
                        "sharpe"
                    ],
            }
        )

    return pd.DataFrame(
        records
    )


def build_selection_frequency(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count model selections by information set.
    """
    records = []

    for (
        feature_set,
        group,
    ) in predictions.groupby(
        "feature_set"
    ):
        counts = Counter(
            group[
                "selected_model"
            ]
        )

        total = len(
            group
        )

        for model in [
            "equal_weight",
            *CANDIDATE_MODELS,
        ]:
            count = (
                counts.get(
                    model,
                    0,
                )
            )

            records.append(
                {
                    "feature_set":
                        feature_set,

                    "model":
                        model,

                    "count":
                        count,

                    "frequency":
                        count
                        / total,
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
        240,
    )

    path = (
        FEATURE_DATA_DIR
        / "meta_learning_dataset_with_macro.csv"
    )

    dataset = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    validate_dataset(
        dataset
    )

    prediction_frames = []

    for (
        feature_set_name,
        features,
    ) in FEATURE_SETS.items():
        result = (
            run_expanding_predictions(
                dataset=
                    dataset,

                feature_set_name=
                    feature_set_name,

                features=
                    features,
            )
        )

        prediction_frames.append(
            result
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    diagnostics = (
        build_prediction_diagnostics(
            predictions
        )
    )

    selector_summary = (
        build_selector_summary(
            predictions
        )
    )

    selection_frequency = (
        build_selection_frequency(
            predictions
        )
    )

    print()
    print("=" * 130)
    print(
        "MARKET VS MACRO META-MODEL COMPARISON"
    )
    print("=" * 130)

    print()
    print(
        "Dataset observations:",
        len(
            dataset
        ),
    )

    print(
        "Initial training periods:",
        INITIAL_TRAIN_PERIODS,
    )

    print(
        "Out-of-sample predictions:",
        len(
            dataset
        )
        - INITIAL_TRAIN_PERIODS,
    )

    print(
        "Ridge alpha:",
        RIDGE_ALPHA,
    )

    print(
        "Selection threshold:",
        SELECTION_THRESHOLD,
    )

    print()
    print(
        "FEATURE SETS"
    )
    print("-" * 130)

    for (
        name,
        features,
    ) in FEATURE_SETS.items():
        print(
            f"{name:<20} "
            f"{len(features):>3} features"
        )

    diagnostic_display = (
        diagnostics.copy()
    )

    for column in [
        "mae",
        "rmse",
        "mean_realized_excess_when_positive",
    ]:
        diagnostic_display[
            column
        ] *= 100

    for column in [
        "directional_accuracy",
        "positive_prediction_win_rate",
    ]:
        diagnostic_display[
            column
        ] *= 100

    print()
    print("=" * 130)
    print(
        "MODEL-LEVEL EXCESS-RETURN PREDICTION"
    )
    print("=" * 130)

    print(
        diagnostic_display[
            [
                "feature_set",
                "model",
                "mae",
                "rmse",
                "pearson_ic",
                "spearman_ic",
                "directional_accuracy",
                "predicted_positive_count",
                "mean_realized_excess_when_positive",
                "positive_prediction_win_rate",
            ]
        ]
        .round(
            {
                "mae": 3,
                "rmse": 3,
                "pearson_ic": 3,
                "spearman_ic": 3,
                "directional_accuracy": 1,
                "mean_realized_excess_when_positive": 3,
                "positive_prediction_win_rate": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    selector_display = (
        selector_summary.copy()
    )

    for column in [
        "selection_accuracy",
        "active_rate",
        "active_win_rate",
    ]:
        selector_display[
            column
        ] *= 100

    for column in [
        "active_mean_excess",
        "average_selected_excess",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "equal_weight_total_return",
    ]:
        selector_display[
            column
        ] *= 100

    print()
    print("=" * 130)
    print(
        "META-SELECTOR PERFORMANCE"
    )
    print("=" * 130)

    print(
        selector_display[
            [
                "feature_set",
                "feature_count",
                "periods",
                "selection_accuracy",
                "active_rate",
                "active_win_rate",
                "active_mean_excess",
                "average_selected_excess",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "sortino",
                "max_drawdown",
                "equal_weight_total_return",
                "equal_weight_sharpe",
            ]
        ]
        .round(
            {
                "selection_accuracy": 1,
                "active_rate": 1,
                "active_win_rate": 1,
                "active_mean_excess": 3,
                "average_selected_excess": 3,
                "total_return": 2,
                "annualized_return": 2,
                "annualized_volatility": 2,
                "sharpe": 3,
                "sortino": 3,
                "max_drawdown": 2,
                "equal_weight_total_return": 2,
                "equal_weight_sharpe": 3,
            }
        )
        .sort_values(
            "sharpe",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    frequency_display = (
        selection_frequency.copy()
    )

    frequency_display[
        "frequency"
    ] *= 100

    print()
    print("=" * 130)
    print(
        "MODEL SELECTION FREQUENCY"
    )
    print("=" * 130)

    print(
        frequency_display
        .round(
            {
                "frequency": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print(
        "MAXIMUM-SHARPE SIGNAL COMPARISON"
    )
    print("=" * 130)

    max_sharpe = (
        diagnostic_display[
            diagnostic_display[
                "model"
            ]
            == "maximum_sharpe"
        ]
        [
            [
                "feature_set",
                "pearson_ic",
                "spearman_ic",
                "directional_accuracy",
                "predicted_positive_count",
                "mean_realized_excess_when_positive",
                "positive_prediction_win_rate",
            ]
        ]
    )

    print(
        max_sharpe
        .round(
            {
                "pearson_ic": 3,
                "spearman_ic": 3,
                "directional_accuracy": 1,
                "mean_realized_excess_when_positive": 3,
                "positive_prediction_win_rate": 1,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
