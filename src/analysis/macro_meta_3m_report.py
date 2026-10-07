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


FEATURE_SETS = {
    "market_only":
        MARKET_FEATURES,

    "macro_only":
        MACRO_CORE_V1,

    "market_plus_macro":
        MARKET_FEATURES
        + MACRO_CORE_V1,
}


CANDIDATE_MODELS = [
    "maximum_sharpe",
    "minimum_variance",
    "hrp",
    "risk_parity",
    "risk_consensus",
]


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
                    alpha=
                        RIDGE_ALPHA
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


def run_purged_predictions(
    dataset: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
) -> pd.DataFrame:
    """
    Expanding-window prediction with target purging.

    A historical row can enter training only when its entire
    forward horizon ended strictly before the current
    rebalance date.
    """
    dataset = (
        dataset
        .sort_index()
        .copy()
    )

    records = []

    for current_date in (
        dataset.index
    ):
        completed_training = (
            dataset[
                dataset[
                    "horizon_end"
                ]
                < current_date
            ]
        )

        if len(
            completed_training
        ) < MIN_TRAINING_LABELS:
            continue

        current = (
            dataset.loc[
                [
                    current_date
                ]
            ]
        )

        predictions = {}

        for model in (
            CANDIDATE_MODELS
        ):
            target = (
                f"{model}"
                "_excess_vs_equal_weight"
            )

            estimator = (
                build_pipeline()
            )

            estimator.fit(
                completed_training[
                    features
                ],
                completed_training[
                    target
                ],
            )

            predictions[
                model
            ] = float(
                estimator.predict(
                    current[
                        features
                    ]
                )[0]
            )

        candidate = max(
            predictions,
            key=
                predictions.get,
        )

        if (
            predictions[
                candidate
            ]
            > SELECTION_THRESHOLD
        ):
            selected_model = (
                candidate
            )
        else:
            selected_model = (
                "equal_weight"
            )

        record = {
            "feature_set":
                feature_set_name,

            "rebalance_date":
                current_date,

            "training_labels":
                len(
                    completed_training
                ),

            "selected_model":
                selected_model,

            "actual_best_model":
                current[
                    "best_model"
                ].iloc[
                    0
                ],

            "selected_excess":
                float(
                    current[
                        f"{selected_model}"
                        "_excess_vs_equal_weight"
                    ].iloc[
                        0
                    ]
                ),
        }

        for model in (
            CANDIDATE_MODELS
        ):
            record[
                f"{model}_predicted"
            ] = predictions[
                model
            ]

            record[
                f"{model}_actual"
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

    return pd.DataFrame(
        records
    )


def build_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
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
                    f"{model}_actual"
                ]
            )

            predicted = (
                group[
                    f"{model}_predicted"
                ]
            )

            positive = (
                predicted > 0
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
                        safe_corr(
                            actual,
                            predicted,
                            "pearson",
                        ),

                    "spearman_ic":
                        safe_corr(
                            actual,
                            predicted,
                            "spearman",
                        ),

                    "directional_accuracy":
                        (
                            np.sign(
                                actual
                            )
                            == np.sign(
                                predicted
                            )
                        ).mean(),

                    "predicted_positive_count":
                        int(
                            positive.sum()
                        ),

                    "mean_realized_excess_when_positive":
                        (
                            actual[
                                positive
                            ].mean()
                            if positive.any()
                            else np.nan
                        ),

                    "positive_prediction_win_rate":
                        (
                            (
                                actual[
                                    positive
                                ]
                                > 0
                            ).mean()
                            if positive.any()
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(
        records
    )


def build_selector_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (
        feature_set,
        group,
    ) in predictions.groupby(
        "feature_set"
    ):
        active = (
            group[
                "selected_model"
            ]
            != "equal_weight"
        )

        records.append(
            {
                "feature_set":
                    feature_set,

                "features":
                    len(
                        FEATURE_SETS[
                            feature_set
                        ]
                    ),

                "oos_predictions":
                    len(
                        group
                    ),

                "first_prediction":
                    group[
                        "rebalance_date"
                    ].min(),

                "last_prediction":
                    group[
                        "rebalance_date"
                    ].max(),

                "selection_accuracy":
                    (
                        group[
                            "selected_model"
                        ]
                        == group[
                            "actual_best_model"
                        ]
                    ).mean(),

                "active_rate":
                    active.mean(),

                "active_win_rate":
                    (
                        (
                            group.loc[
                                active,
                                "selected_excess",
                            ]
                            > 0
                        ).mean()
                        if active.any()
                        else np.nan
                    ),

                "mean_active_excess":
                    (
                        group.loc[
                            active,
                            "selected_excess",
                        ].mean()
                        if active.any()
                        else np.nan
                    ),

                "mean_selected_excess":
                    group[
                        "selected_excess"
                    ].mean(),

                "median_selected_excess":
                    group[
                        "selected_excess"
                    ].median(),
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

    features = pd.read_csv(
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

    # Strip the original one-month label columns from the
    # stored dataset. Retain only observable features.
    feature_columns = [
        column
        for column
        in features.columns
        if (
            column
            in set(
                MARKET_FEATURES
                + MACRO_CORE_V1
            )
        )
    ]

    dataset = (
        features[
            feature_columns
        ]
        .join(
            labels,
            how="inner",
        )
        .sort_index()
    )

    prediction_frames = []

    for (
        feature_set_name,
        selected_features,
    ) in FEATURE_SETS.items():

        prediction_frames.append(
            run_purged_predictions(
                dataset=
                    dataset,

                feature_set_name=
                    feature_set_name,

                features=
                    selected_features,
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    diagnostics = (
        build_diagnostics(
            predictions
        )
    )

    selector = (
        build_selector_summary(
            predictions
        )
    )

    display_diagnostics = (
        diagnostics.copy()
    )

    for column in [
        "mae",
        "rmse",
        "mean_realized_excess_when_positive",
    ]:
        display_diagnostics[
            column
        ] *= 100

    for column in [
        "directional_accuracy",
        "positive_prediction_win_rate",
    ]:
        display_diagnostics[
            column
        ] *= 100

    display_selector = (
        selector.copy()
    )

    for column in [
        "selection_accuracy",
        "active_rate",
        "active_win_rate",
    ]:
        display_selector[
            column
        ] *= 100

    for column in [
        "mean_active_excess",
        "mean_selected_excess",
        "median_selected_excess",
    ]:
        display_selector[
            column
        ] *= 100

    print()
    print("=" * 130)
    print(
        "PURGED 3-MONTH META-MODEL COMPARISON"
    )
    print("=" * 130)

    print()
    print(
        "Forward horizon:",
        HORIZON_PERIODS,
        "months",
    )

    print(
        "Minimum completed training labels:",
        MIN_TRAINING_LABELS,
    )

    print(
        "Ridge alpha:",
        RIDGE_ALPHA,
    )

    print(
        "3-month label rows:",
        len(
            dataset
        ),
    )

    print()
    print(
        "IMPORTANT: overlapping 3-month outcomes "
        "are used for signal diagnostics only."
    )

    print(
        "No total-return or Sharpe statistic is "
        "calculated from overlapping labels."
    )

    print()
    print("=" * 130)
    print(
        "MODEL-LEVEL 3-MONTH EXCESS-RETURN PREDICTION"
    )
    print("=" * 130)

    print(
        display_diagnostics.round(
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

    print()
    print("=" * 130)
    print(
        "3-MONTH SELECTOR DIAGNOSTICS"
    )
    print("=" * 130)

    print(
        display_selector.round(
            {
                "selection_accuracy": 1,
                "active_rate": 1,
                "active_win_rate": 1,
                "mean_active_excess": 3,
                "mean_selected_excess": 3,
                "median_selected_excess": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print(
        "MAXIMUM-SHARPE 3-MONTH SIGNAL"
    )
    print("=" * 130)

    print(
        display_diagnostics[
            display_diagnostics[
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
