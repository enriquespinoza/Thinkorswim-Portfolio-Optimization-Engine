from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_META_FEATURES = [
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

DEFAULT_CANDIDATE_MODELS = [
    "maximum_sharpe",
    "minimum_variance",
    "hrp",
    "risk_parity",
    "risk_consensus",
]


@dataclass(frozen=True)
class MetaRegressionResult:
    """
    Walk-forward meta-model results.
    """

    predictions: pd.DataFrame
    model_metrics: pd.DataFrame
    selection_metrics: pd.Series


def validate_meta_dataset(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    candidate_models: list[str],
) -> pd.DataFrame:
    """
    Validate the meta-learning dataset.
    """
    if not isinstance(
        dataset,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataset must be a pandas DataFrame."
        )

    if dataset.empty:
        raise ValueError(
            "dataset cannot be empty."
        )

    if not isinstance(
        dataset.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "dataset must use a DatetimeIndex."
        )

    required = set(
        feature_columns
    )

    required.add(
        "equal_weight_forward_return"
    )

    required.add(
        "best_model"
    )

    for model in candidate_models:
        required.add(
            f"{model}_forward_return"
        )

        required.add(
            f"{model}_excess_vs_equal_weight"
        )

    missing = (
        required
        - set(dataset.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    cleaned = (
        dataset
        .sort_index()
        .copy()
    )

    numeric_columns = (
        feature_columns
        + [
            "equal_weight_forward_return"
        ]
    )

    for model in candidate_models:
        numeric_columns.extend(
            [
                f"{model}_forward_return",
                f"{model}_excess_vs_equal_weight",
            ]
        )

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    if cleaned[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Meta-learning dataset contains missing "
            "numeric values."
        )

    if not np.isfinite(
        cleaned[
            numeric_columns
        ].to_numpy()
    ).all():
        raise ValueError(
            "Meta-learning dataset contains "
            "non-finite values."
        )

    return cleaned


def build_ridge_pipeline(
    alpha: float = 10.0,
) -> Pipeline:
    """
    Standardized ridge-regression pipeline.

    Ridge is intentionally used as the first baseline because
    the sample is small and many regime features are correlated.
    """
    if alpha < 0:
        raise ValueError(
            "alpha cannot be negative."
        )

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                ),
            ),
        ]
    )


def walk_forward_meta_regression(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
    candidate_models: list[str] | None = None,
    min_train_periods: int = 36,
    alpha: float = 10.0,
    selection_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Run expanding-window meta-model predictions.

    At each date t:

        train only on observations before t
        predict each optimizer's excess return at t
        select the model with the highest prediction

    Equal weight remains selected unless the highest predicted
    optimizer excess return exceeds selection_threshold.

    No future observations are used to train a prediction.
    """
    if feature_columns is None:
        feature_columns = (
            DEFAULT_META_FEATURES.copy()
        )

    if candidate_models is None:
        candidate_models = (
            DEFAULT_CANDIDATE_MODELS.copy()
        )

    dataset = validate_meta_dataset(
        dataset=dataset,
        feature_columns=feature_columns,
        candidate_models=candidate_models,
    )

    if min_train_periods < 12:
        raise ValueError(
            "min_train_periods must be at least 12."
        )

    if len(dataset) <= min_train_periods:
        raise ValueError(
            "Not enough observations for walk-forward "
            "meta regression."
        )

    X = dataset[
        feature_columns
    ]

    prediction_records = []

    for index in range(
        min_train_periods,
        len(dataset),
    ):
        train = dataset.iloc[
            :index
        ]

        test = dataset.iloc[
            index:index + 1
        ]

        prediction_date = (
            dataset.index[index]
        )

        predicted_excess = {}

        for model in candidate_models:
            target_column = (
                f"{model}_excess_vs_equal_weight"
            )

            pipeline = (
                build_ridge_pipeline(
                    alpha=alpha
                )
            )

            pipeline.fit(
                train[
                    feature_columns
                ],
                train[
                    target_column
                ],
            )

            prediction = float(
                pipeline.predict(
                    test[
                        feature_columns
                    ]
                )[0]
            )

            predicted_excess[
                model
            ] = prediction

        best_candidate = max(
            predicted_excess,
            key=predicted_excess.get,
        )

        best_prediction = (
            predicted_excess[
                best_candidate
            ]
        )

        if (
            best_prediction
            > selection_threshold
        ):
            selected_model = (
                best_candidate
            )
        else:
            selected_model = (
                "equal_weight"
            )

        equal_weight_return = float(
            test[
                "equal_weight_forward_return"
            ].iloc[0]
        )

        if (
            selected_model
            == "equal_weight"
        ):
            selected_return = (
                equal_weight_return
            )

            selected_realized_excess = (
                0.0
            )

        else:
            selected_return = float(
                test[
                    f"{selected_model}_forward_return"
                ].iloc[0]
            )

            selected_realized_excess = float(
                test[
                    f"{selected_model}_excess_vs_equal_weight"
                ].iloc[0]
            )

        record = {
            "rebalance_date":
                prediction_date,

            "selected_model":
                selected_model,

            "selected_predicted_excess":
                max(
                    best_prediction,
                    0.0,
                )
                if selected_model
                != "equal_weight"
                else 0.0,

            "selected_realized_excess":
                selected_realized_excess,

            "selected_forward_return":
                selected_return,

            "equal_weight_forward_return":
                equal_weight_return,

            "actual_best_model":
                test[
                    "best_model"
                ].iloc[0],
        }

        for model in candidate_models:
            record[
                f"{model}_predicted_excess"
            ] = (
                predicted_excess[
                    model
                ]
            )

            record[
                f"{model}_realized_excess"
            ] = float(
                test[
                    f"{model}_excess_vs_equal_weight"
                ].iloc[0]
            )

        prediction_records.append(
            record
        )

    predictions = pd.DataFrame(
        prediction_records
    )

    predictions = predictions.set_index(
        "rebalance_date"
    )

    return predictions


def evaluate_model_predictions(
    predictions: pd.DataFrame,
    candidate_models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate each optimizer's predicted excess returns.
    """
    if candidate_models is None:
        candidate_models = (
            DEFAULT_CANDIDATE_MODELS.copy()
        )

    records = []

    for model in candidate_models:
        predicted = predictions[
            f"{model}_predicted_excess"
        ]

        realized = predictions[
            f"{model}_realized_excess"
        ]

        error = (
            predicted
            - realized
        )

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        if (
            predicted.std(ddof=0) > 0
            and realized.std(ddof=0) > 0
        ):
            correlation = float(
                predicted.corr(
                    realized
                )
            )
        else:
            correlation = np.nan

        directional_accuracy = float(
            (
                np.sign(predicted)
                == np.sign(realized)
            ).mean()
        )

        records.append(
            {
                "model":
                    model,

                "mae":
                    mae,

                "rmse":
                    rmse,

                "correlation":
                    correlation,

                "directional_accuracy":
                    directional_accuracy,
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .set_index(
            "model"
        )
    )


def evaluate_model_selection(
    predictions: pd.DataFrame,
) -> pd.Series:
    """
    Evaluate the meta-model's model-selection decisions.

    Note
    ----
    selected_forward_return is a research proxy based on the
    underlying standalone model returns.

    A later integrated backtest must calculate true turnover
    when the meta-model switches from one optimizer to another.
    """
    selected_excess = predictions[
        "selected_realized_excess"
    ]

    selection_accuracy = float(
        (
            predictions[
                "selected_model"
            ]
            == predictions[
                "actual_best_model"
            ]
        ).mean()
    )

    positive_excess_rate = float(
        (
            selected_excess
            > 0
        ).mean()
    )

    non_benchmark = (
        predictions[
            "selected_model"
        ]
        != "equal_weight"
    )

    if non_benchmark.any():
        active_positive_rate = float(
            (
                selected_excess[
                    non_benchmark
                ]
                > 0
            ).mean()
        )
    else:
        active_positive_rate = np.nan

    average_selected_excess = float(
        selected_excess.mean()
    )

    proxy_growth = (
        1.0
        + predictions[
            "selected_forward_return"
        ]
    ).prod()

    benchmark_growth = (
        1.0
        + predictions[
            "equal_weight_forward_return"
        ]
    ).prod()

    return pd.Series(
        {
            "periods":
                len(predictions),

            "selection_accuracy":
                selection_accuracy,

            "positive_excess_rate":
                positive_excess_rate,

            "active_positive_rate":
                active_positive_rate,

            "average_selected_excess":
                average_selected_excess,

            "proxy_total_return":
                proxy_growth - 1.0,

            "equal_weight_total_return":
                benchmark_growth - 1.0,

            "proxy_excess_growth":
                (
                    proxy_growth
                    / benchmark_growth
                    - 1.0
                ),
        }
    )


def run_meta_regression(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
    candidate_models: list[str] | None = None,
    min_train_periods: int = 36,
    alpha: float = 10.0,
    selection_threshold: float = 0.0,
) -> MetaRegressionResult:
    """
    Run and evaluate the complete ridge meta-model baseline.
    """
    if candidate_models is None:
        candidate_models = (
            DEFAULT_CANDIDATE_MODELS.copy()
        )

    predictions = (
        walk_forward_meta_regression(
            dataset=dataset,
            feature_columns=
                feature_columns,
            candidate_models=
                candidate_models,
            min_train_periods=
                min_train_periods,
            alpha=alpha,
            selection_threshold=
                selection_threshold,
        )
    )

    model_metrics = (
        evaluate_model_predictions(
            predictions,
            candidate_models=
                candidate_models,
        )
    )

    selection_metrics = (
        evaluate_model_selection(
            predictions
        )
    )

    return MetaRegressionResult(
        predictions=predictions,
        model_metrics=
            model_metrics,
        selection_metrics=
            selection_metrics,
    )
