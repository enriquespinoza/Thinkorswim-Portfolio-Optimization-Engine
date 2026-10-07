from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    FEATURE_DATA_DIR,
    GENERATED_REPORTS_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
)

from src.analysis.binary_meta_3m_report import (
    HORIZON_PERIODS,
    RIDGE_ALPHA,
    THRESHOLD,
    build_pipeline,
    run_binary_predictions,
)

from src.analysis.macro_meta_3m_report import (
    MARKET_FEATURES,
)

from src.analysis.quarterly_meta_backtest import (
    build_model_assignments,
    build_selector_weight_history,
    build_static_benchmark_weights,
)

from src.analysis.quarterly_meta_robustness import (
    build_phase_decisions,
)

from src.backtest.metrics import (
    summarize_backtest,
)

from src.backtest.portfolio_blend import (
    simulate_weight_history,
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


VERSION = "1.0.0"

TRANSACTION_COST_BPS = 5.0

BOOTSTRAP_SAMPLES = 10000

RANDOM_SEED = 42

PHASES = [
    0,
    1,
    2,
]

TARGET_COLUMN = (
    "maximum_sharpe_excess_vs_equal_weight"
)

ARTIFACT_DIR = (
    MODELS_DIR
    / "meta_allocation_v1"
)

REPORT_OUTPUT_DIR = (
    GENERATED_REPORTS_DIR
    / "meta_allocation_v1"
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def compound_returns(
    values: pd.Series,
) -> float:
    array = (
        values
        .astype(float)
        .to_numpy()
    )

    return float(
        np.prod(
            1.0 + array
        )
        - 1.0
    )


def annualized_return_quarterly(
    values: np.ndarray,
) -> float:
    n = len(
        values
    )

    if n == 0:
        return np.nan

    wealth = np.prod(
        1.0 + values
    )

    if wealth <= 0:
        return np.nan

    return float(
        wealth ** (
            4.0 / n
        )
        - 1.0
    )


def sharpe_quarterly(
    values: np.ndarray,
) -> float:
    if len(
        values
    ) < 2:
        return np.nan

    volatility = np.std(
        values,
        ddof=1,
    )

    if (
        not np.isfinite(
            volatility
        )
        or volatility == 0
    ):
        return np.nan

    return float(
        np.mean(
            values
        )
        / volatility
        * np.sqrt(
            4.0
        )
    )


def build_phase_portfolio_returns(
    returns: pd.DataFrame,
    backtest,
    binary_predictions: pd.DataFrame,
    phase: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build actual monthly and compounded quarterly portfolio
    returns for:

        Meta Allocation V1
        Equal Weight
        Maximum Sharpe

    All returns include the research transaction-cost
    assumption.
    """
    decisions = (
        build_phase_decisions(
            binary_predictions,
            phase_offset=
                phase,
        )
    )

    all_rebalance_dates = (
        backtest.period_results[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    assignments = (
        build_model_assignments(
            decisions=
                decisions,

            rebalance_dates=
                pd.DatetimeIndex(
                    all_rebalance_dates
                ),

            months_per_decision=
                HORIZON_PERIODS,
        )
    )

    valid_decision_dates = (
        assignments[
            "decision_date"
        ]
        .drop_duplicates()
    )

    decisions = (
        decisions[
            decisions[
                "rebalance_date"
            ].isin(
                valid_decision_dates
            )
        ]
        .copy()
    )

    selector_weights = (
        build_selector_weight_history(
            assignments=
                assignments,

            model_weights=
                backtest.weights,
        )
    )

    selector_weights = (
        selector_weights.copy()
    )

    selector_weights[
        "model"
    ] = (
        "meta_allocation_v1"
    )

    evaluation_dates = (
        assignments[
            "rebalance_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    period_map = (
        backtest.period_results[
            [
                "rebalance_date",
                "period_end",
            ]
        ]
        .drop_duplicates()
    )

    period_map = (
        period_map[
            period_map[
                "rebalance_date"
            ].isin(
                evaluation_dates
            )
        ]
        .copy()
    )

    selector_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                selector_weights,
            period_map=
                period_map,
            transaction_cost_bps=
                TRANSACTION_COST_BPS,
        )
    )

    equal_weight_weights = (
        build_static_benchmark_weights(
            model_weights=
                backtest.weights,

            model=
                "equal_weight",

            rebalance_dates=
                evaluation_dates,
        )
    )

    maximum_sharpe_weights = (
        build_static_benchmark_weights(
            model_weights=
                backtest.weights,

            model=
                "maximum_sharpe",

            rebalance_dates=
                evaluation_dates,
        )
    )

    equal_weight_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                equal_weight_weights,
            period_map=
                period_map,
            transaction_cost_bps=
                TRANSACTION_COST_BPS,
        )
    )

    maximum_sharpe_results = (
        simulate_weight_history(
            returns=returns,
            target_weights=
                maximum_sharpe_weights,
            period_map=
                period_map,
            transaction_cost_bps=
                TRANSACTION_COST_BPS,
        )
    )

    required_columns = {
        "rebalance_date",
        "net_return",
    }

    for (
        name,
        result,
    ) in {
        "selector":
            selector_results,

        "equal_weight":
            equal_weight_results,

        "maximum_sharpe":
            maximum_sharpe_results,
    }.items():

        missing = (
            required_columns
            - set(
                result.columns
            )
        )

        if missing:
            raise ValueError(
                f"{name} simulation missing "
                f"columns: {sorted(missing)}"
            )

    monthly = (
        assignments[
            [
                "decision_date",
                "rebalance_date",
            ]
        ]
        .copy()
    )

    monthly = (
        monthly.merge(
            selector_results[
                [
                    "rebalance_date",
                    "net_return",
                ]
            ].rename(
                columns={
                    "net_return":
                        "meta_allocation_v1"
                }
            ),
            on="rebalance_date",
            how="left",
        )
    )

    monthly = (
        monthly.merge(
            equal_weight_results[
                [
                    "rebalance_date",
                    "net_return",
                ]
            ].rename(
                columns={
                    "net_return":
                        "equal_weight"
                }
            ),
            on="rebalance_date",
            how="left",
        )
    )

    monthly = (
        monthly.merge(
            maximum_sharpe_results[
                [
                    "rebalance_date",
                    "net_return",
                ]
            ].rename(
                columns={
                    "net_return":
                        "maximum_sharpe"
                }
            ),
            on="rebalance_date",
            how="left",
        )
    )

    if monthly[
        [
            "meta_allocation_v1",
            "equal_weight",
            "maximum_sharpe",
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing simulated monthly returns."
        )

    quarterly = (
        monthly
        .groupby(
            "decision_date"
        )
        .agg(
            {
                "meta_allocation_v1":
                    compound_returns,

                "equal_weight":
                    compound_returns,

                "maximum_sharpe":
                    compound_returns,
            }
        )
        .reset_index()
    )

    quarterly[
        "phase"
    ] = phase

    quarterly[
        "excess_vs_equal_weight"
    ] = (
        quarterly[
            "meta_allocation_v1"
        ]
        - quarterly[
            "equal_weight"
        ]
    )

    quarterly[
        "excess_vs_maximum_sharpe"
    ] = (
        quarterly[
            "meta_allocation_v1"
        ]
        - quarterly[
            "maximum_sharpe"
        ]
    )

    return (
        quarterly,
        pd.concat(
            [
                selector_results,
                equal_weight_results,
                maximum_sharpe_results,
            ],
            ignore_index=True,
        ),
    )


def contribution_concentration(
    excess: np.ndarray,
) -> float:
    """
    Fraction of total additive excess explained by the
    three largest winning quarters.

    This can exceed 100% when losing quarters offset part
    of the positive contribution.
    """
    excess = np.asarray(
        excess,
        dtype=float,
    )

    total_excess = (
        excess.sum()
    )

    if total_excess <= 0:
        return np.nan

    winners = (
        excess[
            excess > 0
        ]
    )

    if len(
        winners
    ) == 0:
        return np.nan

    top_three = (
        np.sort(
            winners
        )[
            -3:
        ]
        .sum()
    )

    return float(
        top_three
        / total_excess
    )


def bootstrap_phase(
    quarterly: pd.DataFrame,
    phase: int,
) -> dict:
    """
    Paired bootstrap of complete non-overlapping quarterly
    portfolio blocks.
    """
    selector = (
        quarterly[
            "meta_allocation_v1"
        ]
        .to_numpy(
            dtype=float
        )
    )

    equal_weight = (
        quarterly[
            "equal_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    maximum_sharpe = (
        quarterly[
            "maximum_sharpe"
        ]
        .to_numpy(
            dtype=float
        )
    )

    n = len(
        selector
    )

    if n < 2:
        raise ValueError(
            "Not enough quarterly blocks "
            "for bootstrap validation."
        )

    rng = (
        np.random.default_rng(
            RANDOM_SEED
            + phase
        )
    )

    sample_indices = (
        rng.integers(
            0,
            n,
            size=(
                BOOTSTRAP_SAMPLES,
                n,
            ),
        )
    )

    selector_samples = (
        selector[
            sample_indices
        ]
    )

    ew_samples = (
        equal_weight[
            sample_indices
        ]
    )

    ms_samples = (
        maximum_sharpe[
            sample_indices
        ]
    )

    selector_wealth = (
        np.prod(
            1.0
            + selector_samples,
            axis=1,
        )
    )

    ew_wealth = (
        np.prod(
            1.0
            + ew_samples,
            axis=1,
        )
    )

    ms_wealth = (
        np.prod(
            1.0
            + ms_samples,
            axis=1,
        )
    )

    annualization_power = (
        4.0 / n
    )

    selector_ann_return = (
        selector_wealth
        ** annualization_power
        - 1.0
    )

    ew_ann_return = (
        ew_wealth
        ** annualization_power
        - 1.0
    )

    ms_ann_return = (
        ms_wealth
        ** annualization_power
        - 1.0
    )

    def bootstrap_sharpe(
        sample_matrix: np.ndarray,
    ) -> np.ndarray:

        means = (
            sample_matrix.mean(
                axis=1
            )
        )

        stds = (
            sample_matrix.std(
                axis=1,
                ddof=1,
            )
        )

        result = np.full(
            len(
                means
            ),
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(
                stds
            )
            & (
                stds > 0
            )
        )

        result[
            valid
        ] = (
            means[
                valid
            ]
            / stds[
                valid
            ]
            * np.sqrt(
                4.0
            )
        )

        return result

    selector_sharpe = (
        bootstrap_sharpe(
            selector_samples
        )
    )

    ew_sharpe = (
        bootstrap_sharpe(
            ew_samples
        )
    )

    ms_sharpe = (
        bootstrap_sharpe(
            ms_samples
        )
    )

    ew_excess_samples = (
        selector_samples
        - ew_samples
    )

    ms_excess_samples = (
        selector_samples
        - ms_samples
    )

    mean_ew_excess = (
        ew_excess_samples.mean(
            axis=1
        )
    )

    mean_ms_excess = (
        ms_excess_samples.mean(
            axis=1
        )
    )

    actual_ew_excess = (
        selector
        - equal_weight
    )

    actual_ms_excess = (
        selector
        - maximum_sharpe
    )

    return {
        "phase":
            phase,

        "quarterly_blocks":
            n,

        "selector_annualized_return":
            annualized_return_quarterly(
                selector
            ),

        "equal_weight_annualized_return":
            annualized_return_quarterly(
                equal_weight
            ),

        "maximum_sharpe_annualized_return":
            annualized_return_quarterly(
                maximum_sharpe
            ),

        "selector_quarterly_sharpe":
            sharpe_quarterly(
                selector
            ),

        "equal_weight_quarterly_sharpe":
            sharpe_quarterly(
                equal_weight
            ),

        "maximum_sharpe_quarterly_sharpe":
            sharpe_quarterly(
                maximum_sharpe
            ),

        "mean_quarterly_excess_vs_ew":
            float(
                actual_ew_excess.mean()
            ),

        "mean_quarterly_excess_vs_ms":
            float(
                actual_ms_excess.mean()
            ),

        "p_return_gt_equal_weight":
            float(
                (
                    selector_ann_return
                    > ew_ann_return
                ).mean()
            ),

        "p_return_gt_maximum_sharpe":
            float(
                (
                    selector_ann_return
                    > ms_ann_return
                ).mean()
            ),

        "p_sharpe_gt_equal_weight":
            float(
                np.nanmean(
                    selector_sharpe
                    > ew_sharpe
                )
            ),

        "p_sharpe_gt_maximum_sharpe":
            float(
                np.nanmean(
                    selector_sharpe
                    > ms_sharpe
                )
            ),

        "ew_excess_ci_low":
            float(
                np.quantile(
                    mean_ew_excess,
                    0.025,
                )
            ),

        "ew_excess_ci_high":
            float(
                np.quantile(
                    mean_ew_excess,
                    0.975,
                )
            ),

        "ms_excess_ci_low":
            float(
                np.quantile(
                    mean_ms_excess,
                    0.025,
                )
            ),

        "ms_excess_ci_high":
            float(
                np.quantile(
                    mean_ms_excess,
                    0.975,
                )
            ),

        "top3_contribution_vs_ew":
            contribution_concentration(
                actual_ew_excess
            ),

        "top3_contribution_vs_ms":
            contribution_concentration(
                actual_ms_excess
            ),
    }


def get_git_info() -> dict:
    try:
        commit = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip()
        )

        status = (
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                ],
                text=True,
            )
        )

        return {
            "commit":
                commit,

            "working_tree_dirty":
                bool(
                    status.strip()
                ),
        }

    except Exception:
        return {
            "commit":
                None,

            "working_tree_dirty":
                None,
        }


def write_frozen_artifact(
    dataset: pd.DataFrame,
    source_dataset_path: Path,
    validation: pd.DataFrame,
    quarterly_returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    """
    Train the final historical V1 model and create the
    immutable research artifact.
    """
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = (
        dataset[
            dataset[
                "horizon_end"
            ]
            <= as_of_date
        ]
        .copy()
    )

    if completed.empty:
        raise ValueError(
            "No completed training labels "
            "available for frozen model."
        )

    model = (
        build_pipeline()
    )

    model.fit(
        completed[
            MARKET_FEATURES
        ],
        completed[
            TARGET_COLUMN
        ],
    )

    model_path = (
        ARTIFACT_DIR
        / "meta_allocation_v1.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    coefficients = pd.DataFrame(
        {
            "feature":
                MARKET_FEATURES,

            "standardized_ridge_coefficient":
                model.named_steps[
                    "ridge"
                ].coef_,
        }
    )

    coefficients.to_csv(
        ARTIFACT_DIR
        / "coefficients.csv",
        index=False,
    )

    validation.to_csv(
        ARTIFACT_DIR
        / "research_results.csv",
        index=False,
    )

    quarterly_returns.to_csv(
        ARTIFACT_DIR
        / "quarterly_returns.csv",
        index=False,
    )

    validation.to_csv(
        REPORT_OUTPUT_DIR
        / "final_bootstrap_validation.csv",
        index=False,
    )

    quarterly_returns.to_csv(
        REPORT_OUTPUT_DIR
        / "final_quarterly_returns.csv",
        index=False,
    )

    frozen_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    git_info = (
        get_git_info()
    )

    model_spec = {
        "artifact_name":
            "Meta Allocation V1",

        "version":
            VERSION,

        "status":
            "FROZEN_RESEARCH_CANDIDATE",

        "intended_use":
            "forward_test_and_paper_money_only",

        "frozen_at_utc":
            frozen_at,

        "historical_development_closed":
            True,

        "model":
            "Ridge",

        "ridge_alpha":
            RIDGE_ALPHA,

        "scaler":
            "StandardScaler",

        "target":
            TARGET_COLUMN,

        "forecast_horizon_months":
            HORIZON_PERIODS,

        "decision_threshold":
            THRESHOLD,

        "positive_state":
            "maximum_sharpe",

        "negative_state":
            "equal_weight",

        "maximum_asset_weight":
            0.40,

        "research_transaction_cost_bps":
            TRANSACTION_COST_BPS,

        "training_rows":
            int(
                len(
                    completed
                )
            ),

        "training_feature_date_start":
            str(
                completed.index.min().date()
            ),

        "training_feature_date_end":
            str(
                completed.index.max().date()
            ),

        "latest_completed_horizon_end":
            str(
                pd.to_datetime(
                    completed[
                        "horizon_end"
                    ]
                )
                .max()
                .date()
            ),

        "information_as_of":
            str(
                as_of_date.date()
            ),

        "source_git_commit":
            git_info[
                "commit"
            ],

        "source_working_tree_dirty":
            git_info[
                "working_tree_dirty"
            ],
    }

    with (
        ARTIFACT_DIR
        / "model_spec.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            model_spec,
            handle,
            indent=2,
        )

    feature_spec = {
        "version":
            VERSION,

        "feature_count":
            len(
                MARKET_FEATURES
            ),

        "features":
            MARKET_FEATURES,

        "feature_policy":
            (
                "Frozen. Historical feature changes "
                "require a new model version."
            ),
    }

    with (
        ARTIFACT_DIR
        / "feature_list.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            feature_spec,
            handle,
            indent=2,
        )

    average_values = (
        validation[
            [
                "selector_annualized_return",
                "equal_weight_annualized_return",
                "maximum_sharpe_annualized_return",
                "p_return_gt_equal_weight",
                "p_return_gt_maximum_sharpe",
                "p_sharpe_gt_equal_weight",
                "p_sharpe_gt_maximum_sharpe",
            ]
        ]
        .mean()
    )

    validation_text = f"""# Meta Allocation V1 Validation Summary

## Status

**FROZEN RESEARCH CANDIDATE**

Historical model development ended when this artifact was created.

No feature, target, threshold, alpha, optimizer, horizon, or historical
decision-rule changes should be made to V1 after this point.

Any such modification must become a separately named V2 research model.

## Model

- Model: Ridge regression
- Alpha: {RIDGE_ALPHA}
- Features: {len(MARKET_FEATURES)}
- Forecast horizon: {HORIZON_PERIODS} months
- Decision threshold: {THRESHOLD}
- Positive state: Maximum Sharpe
- Negative state: Equal Weight
- Research transaction cost: {TRANSACTION_COST_BPS} bps
- Historical training rows at freeze: {len(completed)}
- Information as of: {as_of_date.date()}

## Final Paired Portfolio Bootstrap

The final validation resamples complete, non-overlapping three-month
portfolio blocks.

It uses realized portfolio returns after the configured transaction-cost
assumption rather than overlapping prediction labels.

{validation.to_string(index=False)}

## Across-Phase Mean Results

Selector annualized return:
{average_values["selector_annualized_return"]:.4%}

Equal Weight annualized return:
{average_values["equal_weight_annualized_return"]:.4%}

Maximum Sharpe annualized return:
{average_values["maximum_sharpe_annualized_return"]:.4%}

Mean bootstrap probability selector return > Equal Weight:
{average_values["p_return_gt_equal_weight"]:.2%}

Mean bootstrap probability selector return > Maximum Sharpe:
{average_values["p_return_gt_maximum_sharpe"]:.2%}

Mean bootstrap probability selector Sharpe > Equal Weight:
{average_values["p_sharpe_gt_equal_weight"]:.2%}

Mean bootstrap probability selector Sharpe > Maximum Sharpe:
{average_values["p_sharpe_gt_maximum_sharpe"]:.2%}

## Interpretation

This artifact is not a production-approved investment strategy.

The historical sample is small and the three phase tests overlap in
calendar time. Phase robustness therefore should not be interpreted as
three independent experiments.

The purpose of freezing V1 is to stop historical optimization and begin
collecting genuinely unseen forward evidence.
"""

    (
        ARTIFACT_DIR
        / "validation_summary.md"
    ).write_text(
        validation_text,
        encoding="utf-8",
    )

    readme_text = f"""# Meta Allocation V1

Version: {VERSION}

Status: **FROZEN RESEARCH CANDIDATE**

Frozen: {frozen_at}

## Purpose

Meta Allocation V1 is a tactical gate between an Equal Weight portfolio
and the Maximum Sharpe optimizer.

The model forecasts Maximum Sharpe's three-month excess return relative
to Equal Weight.

If predicted excess is greater than zero, the allocation state is
Maximum Sharpe.

Otherwise, the allocation state is Equal Weight.

## Frozen Architecture

Market-state features
→ StandardScaler
→ Ridge(alpha={RIDGE_ALPHA})
→ predicted three-month Maximum-Sharpe excess
→ zero threshold
→ Equal Weight or Maximum Sharpe
→ portfolio constraints
→ execution

## Files

- `meta_allocation_v1.joblib` — frozen fitted sklearn pipeline
- `model_spec.json` — model and training specification
- `feature_list.json` — immutable V1 feature set
- `coefficients.csv` — standardized Ridge coefficients at freeze
- `research_results.csv` — final portfolio-bootstrap validation
- `quarterly_returns.csv` — realized non-overlapping validation blocks
- `validation_summary.md` — final historical research summary
- `artifact_manifest.json` — reproducibility hashes
- `FROZEN.txt` — historical-development freeze notice

## Freeze Policy

Historical tuning of V1 is closed.

Do not change any of the following using pre-freeze history:

- feature list
- Ridge alpha
- three-month target horizon
- zero decision threshold
- Equal Weight / Maximum Sharpe decision states
- maximum portfolio weight
- training methodology

Any modification creates Meta Allocation V2.

V1 should now be evaluated using genuinely unseen forward observations
and paperMoney execution.

This artifact is for research and forward testing, not an assertion of
future investment performance.
"""

    (
        ARTIFACT_DIR
        / "README.md"
    ).write_text(
        readme_text,
        encoding="utf-8",
    )

    (
        ARTIFACT_DIR
        / "FROZEN.txt"
    ).write_text(
        (
            "META ALLOCATION V1 HISTORICAL "
            "DEVELOPMENT IS FROZEN.\n\n"
            f"Frozen at: {frozen_at}\n"
            "Any historical redesign must use "
            "a new model version.\n"
        ),
        encoding="utf-8",
    )

    artifact_files = [
        path
        for path in ARTIFACT_DIR.iterdir()
        if (
            path.is_file()
            and path.name
            != "artifact_manifest.json"
        )
    ]

    manifest = {
        "artifact":
            "Meta Allocation V1",

        "version":
            VERSION,

        "frozen_at_utc":
            frozen_at,

        "source_dataset":
            str(
                source_dataset_path
            ),

        "source_dataset_sha256":
            sha256_file(
                source_dataset_path
            ),

        "files":
            {
                path.name:
                    sha256_file(
                        path
                    )
                for path
                in sorted(
                    artifact_files
                )
            },
    }

    with (
        ARTIFACT_DIR
        / "artifact_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        260,
    )

    source_dataset_path = (
        FEATURE_DATA_DIR
        / "meta_learning_dataset_with_macro.csv"
    )

    stored = pd.read_csv(
        source_dataset_path,
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
            transaction_cost_bps=
                TRANSACTION_COST_BPS,
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

    binary = (
        run_binary_predictions(
            dataset
        )
    )

    validation_records = []

    quarterly_frames = []

    monthly_summaries = []

    for phase in PHASES:
        (
            quarterly,
            monthly_results,
        ) = (
            build_phase_portfolio_returns(
                returns=
                    returns,

                backtest=
                    backtest,

                binary_predictions=
                    binary,

                phase=
                    phase,
            )
        )

        quarterly_frames.append(
            quarterly
        )

        validation_records.append(
            bootstrap_phase(
                quarterly=
                    quarterly,

                phase=
                    phase,
            )
        )

        monthly_summary = (
            summarize_backtest(
                monthly_results
            )
            .reset_index()
        )

        monthly_summary[
            "phase"
        ] = phase

        monthly_summaries.append(
            monthly_summary
        )

    validation = pd.DataFrame(
        validation_records
    )

    quarterly_returns = (
        pd.concat(
            quarterly_frames,
            ignore_index=True,
        )
    )

    portfolio_summary = (
        pd.concat(
            monthly_summaries,
            ignore_index=True,
        )
    )

    print()
    print("=" * 170)
    print(
        "FINAL META ALLOCATION V1 "
        "REALIZED-RETURN BOOTSTRAP"
    )
    print("=" * 170)

    display = (
        validation.copy()
    )

    percent_columns = [
        "selector_annualized_return",
        "equal_weight_annualized_return",
        "maximum_sharpe_annualized_return",
        "mean_quarterly_excess_vs_ew",
        "mean_quarterly_excess_vs_ms",
        "p_return_gt_equal_weight",
        "p_return_gt_maximum_sharpe",
        "p_sharpe_gt_equal_weight",
        "p_sharpe_gt_maximum_sharpe",
        "ew_excess_ci_low",
        "ew_excess_ci_high",
        "ms_excess_ci_low",
        "ms_excess_ci_high",
        "top3_contribution_vs_ew",
        "top3_contribution_vs_ms",
    ]

    for column in percent_columns:
        display[
            column
        ] *= 100

    print()
    print(
        display.round(
            {
                "selector_annualized_return": 2,
                "equal_weight_annualized_return": 2,
                "maximum_sharpe_annualized_return": 2,
                "selector_quarterly_sharpe": 3,
                "equal_weight_quarterly_sharpe": 3,
                "maximum_sharpe_quarterly_sharpe": 3,
                "mean_quarterly_excess_vs_ew": 3,
                "mean_quarterly_excess_vs_ms": 3,
                "p_return_gt_equal_weight": 1,
                "p_return_gt_maximum_sharpe": 1,
                "p_sharpe_gt_equal_weight": 1,
                "p_sharpe_gt_maximum_sharpe": 1,
                "ew_excess_ci_low": 3,
                "ew_excess_ci_high": 3,
                "ms_excess_ci_low": 3,
                "ms_excess_ci_high": 3,
                "top3_contribution_vs_ew": 1,
                "top3_contribution_vs_ms": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 170)
    print(
        "MONTHLY REALIZED PORTFOLIO SUMMARY"
    )
    print("=" * 170)

    print(
        portfolio_summary.to_string(
            index=False
        )
    )

    as_of_date = (
        pd.Timestamp(
            returns.index.max()
        )
    )

    write_frozen_artifact(
        dataset=
            dataset,

        source_dataset_path=
            source_dataset_path,

        validation=
            validation,

        quarterly_returns=
            quarterly_returns,

        as_of_date=
            as_of_date,
    )

    print()
    print("=" * 170)
    print(
        "META ALLOCATION V1 FROZEN"
    )
    print("=" * 170)

    print()
    print(
        "Artifact directory:",
        ARTIFACT_DIR,
    )

    print(
        "Historical development closed:",
        True,
    )

    print(
        "Next stage:",
        "unseen forward test / paperMoney",
    )


if __name__ == "__main__":
    main()
