from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    DATA_DIR,
    FEATURE_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
)

from src.features.regime_features import (
    build_regime_feature_matrix,
)

from src.portfolio.returns import (
    build_return_matrix,
)


ARTIFACT_DIR = (
    MODELS_DIR
    / "meta_allocation_v1"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "meta_allocation_v1.joblib"
)

MODEL_SPEC_PATH = (
    ARTIFACT_DIR
    / "model_spec.json"
)

FEATURE_SPEC_PATH = (
    ARTIFACT_DIR
    / "feature_list.json"
)

MANIFEST_PATH = (
    ARTIFACT_DIR
    / "artifact_manifest.json"
)

MARKET_DATA_PATH = (
    RAW_DATA_DIR
    / "portfolio_universe_daily.csv"
)

FORWARD_DATA_DIR = (
    DATA_DIR
    / "forward"
)

SIGNAL_LEDGER_PATH = (
    FORWARD_DATA_DIR
    / "meta_allocation_v1_signals.csv"
)


def sha256_file(
    path: Path,
) -> str:
    """
    Return SHA-256 digest for one file.
    """
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


def sha256_feature_snapshot(
    features: pd.Series,
) -> str:
    """
    Hash the exact ordered feature values used by the model.

    The feature hash makes each forward decision auditable.
    """
    payload = [
        {
            "feature":
                str(feature),

            "value":
                float(
                    features.loc[
                        feature
                    ]
                ),
        }
        for feature
        in features.index
    ]

    encoded = json.dumps(
        payload,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def validate_artifact_files() -> None:
    """
    Confirm the frozen V1 artifact exists and that the model
    file still matches its recorded manifest hash.
    """
    required = [
        MODEL_PATH,
        MODEL_SPEC_PATH,
        FEATURE_SPEC_PATH,
        MANIFEST_PATH,
    ]

    missing = [
        path
        for path
        in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Meta Allocation V1 artifact is incomplete:\n"
            + "\n".join(
                str(path)
                for path
                in missing
            )
        )

    manifest = load_json(
        MANIFEST_PATH
    )

    recorded_hash = (
        manifest
        .get(
            "files",
            {},
        )
        .get(
            MODEL_PATH.name
        )
    )

    if recorded_hash is None:
        raise ValueError(
            "Frozen artifact manifest does not contain "
            "the model hash."
        )

    current_hash = (
        sha256_file(
            MODEL_PATH
        )
    )

    if current_hash != recorded_hash:
        raise RuntimeError(
            "Frozen model integrity check failed.\n"
            f"Expected: {recorded_hash}\n"
            f"Actual:   {current_hash}"
        )


def load_artifact() -> tuple[
    object,
    dict,
    list[str],
]:
    """
    Load the frozen fitted model and frozen model metadata.
    """
    validate_artifact_files()

    model_spec = load_json(
        MODEL_SPEC_PATH
    )

    feature_spec = load_json(
        FEATURE_SPEC_PATH
    )

    status = (
        model_spec.get(
            "status"
        )
    )

    if status != "FROZEN_RESEARCH_CANDIDATE":
        raise ValueError(
            "Unexpected model status: "
            f"{status}"
        )

    features = (
        feature_spec.get(
            "features"
        )
    )

    if (
        not isinstance(
            features,
            list,
        )
        or len(
            features
        )
        == 0
    ):
        raise ValueError(
            "Frozen feature list is missing or invalid."
        )

    model = joblib.load(
        MODEL_PATH
    )

    return (
        model,
        model_spec,
        features,
    )


def load_market_returns() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load current market data and convert prices to returns.
    """
    if not MARKET_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Market data file not found: {MARKET_DATA_PATH}"
        )

    prices = pd.read_csv(
        MARKET_DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    prices = (
        prices
        .sort_index()
    )

    if prices.empty:
        raise ValueError(
            "Market price dataset is empty."
        )

    returns = (
        build_return_matrix(
            prices
        )
        .sort_index()
    )

    if returns.empty:
        raise ValueError(
            "Market return matrix is empty."
        )

    source_hash = (
        sha256_file(
            MARKET_DATA_PATH
        )
    )

    return (
        returns,
        source_hash,
    )


def is_business_month_end(
    date: pd.Timestamp,
) -> bool:
    """
    Determine whether date is the final standard business
    day of its calendar month.

    This handles weekend month ends. It is not intended to
    replace an exchange holiday calendar.
    """
    date = pd.Timestamp(
        date
    ).normalize()

    business_month_end = (
        date
        + pd.offsets.BMonthEnd(
            0
        )
    )

    return (
        date
        == business_month_end.normalize()
    )


def resolve_signal_date(
    returns: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    signal_date: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """
    Resolve the market date on which the forward signal will
    be calculated.

    If signal_date is supplied, use that date explicitly.

    Otherwise, use the most recent completed month-end
    trading observation available as of `as_of`.

    Mid-month runs therefore use the previous completed
    month.
    """
    if returns.empty:
        raise ValueError(
            "returns cannot be empty."
        )

    market_dates = (
        pd.DatetimeIndex(
            returns.index
        )
        .sort_values()
        .normalize()
    )

    if as_of is None:
        as_of_date = (
            market_dates.max()
        )
    else:
        as_of_date = pd.Timestamp(
            as_of
        ).normalize()

    available_dates = (
        market_dates[
            market_dates
            <= as_of_date
        ]
    )

    if len(
        available_dates
    ) == 0:
        raise ValueError(
            f"No market observations exist on or before "
            f"{as_of_date.date()}."
        )

    if signal_date is not None:
        requested = pd.Timestamp(
            signal_date
        ).normalize()

        if requested > as_of_date:
            raise ValueError(
                "signal_date cannot be after as_of."
            )

        if requested not in market_dates:
            raise ValueError(
                f"Requested signal date "
                f"{requested.date()} is not present "
                "in the market return index."
            )

        return requested

    latest = (
        available_dates.max()
    )

    # If the latest observation is a business month-end,
    # it is eligible immediately.
    if is_business_month_end(
        latest
    ):
        return latest

    latest_period = (
        latest.to_period(
            "M"
        )
    )

    prior_dates = (
        available_dates[
            available_dates.to_period(
                "M"
            )
            < latest_period
        ]
    )

    if len(
        prior_dates
    ) == 0:
        raise ValueError(
            "No completed prior month is available."
        )

    return (
        prior_dates.max()
    )


def build_frozen_feature_snapshot(
    returns: pd.DataFrame,
    signal_date: pd.Timestamp,
    frozen_features: list[str],
) -> pd.Series:
    """
    Calculate the same market-state feature family used
    during historical model development.

    Only observations <= signal_date are supplied.
    """
    historical_returns = (
        returns.loc[
            :signal_date
        ]
        .copy()
    )

    if historical_returns.empty:
        raise ValueError(
            "No return history is available for signal date."
        )

    matrix = (
        build_regime_feature_matrix(
            returns=
                historical_returns,

            rebalance_dates=
                pd.DatetimeIndex(
                    [
                        signal_date
                    ]
                ),
        )
    )

    if signal_date not in matrix.index:
        raise ValueError(
            f"Feature matrix did not produce "
            f"{signal_date.date()}."
        )

    missing_features = [
        feature
        for feature
        in frozen_features
        if feature not in matrix.columns
    ]

    if missing_features:
        raise ValueError(
            "Forward feature engine is missing frozen "
            f"features: {missing_features}"
        )

    snapshot = (
        matrix.loc[
            signal_date,
            frozen_features,
        ]
        .astype(
            float
        )
    )

    if snapshot.isna().any():
        missing = (
            snapshot[
                snapshot.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Frozen forward feature snapshot contains "
            f"missing values: {missing}"
        )

    if not np.isfinite(
        snapshot.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Frozen forward feature snapshot contains "
            "non-finite values."
        )

    return snapshot


def generate_signal(
    model,
    model_spec: dict,
    frozen_features: list[str],
    feature_snapshot: pd.Series,
) -> dict:
    """
    Generate the frozen Meta Allocation V1 decision.
    """
    feature_frame = pd.DataFrame(
        [
            feature_snapshot[
                frozen_features
            ].to_dict()
        ],
        columns=
            frozen_features,
    )

    prediction = float(
        model.predict(
            feature_frame
        )[0]
    )

    if not np.isfinite(
        prediction
    ):
        raise ValueError(
            "Model produced a non-finite prediction."
        )

    threshold = float(
        model_spec[
            "decision_threshold"
        ]
    )

    if prediction > threshold:
        state = (
            "MAXIMUM_SHARPE"
        )
    else:
        state = (
            "EQUAL_WEIGHT"
        )

    return {
        "predicted_ms_excess_3m":
            prediction,

        "allocation_state":
            state,

        "decision_threshold":
            threshold,
    }


def validate_forward_only(
    signal_date: pd.Timestamp,
    model_spec: dict,
    allow_in_sample: bool,
) -> None:
    """
    Prevent historical observations from silently entering
    the forward ledger.
    """
    freeze_date = pd.Timestamp(
        model_spec[
            "information_as_of"
        ]
    ).normalize()

    if (
        signal_date
        <= freeze_date
        and not allow_in_sample
    ):
        raise RuntimeError(
            "\nForward-only protection triggered.\n\n"
            f"Signal date: {signal_date.date()}\n"
            f"V1 information freeze: {freeze_date.date()}\n\n"
            "A genuine forward signal must occur after the "
            "historical freeze date.\n"
            "Use --allow-in-sample only for pipeline testing; "
            "those signals should not be treated as forward "
            "evidence."
        )


def build_signal_record(
    signal_date: pd.Timestamp,
    model_spec: dict,
    signal: dict,
    feature_snapshot: pd.Series,
    market_data_hash: str,
    market_data_end: pd.Timestamp,
    source_market_data_end: pd.Timestamp,
    is_forward: bool,
) -> dict:
    """
    Construct the immutable audit record.
    """
    model_hash = (
        sha256_file(
            MODEL_PATH
        )
    )

    feature_hash = (
        sha256_feature_snapshot(
            feature_snapshot
        )
    )

    record = {
        "signal_date":
            signal_date.strftime(
                "%Y-%m-%d"
            ),

        "model_version":
            str(
                model_spec[
                    "version"
                ]
            ),

        "model_status":
            str(
                model_spec[
                    "status"
                ]
            ),

        "is_forward_observation":
            bool(
                is_forward
            ),

        "predicted_ms_excess_3m":
            float(
                signal[
                    "predicted_ms_excess_3m"
                ]
            ),

        "decision_threshold":
            float(
                signal[
                    "decision_threshold"
                ]
            ),

        "allocation_state":
            signal[
                "allocation_state"
            ],

        "feature_snapshot_hash":
            feature_hash,

        "model_artifact_hash":
            model_hash,

        "input_market_data_hash":
            market_data_hash,

        "input_market_data_end":
            pd.Timestamp(
                market_data_end
            ).strftime(
                "%Y-%m-%d"
            ),

        "source_market_data_end":
            pd.Timestamp(
                source_market_data_end
            ).strftime(
                "%Y-%m-%d"
            ),

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    # Store the actual frozen feature values as part of the
    # ledger. This allows exact post-mortem reconstruction.
    for feature in (
        feature_snapshot.index
    ):
        record[
            f"feature__{feature}"
        ] = float(
            feature_snapshot.loc[
                feature
            ]
        )

    return record


def append_signal_record(
    record: dict,
    ledger_path: Path = SIGNAL_LEDGER_PATH,
) -> str:
    """
    Append one signal to the forward ledger.

    The ledger is append-only.

    If the exact same model/date decision already exists,
    return 'EXISTS' rather than adding a duplicate.

    If the same model/date exists with different inputs or
    output, raise an error rather than rewriting history.
    """
    ledger_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_row = pd.DataFrame(
        [
            record
        ]
    )

    if not ledger_path.exists():
        new_row.to_csv(
            ledger_path,
            index=False,
        )

        return "APPENDED"

    existing = pd.read_csv(
        ledger_path,
    )

    key_match = (
        (
            existing[
                "signal_date"
            ].astype(
                str
            )
            == str(
                record[
                    "signal_date"
                ]
            )
        )
        &
        (
            existing[
                "model_version"
            ].astype(
                str
            )
            == str(
                record[
                    "model_version"
                ]
            )
        )
    )

    matches = (
        existing[
            key_match
        ]
    )

    if matches.empty:
        new_row.to_csv(
            ledger_path,
            mode="a",
            header=False,
            index=False,
        )

        return "APPENDED"

    if len(
        matches
    ) > 1:
        raise RuntimeError(
            "Forward ledger already contains duplicate "
            "model/date keys."
        )

    previous = (
        matches.iloc[
            0
        ]
    )

    immutable_fields = [
        "allocation_state",
        "feature_snapshot_hash",
        "model_artifact_hash",
        "input_market_data_hash",
    ]

    for field in immutable_fields:
        if str(
            previous[
                field
            ]
        ) != str(
            record[
                field
            ]
        ):
            raise RuntimeError(
                "Forward ledger conflict detected.\n"
                f"Signal date: {record['signal_date']}\n"
                f"Field: {field}\n"
                f"Existing: {previous[field]}\n"
                f"New: {record[field]}\n\n"
                "The forward ledger is append-only and "
                "will not rewrite a historical decision."
            )

    previous_prediction = float(
        previous[
            "predicted_ms_excess_3m"
        ]
    )

    new_prediction = float(
        record[
            "predicted_ms_excess_3m"
        ]
    )

    if not np.isclose(
        previous_prediction,
        new_prediction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Forward ledger conflict detected: "
            "prediction changed for an existing signal."
        )

    return "EXISTS"


def run_forward_signal(
    as_of: str | pd.Timestamp | None = None,
    signal_date: str | pd.Timestamp | None = None,
    allow_in_sample: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Complete Meta Allocation V1 forward inference pipeline.

    This function NEVER retrains the frozen model.
    """
    (
        model,
        model_spec,
        frozen_features,
    ) = load_artifact()

    (
        returns,
        market_data_hash,
    ) = load_market_returns()

    resolved_signal_date = (
        resolve_signal_date(
            returns=
                returns,

            as_of=
                as_of,

            signal_date=
                signal_date,
        )
    )

    input_returns = (
        returns.loc[
            :resolved_signal_date
        ]
        .copy()
    )

    input_market_data_hash = (
        sha256_dataframe(
            input_returns
        )
    )


    freeze_date = pd.Timestamp(
        model_spec[
            "information_as_of"
        ]
    ).normalize()

    is_forward = (
        resolved_signal_date
        > freeze_date
    )

    validate_forward_only(
        signal_date=
            resolved_signal_date,

        model_spec=
            model_spec,

        allow_in_sample=
            allow_in_sample,
    )

    feature_snapshot = (
        build_frozen_feature_snapshot(
            returns=
                returns,

            signal_date=
                resolved_signal_date,

            frozen_features=
                frozen_features,
        )
    )

    signal = (
        generate_signal(
            model=
                model,

            model_spec=
                model_spec,

            frozen_features=
                frozen_features,

            feature_snapshot=
                feature_snapshot,
        )
    )

    record = (
        build_signal_record(
            signal_date=
                resolved_signal_date,

            model_spec=
                model_spec,

            signal=
                signal,

            feature_snapshot=
                feature_snapshot,

            market_data_hash=
                input_market_data_hash,

            market_data_end=
                resolved_signal_date,

            source_market_data_end=
                returns.index.max(),

            is_forward=
                is_forward,
        )
    )

    if dry_run:
        ledger_action = (
            "DRY_RUN"
        )
    else:
        ledger_action = (
            append_signal_record(
                record
            )
        )

    return {
        **record,

        "ledger_action":
            ledger_action,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a frozen Meta Allocation V1 "
            "forward allocation signal."
        )
    )

    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Information cutoff date, YYYY-MM-DD. "
            "Defaults to latest market observation."
        ),
    )

    parser.add_argument(
        "--signal-date",
        default=None,
        help=(
            "Explicit market signal date. Normally omitted."
        ),
    )

    parser.add_argument(
        "--allow-in-sample",
        action="store_true",
        help=(
            "Allow pre-freeze dates for pipeline testing. "
            "Do not treat these as genuine forward evidence."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate the signal without writing to the "
            "forward ledger."
        ),
    )

    return parser


def main() -> None:
    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    result = (
        run_forward_signal(
            as_of=
                args.as_of,

            signal_date=
                args.signal_date,

            allow_in_sample=
                args.allow_in_sample,

            dry_run=
                args.dry_run,
        )
    )

def sha256_dataframe(
    frame: pd.DataFrame,
) -> str:
    """
    Hash a dataframe deterministically.

    Used to identify the exact historical market-data slice
    supplied to a forward decision.
    """
    canonical = (
        frame
        .sort_index()
        .to_csv(
            index=True,
            float_format="%.17g",
            date_format="%Y-%m-%d",
        )
        .encode(
            "utf-8"
        )
    )

    return hashlib.sha256(
        canonical
    ).hexdigest()

    print()
    print("=" * 100)
    print(
        "META ALLOCATION V1 SIGNAL"
    )
    print("=" * 100)

    print()
    print(
        "Signal date:",
        result[
            "signal_date"
        ],
    )

    print(
        "Model version:",
        result[
            "model_version"
        ],
    )

    print(
        "Forward observation:",
        result[
            "is_forward_observation"
        ],
    )

    print(
        "Predicted MS excess, 3M (%):",
        round(
            result[
                "predicted_ms_excess_3m"
            ]
            * 100,
            3,
        ),
    )

    print(
        "Decision threshold (%):",
        round(
            result[
                "decision_threshold"
            ]
            * 100,
            3,
        ),
    )

    print(
        "Allocation state:",
        result[
            "allocation_state"
        ],
    )

    print(
        "Market data end:",
        result[
            "input_market_data_end"
        ],
    )

    print(
        "Source market data end:",
        result[
            "source_market_data_end"
        ],
    )

    print(
        "Feature hash:",
        result[
            "feature_snapshot_hash"
        ],
    )

    print(
        "Model hash:",
        result[
            "model_artifact_hash"
        ],
    )

    print(
        "Ledger action:",
        result[
            "ledger_action"
        ],
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
