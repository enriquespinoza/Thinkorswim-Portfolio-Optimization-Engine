from __future__ import annotations

import argparse
import hashlib
import json
import exchange_calendars as xcals
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasts.expected_returns import (
    estimate_expected_returns,
)

from src.optimizers.maximum_sharpe import (
    optimize_maximum_sharpe,
)

from src.risk.covariance import (
    calculate_ledoit_wolf_covariance,
)

from config.settings import (
    DATA_DIR,
    DEFAULT_UNIVERSE,
    MODELS_DIR,
    TRADING_DAYS_PER_YEAR,
)

from src.forward.meta_allocation_v1_signal import (
    SIGNAL_LEDGER_PATH,
    load_market_returns,
    resolve_signal_date,
    sha256_dataframe,
)


# ---------------------------------------------------------------------------
# Paths / frozen V1 policy
# ---------------------------------------------------------------------------

ARTIFACT_DIR = MODELS_DIR / "meta_allocation_v1"
MODEL_SPEC_PATH = ARTIFACT_DIR / "model_spec.json"

FORWARD_DATA_DIR = DATA_DIR / "forward"
WEIGHT_LEDGER_PATH = (
    FORWARD_DATA_DIR
    / "meta_allocation_v1_weights.csv"
)

FROZEN_ASSETS = list(DEFAULT_UNIVERSE)

DECISION_MONTHS = {1, 4, 7, 10}
HOLDING_MONTHS = 3

EWMA_SPAN = 126
RISK_FREE_RATE = 0.0

DEFAULT_MAX_WEIGHT = 0.40


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def load_model_spec() -> dict:
    if not MODEL_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Model spec not found: {MODEL_SPEC_PATH}"
        )

    with MODEL_SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def get_model_version(
    model_spec: dict,
) -> str:
    version = model_spec.get(
        "version",
        "1.0.0",
    )

    return str(version)


def get_max_weight(
    model_spec: dict,
) -> float:
    """
    Read the frozen asset cap while tolerating reasonable
    historical naming differences in model_spec.json.
    """
    direct_keys = [
        "maximum_asset_weight",
        "max_weight",
        "maximum_weight",
    ]

    for key in direct_keys:
        if key in model_spec:
            return float(
                model_spec[key]
            )

    constraints = model_spec.get(
        "portfolio_constraints",
        {},
    )

    for key in direct_keys:
        if key in constraints:
            return float(
                constraints[key]
            )

    # Frozen V1 research policy.
    return DEFAULT_MAX_WEIGHT


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def normalize_bool(
    value,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    if isinstance(
        value,
        (int, np.integer),
    ):
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "",
        "nan",
    }:
        return False

    raise ValueError(
        f"Cannot interpret boolean value: {value}"
    )


def month_distance(
    earlier: pd.Timestamp,
    later: pd.Timestamp,
) -> int:
    earlier = pd.Timestamp(earlier)
    later = pd.Timestamp(later)

    return (
        (later.year - earlier.year) * 12
        + later.month
        - earlier.month
    )

def is_quarterly_decision_month(
    date: pd.Timestamp,
) -> bool:
    date = pd.Timestamp(date)

    return (
        date.month
        in DECISION_MONTHS
    )


XNYS_CALENDAR = xcals.get_calendar(
    "XNYS"
)


def _normalize_session(
    value,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tz is not None:
        timestamp = (
            timestamp
            .tz_localize(None)
        )

    return timestamp.normalize()


def expected_trading_month_end(
    date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the final XNYS trading session of the
    calendar month containing date.
    """
    date = pd.Timestamp(
        date
    ).normalize()

    month_start = (
        date
        .replace(day=1)
    )

    month_end = (
        date
        + pd.offsets.MonthEnd(0)
    )

    sessions = (
        XNYS_CALENDAR
        .sessions_in_range(
            month_start,
            month_end,
        )
    )

    if len(sessions) == 0:
        raise RuntimeError(
            "No XNYS trading sessions found "
            f"for {date:%Y-%m}."
        )

    return _normalize_session(
        sessions[-1]
    )


def validate_information_date(
    information_date: pd.Timestamp,
) -> None:
    """
    Require the information date to be the final
    XNYS trading session of the month.
    """
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    expected = (
        expected_trading_month_end(
            information_date
        )
    )

    if information_date != expected:
        raise RuntimeError(
            "Meta Allocation V1 target generation "
            "requires the final XNYS trading "
            "session of the month.\n"
            f"Received: "
            f"{information_date.date()}\n"
            f"Expected: "
            f"{expected.date()}"
        )


def calculate_effective_from(
    information_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the first XNYS trading session strictly
    after the information date.
    """
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    validate_information_date(
        information_date
    )

    current_session = (
        XNYS_CALENDAR
        .date_to_session(
            information_date,
            direction="none",
        )
    )

    next_session = (
        XNYS_CALENDAR
        .next_session(
            current_session
        )
    )

    return _normalize_session(
        next_session
    )

def validate_information_date(
    information_date: pd.Timestamp,
) -> None:
    """
    V1 targets may only be generated from month-end
    information dates.
    """
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    expected = (
        expected_trading_month_end(
            information_date
        )
    )

    if information_date != expected:
        raise RuntimeError(
            "Meta Allocation V1 target generation "
            "requires a month-end information date.\n"
            f"Received: {information_date.date()}\n"
            f"Expected month-end: {expected.date()}"
        )


XNYS_CALENDAR = xcals.get_calendar(
    "XNYS"
)


def _normalize_session(
    value,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tz is not None:
        timestamp = (
            timestamp
            .tz_localize(None)
        )

    return timestamp.normalize()


def expected_trading_month_end(
    date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the final XNYS trading session of the
    calendar month containing date.
    """
    date = pd.Timestamp(
        date
    ).normalize()

    month_start = (
        date
        .replace(day=1)
    )

    month_end = (
        date
        + pd.offsets.MonthEnd(0)
    )

    sessions = (
        XNYS_CALENDAR
        .sessions_in_range(
            month_start,
            month_end,
        )
    )

    if len(sessions) == 0:
        raise RuntimeError(
            "No XNYS trading sessions found "
            f"for {date:%Y-%m}."
        )

    return _normalize_session(
        sessions[-1]
    )


def validate_information_date(
    information_date: pd.Timestamp,
) -> None:
    """
    Require the information date to be the final
    XNYS trading session of the month.
    """
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    expected = (
        expected_trading_month_end(
            information_date
        )
    )

    if information_date != expected:
        raise RuntimeError(
            "Meta Allocation V1 target generation "
            "requires the final XNYS trading "
            "session of the month.\n"
            f"Received: "
            f"{information_date.date()}\n"
            f"Expected: "
            f"{expected.date()}"
        )


def calculate_effective_from(
    information_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Return the first XNYS trading session strictly
    after the information date.
    """
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    validate_information_date(
        information_date
    )

    current_session = (
        XNYS_CALENDAR
        .date_to_session(
            information_date,
            direction="none",
        )
    )

    next_session = (
        XNYS_CALENDAR
        .next_session(
            current_session
        )
    )

    return _normalize_session(
        next_session
    )

# ---------------------------------------------------------------------------
# Return / risk estimates
# ---------------------------------------------------------------------------

def validate_market_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    missing = [
        asset
        for asset in FROZEN_ASSETS
        if asset not in returns.columns
    ]

    if missing:
        raise ValueError(
            "Missing frozen V1 assets: "
            f"{missing}"
        )

    data = (
        returns[
            FROZEN_ASSETS
        ]
        .copy()
        .sort_index()
        .dropna(
            how="any"
        )
    )

    if data.empty:
        raise ValueError(
            "No aligned return history available."
        )

    if len(data) < EWMA_SPAN:
        raise ValueError(
            f"Need at least {EWMA_SPAN} aligned "
            "daily observations."
        )

    values = data.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Return history contains non-finite values."
        )

    if (
        values <= -1.0
    ).any():
        raise ValueError(
            "Return history contains a return <= -100%."
        )

    return data

def robust_expected_returns(
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Use the exact frozen expected-return implementation
    used by the historical walk-forward engine.
    """
    forecasts = (
        estimate_expected_returns(
            returns,
            ewma_span=EWMA_SPAN,
        )
    )

    return forecasts.robust.copy()


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use the exact covariance estimator used by the
    historical walk-forward engine.
    """
    covariance, _ = (
        calculate_ledoit_wolf_covariance(
            returns
        )
    )

    return covariance


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def equal_weight_target(
    assets: list[str],
) -> pd.Series:
    if not assets:
        raise ValueError(
            "Asset list cannot be empty."
        )

    return pd.Series(
        1.0 / len(assets),
        index=assets,
        dtype=float,
        name="weight",
    )


def maximum_sharpe_target(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float,
) -> pd.Series:
    """
    Use the exact Maximum-Sharpe optimizer used by the
    historical walk-forward engine.
    """
    result = (
        optimize_maximum_sharpe(
            expected_returns=
                expected_returns,

            covariance=
                covariance,

            risk_free_rate=
                RISK_FREE_RATE,

            max_weight=
                max_weight,
        )
    )

    weights = (
        result.weights
        .copy()
        .astype(float)
    )

    validate_target_weights(
        weights=
            weights,

        max_weight=
            max_weight,
    )

    return weights


def validate_target_weights(
    weights: pd.Series,
    max_weight: float,
) -> None:
    if set(weights.index) != set(
        FROZEN_ASSETS
    ):
        raise ValueError(
            "Weights do not contain exactly the "
            "frozen V1 universe."
        )

    values = weights.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Weights contain non-finite values."
        )

    if (
        weights < -1e-10
    ).any():
        raise ValueError(
            "Negative target weight detected."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Target weights do not sum to 1.0. "
            f"Actual={weights.sum():.12f}"
        )

    if (
        weights
        > max_weight + 1e-8
    ).any():
        raise ValueError(
            "Target weight exceeds frozen "
            f"{max_weight:.2%} cap."
        )


def portfolio_statistics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> dict[str, float]:
    assets = list(
        expected_returns.index
    )

    weights = weights.reindex(
        assets
    )

    covariance = covariance.loc[
        assets,
        assets,
    ]

    w = weights.to_numpy(
        dtype=float
    )

    mu = expected_returns.to_numpy(
        dtype=float
    )

    cov = covariance.to_numpy(
        dtype=float
    )

    expected_return = float(
        w @ mu
    )

    variance = float(
        w
        @ cov
        @ w
    )

    volatility = float(
        np.sqrt(
            max(
                variance,
                0.0,
            )
        )
    )

    if volatility == 0:
        sharpe = np.nan
    else:
        sharpe = float(
            (
                expected_return
                - RISK_FREE_RATE
            )
            / volatility
        )

    return {
        "expected_return":
            expected_return,
        "volatility":
            volatility,
        "sharpe":
            sharpe,
    }


def build_target_weights(
    allocation_state: str,
    historical_returns: pd.DataFrame,
    max_weight: float,
) -> tuple[
    pd.Series,
    dict[str, float],
]:
    state = (
        allocation_state
        .strip()
        .upper()
    )

    returns = (
        validate_market_returns(
            historical_returns
        )
    )

    if state == "EQUAL_WEIGHT":
        weights = (
            equal_weight_target(
                FROZEN_ASSETS
            )
        )

        validate_target_weights(
            weights=weights,
            max_weight=max_weight,
        )

        return (
            weights,
            {
                "expected_return": np.nan,
                "volatility": np.nan,
                "sharpe": np.nan,
            },
        )

    if state != "MAXIMUM_SHARPE":
        raise ValueError(
            "allocation_state must be "
            "EQUAL_WEIGHT or MAXIMUM_SHARPE."
        )

    expected_returns = (
        robust_expected_returns(
            returns
        )
    )

    covariance = (
        ledoit_wolf_covariance(
            returns
        )
    )

    weights = (
        maximum_sharpe_target(
            expected_returns=
                expected_returns,
            covariance=
                covariance,
            max_weight=
                max_weight,
        )
    )

    statistics = (
        portfolio_statistics(
            weights=weights,
            expected_returns=
                expected_returns,
            covariance=covariance,
        )
    )

    return (
        weights,
        statistics,
    )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def synthetic_test_signal(
    information_date: pd.Timestamp,
    allocation_state: str,
    model_spec: dict,
) -> pd.Series:
    state = (
        allocation_state
        .strip()
        .upper()
    )

    if state not in {
        "EQUAL_WEIGHT",
        "MAXIMUM_SHARPE",
    }:
        raise ValueError(
            f"Unsupported test state: {state}"
        )

    information_date = pd.Timestamp(
        information_date
    ).normalize()

    return pd.Series(
        {
            "signal_date":
                information_date,

            "model_version":
                get_model_version(
                    model_spec
                ),

            "allocation_state":
                state,

            "predicted_ms_excess_3m":
                np.nan,

            "is_forward_observation":
                False,
        }
    )


def load_active_signal(
    information_date: pd.Timestamp,
    model_spec: dict,
    allow_in_sample: bool = False,
) -> pd.Series:
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    if not SIGNAL_LEDGER_PATH.exists():
        raise FileNotFoundError(
            "Signal ledger does not exist: "
            f"{SIGNAL_LEDGER_PATH}"
        )

    ledger = pd.read_csv(
        SIGNAL_LEDGER_PATH
    )

    required = {
        "signal_date",
        "model_version",
        "allocation_state",
        "predicted_ms_excess_3m",
        "is_forward_observation",
    }

    missing = (
        required
        - set(ledger.columns)
    )

    if missing:
        raise ValueError(
            "Signal ledger missing columns: "
            f"{sorted(missing)}"
        )

    ledger[
        "signal_date"
    ] = pd.to_datetime(
        ledger[
            "signal_date"
        ]
    )

    version = get_model_version(
        model_spec
    )

    ledger = (
        ledger[
            ledger[
                "model_version"
            ].astype(str)
            == version
        ]
        .copy()
    )

    # Only quarterly V1 decision signals may control
    # portfolio state.
    ledger = (
        ledger[
            ledger[
                "signal_date"
            ]
            .dt.month
            .isin(
                DECISION_MONTHS
            )
        ]
        .copy()
    )

    ledger = (
        ledger[
            ledger[
                "signal_date"
            ]
            <= information_date
        ]
        .sort_values(
            "signal_date"
        )
    )

    if ledger.empty:
        raise ValueError(
            "No eligible quarterly signal exists "
            f"on or before {information_date.date()}."
        )

    signal = ledger.iloc[-1]

    signal_date = pd.Timestamp(
        signal[
            "signal_date"
        ]
    ).normalize()

    age = month_distance(
        signal_date,
        information_date,
    )

    if (
        age < 0
        or age >= HOLDING_MONTHS
    ):
        raise RuntimeError(
            "Latest allocation signal is stale.\n"
            f"Signal: {signal_date.date()}\n"
            f"Information date: "
            f"{information_date.date()}\n"
            f"Age: {age} months"
        )

    # Jan / Apr / Jul / Oct require a new signal
    # from that exact information date.
    if is_quarterly_decision_month(
        information_date
    ):
        if signal_date != information_date:
            raise RuntimeError(
                "A quarterly decision month requires "
                "a signal generated from the exact "
                "same information date.\n"
                f"Information date: "
                f"{information_date.date()}\n"
                f"Signal date: "
                f"{signal_date.date()}"
            )

    is_forward = normalize_bool(
        signal[
            "is_forward_observation"
        ]
    )

    if (
        not is_forward
        and not allow_in_sample
    ):
        raise RuntimeError(
            "Active signal is not a genuine "
            "forward observation."
        )

    state = str(
        signal[
            "allocation_state"
        ]
    ).strip().upper()

    if state not in {
        "EQUAL_WEIGHT",
        "MAXIMUM_SHARPE",
    }:
        raise ValueError(
            f"Invalid allocation state: {state}"
        )

    return signal


# ---------------------------------------------------------------------------
# Audit ledger
# ---------------------------------------------------------------------------

def sha256_weights(
    weights: pd.Series,
) -> str:
    payload = [
        {
            "asset": asset,
            "weight": float(
                weights.loc[
                    asset
                ]
            ),
        }
        for asset in FROZEN_ASSETS
    ]

    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def build_weight_record(
    information_date: pd.Timestamp,
    signal: pd.Series,
    weights: pd.Series,
    statistics: dict[str, float],
    input_market_data_hash: str,
) -> dict:
    information_date = pd.Timestamp(
        information_date
    ).normalize()

    signal_date = pd.Timestamp(
        signal[
            "signal_date"
        ]
    ).normalize()

    effective_from = (
        calculate_effective_from(
            information_date
        )
    )

    state_age_months = (
        month_distance(
            signal_date,
            information_date,
        )
    )

    record = {
        "information_date":
            information_date.strftime(
                "%Y-%m-%d"
            ),

        "decision_signal_date":
            signal_date.strftime(
                "%Y-%m-%d"
            ),

        "effective_from":
            effective_from.strftime(
                "%Y-%m-%d"
            ),

        "is_quarterly_decision":
            is_quarterly_decision_month(
                information_date
            ),

        "state_age_months":
            state_age_months,

        "model_version":
            str(
                signal[
                    "model_version"
                ]
            ),

        "allocation_state":
            str(
                signal[
                    "allocation_state"
                ]
            ).upper(),

        "predicted_ms_excess_3m":
            float(
                signal[
                    "predicted_ms_excess_3m"
                ]
            ),

        "input_market_data_end":
            information_date.strftime(
                "%Y-%m-%d"
            ),

        "input_market_data_hash":
            input_market_data_hash,

        "target_weights_hash":
            sha256_weights(
                weights
            ),

        "portfolio_expected_return":
            statistics[
                "expected_return"
            ],

        "portfolio_expected_volatility":
            statistics[
                "volatility"
            ],

        "portfolio_expected_sharpe":
            statistics[
                "sharpe"
            ],

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    for asset in FROZEN_ASSETS:
        record[
            f"weight_{asset}"
        ] = float(
            weights.loc[
                asset
            ]
        )

    return record

def append_weight_record(
    record: dict,
    ledger_path: Path = WEIGHT_LEDGER_PATH,
) -> str:
    ledger_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_row = pd.DataFrame(
        [record]
    )

    if not ledger_path.exists():
        new_row.to_csv(
            ledger_path,
            index=False,
        )

        return "APPENDED"

    existing = pd.read_csv(
        ledger_path
    )

    required_columns = {
        "information_date",
        "model_version",
        "decision_signal_date",
        "effective_from",
        "allocation_state",
        "input_market_data_hash",
        "target_weights_hash",
    }

    missing = (
        required_columns
        - set(existing.columns)
    )

    if missing:
        raise RuntimeError(
            "Existing weight ledger uses an "
            "incompatible schema. Missing: "
            f"{sorted(missing)}"
        )

    match = (
        existing[
            "information_date"
        ].astype(str)
        == str(
            record[
                "information_date"
            ]
        )
    ) & (
        existing[
            "model_version"
        ].astype(str)
        == str(
            record[
                "model_version"
            ]
        )
    )

    rows = existing[
        match
    ]

    if rows.empty:
        new_row.to_csv(
            ledger_path,
            mode="a",
            header=False,
            index=False,
        )

        return "APPENDED"

    if len(rows) > 1:
        raise RuntimeError(
            "Duplicate weight-ledger key detected."
        )

    previous = rows.iloc[0]

    immutable_fields = [
        "decision_signal_date",
        "effective_from",
        "allocation_state",
        "input_market_data_hash",
        "target_weights_hash",
    ]

    for field in immutable_fields:
        if str(
            previous[field]
        ) != str(
            record[field]
        ):
            raise RuntimeError(
                "Weight-ledger conflict.\n"
                f"Information date: "
                f"{record['information_date']}\n"
                f"Field: {field}\n"
                f"Existing: {previous[field]}\n"
                f"New: {record[field]}"
            )

    return "EXISTS"

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_forward_weights(
    as_of: str | pd.Timestamp | None = None,
    allow_in_sample: bool = False,
    dry_run: bool = False,
    test_state: str | None = None,
) -> dict:
    model_spec = load_model_spec()

    (
        returns,
        _,
    ) = load_market_returns()

    information_date = (
        resolve_signal_date(
            returns=returns,
            as_of=as_of,
        )
    )

    information_date = pd.Timestamp(
        information_date
    ).normalize()

    # Prevent an incomplete month from becoming
    # an official target-generation date.
    validate_information_date(
        information_date
    )

    historical_returns = (
        returns.loc[
            :information_date,
            FROZEN_ASSETS,
        ]
        .copy()
    )

    input_market_data_hash = (
        sha256_dataframe(
            historical_returns
        )
    )

    if test_state is not None:
        if not dry_run:
            raise RuntimeError(
                "--test-state requires --dry-run."
            )

        signal = synthetic_test_signal(
            information_date=
                information_date,

            allocation_state=
                test_state,

            model_spec=
                model_spec,
        )

    else:
        signal = load_active_signal(
            information_date=
                information_date,

            model_spec=
                model_spec,

            allow_in_sample=
                allow_in_sample,
        )

    state = str(
        signal[
            "allocation_state"
        ]
    ).upper()

    max_weight = get_max_weight(
        model_spec
    )

    (
        weights,
        statistics,
    ) = build_target_weights(
        allocation_state=
            state,

        historical_returns=
            historical_returns,

        max_weight=
            max_weight,
    )

    record = build_weight_record(
        information_date=
            information_date,

        signal=
            signal,

        weights=
            weights,

        statistics=
            statistics,

        input_market_data_hash=
            input_market_data_hash,
    )

    if dry_run:
        ledger_action = "DRY_RUN"

    else:
        ledger_action = (
            append_weight_record(
                record
            )
        )

    return {
        **record,

        "max_weight":
            max_weight,

        "ledger_action":
            ledger_action,
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen Meta Allocation V1 "
            "portfolio target weights."
        )
    )

    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Information cutoff YYYY-MM-DD."
        ),
    )

    parser.add_argument(
        "--allow-in-sample",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--test-state",
        choices=[
            "EQUAL_WEIGHT",
            "MAXIMUM_SHARPE",
        ],
        default=None,
        help=(
            "Testing-only state override. "
            "Requires --dry-run."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_forward_weights(
        as_of=args.as_of,
        allow_in_sample=
            args.allow_in_sample,
        dry_run=args.dry_run,
        test_state=args.test_state,
    )

    print()
    print("=" * 100)
    print(
        "META ALLOCATION V1 TARGET WEIGHTS"
    )
    print("=" * 100)

    print()

    print(
        "Information date:",
        result[
            "information_date"
        ],
    )

    print(
        "Decision signal date:",
        result[
            "decision_signal_date"
        ],
    )

    print(
        "Effective from:",
        result[
            "effective_from"
        ],
    )

    print(
        "Quarterly decision:",
        result[
            "is_quarterly_decision"
        ],
    )

    print(
        "State age, months:",
        result[
            "state_age_months"
        ],
    )

    print(
        "Allocation state:",
        result[
            "allocation_state"
        ],
    )

    prediction = result[
        "predicted_ms_excess_3m"
    ]

    if pd.isna(prediction):
        print(
            "Recorded MS excess, 3M:",
            "TEST OVERRIDE",
        )
    else:
        print(
            "Recorded MS excess, 3M (%):",
            round(
                prediction * 100,
                3,
            ),
        ),

    print(
        "Maximum asset weight:",
        f"{result['max_weight']:.2%}",
    )

    print()
    print("TARGET ALLOCATION")
    print("-" * 100)

    for asset in FROZEN_ASSETS:
        print(
            f"{asset:<8}"
            f"{result[f'weight_{asset}'] * 100:>10.4f}%"
        )

    total = sum(
        result[
            f"weight_{asset}"
        ]
        for asset in FROZEN_ASSETS
    )

    print("-" * 100)
    print(
        f"{'TOTAL':<8}"
        f"{total * 100:>10.4f}%"
    )

    expected_return = result[
        "portfolio_expected_return"
    ]

    if not pd.isna(
        expected_return
    ):
        print()
        print(
            "Expected annual return (%):",
            round(
                expected_return
                * 100,
                4,
            ),
        )

        print(
            "Expected annual volatility (%):",
            round(
                result[
                    "portfolio_expected_volatility"
                ]
                * 100,
                4,
            ),
        )

        print(
            "Expected Sharpe:",
            round(
                result[
                    "portfolio_expected_sharpe"
                ],
                6,
            ),
        )

    print()
    print(
        "Input market data end:",
        result[
            "input_market_data_end"
        ],
    )

    print(
        "Input market data hash:",
        result[
            "input_market_data_hash"
        ],
    )

    print(
        "Target weights hash:",
        result[
            "target_weights_hash"
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