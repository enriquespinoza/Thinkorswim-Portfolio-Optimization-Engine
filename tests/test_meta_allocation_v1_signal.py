from pathlib import Path

import pandas as pd
import pytest

from src.forward.meta_allocation_v1_signal import (
    append_signal_record,
    resolve_signal_date,
)


def build_returns() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2026-08-28",
            "2026-08-31",
            "2026-09-29",
            "2026-09-30",
            "2026-10-01",
            "2026-10-02",
            "2026-10-05",
            "2026-10-06",
        ]
    )

    return pd.DataFrame(
        {
            "SPY": range(
                len(index)
            )
        },
        index=index,
    )


def test_mid_month_uses_prior_completed_month():
    returns = build_returns()

    result = (
        resolve_signal_date(
            returns=
                returns,

            as_of=
                "2026-10-06",
        )
    )

    assert result == pd.Timestamp(
        "2026-09-30"
    )


def test_explicit_signal_date():
    returns = build_returns()

    result = (
        resolve_signal_date(
            returns=
                returns,

            as_of=
                "2026-10-06",

            signal_date=
                "2026-10-02",
        )
    )

    assert result == pd.Timestamp(
        "2026-10-02"
    )


def sample_record() -> dict:
    return {
        "signal_date":
            "2026-10-30",

        "model_version":
            "1.0.0",

        "model_status":
            "FROZEN_RESEARCH_CANDIDATE",

        "is_forward_observation":
            True,

        "predicted_ms_excess_3m":
            0.0123,

        "decision_threshold":
            0.0,

        "allocation_state":
            "MAXIMUM_SHARPE",

        "feature_snapshot_hash":
            "feature123",

        "model_artifact_hash":
            "model123",

        "input_market_data_hash":
            "market123",

        "input_market_data_end":
            "2026-10-30",

        "source_market_data_end":
            "2026-10-30",

        "generated_at_utc":
            "2026-10-30T20:00:00+00:00",

        "feature__SPY_return_21d":
            0.02,
    }


def test_append_signal_is_idempotent(
    tmp_path: Path,
):
    ledger = (
        tmp_path
        / "signals.csv"
    )

    record = (
        sample_record()
    )

    first = (
        append_signal_record(
            record=
                record,

            ledger_path=
                ledger,
        )
    )

    second = (
        append_signal_record(
            record=
                record,

            ledger_path=
                ledger,
        )
    )

    assert first == "APPENDED"
    assert second == "EXISTS"

    saved = pd.read_csv(
        ledger
    )

    assert len(
        saved
    ) == 1


def test_append_signal_rejects_rewrite(
    tmp_path: Path,
):
    ledger = (
        tmp_path
        / "signals.csv"
    )

    original = (
        sample_record()
    )

    append_signal_record(
        record=
            original,

        ledger_path=
            ledger,
    )

    changed = {
        **original,

        "allocation_state":
            "EQUAL_WEIGHT",
    }

    with pytest.raises(
        RuntimeError
    ):
        append_signal_record(
            record=
                changed,

            ledger_path=
                ledger,
        )
 