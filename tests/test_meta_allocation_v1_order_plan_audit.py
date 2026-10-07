from pathlib import Path

import pandas as pd
import pytest

from src.forward.meta_allocation_v1_order_plan import (
    build_order_plan_from_weight_record,
)

from src.forward.meta_allocation_v1_order_plan_audit import (
    append_order_plan_audit,
    build_detail_records,
    build_summary_record,
)


def make_result():
    record = {
        "information_date":
            "2026-10-30",

        "decision_signal_date":
            "2026-10-30",

        "effective_from":
            "2026-11-02",

        "allocation_state":
            "EQUAL_WEIGHT",

        "model_version":
            "1.0.0",

        "target_weights_hash":
            "target123",

        "weight_SPY": 0.20,
        "weight_QQQ": 0.20,
        "weight_TLT": 0.20,
        "weight_GLD": 0.20,
        "weight_SCHD": 0.20,
    }

    return (
        build_order_plan_from_weight_record(
            weight_record=
                record,

            current_shares={
                "SPY": 100,
                "QQQ": 50,
                "TLT": 0,
                "GLD": 0,
                "SCHD": 0,
            },

            current_cash=
                20_000.0,

            reference_prices={
                "SPY": 500.0,
                "QQQ": 600.0,
                "TLT": 100.0,
                "GLD": 200.0,
                "SCHD": 80.0,
            },

            transaction_cost_bps=
                5.0,
        )
    )


def test_summary_contains_provenance():
    result = make_result()

    record = build_summary_record(
        result,
        generated_at_utc=
            "2026-10-30T21:00:00+00:00",
    )

    assert (
        record[
            "order_plan_hash"
        ]
        == result.order_plan_hash
    )

    assert (
        record[
            "holdings_snapshot_hash"
        ]
        == result.holdings_snapshot_hash
    )

    assert (
        record[
            "price_snapshot_hash"
        ]
        == result.price_snapshot_hash
    )

    assert (
        record[
            "execution_mode"
        ]
        == "DRY_RUN"
    )


def test_detail_has_five_assets():
    details = (
        build_detail_records(
            make_result()
        )
    )

    assert len(details) == 5

    assert set(
        details[
            "asset"
        ]
    ) == {
        "SPY",
        "QQQ",
        "TLT",
        "GLD",
        "SCHD",
    }


def test_append_audit_is_idempotent(
    tmp_path: Path,
):
    summary = (
        tmp_path
        / "summary.csv"
    )

    details = (
        tmp_path
        / "details.csv"
    )

    result = make_result()

    first = append_order_plan_audit(
        result=result,
        summary_path=summary,
        detail_path=details,
    )

    second = append_order_plan_audit(
        result=result,
        summary_path=summary,
        detail_path=details,
    )

    assert first == {
        "summary": "APPENDED",
        "details": "APPENDED",
    }

    assert second == {
        "summary": "EXISTS",
        "details": "EXISTS",
    }

    summary_df = pd.read_csv(
        summary
    )

    detail_df = pd.read_csv(
        details
    )

    assert len(summary_df) == 1
    assert len(detail_df) == 5


def test_detail_rows_are_bound_to_plan_hash():
    result = make_result()

    details = (
        build_detail_records(
            result
        )
    )

    assert (
        details[
            "order_plan_hash"
        ]
        == result.order_plan_hash
    ).all()


def test_audit_never_claims_execution():
    record = build_summary_record(
        make_result()
    )

    assert (
        record[
            "execution_mode"
        ]
        == "DRY_RUN"
    )