from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

from src.forward.meta_allocation_v1_order_plan import (
    EXECUTION_MODE,
    OrderPlanResult,
)


FORWARD_DATA_DIR = DATA_DIR / "forward"

ORDER_PLAN_SUMMARY_LEDGER = (
    FORWARD_DATA_DIR
    / "meta_allocation_v1_order_plan_audit.csv"
)

ORDER_PLAN_DETAIL_LEDGER = (
    FORWARD_DATA_DIR
    / "meta_allocation_v1_order_plan_details.csv"
)


DETAIL_COLUMNS = [
    "asset",
    "current_shares",
    "reference_price",
    "current_value",
    "current_weight",
    "target_weight",
    "target_dollars",
    "target_shares",
    "trade_shares",
    "side",
    "trade_notional",
    "signed_trade_value",
    "post_trade_value",
    "post_trade_weight",
]


def validate_order_plan_result(
    result: OrderPlanResult,
) -> None:
    if result.execution_mode != EXECUTION_MODE:
        raise ValueError(
            "Only DRY_RUN order plans may be written "
            "to the V1 audit ledger."
        )

    if result.execution_mode != "DRY_RUN":
        raise ValueError(
            "Execution mode must be DRY_RUN."
        )

    if not result.order_plan_hash:
        raise ValueError(
            "order_plan_hash cannot be empty."
        )

    if not result.target_weights_hash:
        raise ValueError(
            "target_weights_hash cannot be empty."
        )

    if not result.holdings_snapshot_hash:
        raise ValueError(
            "holdings_snapshot_hash cannot be empty."
        )

    if not result.price_snapshot_hash:
        raise ValueError(
            "price_snapshot_hash cannot be empty."
        )

    if (
        not np.isfinite(
            result.portfolio_value_before
        )
        or result.portfolio_value_before <= 0
    ):
        raise ValueError(
            "portfolio_value_before must be "
            "positive and finite."
        )

    if (
        not np.isfinite(
            result.cash_before
        )
        or result.cash_before < 0
    ):
        raise ValueError(
            "cash_before must be finite "
            "and non-negative."
        )

    if (
        not np.isfinite(
            result.cash_after
        )
        or result.cash_after < -1e-8
    ):
        raise ValueError(
            "cash_after must be finite "
            "and non-negative."
        )

    missing = (
        set(DETAIL_COLUMNS)
        - set(
            result.orders.columns
        )
    )

    if missing:
        raise ValueError(
            "Order detail missing columns: "
            f"{sorted(missing)}"
        )


def build_summary_record(
    result: OrderPlanResult,
    generated_at_utc: str | None = None,
) -> dict:
    validate_order_plan_result(
        result
    )

    if generated_at_utc is None:
        generated_at_utc = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    return {
        "information_date":
            pd.Timestamp(
                result.information_date
            ).strftime(
                "%Y-%m-%d"
            ),

        "effective_from":
            pd.Timestamp(
                result.effective_from
            ).strftime(
                "%Y-%m-%d"
            ),

        "model_version":
            str(
                result.model_version
            ),

        "allocation_state":
            str(
                result.allocation_state
            ).upper(),

        "target_weights_hash":
            result.target_weights_hash,

        "holdings_snapshot_hash":
            result.holdings_snapshot_hash,

        "price_snapshot_hash":
            result.price_snapshot_hash,

        "transaction_cost_bps":
            float(
                result.transaction_cost_bps
            ),

        "portfolio_value_before":
            float(
                result.portfolio_value_before
            ),

        "cash_before":
            float(
                result.cash_before
            ),

        "cash_after":
            float(
                result.cash_after
            ),

        "gross_trade_notional":
            float(
                result.gross_trade_notional
            ),

        "turnover_ratio":
            float(
                result.turnover_ratio
            ),

        "estimated_transaction_cost":
            float(
                result.estimated_transaction_cost
            ),

        "order_plan_hash":
            result.order_plan_hash,

        "execution_mode":
            result.execution_mode,

        "generated_at_utc":
            generated_at_utc,
    }


def build_detail_records(
    result: OrderPlanResult,
) -> pd.DataFrame:
    validate_order_plan_result(
        result
    )

    details = (
        result.orders[
            DETAIL_COLUMNS
        ]
        .copy()
    )

    details.insert(
        0,
        "order_plan_hash",
        result.order_plan_hash,
    )

    details.insert(
        1,
        "information_date",
        pd.Timestamp(
            result.information_date
        ).strftime(
            "%Y-%m-%d"
        ),
    )

    details.insert(
        2,
        "effective_from",
        pd.Timestamp(
            result.effective_from
        ).strftime(
            "%Y-%m-%d"
        ),
    )

    details.insert(
        3,
        "model_version",
        str(
            result.model_version
        ),
    )

    details.insert(
        4,
        "allocation_state",
        str(
            result.allocation_state
        ).upper(),
    )

    details.insert(
        5,
        "execution_mode",
        result.execution_mode,
    )

    return details


def append_summary_record(
    record: dict,
    ledger_path: Path = (
        ORDER_PLAN_SUMMARY_LEDGER
    ),
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

    required = {
        "order_plan_hash",
        "information_date",
        "effective_from",
        "model_version",
        "allocation_state",
        "target_weights_hash",
        "holdings_snapshot_hash",
        "price_snapshot_hash",
        "transaction_cost_bps",
        "execution_mode",
    }

    missing = (
        required
        - set(
            existing.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Existing order-plan summary ledger "
            "uses an incompatible schema. "
            f"Missing: {sorted(missing)}"
        )

    rows = existing[
        existing[
            "order_plan_hash"
        ].astype(str)
        == str(
            record[
                "order_plan_hash"
            ]
        )
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
            "Duplicate order_plan_hash detected "
            "in summary ledger."
        )

    previous = rows.iloc[0]

    immutable_fields = [
        "information_date",
        "effective_from",
        "model_version",
        "allocation_state",
        "target_weights_hash",
        "holdings_snapshot_hash",
        "price_snapshot_hash",
        "transaction_cost_bps",
        "execution_mode",
    ]

    for field in immutable_fields:
        if str(
            previous[field]
        ) != str(
            record[field]
        ):
            raise RuntimeError(
                "Order-plan summary conflict.\n"
                f"Order plan hash: "
                f"{record['order_plan_hash']}\n"
                f"Field: {field}\n"
                f"Existing: {previous[field]}\n"
                f"New: {record[field]}"
            )

    return "EXISTS"


def append_detail_records(
    details: pd.DataFrame,
    ledger_path: Path = (
        ORDER_PLAN_DETAIL_LEDGER
    ),
) -> str:
    ledger_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if details.empty:
        raise ValueError(
            "Order-plan details cannot be empty."
        )

    required = {
        "order_plan_hash",
        "asset",
    }

    missing = (
        required
        - set(
            details.columns
        )
    )

    if missing:
        raise ValueError(
            "Order detail missing required fields: "
            f"{sorted(missing)}"
        )

    plan_hashes = (
        details[
            "order_plan_hash"
        ]
        .astype(str)
        .unique()
    )

    if len(plan_hashes) != 1:
        raise ValueError(
            "Detail rows must contain exactly "
            "one order_plan_hash."
        )

    plan_hash = str(
        plan_hashes[0]
    )

    if not ledger_path.exists():
        details.to_csv(
            ledger_path,
            index=False,
        )

        return "APPENDED"

    existing = pd.read_csv(
        ledger_path
    )

    required_existing = {
        "order_plan_hash",
        "asset",
    }

    missing_existing = (
        required_existing
        - set(
            existing.columns
        )
    )

    if missing_existing:
        raise RuntimeError(
            "Existing order-plan detail ledger "
            "uses an incompatible schema."
        )

    existing_plan = (
        existing[
            existing[
                "order_plan_hash"
            ].astype(str)
            == plan_hash
        ]
        .copy()
    )

    if existing_plan.empty:
        details.to_csv(
            ledger_path,
            mode="a",
            header=False,
            index=False,
        )

        return "APPENDED"

    expected_assets = (
        set(
            details[
                "asset"
            ].astype(str)
        )
    )

    existing_assets = (
        set(
            existing_plan[
                "asset"
            ].astype(str)
        )
    )

    if (
        existing_assets
        != expected_assets
    ):
        raise RuntimeError(
            "Existing detail rows conflict "
            "with the proposed order plan."
        )

    comparison_columns = [
        column
        for column in details.columns
        if column
        != "generated_at_utc"
    ]

    expected = (
        details[
            comparison_columns
        ]
        .sort_values(
            "asset"
        )
        .reset_index(
            drop=True
        )
    )

    previous = (
        existing_plan[
            comparison_columns
        ]
        .sort_values(
            "asset"
        )
        .reset_index(
            drop=True
        )
    )

    for column in comparison_columns:
        left = previous[
            column
        ]

        right = expected[
            column
        ]

        if pd.api.types.is_numeric_dtype(
            right
        ):
            if not np.allclose(
                left.astype(float),
                right.astype(float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise RuntimeError(
                    "Order-plan detail conflict.\n"
                    f"Order plan hash: "
                    f"{plan_hash}\n"
                    f"Column: {column}"
                )

        else:
            if not (
                left.astype(str)
                .equals(
                    right.astype(str)
                )
            ):
                raise RuntimeError(
                    "Order-plan detail conflict.\n"
                    f"Order plan hash: "
                    f"{plan_hash}\n"
                    f"Column: {column}"
                )

    return "EXISTS"


def append_order_plan_audit(
    result: OrderPlanResult,
    summary_path: Path = (
        ORDER_PLAN_SUMMARY_LEDGER
    ),
    detail_path: Path = (
        ORDER_PLAN_DETAIL_LEDGER
    ),
) -> dict[str, str]:
    """
    Append a DRY_RUN order plan to two immutable ledgers:

    summary:
        one row per unique order_plan_hash

    detail:
        one row per asset per order_plan_hash

    No brokerage actions occur here.
    """
    summary_record = (
        build_summary_record(
            result
        )
    )

    detail_records = (
        build_detail_records(
            result
        )
    )

    summary_action = (
        append_summary_record(
            record=
                summary_record,
            ledger_path=
                summary_path,
        )
    )

    detail_action = (
        append_detail_records(
            details=
                detail_records,
            ledger_path=
                detail_path,
        )
    )

    return {
        "summary":
            summary_action,

        "details":
            detail_action,
    }