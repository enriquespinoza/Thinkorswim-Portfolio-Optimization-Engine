from __future__ import annotations

from config.scanner_allocation import (
    ALLOCATION_CONTRACT_VERSION,
)


PORTFOLIO_TARGET_HANDOFF_VERSION = "portfolio-target-handoff-v1"
TARGET_BUILDER_VERSION = "1.2.0"

EXPECTED_ALLOCATION_CONTRACT_VERSION = (
    ALLOCATION_CONTRACT_VERSION
)

EXECUTION_STATES = (
    "DRY_RUN",
    "PAPER",
)

REFERENCE_PRICE_SOURCE_ALLOCATION = (
    "scanner_allocation_close"
)
REFERENCE_PRICE_SOURCE_POSITION = (
    "position_snapshot"
)

SHARE_ROUNDING_POLICY = (
    "floor_nonnegative_whole_shares"
)

TARGET_STATUSES = (
    "TARGETED",
    "ZERO_WEIGHT",
    "ROUNDING_TO_ZERO",
    "EXIT",
)

TARGET_COLUMNS = (
    "handoff_contract_version",
    "target_builder_version",
    "as_of_session",
    "allocation_contract_version",
    "allocation_sha256",
    "allocation_manifest_sha256",
    "scanner_snapshot_sha256",
    "symbol",
    "scanner_rank",
    "approved_target_weight",
    "reference_price",
    "reference_price_source",
    "portfolio_capital",
    "target_dollar_value",
    "target_shares",
    "target_share_notional",
    "realized_target_weight",
    "weight_rounding_error",
    "target_status",
    "current_position",
    "share_delta",
    "trade_side",
    "execution_state",
)


PAPER_POSITION_SNAPSHOT_CONTRACT = (
    "paper-optimizer-position-snapshot-v1"
)
