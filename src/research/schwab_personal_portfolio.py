from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PersonalPortfolioCapture:
    account_index: int
    positions: pd.DataFrame
    fetched_at_utc: pd.Timestamp


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return value


def normalize_account_positions(
    payload: Mapping[str, Any],
) -> pd.DataFrame:
    root = _require_mapping(payload, "Schwab account payload")

    account = root.get("securitiesAccount", root)
    account = _require_mapping(account, "Schwab securities account")

    raw_positions = account.get("positions", [])
    if raw_positions is None:
        raw_positions = []
    if not isinstance(raw_positions, list):
        raise RuntimeError("Schwab positions field must be a list")

    rows: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_positions):
        row = _require_mapping(raw, f"Schwab position {index}")
        instrument = _require_mapping(
            row.get("instrument", {}),
            f"Schwab position instrument {index}",
        )

        symbol = str(instrument.get("symbol", "")).strip().upper()
        if not symbol:
            continue

        long_qty = float(row.get("longQuantity", 0.0) or 0.0)
        short_qty = float(row.get("shortQuantity", 0.0) or 0.0)
        quantity = long_qty - short_qty
        market_value = float(row.get("marketValue", 0.0) or 0.0)

        if quantity == 0 and market_value == 0:
            continue

        rows.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "market_value": market_value,
                "asset_type": str(
                    instrument.get("assetType", "")
                ).strip().upper(),
            }
        )

    columns = [
        "symbol",
        "quantity",
        "market_value",
        "asset_type",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows, columns=columns)
    frame = (
        frame.groupby(
            ["symbol", "asset_type"],
            as_index=False,
            dropna=False,
        )
        .agg(
            quantity=("quantity", "sum"),
            market_value=("market_value", "sum"),
        )
        .loc[:, columns]
        .sort_values(["symbol", "asset_type"], kind="stable")
        .reset_index(drop=True)
    )

    return frame


def capture_account_by_index(
    client,
    *,
    account_index: int,
) -> PersonalPortfolioCapture:
    if account_index < 1:
        raise ValueError("account_index must be 1 or greater")

    records = client.get_account_number_records()

    if account_index > len(records):
        raise ValueError(
            f"account_index {account_index} is out of range; "
            f"{len(records)} accounts are available"
        )

    record = records[account_index - 1]
    account_hash = str(record.get("hashValue", "")).strip()
    if not account_hash:
        raise RuntimeError("selected Schwab account is missing a hash")

    payload = client.get_account_payload(account_hash)
    positions = normalize_account_positions(payload)

    return PersonalPortfolioCapture(
        account_index=account_index,
        positions=positions,
        fetched_at_utc=pd.Timestamp.now(tz="UTC"),
    )
