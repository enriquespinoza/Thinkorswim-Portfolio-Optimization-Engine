from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


SNAPSHOT_COLUMNS = (
    "symbol",
    "quantity",
    "market_value",
    "asset_type",
)


@dataclass(frozen=True)
class PersonalPortfolioSnapshot:
    positions: pd.DataFrame
    cash_balance: float
    securities_market_value: float
    total_portfolio_value: float


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def normalize_schwab_account_payload(
    payload: Mapping[str, Any],
    *,
    include_cash: bool = True,
) -> PersonalPortfolioSnapshot:
    root = _as_mapping(payload, "Schwab account payload")
    account = _as_mapping(
        root.get("securitiesAccount"),
        "Schwab securitiesAccount",
    )

    raw_positions = account.get("positions", [])
    if raw_positions is None:
        raw_positions = []
    if not isinstance(raw_positions, list):
        raise ValueError("Schwab positions must be a list")

    rows: list[dict[str, Any]] = []

    for index, raw_position in enumerate(raw_positions):
        position = _as_mapping(
            raw_position,
            f"Schwab position {index}",
        )
        instrument = _as_mapping(
            position.get("instrument"),
            f"Schwab position {index} instrument",
        )

        symbol = str(instrument.get("symbol", "")).strip().upper()
        if not symbol:
            raise ValueError(f"Schwab position {index} is missing symbol")

        asset_type = str(instrument.get("assetType", "UNKNOWN")).strip().upper()
        long_quantity = float(position.get("longQuantity", 0.0) or 0.0)
        short_quantity = float(position.get("shortQuantity", 0.0) or 0.0)

        if short_quantity > 0:
            raise ValueError(
                "Personal Portfolio Benchmark V1 does not support short positions"
            )
        if long_quantity < 0:
            raise ValueError("Schwab longQuantity cannot be negative")

        market_value = position.get("marketValue")
        if market_value is None:
            average_price = position.get("averagePrice")
            if average_price is None:
                raise ValueError(
                    f"Schwab position {symbol} is missing marketValue"
                )
            market_value = float(average_price) * long_quantity
        else:
            market_value = float(market_value)

        if market_value < 0:
            raise ValueError(
                f"Schwab position {symbol} has negative marketValue"
            )

        rows.append(
            {
                "symbol": symbol,
                "quantity": long_quantity,
                "market_value": market_value,
                "asset_type": asset_type,
            }
        )

    positions = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)

    if not positions.empty:
        positions = (
            positions.groupby(
                ["symbol", "asset_type"],
                as_index=False,
                sort=True,
            )
            .agg(
                quantity=("quantity", "sum"),
                market_value=("market_value", "sum"),
            )
        )
        positions = positions[
            ["symbol", "quantity", "market_value", "asset_type"]
        ]

    balances = account.get("currentBalances", {})
    if balances is None:
        balances = {}
    balances = _as_mapping(balances, "Schwab currentBalances")

    cash_balance = float(balances.get("cashBalance", 0.0) or 0.0)
    if cash_balance < 0:
        raise ValueError(
            "Personal Portfolio Benchmark V1 does not support negative cash/margin"
        )

    securities_market_value = float(
        positions["market_value"].sum()
        if not positions.empty
        else 0.0
    )

    if include_cash and cash_balance > 0:
        cash_row = pd.DataFrame(
            [
                {
                    "symbol": "CASH",
                    "quantity": cash_balance,
                    "market_value": cash_balance,
                    "asset_type": "CASH",
                }
            ]
        )
        positions = pd.concat(
            [positions, cash_row],
            ignore_index=True,
        ).sort_values("symbol", kind="stable").reset_index(drop=True)

    total_portfolio_value = securities_market_value + cash_balance

    if total_portfolio_value <= 0:
        raise ValueError("Schwab account has no positive portfolio value")

    return PersonalPortfolioSnapshot(
        positions=positions,
        cash_balance=cash_balance,
        securities_market_value=securities_market_value,
        total_portfolio_value=total_portfolio_value,
    )
