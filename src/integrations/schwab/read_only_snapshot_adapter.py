from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.forward.meta_allocation_v1_portfolio_snapshot import (
    FROZEN_ASSETS,
    PortfolioSnapshot,
    build_portfolio_snapshot,
)

from src.integrations.schwab.read_only_client import (
    SchwabReadOnlyPayloads,
)


SCHWAB_READ_ONLY_MODE = "SCHWAB_READ_ONLY"

DEFAULT_MAX_PRICE_AGE = pd.Timedelta(
    days=7
)


def _require_mapping(
    value,
    field_name: str,
) -> Mapping:
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{field_name} must be a mapping."
        )

    return value


def _coerce_float(
    value,
    field_name: str,
) -> float:
    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc

    if not np.isfinite(
        result
    ):
        raise ValueError(
            f"{field_name} must be finite."
        )

    return result


def _normalize_timestamp(
    value,
    field_name: str,
) -> pd.Timestamp:
    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric = float(
            value
        )

        if not np.isfinite(
            numeric
        ):
            raise ValueError(
                f"{field_name} must be finite."
            )

        unit = (
            "ms"
            if abs(
                numeric
            ) >= 1e11
            else "s"
        )

        return pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
        )

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:
        raise ValueError(
            f"{field_name} must be timezone-aware."
        )

    return timestamp.tz_convert(
        "UTC"
    )


def _extract_securities_account(
    account_payload: Mapping,
) -> Mapping:
    account_payload = (
        _require_mapping(
            account_payload,
            "account_payload",
        )
    )

    if (
        "securitiesAccount"
        not in account_payload
    ):
        raise ValueError(
            "Schwab account payload missing "
            "'securitiesAccount'."
        )

    return _require_mapping(
        account_payload[
            "securitiesAccount"
        ],
        "securitiesAccount",
    )


def extract_schwab_cash(
    account_payload: Mapping,
) -> float:
    """
    Extract accounting cash.

    V1 deliberately uses cashBalance rather than
    buying power so the dry-run planner does not
    intentionally introduce margin.
    """
    account = (
        _extract_securities_account(
            account_payload
        )
    )

    if (
        "currentBalances"
        not in account
    ):
        raise ValueError(
            "securitiesAccount missing "
            "'currentBalances'."
        )

    balances = _require_mapping(
        account[
            "currentBalances"
        ],
        "currentBalances",
    )

    if (
        "cashBalance"
        not in balances
    ):
        raise ValueError(
            "currentBalances missing "
            "'cashBalance'."
        )

    cash = _coerce_float(
        balances[
            "cashBalance"
        ],
        "cashBalance",
    )

    if cash < 0:
        raise ValueError(
            "Negative Schwab cash balance is "
            "not supported by Meta Allocation V1."
        )

    return cash


def extract_schwab_holdings(
    account_payload: Mapping,
) -> pd.Series:
    account = (
        _extract_securities_account(
            account_payload
        )
    )

    positions = account.get(
        "positions",
        [],
    )

    if positions is None:
        positions = []

    if not isinstance(
        positions,
        (
            list,
            tuple,
        ),
    ):
        raise ValueError(
            "Schwab positions must be a list."
        )

    holdings = {
        asset: 0
        for asset in FROZEN_ASSETS
    }

    seen_symbols = set()

    for index, position in enumerate(
        positions,
        start=1,
    ):
        position = _require_mapping(
            position,
            f"positions[{index}]",
        )

        instrument = _require_mapping(
            position.get(
                "instrument"
            ),
            (
                f"positions[{index}]"
                ".instrument"
            ),
        )

        symbol = (
            str(
                instrument.get(
                    "symbol",
                    "",
                )
            )
            .strip()
            .upper()
        )

        if not symbol:
            raise ValueError(
                "Schwab position missing symbol."
            )

        if symbol in seen_symbols:
            raise ValueError(
                "Duplicate Schwab position "
                f"detected: {symbol}"
            )

        seen_symbols.add(
            symbol
        )

        long_quantity = _coerce_float(
            position.get(
                "longQuantity",
                0.0,
            ),
            f"{symbol}.longQuantity",
        )

        short_quantity = _coerce_float(
            position.get(
                "shortQuantity",
                0.0,
            ),
            f"{symbol}.shortQuantity",
        )

        if long_quantity < 0:
            raise ValueError(
                f"{symbol} longQuantity "
                "cannot be negative."
            )

        if short_quantity < 0:
            raise ValueError(
                f"{symbol} shortQuantity "
                "cannot be negative."
            )

        if short_quantity > 1e-12:
            raise ValueError(
                "Short positions are not "
                "supported by Meta Allocation V1. "
                f"Symbol: {symbol}"
            )

        if np.isclose(
            long_quantity,
            0.0,
            atol=1e-12,
            rtol=0.0,
        ):
            continue

        if symbol not in FROZEN_ASSETS:
            raise ValueError(
                "Schwab account contains a "
                "non-zero position outside the "
                "Meta Allocation V1 universe: "
                f"{symbol}"
            )

        if not np.isclose(
            long_quantity,
            round(
                long_quantity
            ),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "Fractional holdings are not "
                "supported by Meta Allocation V1. "
                f"Symbol: {symbol}"
            )

        holdings[
            symbol
        ] = int(
            round(
                long_quantity
            )
        )

    return pd.Series(
        holdings,
        index=FROZEN_ASSETS,
        dtype=int,
        name="current_shares",
    )


def extract_schwab_reference_prices(
    quote_payload: Mapping,
    snapshot_timestamp: pd.Timestamp,
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> tuple[
    pd.Series,
    pd.Timestamp,
]:
    quote_payload = (
        _require_mapping(
            quote_payload,
            "quote_payload",
        )
    )

    snapshot_timestamp = (
        _normalize_timestamp(
            snapshot_timestamp,
            "snapshot_timestamp",
        )
    )

    max_price_age = pd.Timedelta(
        max_price_age
    )

    if max_price_age < pd.Timedelta(0):
        raise ValueError(
            "max_price_age cannot be negative."
        )

    prices = {}
    quote_times = {}

    for symbol in FROZEN_ASSETS:
        if symbol not in quote_payload:
            raise ValueError(
                "Missing Schwab quote for: "
                f"{symbol}"
            )

        symbol_payload = (
            _require_mapping(
                quote_payload[
                    symbol
                ],
                f"quote_payload[{symbol}]",
            )
        )

        if (
            "quote"
            not in symbol_payload
        ):
            raise ValueError(
                f"Schwab quote for {symbol} "
                "missing 'quote' section."
            )

        quote = _require_mapping(
            symbol_payload[
                "quote"
            ],
            f"{symbol}.quote",
        )

        if (
            "lastPrice"
            not in quote
        ):
            raise ValueError(
                f"Schwab quote for {symbol} "
                "missing 'lastPrice'."
            )

        price = _coerce_float(
            quote[
                "lastPrice"
            ],
            f"{symbol}.lastPrice",
        )

        if price <= 0:
            raise ValueError(
                f"{symbol} lastPrice "
                "must be positive."
            )

        if (
            "quoteTime"
            not in quote
        ):
            raise ValueError(
                f"Schwab quote for {symbol} "
                "missing 'quoteTime'."
            )

        quote_time = (
            _normalize_timestamp(
                quote[
                    "quoteTime"
                ],
                f"{symbol}.quoteTime",
            )
        )

        if (
            quote_time
            > snapshot_timestamp
        ):
            raise ValueError(
                f"{symbol} quote timestamp "
                "is later than the snapshot."
            )

        quote_age = (
            snapshot_timestamp
            - quote_time
        )

        if quote_age > max_price_age:
            raise ValueError(
                f"{symbol} Schwab quote is stale.\n"
                f"Quote age: {quote_age}\n"
                f"Maximum allowed age: "
                f"{max_price_age}"
            )

        prices[
            symbol
        ] = price

        quote_times[
            symbol
        ] = quote_time

    reference_prices = pd.Series(
        prices,
        index=FROZEN_ASSETS,
        dtype=float,
        name="reference_price",
    )

    # Conservative freshness timestamp:
    # oldest quote among all five assets.
    price_as_of = min(
        quote_times.values()
    )

    return (
        reference_prices,
        price_as_of,
    )


def build_schwab_read_only_snapshot(
    payloads: SchwabReadOnlyPayloads,
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> PortfolioSnapshot:
    """
    Convert raw data retrieved by SchwabReadOnlyClient
    into Meta Allocation V1's broker-independent
    PortfolioSnapshot.

    This adapter performs no network requests.

    It cannot generate or submit brokerage orders.
    """
    if not isinstance(
        payloads,
        SchwabReadOnlyPayloads,
    ):
        raise TypeError(
            "payloads must be "
            "SchwabReadOnlyPayloads."
        )

    snapshot_timestamp = (
        _normalize_timestamp(
            payloads.fetched_at_utc,
            "fetched_at_utc",
        )
    )

    cash = extract_schwab_cash(
        payloads.account_payload
    )

    holdings = extract_schwab_holdings(
        payloads.account_payload
    )

    (
        reference_prices,
        price_as_of,
    ) = extract_schwab_reference_prices(
        quote_payload=
            payloads.quote_payload,

        snapshot_timestamp=
            snapshot_timestamp,

        max_price_age=
            max_price_age,
    )

    return build_portfolio_snapshot(
        holdings=
            holdings,

        cash=
            cash,

        reference_prices=
            reference_prices,

        snapshot_timestamp=
            snapshot_timestamp,

        price_as_of=
            price_as_of,

        account_mode=
            SCHWAB_READ_ONLY_MODE,

        max_price_age=
            max_price_age,
    )