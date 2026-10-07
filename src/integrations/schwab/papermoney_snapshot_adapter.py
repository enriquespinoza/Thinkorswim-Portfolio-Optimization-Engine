from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.forward.meta_allocation_v1_portfolio_snapshot import (
    FROZEN_ASSETS,
    PortfolioSnapshot,
    build_portfolio_snapshot,
)


PAPERMONEY_MODE = "PAPERMONEY"

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
    """
    Normalize an ISO-style or epoch timestamp to UTC.

    Numeric timestamps are interpreted as:
        >= 1e11 -> epoch milliseconds
        <  1e11 -> epoch seconds
    """
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

        timestamp = pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
        )

    else:
        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            raise ValueError(
                f"{field_name} must be "
                "timezone-aware."
            )

        timestamp = (
            timestamp
            .tz_convert(
                "UTC"
            )
        )

    return timestamp


def _extract_securities_account(
    account_payload: Mapping,
) -> Mapping:
    """
    Require an explicit PAPERMONEY marker.

    The marker is intentionally adapter metadata rather
    than something inferred from the broker payload.
    """
    account_payload = (
        _require_mapping(
            account_payload,
            "account_payload",
        )
    )

    account_mode = (
        str(
            account_payload.get(
                "account_mode",
                "",
            )
        )
        .strip()
        .upper()
    )

    if account_mode != PAPERMONEY_MODE:
        raise ValueError(
            "PaperMoney adapter requires "
            "account_mode='PAPERMONEY'. "
            "LIVE accounts are not supported."
        )

    if (
        "securitiesAccount"
        not in account_payload
    ):
        raise ValueError(
            "account_payload missing "
            "'securitiesAccount'."
        )

    return _require_mapping(
        account_payload[
            "securitiesAccount"
        ],
        "securitiesAccount",
    )


def extract_cash(
    account_payload: Mapping,
) -> float:
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
            "cashBalance cannot be negative."
        )

    return cash


def extract_holdings(
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
            "positions must be a list."
        )

    holdings = {
        asset: 0
        for asset in FROZEN_ASSETS
    }

    seen_symbols = set()

    for position_number, position in enumerate(
        positions,
        start=1,
    ):
        position = _require_mapping(
            position,
            (
                "positions"
                f"[{position_number}]"
            ),
        )

        instrument = _require_mapping(
            position.get(
                "instrument"
            ),
            (
                "positions"
                f"[{position_number}]"
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
                "Position instrument "
                "is missing symbol."
            )

        if symbol in seen_symbols:
            raise ValueError(
                "Duplicate position symbol "
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
            (
                f"{symbol}.longQuantity"
            ),
        )

        short_quantity = _coerce_float(
            position.get(
                "shortQuantity",
                0.0,
            ),
            (
                f"{symbol}.shortQuantity"
            ),
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
            # A broker payload may contain stale/closed
            # zero-quantity positions. They have no effect
            # on the managed portfolio.
            continue

        if symbol not in FROZEN_ASSETS:
            raise ValueError(
                "PaperMoney account contains "
                "a non-zero unsupported position: "
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
                "Fractional positions are not "
                "supported by Meta Allocation V1. "
                f"Symbol: {symbol}, "
                f"quantity: {long_quantity}"
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


def _extract_quote_map(
    quote_payload: Mapping,
) -> Mapping:
    quote_payload = _require_mapping(
        quote_payload,
        "quote_payload",
    )

    if (
        "quotes"
        in quote_payload
    ):
        return _require_mapping(
            quote_payload[
                "quotes"
            ],
            "quotes",
        )

    return quote_payload


def _extract_quote_body(
    quote_entry: Mapping,
    symbol: str,
) -> Mapping:
    quote_entry = _require_mapping(
        quote_entry,
        f"quote[{symbol}]",
    )

    if (
        "quote"
        in quote_entry
    ):
        return _require_mapping(
            quote_entry[
                "quote"
            ],
            f"quote[{symbol}].quote",
        )

    return quote_entry


def extract_reference_prices(
    quote_payload: Mapping,
    snapshot_timestamp: (
        str
        | pd.Timestamp
    ),
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> tuple[
    pd.Series,
    pd.Timestamp,
]:
    """
    Extract last prices for the five frozen assets.

    price_as_of is the OLDEST quote timestamp among the
    five strategy assets. This gives the downstream
    freshness check a conservative timestamp.
    """
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

    quotes = (
        _extract_quote_map(
            quote_payload
        )
    )

    prices = {}
    quote_times = {}

    for symbol in FROZEN_ASSETS:
        if symbol not in quotes:
            raise ValueError(
                "Missing required quote for: "
                f"{symbol}"
            )

        quote = (
            _extract_quote_body(
                quotes[
                    symbol
                ],
                symbol,
            )
        )

        if (
            "lastPrice"
            not in quote
        ):
            raise ValueError(
                f"Quote for {symbol} "
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
                f"Quote for {symbol} "
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
                "is later than the "
                "snapshot timestamp."
            )

        quote_age = (
            snapshot_timestamp
            - quote_time
        )

        if quote_age > max_price_age:
            raise ValueError(
                f"{symbol} quote is stale.\n"
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

    price_series = pd.Series(
        prices,
        index=FROZEN_ASSETS,
        dtype=float,
        name="reference_price",
    )

    price_as_of = min(
        quote_times.values()
    )

    return (
        price_series,
        price_as_of,
    )


def build_papermoney_snapshot(
    account_payload: Mapping,
    quote_payload: Mapping,
    snapshot_timestamp: (
        str
        | pd.Timestamp
    ),
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> PortfolioSnapshot:
    """
    Convert a Schwab-style paperMoney account/quote
    payload into the broker-independent
    PortfolioSnapshot used by Meta Allocation V1.

    This function:
        - does not authenticate,
        - does not connect to Schwab,
        - does not generate orders,
        - does not submit orders.
    """
    snapshot_timestamp = (
        _normalize_timestamp(
            snapshot_timestamp,
            "snapshot_timestamp",
        )
    )

    cash = extract_cash(
        account_payload
    )

    holdings = extract_holdings(
        account_payload
    )

    (
        reference_prices,
        price_as_of,
    ) = extract_reference_prices(
        quote_payload=
            quote_payload,

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
            PAPERMONEY_MODE,

        max_price_age=
            max_price_age,
    )