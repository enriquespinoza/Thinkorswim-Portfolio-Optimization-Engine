import pandas as pd
import pytest

from src.integrations.schwab.papermoney_snapshot_adapter import (
    build_papermoney_snapshot,
)


def sample_account_payload():
    return {
        "account_mode":
            "PAPERMONEY",

        "securitiesAccount": {
            "currentBalances": {
                "cashBalance":
                    20_000.0,
            },

            "positions": [
                {
                    "instrument": {
                        "symbol":
                            "SPY",
                    },
                    "longQuantity":
                        100.0,
                    "shortQuantity":
                        0.0,
                },
                {
                    "instrument": {
                        "symbol":
                            "QQQ",
                    },
                    "longQuantity":
                        50.0,
                    "shortQuantity":
                        0.0,
                },
            ],
        },
    }


def sample_quote_payload():
    quote_time = (
        "2026-11-02T14:29:00+00:00"
    )

    return {
        "SPY": {
            "quote": {
                "lastPrice":
                    500.0,
                "quoteTime":
                    quote_time,
            },
        },

        "QQQ": {
            "quote": {
                "lastPrice":
                    600.0,
                "quoteTime":
                    quote_time,
            },
        },

        "TLT": {
            "quote": {
                "lastPrice":
                    100.0,
                "quoteTime":
                    quote_time,
            },
        },

        "GLD": {
            "quote": {
                "lastPrice":
                    200.0,
                "quoteTime":
                    quote_time,
            },
        },

        "SCHD": {
            "quote": {
                "lastPrice":
                    80.0,
                "quoteTime":
                    quote_time,
            },
        },
    }


def build_sample_snapshot():
    return build_papermoney_snapshot(
        account_payload=
            sample_account_payload(),

        quote_payload=
            sample_quote_payload(),

        snapshot_timestamp=
            "2026-11-02T14:30:00+00:00",
    )


def test_builds_papermoney_snapshot():
    snapshot = (
        build_sample_snapshot()
    )

    assert (
        snapshot.account_mode
        == "PAPERMONEY"
    )

    assert (
        snapshot.cash
        == 20_000.0
    )

    assert (
        snapshot.current_shares[
            "SPY"
        ]
        == 100
    )

    assert (
        snapshot.current_shares[
            "QQQ"
        ]
        == 50
    )

    assert (
        snapshot.current_shares[
            "TLT"
        ]
        == 0
    )

    assert (
        snapshot.portfolio_value
        == 100_000.0
    )


def test_account_mode_must_be_papermoney():
    account = (
        sample_account_payload()
    )

    account[
        "account_mode"
    ] = "LIVE"

    with pytest.raises(
        ValueError,
        match="PAPERMONEY",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_missing_cash_balance_rejected():
    account = (
        sample_account_payload()
    )

    del account[
        "securitiesAccount"
    ][
        "currentBalances"
    ][
        "cashBalance"
    ]

    with pytest.raises(
        ValueError,
        match="cashBalance",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_unsupported_nonzero_position_rejected():
    account = (
        sample_account_payload()
    )

    account[
        "securitiesAccount"
    ][
        "positions"
    ].append(
        {
            "instrument": {
                "symbol":
                    "TMF",
            },
            "longQuantity":
                100.0,
            "shortQuantity":
                0.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_short_position_rejected():
    account = (
        sample_account_payload()
    )

    account[
        "securitiesAccount"
    ][
        "positions"
    ][0][
        "shortQuantity"
    ] = 1.0

    with pytest.raises(
        ValueError,
        match="Short positions",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_fractional_position_rejected():
    account = (
        sample_account_payload()
    )

    account[
        "securitiesAccount"
    ][
        "positions"
    ][0][
        "longQuantity"
    ] = 100.5

    with pytest.raises(
        ValueError,
        match="Fractional",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_duplicate_position_rejected():
    account = (
        sample_account_payload()
    )

    account[
        "securitiesAccount"
    ][
        "positions"
    ].append(
        {
            "instrument": {
                "symbol":
                    "SPY",
            },
            "longQuantity":
                1.0,
            "shortQuantity":
                0.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        build_papermoney_snapshot(
            account_payload=
                account,

            quote_payload=
                sample_quote_payload(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_missing_required_quote_rejected():
    quotes = (
        sample_quote_payload()
    )

    del quotes[
        "TLT"
    ]

    with pytest.raises(
        ValueError,
        match="TLT",
    ):
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_nonpositive_quote_rejected():
    quotes = (
        sample_quote_payload()
    )

    quotes[
        "SPY"
    ][
        "quote"
    ][
        "lastPrice"
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_stale_quote_rejected():
    quotes = (
        sample_quote_payload()
    )

    quotes[
        "SPY"
    ][
        "quote"
    ][
        "quoteTime"
    ] = (
        "2026-10-20T20:00:00+00:00"
    )

    with pytest.raises(
        ValueError,
        match="stale",
    ):
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            max_price_age=
                "7D",
        )


def test_future_quote_rejected():
    quotes = (
        sample_quote_payload()
    )

    quotes[
        "SPY"
    ][
        "quote"
    ][
        "quoteTime"
    ] = (
        "2026-11-02T14:31:00+00:00"
    )

    with pytest.raises(
        ValueError,
        match="later than",
    ):
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )


def test_snapshot_hash_is_deterministic():
    first = (
        build_sample_snapshot()
    )

    second = (
        build_sample_snapshot()
    )

    assert (
        first.portfolio_snapshot_hash
        == second.portfolio_snapshot_hash
    )

    assert (
        first.holdings_snapshot_hash
        == second.holdings_snapshot_hash
    )

    assert (
        first.price_snapshot_hash
        == second.price_snapshot_hash
    )


def test_extra_quote_symbol_is_ignored():
    quotes = (
        sample_quote_payload()
    )

    quotes[
        "TMF"
    ] = {
        "quote": {
            "lastPrice":
                25.0,
            "quoteTime":
                "2026-11-02T14:29:00+00:00",
        },
    }

    snapshot = (
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )
    )

    assert list(
        snapshot.reference_prices.index
    ) == [
        "SPY",
        "QQQ",
        "TLT",
        "GLD",
        "SCHD",
    ]


def test_epoch_millisecond_quote_time_supported():
    quotes = (
        sample_quote_payload()
    )

    epoch_ms = int(
        pd.Timestamp(
            "2026-11-02T14:29:00+00:00"
        ).timestamp()
        * 1000
    )

    for symbol in quotes:
        quotes[
            symbol
        ][
            "quote"
        ][
            "quoteTime"
        ] = epoch_ms

    snapshot = (
        build_papermoney_snapshot(
            account_payload=
                sample_account_payload(),

            quote_payload=
                quotes,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",
        )
    )

    assert (
        snapshot.price_as_of
        == pd.Timestamp(
            "2026-11-02T14:29:00+00:00"
        )
    )