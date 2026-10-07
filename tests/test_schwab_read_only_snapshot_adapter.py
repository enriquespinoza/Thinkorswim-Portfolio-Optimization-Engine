import pandas as pd
import pytest

from src.integrations.schwab.read_only_client import (
    SchwabReadOnlyPayloads,
)

from src.integrations.schwab.read_only_snapshot_adapter import (
    build_schwab_read_only_snapshot,
)


def quote_time_ms(
    value: str,
) -> int:
    return int(
        pd.Timestamp(
            value
        ).timestamp()
        * 1000
    )


def sample_account_payload():
    return {
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
    timestamp = quote_time_ms(
        "2026-11-02T14:29:00+00:00"
    )

    return {
        "SPY": {
            "quote": {
                "lastPrice": 500.0,
                "quoteTime": timestamp,
            },
        },

        "QQQ": {
            "quote": {
                "lastPrice": 600.0,
                "quoteTime": timestamp,
            },
        },

        "TLT": {
            "quote": {
                "lastPrice": 100.0,
                "quoteTime": timestamp,
            },
        },

        "GLD": {
            "quote": {
                "lastPrice": 200.0,
                "quoteTime": timestamp,
            },
        },

        "SCHD": {
            "quote": {
                "lastPrice": 80.0,
                "quoteTime": timestamp,
            },
        },
    }


def sample_payload_bundle():
    return SchwabReadOnlyPayloads(
        account_hash=
            "HASH-ONE",

        account_payload=
            sample_account_payload(),

        quote_payload=
            sample_quote_payload(),

        fetched_at_utc=
            pd.Timestamp(
                "2026-11-02T14:30:00+00:00"
            ),
    )


def test_builds_schwab_read_only_snapshot():
    snapshot = (
        build_schwab_read_only_snapshot(
            sample_payload_bundle()
        )
    )

    assert (
        snapshot.account_mode
        == "SCHWAB_READ_ONLY"
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
        snapshot.portfolio_value
        == 100_000.0
    )


def test_snapshot_hash_is_deterministic():
    first = (
        build_schwab_read_only_snapshot(
            sample_payload_bundle()
        )
    )

    second = (
        build_schwab_read_only_snapshot(
            sample_payload_bundle()
        )
    )

    assert (
        first.portfolio_snapshot_hash
        == second.portfolio_snapshot_hash
    )


def test_account_hash_not_propagated_to_snapshot():
    snapshot = (
        build_schwab_read_only_snapshot(
            sample_payload_bundle()
        )
    )

    assert not hasattr(
        snapshot,
        "account_hash",
    )


def test_missing_cash_rejected():
    payloads = (
        sample_payload_bundle()
    )

    del payloads.account_payload[
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
        build_schwab_read_only_snapshot(
            payloads
        )


def test_unsupported_position_rejected():
    payloads = (
        sample_payload_bundle()
    )

    payloads.account_payload[
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
        match="outside",
    ):
        build_schwab_read_only_snapshot(
            payloads
        )


def test_short_position_rejected():
    payloads = (
        sample_payload_bundle()
    )

    payloads.account_payload[
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
        build_schwab_read_only_snapshot(
            payloads
        )


def test_fractional_position_rejected():
    payloads = (
        sample_payload_bundle()
    )

    payloads.account_payload[
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
        build_schwab_read_only_snapshot(
            payloads
        )


def test_missing_quote_rejected():
    payloads = (
        sample_payload_bundle()
    )

    del payloads.quote_payload[
        "TLT"
    ]

    with pytest.raises(
        ValueError,
        match="TLT",
    ):
        build_schwab_read_only_snapshot(
            payloads
        )


def test_stale_quote_rejected():
    payloads = (
        sample_payload_bundle()
    )

    payloads.quote_payload[
        "SPY"
    ][
        "quote"
    ][
        "quoteTime"
    ] = quote_time_ms(
        "2026-10-20T20:00:00+00:00"
    )

    with pytest.raises(
        ValueError,
        match="stale",
    ):
        build_schwab_read_only_snapshot(
            payloads,
            max_price_age="7D",
        )


def test_future_quote_rejected():
    payloads = (
        sample_payload_bundle()
    )

    payloads.quote_payload[
        "SPY"
    ][
        "quote"
    ][
        "quoteTime"
    ] = quote_time_ms(
        "2026-11-02T14:31:00+00:00"
    )

    with pytest.raises(
        ValueError,
        match="later than",
    ):
        build_schwab_read_only_snapshot(
            payloads
        )


def test_oldest_quote_sets_price_as_of():
    payloads = (
        sample_payload_bundle()
    )

    payloads.quote_payload[
        "TLT"
    ][
        "quote"
    ][
        "quoteTime"
    ] = quote_time_ms(
        "2026-11-02T14:28:00+00:00"
    )

    snapshot = (
        build_schwab_read_only_snapshot(
            payloads
        )
    )

    assert (
        snapshot.price_as_of
        == pd.Timestamp(
            "2026-11-02T14:28:00+00:00"
        )
    )