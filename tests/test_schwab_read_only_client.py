import pandas as pd
import pytest

from src.integrations.schwab.read_only_client import (
    FROZEN_ASSETS,
    SchwabReadOnlyClient,
    SchwabReadOnlyPayloads,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        status_code=200,
    ):
        self._payload = payload
        self.status_code = status_code

    def json(
        self,
    ):
        return self._payload


class FakeSchwabClient:
    class Account:
        class Fields:
            POSITIONS = "positions"

    def __init__(
        self,
    ):
        self.account_records = [
            {
                "accountNumber":
                    "11111111",

                "hashValue":
                    "HASH-ONE",
            }
        ]

        self.account_payloads = {
            "HASH-ONE": {
                "securitiesAccount": {
                    "currentBalances": {
                        "cashBalance":
                            20_000.0,
                    },

                    "positions": [],
                },
            },
        }

        self.quote_payload = {
            "SPY": {
                "quote": {
                    "lastPrice":
                        500.0,
                },
            },

            "QQQ": {
                "quote": {
                    "lastPrice":
                        600.0,
                },
            },

            "TLT": {
                "quote": {
                    "lastPrice":
                        100.0,
                },
            },

            "GLD": {
                "quote": {
                    "lastPrice":
                        200.0,
                },
            },

            "SCHD": {
                "quote": {
                    "lastPrice":
                        80.0,
                },
            },
        }

        self.account_status_code = 200
        self.quote_status_code = 200

        self.last_account_hash = None
        self.last_account_fields = None
        self.last_quote_symbols = None

    def get_account_numbers(
        self,
    ):
        return FakeResponse(
            self.account_records
        )

    def get_account(
        self,
        account_hash,
        fields=None,
    ):
        self.last_account_hash = (
            account_hash
        )

        self.last_account_fields = (
            fields
        )

        return FakeResponse(
            self.account_payloads.get(
                account_hash,
                {},
            ),
            status_code=
                self.account_status_code,
        )

    def get_quotes(
        self,
        symbols,
    ):
        self.last_quote_symbols = list(
            symbols
        )

        requested = {
            symbol:
                self.quote_payload[
                    symbol
                ]
            for symbol in symbols
            if symbol
            in self.quote_payload
        }

        return FakeResponse(
            requested,
            status_code=
                self.quote_status_code,
        )


def test_resolves_single_account_hash():
    broker = (
        FakeSchwabClient()
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    assert (
        client.resolve_account_hash()
        == "HASH-ONE"
    )


def test_multiple_accounts_require_explicit_selection():
    broker = (
        FakeSchwabClient()
    )

    broker.account_records.append(
        {
            "accountNumber":
                "22222222",

            "hashValue":
                "HASH-TWO",
        }
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    with pytest.raises(
        ValueError,
        match="Multiple",
    ):
        client.resolve_account_hash()


def test_explicit_account_selection():
    broker = (
        FakeSchwabClient()
    )

    broker.account_records.append(
        {
            "accountNumber":
                "22222222",

            "hashValue":
                "HASH-TWO",
        }
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    assert (
        client.resolve_account_hash(
            account_number=
                "22222222"
        )
        == "HASH-TWO"
    )


def test_account_request_includes_positions():
    broker = (
        FakeSchwabClient()
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    client.get_account_payload(
        "HASH-ONE"
    )

    assert (
        broker.last_account_hash
        == "HASH-ONE"
    )

    assert (
        broker.last_account_fields
        == [
            "positions"
        ]
    )


def test_quote_request_uses_frozen_assets():
    broker = (
        FakeSchwabClient()
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    client.get_quote_payload(
        FROZEN_ASSETS
    )

    assert (
        broker.last_quote_symbols
        == list(
            FROZEN_ASSETS
        )
    )


def test_failed_account_request_rejected():
    broker = (
        FakeSchwabClient()
    )

    broker.account_status_code = 500

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    with pytest.raises(
        RuntimeError,
        match="HTTP 500",
    ):
        client.get_account_payload(
            "HASH-ONE"
        )


def test_failed_quote_request_rejected():
    broker = (
        FakeSchwabClient()
    )

    broker.quote_status_code = 429

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    with pytest.raises(
        RuntimeError,
        match="HTTP 429",
    ):
        client.get_quote_payload(
            FROZEN_ASSETS
        )


def test_missing_quote_rejected():
    broker = (
        FakeSchwabClient()
    )

    del broker.quote_payload[
        "TLT"
    ]

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    with pytest.raises(
        RuntimeError,
        match="TLT",
    ):
        client.get_quote_payload(
            FROZEN_ASSETS
        )


def test_fetches_meta_allocation_payload_bundle():
    broker = (
        FakeSchwabClient()
    )

    client = (
        SchwabReadOnlyClient(
            broker
        )
    )

    result = (
        client
        .fetch_meta_allocation_v1_payloads()
    )

    assert isinstance(
        result,
        SchwabReadOnlyPayloads,
    )

    assert (
        result.account_hash
        == "HASH-ONE"
    )

    assert (
        "securitiesAccount"
        in result.account_payload
    )

    assert set(
        result.quote_payload
    ) == set(
        FROZEN_ASSETS
    )

    assert (
        result.fetched_at_utc.tzinfo
        is not None
    )


def test_wrapper_exposes_no_order_methods():
    client = (
        SchwabReadOnlyClient(
            FakeSchwabClient()
        )
    )

    assert not hasattr(
        client,
        "place_order",
    )

    assert not hasattr(
        client,
        "replace_order",
    )

    assert not hasattr(
        client,
        "cancel_order",
    )

    assert not hasattr(
        client,
        "preview_order",
    )