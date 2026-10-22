from __future__ import annotations

import pandas as pd
import pytest

from src.research.schwab_personal_portfolio import (
    capture_account_by_index,
    normalize_account_positions,
)


def test_normalize_account_positions() -> None:
    payload = {
        "securitiesAccount": {
            "positions": [
                {
                    "longQuantity": 10,
                    "shortQuantity": 0,
                    "marketValue": 1250.0,
                    "instrument": {
                        "symbol": "aaa",
                        "assetType": "EQUITY",
                    },
                },
                {
                    "longQuantity": 2,
                    "shortQuantity": 0,
                    "marketValue": 500.0,
                    "instrument": {
                        "symbol": "BBB",
                        "assetType": "ETF",
                    },
                },
            ]
        }
    }

    frame = normalize_account_positions(payload)

    assert frame["symbol"].tolist() == ["AAA", "BBB"]
    assert frame["quantity"].tolist() == [10.0, 2.0]
    assert frame["market_value"].tolist() == [1250.0, 500.0]


class FakeClient:
    def get_account_number_records(self):
        return [
            {"accountNumber": "PRIVATE1", "hashValue": "HASH1"},
            {"accountNumber": "PRIVATE2", "hashValue": "HASH2"},
            {"accountNumber": "PRIVATE3", "hashValue": "HASH3"},
        ]

    def get_account_payload(self, account_hash):
        assert account_hash == "HASH2"
        return {
            "securitiesAccount": {
                "positions": [
                    {
                        "longQuantity": 5,
                        "shortQuantity": 0,
                        "marketValue": 750.0,
                        "instrument": {
                            "symbol": "XYZ",
                            "assetType": "EQUITY",
                        },
                    }
                ]
            }
        }


def test_capture_account_by_index_uses_ordinal_selection() -> None:
    capture = capture_account_by_index(
        FakeClient(),
        account_index=2,
    )

    assert capture.account_index == 2
    assert capture.positions["symbol"].tolist() == ["XYZ"]
    assert capture.positions["market_value"].sum() == 750.0


def test_capture_account_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        capture_account_by_index(
            FakeClient(),
            account_index=4,
        )
