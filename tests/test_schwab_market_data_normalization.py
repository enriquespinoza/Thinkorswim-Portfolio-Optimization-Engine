from pathlib import Path
import json

import pandas as pd
import pytest

from src.integrations.schwab.market_data_normalization import (
    NORMALIZED_SCHEMA_VERSION,
    SchwabMarketDataNormalizer,
)


def write_raw_artifact(
    path: Path,
    *,
    dataset: str,
    record_id: str,
    payload: dict,
):
    import hashlib

    def canonical_json_bytes(
        value,
    ):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    payload_sha = hashlib.sha256(
        canonical_json_bytes(
            payload
        )
    ).hexdigest()

    envelope = {
        "schema_version": "1.0",
        "source": "Charles Schwab Trader API - Individual / Market Data Production",
        "dataset": dataset,
        "captured_at_utc": "2026-10-09T16:30:00+00:00",
        "record_id": record_id,
        "request_sha256": "REQHASH",
        "payload_sha256": payload_sha,
        "request": {
            "dataset": dataset,
            "parameters": {},
        },
        "payload": payload,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            envelope,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return envelope


def make_normalizer(
    tmp_path: Path,
):
    return SchwabMarketDataNormalizer(
        raw_root=(
            tmp_path
            / "raw"
            / "schwab"
        ),
        processed_root=(
            tmp_path
            / "processed"
            / "schwab"
        ),
    )


def test_price_history_normalizes_to_one_row_per_candle(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "price_history"
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset="price_history",
        record_id="RID1",
        payload={
            "symbol": "SPY",
            "candles": [
                {
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1000,
                    "datetime": 1760000000000,
                },
                {
                    "open": 101.0,
                    "high": 103.0,
                    "low": 100.0,
                    "close": 102.0,
                    "volume": 1200,
                    "datetime": 1760086400000,
                },
            ],
        },
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    assert artifact.dataset == "price_history"
    assert artifact.row_count == 2
    assert artifact.status == "WRITTEN"

    frame = pd.read_csv(
        normalizer.processed_root
        / artifact.relative_path
    )

    assert list(
        frame[
            "symbol"
        ]
    ) == [
        "SPY",
        "SPY",
    ]

    assert list(
        frame[
            "candle_index"
        ]
    ) == [
        0,
        1,
    ]

    assert "datetime_utc" in frame.columns
    assert (
        frame[
            "raw_record_id"
        ]
        .nunique()
        == 1
    )


def test_quotes_flatten_nested_sections(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "quotes"
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset="quotes",
        record_id="RID2",
        payload={
            "SPY": {
                "quote": {
                    "lastPrice": 500.0,
                    "bidPrice": 499.9,
                },
                "fundamental": {
                    "peRatio": 25.0,
                },
            }
        },
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    frame = pd.read_csv(
        normalizer.processed_root
        / artifact.relative_path
    )

    assert artifact.row_count == 1
    assert frame.loc[
        0,
        "symbol",
    ] == "SPY"

    assert frame.loc[
        0,
        "quote__lastPrice",
    ] == 500.0

    assert frame.loc[
        0,
        "fundamental__peRatio",
    ] == 25.0


def test_option_chain_explodes_contracts_by_expiration_and_strike(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "option_chains"
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset="option_chains",
        record_id="RID3",
        payload={
            "symbol": "SPY",
            "callExpDateMap": {
                "2026-10-16:7": {
                    "500.0": [
                        {
                            "symbol": "SPY_CALL",
                            "bid": 5.0,
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2026-10-16:7": {
                    "500.0": [
                        {
                            "symbol": "SPY_PUT",
                            "bid": 4.5,
                        }
                    ]
                }
            },
        },
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    frame = pd.read_csv(
        normalizer.processed_root
        / artifact.relative_path
    )

    assert artifact.row_count == 2
    assert set(
        frame[
            "put_call"
        ]
    ) == {
        "CALL",
        "PUT",
    }

    assert set(
        frame[
            "expiration_dte"
        ]
    ) == {
        7,
    }

    assert set(
        frame[
            "strike"
        ]
    ) == {
        500.0,
    }


@pytest.mark.parametrize(
    (
        "dataset",
        "payload",
        "expected_rows",
    ),
    [
        (
            "expiration_chains",
            {
                "expirationList": [
                    {
                        "expirationDate":
                            "2026-10-16",
                    },
                    {
                        "expirationDate":
                            "2026-10-23",
                    },
                ]
            },
            2,
        ),
        (
            "movers",
            {
                "screeners": [
                    {
                        "symbol": "AAA",
                    },
                    {
                        "symbol": "BBB",
                    },
                ]
            },
            2,
        ),
        (
            "instruments",
            {
                "instruments": [
                    {
                        "symbol": "SPY",
                    }
                ]
            },
            1,
        ),
    ],
)
def test_list_based_datasets_normalize(
    tmp_path: Path,
    dataset,
    payload,
    expected_rows,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / dataset
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset=dataset,
        record_id=(
            "RID-"
            + dataset
        ),
        payload=payload,
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    assert (
        artifact.row_count
        == expected_rows
    )


def test_market_hours_explodes_session_intervals(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "market_hours"
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset="market_hours",
        record_id="RID4",
        payload={
            "equity": {
                "EQ": {
                    "date": "2026-10-09",
                    "isOpen": True,
                    "sessionHours": {
                        "regularMarket": [
                            {
                                "start":
                                    "2026-10-09T09:30:00-04:00",
                                "end":
                                    "2026-10-09T16:00:00-04:00",
                            }
                        ]
                    },
                }
            }
        },
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    frame = pd.read_csv(
        normalizer.processed_root
        / artifact.relative_path
    )

    assert artifact.row_count == 1
    assert (
        frame.loc[
            0,
            "session_type",
        ]
        == "regularMarket"
    )
    assert (
        frame.loc[
            0,
            "session__start",
        ]
        == "2026-10-09T09:30:00-04:00"
    )


def test_normalization_is_idempotent(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "quotes"
        / "capture.json"
    )

    write_raw_artifact(
        raw_path,
        dataset="quotes",
        record_id="RID5",
        payload={
            "SPY": {
                "quote": {
                    "lastPrice": 500.0,
                }
            }
        },
    )

    first = normalizer.normalize_file(
        raw_path
    )
    second = normalizer.normalize_file(
        raw_path
    )

    assert first.normalized_sha256 == (
        second.normalized_sha256
    )
    assert second.status == "EXISTS"

    manifest_lines = (
        normalizer
        .normalization_manifest_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(
        [
            line
            for line
            in manifest_lines
            if line.strip()
        ]
    ) == 1


def test_tampered_raw_payload_is_rejected(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "quotes"
        / "capture.json"
    )

    envelope = write_raw_artifact(
        raw_path,
        dataset="quotes",
        record_id="RID6",
        payload={
            "SPY": {
                "quote": {
                    "lastPrice": 500.0,
                }
            }
        },
    )

    envelope[
        "payload"
    ][
        "SPY"
    ][
        "quote"
    ][
        "lastPrice"
    ] = 999.0

    raw_path.write_text(
        json.dumps(
            envelope,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="integrity",
    ):
        normalizer.normalize_file(
            raw_path
        )


def test_normalize_manifest_processes_raw_manifest_order(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    first_path = (
        normalizer.raw_root
        / "quotes"
        / "first.json"
    )
    second_path = (
        normalizer.raw_root
        / "price_history"
        / "second.json"
    )

    first = write_raw_artifact(
        first_path,
        dataset="quotes",
        record_id="RID7",
        payload={
            "SPY": {
                "quote": {
                    "lastPrice": 500.0,
                }
            }
        },
    )

    second = write_raw_artifact(
        second_path,
        dataset="price_history",
        record_id="RID8",
        payload={
            "symbol": "SPY",
            "candles": [],
        },
    )

    normalizer.raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = [
        {
            "record_id":
                first[
                    "record_id"
                ],
            "relative_path":
                "quotes/first.json",
        },
        {
            "record_id":
                second[
                    "record_id"
                ],
            "relative_path":
                "price_history/second.json",
        },
    ]

    (
        normalizer
        .raw_manifest_path
        .write_text(
            "".join(
                json.dumps(row)
                + "\n"
                for row
                in manifest_rows
            ),
            encoding="utf-8",
        )
    )

    artifacts = (
        normalizer
        .normalize_manifest()
    )

    assert [
        artifact.raw_record_id
        for artifact
        in artifacts
    ] == [
        "RID7",
        "RID8",
    ]


def test_provenance_columns_are_added(
    tmp_path: Path,
):
    normalizer = make_normalizer(
        tmp_path
    )

    raw_path = (
        normalizer.raw_root
        / "quotes"
        / "capture.json"
    )

    envelope = write_raw_artifact(
        raw_path,
        dataset="quotes",
        record_id="RID9",
        payload={
            "SPY": {
                "quote": {
                    "lastPrice": 500.0,
                }
            }
        },
    )

    artifact = normalizer.normalize_file(
        raw_path
    )

    frame = pd.read_csv(
        normalizer.processed_root
        / artifact.relative_path
    )

    assert (
        frame.loc[
            0,
            "raw_record_id",
        ]
        == "RID9"
    )

    assert (
        frame.loc[
            0,
            "raw_payload_sha256",
        ]
        == envelope[
            "payload_sha256"
        ]
    )

    assert (
        frame.loc[
            0,
            "normalized_schema_version",
        ]
        == NORMALIZED_SCHEMA_VERSION
    )
