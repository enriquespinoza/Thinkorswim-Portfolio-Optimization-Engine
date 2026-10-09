from datetime import date, datetime, timezone
import json
from pathlib import Path

import pytest

from src.integrations.schwab.market_data_client import (
    SchwabMarketDataClient,
)
from src.integrations.schwab.market_data_ingestion import (
    DATASET_DIRECTORIES,
    RAW_SCHEMA_VERSION,
    SchwabMarketDataIngestor,
)


FIXED_TIME = datetime(
    2026,
    10,
    9,
    16,
    30,
    tzinfo=timezone.utc,
)


class FakeMarketDataClient(
    SchwabMarketDataClient
):
    def __init__(self):
        super().__init__(
            object()
        )
        self.calls = []

    def get_quotes(
        self,
        symbols,
        *,
        fields=None,
        indicative=None,
    ):
        self.calls.append(
            (
                "quotes",
                symbols,
                fields,
                indicative,
            )
        )
        values = (
            [symbols]
            if isinstance(
                symbols,
                str,
            )
            else list(symbols)
        )
        return {
            str(symbol).upper(): {
                "quote": {
                    "lastPrice": 100.0,
                }
            }
            for symbol in values
        }

    def get_price_history(
        self,
        symbol,
        **kwargs,
    ):
        self.calls.append(
            (
                "price_history",
                symbol,
                kwargs,
            )
        )
        return {
            "symbol": str(
                symbol
            ).upper(),
            "candles": [
                {
                    "open": 99.0,
                    "high": 101.0,
                    "low": 98.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                    "datetime": 1,
                }
            ],
        }

    def get_option_chain(
        self,
        symbol,
        **kwargs,
    ):
        self.calls.append(
            (
                "option_chain",
                symbol,
                kwargs,
            )
        )
        return {
            "symbol": str(
                symbol
            ).upper(),
            "status": "SUCCESS",
        }

    def get_option_expiration_chain(
        self,
        symbol,
    ):
        self.calls.append(
            (
                "expiration_chain",
                symbol,
            )
        )
        return {
            "expirationList": [
                {
                    "expirationDate":
                        "2026-10-16",
                }
            ]
        }

    def get_movers(
        self,
        index,
        **kwargs,
    ):
        self.calls.append(
            (
                "movers",
                index,
                kwargs,
            )
        )
        return {
            "screeners": []
        }

    def get_market_hours(
        self,
        markets,
        *,
        date_value=None,
    ):
        self.calls.append(
            (
                "market_hours",
                markets,
                date_value,
            )
        )
        return {
            "equity": {
                "isOpen": True,
            }
        }

    def get_instruments(
        self,
        symbols,
        *,
        projection,
    ):
        self.calls.append(
            (
                "instruments",
                symbols,
                projection,
            )
        )
        return {
            "instruments": []
        }

    def get_instrument_by_cusip(
        self,
        cusip,
    ):
        self.calls.append(
            (
                "cusip",
                cusip,
            )
        )
        return {
            "instruments": [
                {
                    "cusip": cusip,
                }
            ]
        }


def make_ingestor(
    tmp_path: Path,
):
    return SchwabMarketDataIngestor(
        FakeMarketDataClient(),
        root_dir=(
            tmp_path
            / "schwab"
        ),
        clock=lambda:
            FIXED_TIME,
    )


def read_artifact(
    ingestor,
    artifact,
):
    path = (
        ingestor.root_dir
        / artifact.relative_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def test_ensure_directories_builds_expected_layout(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    ingestor.ensure_directories()

    assert ingestor.root_dir.is_dir()

    for directory in (
        DATASET_DIRECTORIES
        .values()
    ):
        assert (
            ingestor.root_dir
            / directory
        ).is_dir()


def test_quotes_are_persisted_as_raw_envelope(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    artifact = ingestor.ingest_quotes(
        [
            "SPY",
            "QQQ",
        ]
    )

    assert artifact.dataset == "quotes"
    assert artifact.status == "WRITTEN"
    assert (
        artifact.relative_path
        .startswith(
            "quotes/"
        )
    )

    envelope = read_artifact(
        ingestor,
        artifact,
    )

    assert (
        envelope[
            "schema_version"
        ]
        == RAW_SCHEMA_VERSION
    )
    assert (
        envelope[
            "dataset"
        ]
        == "quotes"
    )
    assert set(
        envelope[
            "payload"
        ]
    ) == {
        "SPY",
        "QQQ",
    }

    assert (
        envelope[
            "payload_sha256"
        ]
        == artifact.payload_sha256
    )
    assert (
        envelope[
            "request_sha256"
        ]
        == artifact.request_sha256
    )


def test_manifest_records_capture(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    artifact = (
        ingestor
        .ingest_price_history(
            "SPY",
            period_type="year",
            period=20,
        )
    )

    rows = [
        json.loads(line)
        for line in (
            ingestor
            .manifest_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]

    assert len(rows) == 1
    assert (
        rows[0][
            "record_id"
        ]
        == artifact.record_id
    )
    assert (
        rows[0][
            "dataset"
        ]
        == "price_history"
    )
    assert (
        rows[0][
            "relative_path"
        ]
        == artifact.relative_path
    )


def test_identical_capture_is_idempotent(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    first = ingestor.ingest_quotes(
        "SPY"
    )
    second = ingestor.ingest_quotes(
        "SPY"
    )

    assert (
        first.record_id
        == second.record_id
    )
    assert (
        second.status
        == "EXISTS"
    )

    files = list(
        (
            ingestor.root_dir
            / "quotes"
        ).glob(
            "*.json"
        )
    )

    assert len(files) == 1

    manifest_lines = (
        ingestor.manifest_path
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


@pytest.mark.parametrize(
    (
        "method_name",
        "expected_dataset",
        "args",
        "kwargs",
    ),
    [
        (
            "ingest_price_history",
            "price_history",
            ("SPY",),
            {
                "period_type":
                    "year",
            },
        ),
        (
            "ingest_option_chain",
            "option_chains",
            ("SPY",),
            {
                "strike_count":
                    10,
            },
        ),
        (
            "ingest_option_expiration_chain",
            "expiration_chains",
            ("SPY",),
            {},
        ),
        (
            "ingest_movers",
            "movers",
            ("$SPX",),
            {
                "frequency":
                    5,
            },
        ),
        (
            "ingest_market_hours",
            "market_hours",
            ("equity",),
            {
                "date_value":
                    date(
                        2026,
                        10,
                        9,
                    ),
            },
        ),
        (
            "ingest_instruments",
            "instruments",
            ("SPY",),
            {
                "projection":
                    "fundamental",
            },
        ),
        (
            "ingest_instrument_by_cusip",
            "instruments",
            ("037833100",),
            {},
        ),
    ],
)
def test_each_market_data_family_routes_to_expected_dataset(
    tmp_path: Path,
    method_name,
    expected_dataset,
    args,
    kwargs,
):
    ingestor = make_ingestor(
        tmp_path
    )

    method = getattr(
        ingestor,
        method_name,
    )

    artifact = method(
        *args,
        **kwargs,
    )

    assert (
        artifact.dataset
        == expected_dataset
    )

    assert (
        ingestor.root_dir
        / artifact.relative_path
    ).is_file()


def test_capture_timestamp_is_utc_and_stable(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    artifact = ingestor.ingest_quotes(
        "SPY"
    )

    assert (
        artifact.captured_at_utc
        == FIXED_TIME.isoformat()
    )

    assert (
        "20261009T163000.000000Z"
        in artifact.relative_path
    )


def test_raw_payload_is_not_feature_engineered(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    artifact = (
        ingestor
        .ingest_price_history(
            "SPY"
        )
    )

    envelope = read_artifact(
        ingestor,
        artifact,
    )

    candle = (
        envelope[
            "payload"
        ][
            "candles"
        ][0]
    )

    assert candle == {
        "close": 100.0,
        "datetime": 1,
        "high": 101.0,
        "low": 98.0,
        "open": 99.0,
        "volume": 1_000_000,
    }


def test_ingestor_requires_market_data_client(
    tmp_path: Path,
):
    with pytest.raises(
        TypeError,
        match="SchwabMarketDataClient",
    ):
        SchwabMarketDataIngestor(
            object(),
            root_dir=tmp_path,
        )


def test_no_account_or_order_operations_are_exposed(
    tmp_path: Path,
):
    ingestor = make_ingestor(
        tmp_path
    )

    forbidden = (
        "get_account",
        "get_account_numbers",
        "place_order",
        "replace_order",
        "cancel_order",
        "preview_order",
    )

    for method_name in forbidden:
        assert not hasattr(
            ingestor,
            method_name,
        )
