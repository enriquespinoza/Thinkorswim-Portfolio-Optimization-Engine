from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.integrations.schwab.auth_bootstrap import (
    create_market_data_schwab_client,
    load_schwab_auth_config,
)
from src.integrations.schwab.market_data_ingestion import (
    SchwabMarketDataIngestor,
)
from src.integrations.schwab.market_data_normalization import (
    SchwabMarketDataNormalizer,
)


RUN_LIVE = (
    os.environ.get(
        "RUN_SCHWAB_LIVE_TESTS",
        "",
    ).strip()
    == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_LIVE,
    reason=(
        "Live Schwab Market Data tests are opt-in. "
        "Set RUN_SCHWAB_LIVE_TESTS=1 to enable."
    ),
)


@pytest.fixture(
    scope="session",
)
def live_market_data_client():
    """
    Create one authenticated Market Data client for the
    live-test session.

    The fixture reads the existing local Schwab OAuth
    configuration. It never prints credentials, tokens,
    account identifiers, or response bodies.
    """
    try:
        config = (
            load_schwab_auth_config()
        )

        return (
            create_market_data_schwab_client(
                config
            )
        )

    except Exception as exc:
        pytest.fail(
            "Could not initialize the live Schwab Market "
            "Data client. Verify the local .env values, "
            "OAuth callback configuration, token state, "
            "and Schwab app access. "
            f"Error type: {type(exc).__name__}"
        )


@pytest.fixture
def live_pipeline(
    live_market_data_client,
    tmp_path: Path,
):
    """
    Route live responses through temporary raw and processed
    directories so smoke tests never alter the permanent
    research archive.
    """
    raw_root = (
        tmp_path
        / "raw"
        / "schwab"
    )

    processed_root = (
        tmp_path
        / "processed"
        / "schwab"
    )

    ingestor = (
        SchwabMarketDataIngestor(
            live_market_data_client,
            root_dir=
                raw_root,
        )
    )

    normalizer = (
        SchwabMarketDataNormalizer(
            raw_root=
                raw_root,

            processed_root=
                processed_root,
        )
    )

    return (
        ingestor,
        normalizer,
    )


def _normalize_and_read(
    ingestor,
    normalizer,
    artifact,
) -> pd.DataFrame:
    raw_path = (
        ingestor.root_dir
        / artifact.relative_path
    )

    normalized = (
        normalizer.normalize_file(
            raw_path
        )
    )

    processed_path = (
        normalizer.processed_root
        / normalized.relative_path
    )

    assert raw_path.is_file()
    assert processed_path.is_file()

    assert (
        normalized.raw_record_id
        == artifact.record_id
    )

    assert (
        normalized.source_payload_sha256
        == artifact.payload_sha256
    )

    return pd.read_csv(
        processed_path
    )


def test_live_quote_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor.ingest_quotes(
            "SPY"
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert artifact.dataset == "quotes"
    assert not frame.empty
    assert "symbol" in frame.columns

    symbols = (
        frame[
            "symbol"
        ]
        .astype(str)
        .str.upper()
        .tolist()
    )

    assert "SPY" in symbols


def test_live_price_history_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor.ingest_price_history(
            "SPY"
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert (
        artifact.dataset
        == "price_history"
    )

    assert not frame.empty

    required = {
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "datetime",
    }

    assert required.issubset(
        frame.columns
    )


def test_live_option_chain_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor.ingest_option_chain(
            "SPY",
            strike_count=4,
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert (
        artifact.dataset
        == "option_chains"
    )

    assert not frame.empty

    required = {
        "underlying_symbol",
        "put_call",
        "expiration_date",
        "strike",
    }

    assert required.issubset(
        frame.columns
    )

    assert set(
        frame[
            "put_call"
        ]
        .dropna()
        .astype(str)
        .str.upper()
    ).issubset(
        {
            "CALL",
            "PUT",
        }
    )


def test_live_expiration_chain_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor
        .ingest_option_expiration_chain(
            "SPY"
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert (
        artifact.dataset
        == "expiration_chains"
    )

    assert not frame.empty

    assert (
        "expirationDate"
        in frame.columns
    )


def test_live_market_hours_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor.ingest_market_hours(
            "EQUITY"
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert (
        artifact.dataset
        == "market_hours"
    )

    assert not frame.empty

    assert (
        "market_group"
        in frame.columns
    )


def test_live_instrument_ingestion_and_normalization(
    live_pipeline,
):
    ingestor, normalizer = (
        live_pipeline
    )

    artifact = (
        ingestor.ingest_instruments(
            "SPY",
            projection=
                "FUNDAMENTAL",
        )
    )

    frame = _normalize_and_read(
        ingestor,
        normalizer,
        artifact,
    )

    assert (
        artifact.dataset
        == "instruments"
    )

    assert not frame.empty

    symbol_columns = [
        column
        for column in frame.columns
        if (
            column == "symbol"
            or column.endswith(
                "__symbol"
            )
        )
    ]

    assert symbol_columns


def test_live_pipeline_exposes_no_trading_surface(
    live_market_data_client,
):
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
            live_market_data_client,
            method_name,
        )
