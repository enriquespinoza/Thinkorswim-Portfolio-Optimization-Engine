from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from config.settings import (
    DEFAULT_UNIVERSE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.integrations.schwab.auth_bootstrap import (
    create_market_data_schwab_client,
    load_schwab_auth_config,
)
from src.integrations.schwab.market_data_ingestion import (
    RawMarketDataArtifact,
    SchwabMarketDataIngestor,
)
from src.integrations.schwab.market_data_normalization import (
    NormalizedMarketDataArtifact,
    SchwabMarketDataNormalizer,
)


DEFAULT_DATASETS = (
    "quotes",
    "price_history",
    "option_chains",
    "expiration_chains",
    "instruments",
    "market_hours",
)

SUPPORTED_DATASETS = (
    "quotes",
    "price_history",
    "option_chains",
    "expiration_chains",
    "instruments",
    "market_hours",
    "movers",
)

DEFAULT_OPTION_STRIKE_COUNT = 10
DEFAULT_MARKET = "EQUITY"

DEFAULT_RAW_ROOT = (
    RAW_DATA_DIR
    / "schwab"
)

DEFAULT_PROCESSED_ROOT = (
    PROCESSED_DATA_DIR
    / "schwab"
)

CAPTURE_RUNS_DIRECTORY = "capture_runs"
CAPTURE_RUN_SCHEMA_VERSION = "1.0.0"


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _normalize_symbols(
    symbols: Iterable[str],
) -> tuple[str, ...]:
    normalized: list[str] = []

    for symbol in symbols:
        value = str(
            symbol
        ).strip().upper()

        if not value:
            raise ValueError(
                "symbols cannot contain empty values."
            )

        if value not in normalized:
            normalized.append(
                value
            )

    if not normalized:
        raise ValueError(
            "At least one symbol is required."
        )

    return tuple(
        normalized
    )


def _normalize_datasets(
    datasets: Iterable[str],
) -> tuple[str, ...]:
    normalized: list[str] = []

    for dataset in datasets:
        value = str(
            dataset
        ).strip().lower()

        if value not in SUPPORTED_DATASETS:
            raise ValueError(
                "Unsupported dataset: "
                f"{dataset!r}. "
                f"Expected one of {SUPPORTED_DATASETS}."
            )

        if value not in normalized:
            normalized.append(
                value
            )

    if not normalized:
        raise ValueError(
            "At least one dataset is required."
        )

    return tuple(
        normalized
    )


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            value
        )
    ).hexdigest()


def _artifact_record(
    *,
    scope: str,
    raw: RawMarketDataArtifact,
    normalized: NormalizedMarketDataArtifact,
) -> dict[str, Any]:
    return {
        "scope":
            scope,

        "dataset":
            raw.dataset,

        "raw":
            asdict(
                raw
            ),

        "normalized":
            asdict(
                normalized
            ),
    }


def _normalize_raw_artifact(
    *,
    ingestor: SchwabMarketDataIngestor,
    normalizer: SchwabMarketDataNormalizer,
    artifact: RawMarketDataArtifact,
) -> NormalizedMarketDataArtifact:
    raw_path = (
        ingestor.root_dir
        / artifact.relative_path
    )

    return normalizer.normalize_file(
        raw_path
    )


def _capture_one(
    *,
    ingestor: SchwabMarketDataIngestor,
    normalizer: SchwabMarketDataNormalizer,
    scope: str,
    raw_artifact: RawMarketDataArtifact,
) -> dict[str, Any]:
    normalized_artifact = (
        _normalize_raw_artifact(
            ingestor=
                ingestor,

            normalizer=
                normalizer,

            artifact=
                raw_artifact,
        )
    )

    return _artifact_record(
        scope=
            scope,

        raw=
            raw_artifact,

        normalized=
            normalized_artifact,
    )


def _write_run_summary(
    *,
    processed_root: Path,
    summary: dict[str, Any],
) -> Path:
    run_dir = (
        Path(
            processed_root
        )
        / CAPTURE_RUNS_DIRECTORY
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = datetime.fromisoformat(
        str(
            summary[
                "started_at_utc"
            ]
        )
    )

    token = started_at.astimezone(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )

    path = (
        run_dir
        / (
            f"{token}"
            f"__{summary['run_id'][:12]}"
            ".json"
        )
    )

    encoded = (
        json.dumps(
            summary,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    try:
        with path.open(
            "xb"
        ) as handle:
            handle.write(
                encoded
            )

    except FileExistsError:
        existing = (
            path.read_bytes()
        )

        if existing != encoded:
            raise RuntimeError(
                "Capture-run summary path already exists "
                "with different content: "
                f"{path}"
            )

    return path


def capture_schwab_market_data(
    *,
    client,
    symbols: Iterable[str] = DEFAULT_UNIVERSE,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    raw_root: Path = DEFAULT_RAW_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    option_strike_count: int = DEFAULT_OPTION_STRIKE_COUNT,
    market: str = DEFAULT_MARKET,
    movers_index: str | None = None,
    clock=_utc_now,
) -> dict[str, Any]:
    """
    Execute one permanent research-data capture.

    The function performs read-only Schwab Market Data calls,
    persists immutable raw JSON, immediately normalizes each
    artifact to a deterministic CSV table, and writes one run
    summary tying the artifacts together.

    It never calls Schwab account or trading endpoints.
    """
    symbols = _normalize_symbols(
        symbols
    )

    datasets = _normalize_datasets(
        datasets
    )

    if (
        not isinstance(
            option_strike_count,
            int,
        )
        or option_strike_count <= 0
    ):
        raise ValueError(
            "option_strike_count must be a "
            "positive integer."
        )

    market = str(
        market
    ).strip().upper()

    if not market:
        raise ValueError(
            "market cannot be empty."
        )

    if (
        "movers"
        in datasets
        and not movers_index
    ):
        raise ValueError(
            "movers_index is required when the "
            "movers dataset is selected."
        )

    started_at = clock()

    if (
        not isinstance(
            started_at,
            datetime,
        )
    ):
        raise TypeError(
            "clock must return datetime."
        )

    if started_at.tzinfo is None:
        started_at = (
            started_at.replace(
                tzinfo=timezone.utc
            )
        )

    started_at = (
        started_at.astimezone(
            timezone.utc
        )
    )

    ingestor = (
        SchwabMarketDataIngestor(
            client,
            root_dir=
                Path(
                    raw_root
                ),
        )
    )

    normalizer = (
        SchwabMarketDataNormalizer(
            raw_root=
                Path(
                    raw_root
                ),

            processed_root=
                Path(
                    processed_root
                ),
        )
    )

    artifacts: list[
        dict[
            str,
            Any,
        ]
    ] = []

    if "quotes" in datasets:
        raw = (
            ingestor.ingest_quotes(
                symbols
            )
        )

        artifacts.append(
            _capture_one(
                ingestor=
                    ingestor,

                normalizer=
                    normalizer,

                scope=
                    ",".join(
                        symbols
                    ),

                raw_artifact=
                    raw,
            )
        )

    if "price_history" in datasets:
        for symbol in symbols:
            raw = (
                ingestor.ingest_price_history(
                    symbol
                )
            )

            artifacts.append(
                _capture_one(
                    ingestor=
                        ingestor,

                    normalizer=
                        normalizer,

                    scope=
                        symbol,

                    raw_artifact=
                        raw,
                )
            )

    if "option_chains" in datasets:
        for symbol in symbols:
            raw = (
                ingestor.ingest_option_chain(
                    symbol,
                    strike_count=
                        option_strike_count,
                )
            )

            artifacts.append(
                _capture_one(
                    ingestor=
                        ingestor,

                    normalizer=
                        normalizer,

                    scope=
                        symbol,

                    raw_artifact=
                        raw,
                )
            )

    if "expiration_chains" in datasets:
        for symbol in symbols:
            raw = (
                ingestor
                .ingest_option_expiration_chain(
                    symbol
                )
            )

            artifacts.append(
                _capture_one(
                    ingestor=
                        ingestor,

                    normalizer=
                        normalizer,

                    scope=
                        symbol,

                    raw_artifact=
                        raw,
                )
            )

    if "instruments" in datasets:
        raw = (
            ingestor.ingest_instruments(
                symbols,
                projection=
                    "FUNDAMENTAL",
            )
        )

        artifacts.append(
            _capture_one(
                ingestor=
                    ingestor,

                normalizer=
                    normalizer,

                scope=
                    ",".join(
                        symbols
                    ),

                raw_artifact=
                    raw,
            )
        )

    if "market_hours" in datasets:
        raw = (
            ingestor.ingest_market_hours(
                market
            )
        )

        artifacts.append(
            _capture_one(
                ingestor=
                    ingestor,

                normalizer=
                    normalizer,

                scope=
                    market,

                raw_artifact=
                    raw,
            )
        )

    if "movers" in datasets:
        raw = (
            ingestor.ingest_movers(
                movers_index
            )
        )

        artifacts.append(
            _capture_one(
                ingestor=
                    ingestor,

                normalizer=
                    normalizer,

                scope=
                    str(
                        movers_index
                    ),

                raw_artifact=
                    raw,
            )
        )

    completed_at = clock()

    if (
        not isinstance(
            completed_at,
            datetime,
        )
    ):
        raise TypeError(
            "clock must return datetime."
        )

    if completed_at.tzinfo is None:
        completed_at = (
            completed_at.replace(
                tzinfo=timezone.utc
            )
        )

    completed_at = (
        completed_at.astimezone(
            timezone.utc
        )
    )

    run_identity = {
        "started_at_utc":
            started_at.isoformat(),

        "symbols":
            list(
                symbols
            ),

        "datasets":
            list(
                datasets
            ),

        "artifact_record_ids":
            [
                row[
                    "raw"
                ][
                    "record_id"
                ]
                for row
                in artifacts
            ],
    }

    run_id = (
        _sha256_json(
            run_identity
        )
    )

    summary = {
        "schema_version":
            CAPTURE_RUN_SCHEMA_VERSION,

        "run_id":
            run_id,

        "started_at_utc":
            started_at.isoformat(),

        "completed_at_utc":
            completed_at.isoformat(),

        "source":
            (
                "Charles Schwab Trader API - "
                "Individual / Market Data Production"
            ),

        "mode":
            "LIVE_READ_ONLY_RESEARCH_CAPTURE",

        "symbols":
            list(
                symbols
            ),

        "datasets":
            list(
                datasets
            ),

        "option_strike_count":
            option_strike_count,

        "market":
            market,

        "movers_index":
            movers_index,

        "artifact_count":
            len(
                artifacts
            ),

        "artifacts":
            artifacts,
    }

    summary_path = (
        _write_run_summary(
            processed_root=
                Path(
                    processed_root
                ),

            summary=
                summary,
        )
    )

    summary[
        "summary_path"
    ] = str(
        summary_path
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture read-only Schwab Market Data into "
            "the permanent raw and normalized research "
            "archives."
        )
    )

    parser.add_argument(
        "--execute-live",
        action="store_true",
        help=(
            "Actually authenticate and call Schwab. "
            "Without this flag the command prints the "
            "capture plan only."
        ),
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(
            DEFAULT_UNIVERSE
        ),
        help=(
            "Symbols to capture. Defaults to the project "
            "portfolio research universe."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=
            SUPPORTED_DATASETS,
        default=list(
            DEFAULT_DATASETS
        ),
        help=(
            "Dataset families to capture. Movers is "
            "opt-in because it is market/index scoped."
        ),
    )

    parser.add_argument(
        "--option-strike-count",
        type=int,
        default=
            DEFAULT_OPTION_STRIKE_COUNT,
        help=(
            "Number of option strikes above/below ATM "
            "requested from Schwab."
        ),
    )

    parser.add_argument(
        "--market",
        default=
            DEFAULT_MARKET,
        help=(
            "Market-hours market. Default: EQUITY."
        ),
    )

    parser.add_argument(
        "--movers-index",
        default=None,
        help=(
            "Schwab movers index/category, for example "
            "$SPX. Required only when movers is selected."
        ),
    )

    return parser


def _print_plan(
    *,
    symbols: tuple[str, ...],
    datasets: tuple[str, ...],
    option_strike_count: int,
    market: str,
    movers_index: str | None,
) -> None:
    print()
    print("=" * 88)
    print(
        "SCHWAB MARKET DATA RESEARCH CAPTURE PLAN"
    )
    print("=" * 88)
    print(
        "Mode: PLAN ONLY"
    )
    print(
        "Symbols:",
        ", ".join(
            symbols
        ),
    )
    print(
        "Datasets:",
        ", ".join(
            datasets
        ),
    )
    print(
        "Option strike count:",
        option_strike_count,
    )
    print(
        "Market:",
        market,
    )
    print(
        "Movers index:",
        movers_index,
    )
    print(
        "Raw archive:",
        DEFAULT_RAW_ROOT,
    )
    print(
        "Processed archive:",
        DEFAULT_PROCESSED_ROOT,
    )
    print()
    print(
        "No authentication, network calls, or files "
        "were created."
    )
    print(
        "Add --execute-live to perform the capture."
    )
    print("=" * 88)


def main() -> None:
    parser = (
        build_parser()
    )

    args = (
        parser.parse_args()
    )

    symbols = _normalize_symbols(
        args.symbols
    )

    datasets = _normalize_datasets(
        args.datasets
    )

    if (
        "movers"
        in datasets
        and not args.movers_index
    ):
        parser.error(
            "--movers-index is required when movers "
            "is selected."
        )

    if not args.execute_live:
        _print_plan(
            symbols=
                symbols,

            datasets=
                datasets,

            option_strike_count=
                args.option_strike_count,

            market=
                args.market,

            movers_index=
                args.movers_index,
        )

        return

    config = (
        load_schwab_auth_config()
    )

    client = (
        create_market_data_schwab_client(
            config
        )
    )

    result = (
        capture_schwab_market_data(
            client=
                client,

            symbols=
                symbols,

            datasets=
                datasets,

            option_strike_count=
                args.option_strike_count,

            market=
                args.market,

            movers_index=
                args.movers_index,
        )
    )

    print()
    print("=" * 88)
    print(
        "SCHWAB MARKET DATA RESEARCH CAPTURE COMPLETE"
    )
    print("=" * 88)
    print(
        "Run ID:",
        result[
            "run_id"
        ],
    )
    print(
        "Artifacts:",
        result[
            "artifact_count"
        ],
    )
    print(
        "Started UTC:",
        result[
            "started_at_utc"
        ],
    )
    print(
        "Completed UTC:",
        result[
            "completed_at_utc"
        ],
    )
    print(
        "Summary:",
        result[
            "summary_path"
        ],
    )
    print()
    print(
        "Raw provider payloads were not printed."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
