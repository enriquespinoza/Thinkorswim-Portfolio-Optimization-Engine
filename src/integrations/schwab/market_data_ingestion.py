from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from config.settings import RAW_DATA_DIR

from src.integrations.schwab.market_data_client import (
    SchwabMarketDataClient,
)


SCHWAB_RAW_DATA_DIR = (
    RAW_DATA_DIR
    / "schwab"
)

MANIFEST_FILENAME = (
    "manifest.jsonl"
)

RAW_SCHEMA_VERSION = "1.0"

DATASET_DIRECTORIES = {
    "quotes":
        "quotes",

    "price_history":
        "price_history",

    "option_chains":
        "option_chains",

    "expiration_chains":
        "expiration_chains",

    "movers":
        "movers",

    "market_hours":
        "market_hours",

    "instruments":
        "instruments",
}


@dataclass(frozen=True)
class RawMarketDataArtifact:
    """
    Metadata for one immutable raw Schwab
    market-data capture.
    """

    record_id: str
    dataset: str
    captured_at_utc: str
    relative_path: str
    payload_sha256: str
    request_sha256: str
    byte_size: int
    status: str


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _canonicalize(
    value: Any,
) -> Any:
    """
    Convert supported Python values into stable
    JSON-compatible values.

    This is used for request provenance and hashing.
    """
    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):
        return _canonicalize(
            value.value
        )

    if isinstance(
        value,
        datetime,
    ):
        if value.tzinfo is None:
            return (
                value
                .replace(
                    tzinfo=timezone.utc
                )
                .isoformat()
            )

        return (
            value
            .astimezone(
                timezone.utc
            )
            .isoformat()
        )

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key):
                _canonicalize(
                    item
                )
            for key, item
            in sorted(
                value.items(),
                key=lambda pair:
                    str(
                        pair[0]
                    ),
            )
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _canonicalize(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    raise TypeError(
        "Unsupported value in raw Schwab "
        "market-data provenance: "
        f"{type(value).__name__}"
    )


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    canonical = _canonicalize(
        value
    )

    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
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


def _slugify(
    value: str,
) -> str:
    candidate = (
        str(
            value
        )
        .strip()
        .upper()
    )

    candidate = re.sub(
        r"[^A-Z0-9._-]+",
        "-",
        candidate,
    )

    candidate = (
        candidate
        .strip(
            "-._"
        )
    )

    if not candidate:
        return "CAPTURE"

    return candidate[
        :80
    ]


def _normalize_timestamp(
    value: datetime,
) -> datetime:
    if not isinstance(
        value,
        datetime,
    ):
        raise TypeError(
            "clock must return datetime."
        )

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


class SchwabMarketDataIngestor:
    """
    Persist immutable Schwab Market Data Production
    payloads for research.

    This layer sits above SchwabMarketDataClient:

        authenticated schwab-py client
            -> SchwabMarketDataClient
            -> SchwabMarketDataIngestor
            -> data/raw/schwab/...

    The stored JSON envelope contains both the decoded
    provider payload and request provenance. The payload
    itself is not transformed into research features here.

    No account or order operations exist in this class.
    """

    def __init__(
        self,
        client: SchwabMarketDataClient,
        *,
        root_dir: Path = SCHWAB_RAW_DATA_DIR,
        clock: Callable[
            [],
            datetime,
        ] = _utc_now,
    ) -> None:
        if not isinstance(
            client,
            SchwabMarketDataClient,
        ):
            raise TypeError(
                "client must be SchwabMarketDataClient."
            )

        self._client = client
        self._root_dir = Path(
            root_dir
        )
        self._clock = clock

    @property
    def root_dir(
        self,
    ) -> Path:
        return self._root_dir

    @property
    def manifest_path(
        self,
    ) -> Path:
        return (
            self._root_dir
            / MANIFEST_FILENAME
        )

    def ensure_directories(
        self,
    ) -> None:
        self._root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for directory_name in (
            DATASET_DIRECTORIES
            .values()
        ):
            (
                self._root_dir
                / directory_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    def _manifest_contains(
        self,
        record_id: str,
    ) -> bool:
        if not self.manifest_path.exists():
            return False

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    row = json.loads(
                        stripped
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Raw Schwab manifest contains "
                        "invalid JSON on line "
                        f"{line_number}."
                    ) from exc

                if (
                    str(
                        row.get(
                            "record_id",
                            "",
                        )
                    )
                    == record_id
                ):
                    return True

        return False

    def _append_manifest(
        self,
        artifact: RawMarketDataArtifact,
    ) -> None:
        if self._manifest_contains(
            artifact.record_id
        ):
            return

        manifest_row = {
            "record_id":
                artifact.record_id,

            "dataset":
                artifact.dataset,

            "captured_at_utc":
                artifact.captured_at_utc,

            "relative_path":
                artifact.relative_path,

            "payload_sha256":
                artifact.payload_sha256,

            "request_sha256":
                artifact.request_sha256,

            "byte_size":
                artifact.byte_size,
        }

        encoded = (
            json.dumps(
                manifest_row,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )

        with self.manifest_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                encoded
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )

    def _persist(
        self,
        *,
        dataset: str,
        request: Mapping[
            str,
            Any,
        ],
        payload: Mapping[
            str,
            Any,
        ],
        identity: str,
    ) -> RawMarketDataArtifact:
        if dataset not in (
            DATASET_DIRECTORIES
        ):
            raise ValueError(
                "Unsupported Schwab raw dataset: "
                f"{dataset}"
            )

        if not isinstance(
            payload,
            Mapping,
        ):
            raise TypeError(
                "payload must be a mapping."
            )

        self.ensure_directories()

        captured_at = (
            _normalize_timestamp(
                self._clock()
            )
        )

        captured_at_utc = (
            captured_at.isoformat()
        )

        request_payload = {
            "dataset":
                dataset,

            "parameters":
                _canonicalize(
                    request
                ),
        }

        request_sha256 = (
            _sha256_json(
                request_payload
            )
        )

        payload_sha256 = (
            _sha256_json(
                payload
            )
        )

        identity_material = {
            "dataset":
                dataset,

            "captured_at_utc":
                captured_at_utc,

            "request_sha256":
                request_sha256,

            "payload_sha256":
                payload_sha256,
        }

        record_id = (
            _sha256_json(
                identity_material
            )
        )

        envelope = {
            "schema_version":
                RAW_SCHEMA_VERSION,

            "source":
                (
                    "Charles Schwab Trader API - "
                    "Individual / Market Data Production"
                ),

            "dataset":
                dataset,

            "captured_at_utc":
                captured_at_utc,

            "record_id":
                record_id,

            "request_sha256":
                request_sha256,

            "payload_sha256":
                payload_sha256,

            "request":
                request_payload,

            "payload":
                _canonicalize(
                    payload
                ),
        }

        encoded = (
            json.dumps(
                envelope,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(
            "utf-8"
        )

        timestamp_token = (
            captured_at
            .strftime(
                "%Y%m%dT%H%M%S.%fZ"
            )
        )

        filename = (
            f"{timestamp_token}"
            f"__{_slugify(identity)}"
            f"__{record_id[:12]}"
            ".json"
        )

        dataset_dir = (
            self._root_dir
            / DATASET_DIRECTORIES[
                dataset
            ]
        )

        path = (
            dataset_dir
            / filename
        )

        status = "WRITTEN"

        try:
            with path.open(
                "xb",
            ) as handle:
                handle.write(
                    encoded
                )
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

        except FileExistsError:
            existing = (
                path.read_bytes()
            )

            if existing != encoded:
                raise RuntimeError(
                    "Immutable Schwab raw-data "
                    "path already exists with "
                    "different content: "
                    f"{path}"
                )

            status = "EXISTS"

        relative_path = str(
            path.relative_to(
                self._root_dir
            )
        ).replace(
            "\\",
            "/",
        )

        artifact = (
            RawMarketDataArtifact(
                record_id=
                    record_id,

                dataset=
                    dataset,

                captured_at_utc=
                    captured_at_utc,

                relative_path=
                    relative_path,

                payload_sha256=
                    payload_sha256,

                request_sha256=
                    request_sha256,

                byte_size=
                    len(
                        encoded
                    ),

                status=
                    status,
            )
        )

        self._append_manifest(
            artifact
        )

        return artifact

    def ingest_quotes(
        self,
        symbols: Sequence[str] | str,
        *,
        fields: Sequence[Any] | Any | None = None,
        indicative: bool | None = None,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_quotes(
                symbols,
                fields=fields,
                indicative=indicative,
            )
        )

        if isinstance(
            symbols,
            str,
        ):
            identity = symbols
        else:
            identity = "-".join(
                str(symbol)
                for symbol in symbols
            )

        return self._persist(
            dataset="quotes",
            request={
                "symbols":
                    symbols,

                "fields":
                    fields,

                "indicative":
                    indicative,
            },
            payload=payload,
            identity=identity,
        )

    def ingest_price_history(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_price_history(
                symbol,
                **kwargs,
            )
        )

        return self._persist(
            dataset="price_history",
            request={
                "symbol":
                    symbol,

                **kwargs,
            },
            payload=payload,
            identity=symbol,
        )

    def ingest_option_chain(
        self,
        symbol: str,
        **kwargs: Any,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_option_chain(
                symbol,
                **kwargs,
            )
        )

        return self._persist(
            dataset="option_chains",
            request={
                "symbol":
                    symbol,

                **kwargs,
            },
            payload=payload,
            identity=symbol,
        )

    def ingest_option_expiration_chain(
        self,
        symbol: str,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_option_expiration_chain(
                symbol
            )
        )

        return self._persist(
            dataset="expiration_chains",
            request={
                "symbol":
                    symbol,
            },
            payload=payload,
            identity=symbol,
        )

    def ingest_movers(
        self,
        index: Any,
        **kwargs: Any,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_movers(
                index,
                **kwargs,
            )
        )

        return self._persist(
            dataset="movers",
            request={
                "index":
                    index,

                **kwargs,
            },
            payload=payload,
            identity=str(
                getattr(
                    index,
                    "value",
                    index,
                )
            ),
        )

    def ingest_market_hours(
        self,
        markets: Sequence[Any] | Any,
        *,
        date_value: date | None = None,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_market_hours(
                markets,
                date_value=date_value,
            )
        )

        if isinstance(
            markets,
            str,
        ):
            identity = markets
        else:
            try:
                identity = "-".join(
                    str(
                        getattr(
                            market,
                            "value",
                            market,
                        )
                    )
                    for market
                    in markets
                )
            except TypeError:
                identity = str(
                    getattr(
                        markets,
                        "value",
                        markets,
                    )
                )

        return self._persist(
            dataset="market_hours",
            request={
                "markets":
                    markets,

                "date_value":
                    date_value,
            },
            payload=payload,
            identity=identity,
        )

    def ingest_instruments(
        self,
        symbols: Sequence[str] | str,
        *,
        projection: Any,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_instruments(
                symbols,
                projection=projection,
            )
        )

        if isinstance(
            symbols,
            str,
        ):
            identity = symbols
        else:
            identity = "-".join(
                str(symbol)
                for symbol in symbols
            )

        return self._persist(
            dataset="instruments",
            request={
                "symbols":
                    symbols,

                "projection":
                    projection,
            },
            payload=payload,
            identity=identity,
        )

    def ingest_instrument_by_cusip(
        self,
        cusip: str,
    ) -> RawMarketDataArtifact:
        payload = (
            self._client
            .get_instrument_by_cusip(
                cusip
            )
        )

        return self._persist(
            dataset="instruments",
            request={
                "cusip":
                    cusip,
            },
            payload=payload,
            identity=cusip,
        )
