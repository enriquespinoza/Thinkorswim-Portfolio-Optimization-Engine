from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


SCHWAB_RAW_DATA_DIR = (
    RAW_DATA_DIR
    / "schwab"
)

SCHWAB_PROCESSED_DATA_DIR = (
    PROCESSED_DATA_DIR
    / "schwab"
)

RAW_MANIFEST_FILENAME = "manifest.jsonl"
NORMALIZATION_MANIFEST_FILENAME = "manifest.jsonl"
NORMALIZED_SCHEMA_VERSION = "1.0"

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
class NormalizedMarketDataArtifact:
    """
    Metadata for one immutable normalized table
    derived from one raw Schwab capture.
    """

    dataset: str
    raw_record_id: str
    source_payload_sha256: str
    normalized_sha256: str
    relative_path: str
    row_count: int
    column_count: int
    status: str


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
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


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _flatten_mapping(
    value: Mapping[
        str,
        Any,
    ],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten nested mappings into stable column names.

    Lists are serialized to canonical JSON rather than
    exploded implicitly. Dataset-specific normalizers
    explicitly explode the record collections that define
    table grain.
    """
    flattened: dict[
        str,
        Any,
    ] = {}

    for key in sorted(
        value.keys(),
        key=str,
    ):
        item = value[
            key
        ]

        key_text = str(
            key
        )

        column = (
            f"{prefix}__{key_text}"
            if prefix
            else key_text
        )

        if isinstance(
            item,
            Mapping,
        ):
            flattened.update(
                _flatten_mapping(
                    item,
                    prefix=column,
                )
            )

        elif isinstance(
            item,
            (
                list,
                tuple,
            ),
        ):
            flattened[
                column
            ] = json.dumps(
                item,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                ensure_ascii=True,
                allow_nan=False,
            )

        else:
            flattened[
                column
            ] = item

    return flattened


def _require_mapping(
    value: Any,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise RuntimeError(
            f"{context} must be a mapping."
        )

    return value


def _require_sequence(
    value: Any,
    context: str,
) -> Sequence[Any]:
    if (
        isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
        or not isinstance(
            value,
            Sequence,
        )
    ):
        raise RuntimeError(
            f"{context} must be a sequence."
        )

    return value


def _load_raw_envelope(
    path: Path,
) -> dict[str, Any]:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw Schwab artifact not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            envelope = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Raw Schwab artifact contains invalid JSON: "
            f"{path}"
        ) from exc

    envelope = dict(
        _require_mapping(
            envelope,
            "Raw Schwab envelope",
        )
    )

    required = (
        "schema_version",
        "dataset",
        "captured_at_utc",
        "record_id",
        "request_sha256",
        "payload_sha256",
        "request",
        "payload",
    )

    missing = [
        field
        for field in required
        if field not in envelope
    ]

    if missing:
        raise RuntimeError(
            "Raw Schwab artifact is missing required "
            f"fields: {missing}"
        )

    dataset = str(
        envelope[
            "dataset"
        ]
    )

    if dataset not in DATASET_DIRECTORIES:
        raise RuntimeError(
            "Unsupported raw Schwab dataset: "
            f"{dataset}"
        )

    payload = _require_mapping(
        envelope[
            "payload"
        ],
        "Raw Schwab payload",
    )

    actual_payload_hash = (
        _sha256_json(
            payload
        )
    )

    expected_payload_hash = str(
        envelope[
            "payload_sha256"
        ]
    )

    if (
        actual_payload_hash
        != expected_payload_hash
    ):
        raise RuntimeError(
            "Raw Schwab payload integrity check failed."
        )

    return envelope


def _provenance_columns(
    envelope: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    return {
        "raw_record_id":
            str(
                envelope[
                    "record_id"
                ]
            ),

        "raw_dataset":
            str(
                envelope[
                    "dataset"
                ]
            ),

        "raw_captured_at_utc":
            str(
                envelope[
                    "captured_at_utc"
                ]
            ),

        "raw_request_sha256":
            str(
                envelope[
                    "request_sha256"
                ]
            ),

        "raw_payload_sha256":
            str(
                envelope[
                    "payload_sha256"
                ]
            ),

        "normalized_schema_version":
            NORMALIZED_SCHEMA_VERSION,
    }


def _attach_provenance(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    envelope: Mapping[
        str,
        Any,
    ],
) -> pd.DataFrame:
    provenance = (
        _provenance_columns(
            envelope
        )
    )

    combined = [
        {
            **provenance,
            **row,
        }
        for row in rows
    ]

    if not combined:
        return pd.DataFrame(
            columns=list(
                provenance.keys()
            )
        )

    frame = pd.DataFrame(
        combined
    )

    return frame.reindex(
        sorted(
            frame.columns
        ),
        axis=1,
    )


def _normalize_quotes(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for symbol in sorted(
        payload.keys(),
        key=str,
    ):
        quote_payload = (
            _require_mapping(
                payload[
                    symbol
                ],
                (
                    "Schwab quote payload for "
                    f"{symbol}"
                ),
            )
        )

        rows.append(
            {
                "symbol":
                    str(
                        symbol
                    ).upper(),

                **_flatten_mapping(
                    quote_payload
                ),
            }
        )

    return rows


def _epoch_millis_to_utc(
    value: Any,
) -> str | None:
    if value is None:
        return None

    try:
        numeric = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    try:
        timestamp = pd.to_datetime(
            numeric,
            unit="ms",
            utc=True,
        )
    except Exception:
        return None

    return timestamp.isoformat()


def _normalize_price_history(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    symbol = str(
        payload.get(
            "symbol",
            "",
        )
    ).upper()

    candles = payload.get(
        "candles",
        [],
    )

    candles = _require_sequence(
        candles,
        "Schwab price-history candles",
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, candle in enumerate(
        candles
    ):
        candle = _require_mapping(
            candle,
            (
                "Schwab price-history candle "
                f"{index}"
            ),
        )

        row = {
            "symbol":
                symbol,

            "candle_index":
                index,

            **_flatten_mapping(
                candle
            ),
        }

        if "datetime" in candle:
            row[
                "datetime_utc"
            ] = (
                _epoch_millis_to_utc(
                    candle[
                        "datetime"
                    ]
                )
            )

        rows.append(
            row
        )

    return rows


def _parse_expiration_key(
    value: str,
) -> tuple[
    str | None,
    int | None,
]:
    text = str(
        value
    )

    if ":" not in text:
        return (
            text or None,
            None,
        )

    date_text, dte_text = (
        text.split(
            ":",
            1,
        )
    )

    try:
        dte = int(
            dte_text
        )
    except (
        TypeError,
        ValueError,
    ):
        dte = None

    return (
        date_text or None,
        dte,
    )


def _parse_float(
    value: Any,
) -> float | None:
    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def _normalize_option_side(
    payload: Mapping[
        str,
        Any,
    ],
    *,
    map_key: str,
    put_call: str,
) -> list[dict[str, Any]]:
    expiration_map = payload.get(
        map_key,
        {},
    )

    if expiration_map is None:
        return []

    expiration_map = (
        _require_mapping(
            expiration_map,
            (
                "Schwab option-chain "
                f"{map_key}"
            ),
        )
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for expiration_key in sorted(
        expiration_map.keys(),
        key=str,
    ):
        strike_map = _require_mapping(
            expiration_map[
                expiration_key
            ],
            (
                "Schwab option-chain strike map "
                f"{expiration_key}"
            ),
        )

        (
            expiration_date,
            expiration_dte,
        ) = _parse_expiration_key(
            str(
                expiration_key
            )
        )

        for strike_key in sorted(
            strike_map.keys(),
            key=lambda item:
                (
                    _parse_float(
                        item
                    )
                    is None,
                    _parse_float(
                        item
                    )
                    or 0.0,
                    str(
                        item
                    ),
                ),
        ):
            contracts = (
                _require_sequence(
                    strike_map[
                        strike_key
                    ],
                    (
                        "Schwab option contracts "
                        f"{expiration_key} "
                        f"{strike_key}"
                    ),
                )
            )

            for contract_index, contract in enumerate(
                contracts
            ):
                contract = (
                    _require_mapping(
                        contract,
                        (
                            "Schwab option contract "
                            f"{expiration_key} "
                            f"{strike_key} "
                            f"{contract_index}"
                        ),
                    )
                )

                rows.append(
                    {
                        "underlying_symbol":
                            str(
                                payload.get(
                                    "symbol",
                                    "",
                                )
                            ).upper(),

                        "put_call":
                            put_call,

                        "expiration_key":
                            str(
                                expiration_key
                            ),

                        "expiration_date":
                            expiration_date,

                        "expiration_dte":
                            expiration_dte,

                        "strike_key":
                            str(
                                strike_key
                            ),

                        "strike":
                            _parse_float(
                                strike_key
                            ),

                        "contract_index":
                            contract_index,

                        **_flatten_mapping(
                            contract
                        ),
                    }
                )

    return rows


def _normalize_option_chains(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    rows = (
        _normalize_option_side(
            payload,
            map_key=
                "callExpDateMap",
            put_call=
                "CALL",
        )
    )

    rows.extend(
        _normalize_option_side(
            payload,
            map_key=
                "putExpDateMap",
            put_call=
                "PUT",
        )
    )

    return rows


def _normalize_expiration_chains(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    values = payload.get(
        "expirationList",
        [],
    )

    values = _require_sequence(
        values,
        "Schwab expirationList",
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, value in enumerate(
        values
    ):
        value = _require_mapping(
            value,
            (
                "Schwab expiration record "
                f"{index}"
            ),
        )

        rows.append(
            {
                "expiration_index":
                    index,

                **_flatten_mapping(
                    value
                ),
            }
        )

    return rows


def _normalize_movers(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    values = payload.get(
        "screeners",
        [],
    )

    values = _require_sequence(
        values,
        "Schwab movers screeners",
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, value in enumerate(
        values
    ):
        value = _require_mapping(
            value,
            (
                "Schwab mover record "
                f"{index}"
            ),
        )

        rows.append(
            {
                "mover_index":
                    index,

                **_flatten_mapping(
                    value
                ),
            }
        )

    return rows


def _market_hour_records(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[
    tuple[
        str,
        str,
        Mapping[
            str,
            Any,
        ],
    ]
]:
    records: list[
        tuple[
            str,
            str,
            Mapping[
                str,
                Any,
            ],
        ]
    ] = []

    for market_group in sorted(
        payload.keys(),
        key=str,
    ):
        group_value = payload[
            market_group
        ]

        if not isinstance(
            group_value,
            Mapping,
        ):
            continue

        if (
            "sessionHours"
            in group_value
            or "isOpen"
            in group_value
        ):
            records.append(
                (
                    str(
                        market_group
                    ),
                    str(
                        market_group
                    ),
                    group_value,
                )
            )
            continue

        for market_key in sorted(
            group_value.keys(),
            key=str,
        ):
            market_value = (
                group_value[
                    market_key
                ]
            )

            if isinstance(
                market_value,
                Mapping,
            ):
                records.append(
                    (
                        str(
                            market_group
                        ),
                        str(
                            market_key
                        ),
                        market_value,
                    )
                )

    return records


def _normalize_market_hours(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for (
        market_group,
        market_key,
        market_payload,
    ) in _market_hour_records(
        payload
    ):
        base = {
            "market_group":
                market_group,

            "market_key":
                market_key,
        }

        non_session_fields = {
            key:
                value
            for key, value
            in market_payload.items()
            if key
            != "sessionHours"
        }

        base.update(
            _flatten_mapping(
                non_session_fields
            )
        )

        sessions = market_payload.get(
            "sessionHours",
            {},
        )

        if not sessions:
            rows.append(
                base
            )
            continue

        sessions = _require_mapping(
            sessions,
            (
                "Schwab market-hours "
                "sessionHours"
            ),
        )

        emitted_session = False

        for session_type in sorted(
            sessions.keys(),
            key=str,
        ):
            intervals = sessions[
                session_type
            ]

            intervals = _require_sequence(
                intervals,
                (
                    "Schwab market-hours "
                    f"{session_type}"
                ),
            )

            for interval_index, interval in enumerate(
                intervals
            ):
                interval = (
                    _require_mapping(
                        interval,
                        (
                            "Schwab market-hours "
                            f"{session_type} interval "
                            f"{interval_index}"
                        ),
                    )
                )

                rows.append(
                    {
                        **base,

                        "session_type":
                            str(
                                session_type
                            ),

                        "session_index":
                            interval_index,

                        **_flatten_mapping(
                            interval,
                            prefix=
                                "session",
                        ),
                    }
                )

                emitted_session = True

        if not emitted_session:
            rows.append(
                base
            )

    return rows


def _normalize_instruments(
    payload: Mapping[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    values = payload.get(
        "instruments",
        [],
    )

    values = _require_sequence(
        values,
        "Schwab instruments",
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, value in enumerate(
        values
    ):
        value = _require_mapping(
            value,
            (
                "Schwab instrument record "
                f"{index}"
            ),
        )

        rows.append(
            {
                "instrument_index":
                    index,

                **_flatten_mapping(
                    value
                ),
            }
        )

    return rows


NORMALIZERS = {
    "quotes":
        _normalize_quotes,

    "price_history":
        _normalize_price_history,

    "option_chains":
        _normalize_option_chains,

    "expiration_chains":
        _normalize_expiration_chains,

    "movers":
        _normalize_movers,

    "market_hours":
        _normalize_market_hours,

    "instruments":
        _normalize_instruments,
}


class SchwabMarketDataNormalizer:
    """
    Convert immutable raw Schwab API captures into
    deterministic flat research tables.

    Pipeline boundary:

        data/raw/schwab/*.json
            -> normalization only
            -> data/processed/schwab/*.csv
            -> later feature engineering

    This class does not call Schwab APIs and does not
    calculate predictive or portfolio features.
    """

    def __init__(
        self,
        *,
        raw_root: Path = SCHWAB_RAW_DATA_DIR,
        processed_root: Path = SCHWAB_PROCESSED_DATA_DIR,
    ) -> None:
        self._raw_root = Path(
            raw_root
        )

        self._processed_root = Path(
            processed_root
        )

    @property
    def raw_root(
        self,
    ) -> Path:
        return self._raw_root

    @property
    def processed_root(
        self,
    ) -> Path:
        return self._processed_root

    @property
    def raw_manifest_path(
        self,
    ) -> Path:
        return (
            self._raw_root
            / RAW_MANIFEST_FILENAME
        )

    @property
    def normalization_manifest_path(
        self,
    ) -> Path:
        return (
            self._processed_root
            / NORMALIZATION_MANIFEST_FILENAME
        )

    def ensure_directories(
        self,
    ) -> None:
        self._processed_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for directory in (
            DATASET_DIRECTORIES
            .values()
        ):
            (
                self._processed_root
                / directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    def normalize_envelope(
        self,
        envelope: Mapping[
            str,
            Any,
        ],
    ) -> pd.DataFrame:
        envelope = dict(
            _require_mapping(
                envelope,
                "Raw Schwab envelope",
            )
        )

        dataset = str(
            envelope.get(
                "dataset",
                "",
            )
        )

        if dataset not in NORMALIZERS:
            raise RuntimeError(
                "Unsupported raw Schwab dataset: "
                f"{dataset}"
            )

        payload = _require_mapping(
            envelope.get(
                "payload",
            ),
            "Raw Schwab payload",
        )

        rows = NORMALIZERS[
            dataset
        ](
            payload
        )

        return _attach_provenance(
            rows,
            envelope,
        )

    def _csv_bytes(
        self,
        frame: pd.DataFrame,
    ) -> bytes:
        return (
            frame
            .to_csv(
                index=False,
                lineterminator="\n",
            )
            .encode(
                "utf-8"
            )
        )

    def _manifest_contains(
        self,
        raw_record_id: str,
    ) -> bool:
        path = (
            self
            .normalization_manifest_path
        )

        if not path.exists():
            return False

        with path.open(
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
                        "Schwab normalization manifest "
                        "contains invalid JSON on line "
                        f"{line_number}."
                    ) from exc

                if (
                    str(
                        row.get(
                            "raw_record_id",
                            "",
                        )
                    )
                    == raw_record_id
                ):
                    return True

        return False

    def _append_manifest(
        self,
        artifact: NormalizedMarketDataArtifact,
    ) -> None:
        if self._manifest_contains(
            artifact.raw_record_id
        ):
            return

        row = {
            "dataset":
                artifact.dataset,

            "raw_record_id":
                artifact.raw_record_id,

            "source_payload_sha256":
                artifact.source_payload_sha256,

            "normalized_sha256":
                artifact.normalized_sha256,

            "relative_path":
                artifact.relative_path,

            "row_count":
                artifact.row_count,

            "column_count":
                artifact.column_count,

            "normalized_schema_version":
                NORMALIZED_SCHEMA_VERSION,
        }

        encoded = (
            json.dumps(
                row,
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

        with (
            self
            .normalization_manifest_path
            .open(
                "a",
                encoding="utf-8",
                newline="\n",
            )
        ) as handle:
            handle.write(
                encoded
            )

    def normalize_file(
        self,
        raw_path: Path,
    ) -> NormalizedMarketDataArtifact:
        raw_path = Path(
            raw_path
        )

        envelope = _load_raw_envelope(
            raw_path
        )

        dataset = str(
            envelope[
                "dataset"
            ]
        )

        raw_record_id = str(
            envelope[
                "record_id"
            ]
        )

        if not raw_record_id:
            raise RuntimeError(
                "Raw Schwab record_id cannot be empty."
            )

        frame = self.normalize_envelope(
            envelope
        )

        encoded = self._csv_bytes(
            frame
        )

        normalized_sha256 = (
            _sha256_bytes(
                encoded
            )
        )

        self.ensure_directories()

        filename = (
            f"{raw_record_id}.csv"
        )

        output_path = (
            self._processed_root
            / DATASET_DIRECTORIES[
                dataset
            ]
            / filename
        )

        status = "WRITTEN"

        try:
            with output_path.open(
                "xb",
            ) as handle:
                handle.write(
                    encoded
                )

        except FileExistsError:
            existing = (
                output_path
                .read_bytes()
            )

            if existing != encoded:
                raise RuntimeError(
                    "Immutable normalized Schwab "
                    "table already exists with "
                    "different content: "
                    f"{output_path}"
                )

            status = "EXISTS"

        relative_path = str(
            output_path.relative_to(
                self._processed_root
            )
        ).replace(
            "\\",
            "/",
        )

        artifact = (
            NormalizedMarketDataArtifact(
                dataset=
                    dataset,

                raw_record_id=
                    raw_record_id,

                source_payload_sha256=
                    str(
                        envelope[
                            "payload_sha256"
                        ]
                    ),

                normalized_sha256=
                    normalized_sha256,

                relative_path=
                    relative_path,

                row_count=
                    int(
                        len(
                            frame
                        )
                    ),

                column_count=
                    int(
                        len(
                            frame.columns
                        )
                    ),

                status=
                    status,
            )
        )

        self._append_manifest(
            artifact
        )

        return artifact

    def normalize_manifest(
        self,
    ) -> list[
        NormalizedMarketDataArtifact
    ]:
        """
        Normalize every raw artifact referenced by the
        immutable raw-data manifest, in manifest order.
        """
        if not self.raw_manifest_path.exists():
            raise FileNotFoundError(
                "Raw Schwab manifest not found: "
                f"{self.raw_manifest_path}"
            )

        artifacts: list[
            NormalizedMarketDataArtifact
        ] = []

        with self.raw_manifest_path.open(
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

                row = _require_mapping(
                    row,
                    (
                        "Raw Schwab manifest row "
                        f"{line_number}"
                    ),
                )

                relative_path = str(
                    row.get(
                        "relative_path",
                        "",
                    )
                ).strip()

                if not relative_path:
                    raise RuntimeError(
                        "Raw Schwab manifest row "
                        f"{line_number} is missing "
                        "relative_path."
                    )

                raw_path = (
                    self._raw_root
                    / relative_path
                )

                artifact = (
                    self.normalize_file(
                        raw_path
                    )
                )

                expected_record_id = str(
                    row.get(
                        "record_id",
                        "",
                    )
                )

                if (
                    expected_record_id
                    and artifact.raw_record_id
                    != expected_record_id
                ):
                    raise RuntimeError(
                        "Raw Schwab manifest record_id "
                        "does not match artifact: "
                        f"{relative_path}"
                    )

                artifacts.append(
                    artifact
                )

        return artifacts
