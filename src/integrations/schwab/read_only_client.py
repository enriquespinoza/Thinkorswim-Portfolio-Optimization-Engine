from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from config.settings import DEFAULT_UNIVERSE


FROZEN_ASSETS = tuple(
    DEFAULT_UNIVERSE
)


@dataclass(frozen=True)
class SchwabReadOnlyPayloads:
    """
    Raw read-only Schwab data captured at one
    logical retrieval point.

    No order functionality exists in this object.
    """

    account_hash: str

    account_payload: dict[
        str,
        Any,
    ]

    quote_payload: dict[
        str,
        Any,
    ]

    fetched_at_utc: pd.Timestamp


def _require_success(
    response,
    context: str,
) -> None:
    status_code = getattr(
        response,
        "status_code",
        None,
    )

    if status_code is None:
        raise RuntimeError(
            f"{context} response has no "
            "status_code."
        )

    try:
        status_code = int(
            status_code
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"{context} returned an invalid "
            "status_code."
        ) from exc

    if not (
        200
        <= status_code
        < 300
    ):
        # Do not include response bodies here.
        # Broker responses may contain account data.
        raise RuntimeError(
            f"{context} failed with "
            f"HTTP {status_code}."
        )


def _response_json(
    response,
    context: str,
):
    _require_success(
        response,
        context,
    )

    try:
        return response.json()

    except Exception as exc:
        raise RuntimeError(
            f"{context} returned invalid JSON."
        ) from exc


def _require_mapping(
    value,
    context: str,
) -> Mapping:
    if not isinstance(
        value,
        Mapping,
    ):
        raise RuntimeError(
            f"{context} must be a mapping."
        )

    return value


class SchwabReadOnlyClient:
    """
    Narrow wrapper around an authenticated Schwab
    client.

    Exposed capabilities:

        - discover account hashes
        - retrieve balances/positions
        - retrieve quotes

    Deliberately absent:

        - place_order
        - replace_order
        - cancel_order
        - preview_order
        - any other write operation

    Authentication is intentionally handled outside
    this class.
    """

    def __init__(
        self,
        client,
    ) -> None:
        if client is None:
            raise ValueError(
                "client cannot be None."
            )

        self._client = client

    def _positions_field(
        self,
    ):
        account_namespace = getattr(
            self._client,
            "Account",
            None,
        )

        fields_namespace = getattr(
            account_namespace,
            "Fields",
            None,
        )

        positions_field = getattr(
            fields_namespace,
            "POSITIONS",
            None,
        )

        if positions_field is None:
            raise RuntimeError(
                "Authenticated Schwab client "
                "does not expose "
                "Account.Fields.POSITIONS."
            )

        return positions_field

    def get_account_number_records(
        self,
    ) -> list[dict[str, Any]]:
        """
        Retrieve Schwab's account-number-to-hash
        records.

        Downstream code should use the hash rather
        than raw account numbers.
        """
        response = (
            self._client
            .get_account_numbers()
        )

        payload = _response_json(
            response,
            "Schwab account-number request",
        )

        if not isinstance(
            payload,
            list,
        ):
            raise RuntimeError(
                "Schwab account-number response "
                "must be a list."
            )

        records = []

        for index, row in enumerate(
            payload
        ):
            row = _require_mapping(
                row,
                (
                    "Schwab account-number "
                    f"record {index}"
                ),
            )

            hash_value = str(
                row.get(
                    "hashValue",
                    "",
                )
            ).strip()

            if not hash_value:
                raise RuntimeError(
                    "Schwab account-number "
                    "record missing hashValue."
                )

            account_number = str(
                row.get(
                    "accountNumber",
                    "",
                )
            ).strip()

            records.append(
                {
                    "accountNumber":
                        account_number,

                    "hashValue":
                        hash_value,
                }
            )

        if not records:
            raise RuntimeError(
                "No Schwab accounts were "
                "returned."
            )

        return records

    def resolve_account_hash(
        self,
        account_number: str | None = None,
    ) -> str:
        """
        Resolve an account hash.

        If there is exactly one accessible account,
        account_number may be omitted.

        If there are multiple accounts, explicit
        selection is required.
        """
        records = (
            self.get_account_number_records()
        )

        if account_number is None:
            if len(
                records
            ) != 1:
                raise ValueError(
                    "Multiple Schwab accounts are "
                    "available. Explicit account "
                    "selection is required."
                )

            return str(
                records[0][
                    "hashValue"
                ]
            )

        account_number = str(
            account_number
        ).strip()

        if not account_number:
            raise ValueError(
                "account_number cannot be empty."
            )

        matches = [
            record
            for record in records
            if str(
                record[
                    "accountNumber"
                ]
            )
            == account_number
        ]

        if not matches:
            raise ValueError(
                "Requested Schwab account "
                "was not found."
            )

        if len(
            matches
        ) != 1:
            raise RuntimeError(
                "Schwab returned duplicate "
                "account-number records."
            )

        return str(
            matches[0][
                "hashValue"
            ]
        )

    def get_account_payload(
        self,
        account_hash: str,
    ) -> dict[str, Any]:
        """
        Retrieve balances plus positions for one
        account hash.
        """
        account_hash = str(
            account_hash
        ).strip()

        if not account_hash:
            raise ValueError(
                "account_hash cannot be empty."
            )

        response = (
            self._client
            .get_account(
                account_hash,

                fields=[
                    self._positions_field()
                ],
            )
        )

        payload = _response_json(
            response,
            "Schwab account request",
        )

        payload = _require_mapping(
            payload,
            "Schwab account payload",
        )

        return dict(
            payload
        )

    def get_quote_payload(
        self,
        symbols: Sequence[str],
    ) -> dict[str, Any]:
        """
        Retrieve quotes for strategy symbols.

        This method is read-only.
        """
        normalized = [
            str(
                symbol
            )
            .strip()
            .upper()
            for symbol in symbols
        ]

        if not normalized:
            raise ValueError(
                "symbols cannot be empty."
            )

        if any(
            not symbol
            for symbol in normalized
        ):
            raise ValueError(
                "symbols cannot contain "
                "empty values."
            )

        if len(
            normalized
        ) != len(
            set(
                normalized
            )
        ):
            raise ValueError(
                "symbols cannot contain "
                "duplicates."
            )

        unsupported = (
            set(
                normalized
            )
            - set(
                FROZEN_ASSETS
            )
        )

        if unsupported:
            raise ValueError(
                "Unsupported Meta Allocation V1 "
                "symbols requested: "
                f"{sorted(unsupported)}"
            )

        response = (
            self._client
            .get_quotes(
                normalized
            )
        )

        payload = _response_json(
            response,
            "Schwab quote request",
        )

        payload = _require_mapping(
            payload,
            "Schwab quote payload",
        )

        missing = (
            set(
                normalized
            )
            - set(
                str(
                    key
                ).upper()
                for key in payload.keys()
            )
        )

        if missing:
            raise RuntimeError(
                "Schwab quote response missing "
                "requested symbols: "
                f"{sorted(missing)}"
            )

        return dict(
            payload
        )

    def fetch_meta_allocation_v1_payloads(
        self,
        account_number: str | None = None,
    ) -> SchwabReadOnlyPayloads:
        """
        Retrieve exactly the raw account and quote
        payloads needed by Meta Allocation V1.

        No normalization into PortfolioSnapshot
        occurs here.

        No order generation occurs here.

        No broker writes occur here.
        """
        account_hash = (
            self.resolve_account_hash(
                account_number=
                    account_number
            )
        )

        account_payload = (
            self.get_account_payload(
                account_hash=
                    account_hash
            )
        )

        quote_payload = (
            self.get_quote_payload(
                symbols=
                    FROZEN_ASSETS
            )
        )

        return SchwabReadOnlyPayloads(
            account_hash=
                account_hash,

            account_payload=
                account_payload,

            quote_payload=
                quote_payload,

            fetched_at_utc=
                pd.Timestamp.now(
                    tz="UTC"
                ),
        )