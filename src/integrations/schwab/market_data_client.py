from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any


MARKET_DATA_BASE_URL = "https://api.schwabapi.com/marketdata/v1"

DOCUMENTED_READ_ENDPOINTS = (
    "GET /quotes",
    "GET /{symbol_id}/quotes",
    "GET /chains",
    "GET /expirationchain",
    "GET /pricehistory",
    "GET /movers/{symbol_id}",
    "GET /markets",
    "GET /markets/{market_id}",
    "GET /instruments",
    "GET /instruments/{cusip_id}",
)


def _require_success(response: Any, context: str) -> None:
    status_code = getattr(response, "status_code", None)

    if status_code is None:
        raise RuntimeError(
            f"{context} response has no status_code."
        )

    try:
        status_code = int(status_code)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{context} returned an invalid status_code."
        ) from exc

    if not 200 <= status_code < 300:
        # Do not include the response body. Market-data responses can
        # still contain entitlement or request details that should not
        # be copied into logs automatically.
        raise RuntimeError(
            f"{context} failed with HTTP {status_code}."
        )


def _response_json(response: Any, context: str) -> Any:
    _require_success(response, context)

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"{context} returned invalid JSON."
        ) from exc


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"{context} must be a mapping."
        )

    return value


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()

    if not normalized:
        raise ValueError("symbol cannot be empty.")

    return normalized


def _normalize_symbols(symbols: Sequence[str] | str) -> list[str]:
    if isinstance(symbols, str):
        values = [symbols]
    else:
        values = list(symbols)

    if not values:
        raise ValueError("symbols cannot be empty.")

    normalized = [
        _normalize_symbol(symbol)
        for symbol in values
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "symbols cannot contain duplicates."
        )

    return normalized


def _normalize_strings(
    values: Sequence[str] | str,
    *,
    context: str,
) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    if not raw_values:
        raise ValueError(
            f"{context} cannot be empty."
        )

    normalized = [
        str(value).strip()
        for value in raw_values
    ]

    if any(not value for value in normalized):
        raise ValueError(
            f"{context} cannot contain empty values."
        )

    return normalized


class SchwabMarketDataClient:
    """
    Read-only research wrapper around an authenticated schwab-py client.

    Schwab's Market Data Production API documents read interfaces for
    quotes, option chains, expiration chains, price history, movers,
    market hours, and instruments.

    This wrapper deliberately contains no account or order operations.
    Authentication remains outside this class.

    Public methods return the decoded JSON payload so ingestion code can
    archive the original response before normalization or feature
    engineering.
    """

    def __init__(self, client: Any) -> None:
        if client is None:
            raise ValueError(
                "client cannot be None."
            )

        self._client = client

    def _enum_class(
        self,
        namespace_name: str,
        enum_name: str,
    ) -> type[Enum]:
        namespace = getattr(
            self._client,
            namespace_name,
            None,
        )

        enum_class = getattr(
            namespace,
            enum_name,
            None,
        )

        if enum_class is None:
            raise RuntimeError(
                "Authenticated Schwab client does not expose "
                f"{namespace_name}.{enum_name}."
            )

        return enum_class

    def _resolve_enum(
        self,
        namespace_name: str,
        enum_name: str,
        value: Any,
        *,
        context: str,
    ) -> Any:
        if value is None:
            return None

        enum_class = self._enum_class(
            namespace_name,
            enum_name,
        )

        if isinstance(value, enum_class):
            return value

        if isinstance(value, str):
            candidate = value.strip()

            if not candidate:
                raise ValueError(
                    f"{context} cannot be empty."
                )

            candidate_upper = candidate.upper()

            for member in enum_class:
                if member.name.upper() == candidate_upper:
                    return member

                if str(member.value).upper() == candidate_upper:
                    return member
        else:
            for member in enum_class:
                if member.value == value:
                    return member

        choices = sorted(
            {
                member.name
                for member in enum_class
            }
            | {
                str(member.value)
                for member in enum_class
            }
        )

        raise ValueError(
            f"Unsupported {context}: {value!r}. "
            f"Expected one of {choices}."
        )

    def _resolve_enum_list(
        self,
        namespace_name: str,
        enum_name: str,
        values: Sequence[Any] | Any,
        *,
        context: str,
    ) -> list[Any]:
        if isinstance(values, (str, Enum)):
            raw_values = [values]
        else:
            try:
                raw_values = list(values)
            except TypeError:
                raw_values = [values]

        if not raw_values:
            raise ValueError(
                f"{context} cannot be empty."
            )

        return [
            self._resolve_enum(
                namespace_name,
                enum_name,
                value,
                context=context,
            )
            for value in raw_values
        ]

    def get_quotes(
        self,
        symbols: Sequence[str] | str,
        *,
        fields: Sequence[Any] | Any | None = None,
        indicative: bool | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve current quotes for one or more symbols.

        Schwab endpoint:
            GET /marketdata/v1/quotes
        """
        normalized = _normalize_symbols(
            symbols
        )

        if indicative is not None and not isinstance(
            indicative,
            bool,
        ):
            raise ValueError(
                "indicative must be True, False, or None."
            )

        resolved_fields = None

        if fields is not None:
            resolved_fields = self._resolve_enum_list(
                "Quote",
                "Fields",
                fields,
                context="quote field",
            )

        response = self._client.get_quotes(
            normalized,
            fields=resolved_fields,
            indicative=indicative,
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab quote request",
            ),
            "Schwab quote payload",
        )

        returned_symbols = {
            str(key).strip().upper()
            for key in payload.keys()
        }

        missing = (
            set(normalized)
            - returned_symbols
        )

        if missing:
            raise RuntimeError(
                "Schwab quote response missing requested symbols: "
                f"{sorted(missing)}"
            )

        return dict(payload)

    def get_price_history(
        self,
        symbol: str,
        *,
        period_type: Any | None = None,
        period: Any | None = None,
        frequency_type: Any | None = None,
        frequency: Any | None = None,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        need_extended_hours_data: bool | None = None,
        need_previous_close: bool | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve historical candles for one symbol.

        Schwab endpoint:
            GET /marketdata/v1/pricehistory
        """
        symbol = _normalize_symbol(
            symbol
        )

        if (
            need_extended_hours_data is not None
            and not isinstance(
                need_extended_hours_data,
                bool,
            )
        ):
            raise ValueError(
                "need_extended_hours_data must be "
                "True, False, or None."
            )

        if (
            need_previous_close is not None
            and not isinstance(
                need_previous_close,
                bool,
            )
        ):
            raise ValueError(
                "need_previous_close must be "
                "True, False, or None."
            )

        resolved_period_type = self._resolve_enum(
            "PriceHistory",
            "PeriodType",
            period_type,
            context="price-history period type",
        )

        resolved_period = self._resolve_enum(
            "PriceHistory",
            "Period",
            period,
            context="price-history period",
        )

        resolved_frequency_type = self._resolve_enum(
            "PriceHistory",
            "FrequencyType",
            frequency_type,
            context="price-history frequency type",
        )

        resolved_frequency = self._resolve_enum(
            "PriceHistory",
            "Frequency",
            frequency,
            context="price-history frequency",
        )

        response = self._client.get_price_history(
            symbol,
            period_type=resolved_period_type,
            period=resolved_period,
            frequency_type=resolved_frequency_type,
            frequency=resolved_frequency,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=need_extended_hours_data,
            need_previous_close=need_previous_close,
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab price-history request",
            ),
            "Schwab price-history payload",
        )

        return dict(payload)

    def get_option_chain(
        self,
        symbol: str,
        *,
        contract_type: Any | None = None,
        strike_count: int | None = None,
        include_underlying_quote: bool | None = None,
        strategy: Any | None = None,
        interval: float | None = None,
        strike: float | None = None,
        strike_range: Any | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        volatility: float | None = None,
        underlying_price: float | None = None,
        interest_rate: float | None = None,
        days_to_expiration: int | None = None,
        exp_month: Any | None = None,
        option_type: Any | None = None,
        entitlement: Any | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve an option chain.

        Schwab endpoint:
            GET /marketdata/v1/chains
        """
        symbol = _normalize_symbol(
            symbol
        )

        if strike_count is not None:
            if (
                not isinstance(strike_count, int)
                or strike_count <= 0
            ):
                raise ValueError(
                    "strike_count must be a positive integer."
                )

        if (
            include_underlying_quote is not None
            and not isinstance(
                include_underlying_quote,
                bool,
            )
        ):
            raise ValueError(
                "include_underlying_quote must be "
                "True, False, or None."
            )

        if (
            from_date is not None
            and to_date is not None
            and from_date > to_date
        ):
            raise ValueError(
                "from_date cannot be after to_date."
            )

        if (
            days_to_expiration is not None
            and days_to_expiration < 0
        ):
            raise ValueError(
                "days_to_expiration cannot be negative."
            )

        response = self._client.get_option_chain(
            symbol,
            contract_type=self._resolve_enum(
                "Options",
                "ContractType",
                contract_type,
                context="option contract type",
            ),
            strike_count=strike_count,
            include_underlying_quote=include_underlying_quote,
            strategy=self._resolve_enum(
                "Options",
                "Strategy",
                strategy,
                context="option strategy",
            ),
            interval=interval,
            strike=strike,
            strike_range=self._resolve_enum(
                "Options",
                "StrikeRange",
                strike_range,
                context="option strike range",
            ),
            from_date=from_date,
            to_date=to_date,
            volatility=volatility,
            underlying_price=underlying_price,
            interest_rate=interest_rate,
            days_to_expiration=days_to_expiration,
            exp_month=self._resolve_enum(
                "Options",
                "ExpirationMonth",
                exp_month,
                context="option expiration month",
            ),
            option_type=self._resolve_enum(
                "Options",
                "Type",
                option_type,
                context="option type",
            ),
            entitlement=self._resolve_enum(
                "Options",
                "Entitlement",
                entitlement,
                context="option entitlement",
            ),
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab option-chain request",
            ),
            "Schwab option-chain payload",
        )

        return dict(payload)

    def get_option_expiration_chain(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """
        Retrieve available option expirations.

        Schwab endpoint:
            GET /marketdata/v1/expirationchain
        """
        symbol = _normalize_symbol(
            symbol
        )

        response = (
            self._client
            .get_option_expiration_chain(
                symbol
            )
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab option-expiration request",
            ),
            "Schwab option-expiration payload",
        )

        return dict(payload)

    def get_movers(
        self,
        index: Any,
        *,
        sort_order: Any | None = None,
        frequency: Any | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve market movers for a Schwab-supported index/category.

        Schwab endpoint:
            GET /marketdata/v1/movers/{symbol_id}
        """
        response = self._client.get_movers(
            self._resolve_enum(
                "Movers",
                "Index",
                index,
                context="movers index",
            ),
            sort_order=self._resolve_enum(
                "Movers",
                "SortOrder",
                sort_order,
                context="movers sort order",
            ),
            frequency=self._resolve_enum(
                "Movers",
                "Frequency",
                frequency,
                context="movers frequency",
            ),
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab movers request",
            ),
            "Schwab movers payload",
        )

        return dict(payload)

    def get_market_hours(
        self,
        markets: Sequence[Any] | Any,
        *,
        date_value: date | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve market hours.

        Schwab endpoint:
            GET /marketdata/v1/markets
        """
        resolved_markets = (
            self._resolve_enum_list(
                "MarketHours",
                "Market",
                markets,
                context="market-hours market",
            )
        )

        response = (
            self._client
            .get_market_hours(
                resolved_markets,
                date=date_value,
            )
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab market-hours request",
            ),
            "Schwab market-hours payload",
        )

        return dict(payload)

    def get_instruments(
        self,
        symbols: Sequence[str] | str,
        *,
        projection: Any,
    ) -> dict[str, Any]:
        """
        Search instruments or request fundamentals.

        Schwab endpoint:
            GET /marketdata/v1/instruments
        """
        normalized = _normalize_strings(
            symbols,
            context="instrument symbols/search terms",
        )

        resolved_projection = self._resolve_enum(
            "Instrument",
            "Projection",
            projection,
            context="instrument projection",
        )

        response = (
            self._client
            .get_instruments(
                normalized,
                resolved_projection,
            )
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab instrument request",
            ),
            "Schwab instrument payload",
        )

        return dict(payload)

    def get_instrument_by_cusip(
        self,
        cusip: str,
    ) -> dict[str, Any]:
        """
        Retrieve one instrument by CUSIP.

        Schwab endpoint:
            GET /marketdata/v1/instruments/{cusip_id}
        """
        if not isinstance(cusip, str):
            raise ValueError(
                "cusip must be passed as str so leading "
                "zeroes are preserved."
            )

        cusip = cusip.strip()

        if not cusip:
            raise ValueError(
                "cusip cannot be empty."
            )

        response = (
            self._client
            .get_instrument_by_cusip(
                cusip
            )
        )

        payload = _require_mapping(
            _response_json(
                response,
                "Schwab CUSIP instrument request",
            ),
            "Schwab CUSIP instrument payload",
        )

        return dict(payload)
