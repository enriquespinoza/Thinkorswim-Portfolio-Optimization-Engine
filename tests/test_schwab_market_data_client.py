from datetime import date, datetime
from enum import Enum

import pytest

from src.integrations.schwab.market_data_client import (
    MARKET_DATA_BASE_URL,
    SchwabMarketDataClient,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        status_code=200,
    ):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeSchwabClient:
    class Quote:
        class Fields(Enum):
            QUOTE = "quote"
            FUNDAMENTAL = "fundamental"
            EXTENDED = "extended"
            REFERENCE = "reference"
            REGULAR = "regular"

    class Options:
        class ContractType(Enum):
            CALL = "CALL"
            PUT = "PUT"
            ALL = "ALL"

        class Strategy(Enum):
            SINGLE = "SINGLE"
            ANALYTICAL = "ANALYTICAL"

        class StrikeRange(Enum):
            NEAR_THE_MONEY = "NTM"
            ALL = "ALL"

        class ExpirationMonth(Enum):
            JANUARY = "JAN"
            OCTOBER = "OCT"
            ALL = "ALL"

        class Type(Enum):
            STANDARD = "S"
            NON_STANDARD = "NS"
            ALL = "ALL"

        class Entitlement(Enum):
            PAYING_PRO = "PP"
            NON_PRO = "NP"
            NON_PAYING_PRO = "PN"

    class PriceHistory:
        class PeriodType(Enum):
            DAY = "day"
            MONTH = "month"
            YEAR = "year"
            YEAR_TO_DATE = "ytd"

        class Period(Enum):
            ONE_DAY = 1
            ONE_MONTH = 1
            ONE_YEAR = 1
            TWENTY_YEARS = 20

        class FrequencyType(Enum):
            MINUTE = "minute"
            DAILY = "daily"
            WEEKLY = "weekly"
            MONTHLY = "monthly"

        class Frequency(Enum):
            EVERY_MINUTE = 1
            EVERY_FIVE_MINUTES = 5
            EVERY_FIFTEEN_MINUTES = 15
            EVERY_THIRTY_MINUTES = 30

    class Movers:
        class Index(Enum):
            SPX = "$SPX"
            NYSE = "NYSE"
            NASDAQ = "NASDAQ"

        class SortOrder(Enum):
            VOLUME = "VOLUME"
            TRADES = "TRADES"
            PERCENT_CHANGE_UP = "PERCENT_CHANGE_UP"
            PERCENT_CHANGE_DOWN = "PERCENT_CHANGE_DOWN"

        class Frequency(Enum):
            ZERO = 0
            ONE = 1
            FIVE = 5
            TEN = 10

    class MarketHours:
        class Market(Enum):
            EQUITY = "equity"
            OPTION = "option"
            BOND = "bond"
            FUTURE = "future"
            FOREX = "forex"

    class Instrument:
        class Projection(Enum):
            SYMBOL_SEARCH = "symbol-search"
            SYMBOL_REGEX = "symbol-regex"
            DESCRIPTION_SEARCH = "desc-search"
            DESCRIPTION_REGEX = "desc-regex"
            SEARCH = "search"
            FUNDAMENTAL = "fundamental"

    def __init__(self):
        self.status_code = 200
        self.last_call = None

    def get_quotes(
        self,
        symbols,
        *,
        fields=None,
        indicative=None,
    ):
        self.last_call = (
            "get_quotes",
            list(symbols),
            fields,
            indicative,
        )

        return FakeResponse(
            {
                symbol: {
                    "quote": {
                        "lastPrice": 100.0,
                    }
                }
                for symbol in symbols
            },
            status_code=self.status_code,
        )

    def get_price_history(
        self,
        symbol,
        *,
        period_type=None,
        period=None,
        frequency_type=None,
        frequency=None,
        start_datetime=None,
        end_datetime=None,
        need_extended_hours_data=None,
        need_previous_close=None,
    ):
        self.last_call = (
            "get_price_history",
            symbol,
            period_type,
            period,
            frequency_type,
            frequency,
            start_datetime,
            end_datetime,
            need_extended_hours_data,
            need_previous_close,
        )

        return FakeResponse(
            {
                "symbol": symbol,
                "candles": [
                    {
                        "open": 99.0,
                        "high": 101.0,
                        "low": 98.5,
                        "close": 100.0,
                        "volume": 1_000_000,
                        "datetime": 0,
                    }
                ],
            },
            status_code=self.status_code,
        )

    def get_option_chain(
        self,
        symbol,
        **kwargs,
    ):
        self.last_call = (
            "get_option_chain",
            symbol,
            kwargs,
        )

        return FakeResponse(
            {
                "symbol": symbol,
                "status": "SUCCESS",
            },
            status_code=self.status_code,
        )

    def get_option_expiration_chain(
        self,
        symbol,
    ):
        self.last_call = (
            "get_option_expiration_chain",
            symbol,
        )

        return FakeResponse(
            {
                "expirationList": [],
            },
            status_code=self.status_code,
        )

    def get_movers(
        self,
        index,
        *,
        sort_order=None,
        frequency=None,
    ):
        self.last_call = (
            "get_movers",
            index,
            sort_order,
            frequency,
        )

        return FakeResponse(
            {
                "screeners": [],
            },
            status_code=self.status_code,
        )

    def get_market_hours(
        self,
        markets,
        *,
        date=None,
    ):
        self.last_call = (
            "get_market_hours",
            list(markets),
            date,
        )

        return FakeResponse(
            {
                "equity": {},
            },
            status_code=self.status_code,
        )

    def get_instruments(
        self,
        symbols,
        projection,
    ):
        self.last_call = (
            "get_instruments",
            list(symbols),
            projection,
        )

        return FakeResponse(
            {
                "instruments": [],
            },
            status_code=self.status_code,
        )

    def get_instrument_by_cusip(
        self,
        cusip,
    ):
        self.last_call = (
            "get_instrument_by_cusip",
            cusip,
        )

        return FakeResponse(
            {
                "instruments": [],
            },
            status_code=self.status_code,
        )


def test_market_data_base_url_matches_schwab_production_surface():
    assert (
        MARKET_DATA_BASE_URL
        == "https://api.schwabapi.com/marketdata/v1"
    )


def test_client_rejects_none():
    with pytest.raises(
        ValueError,
        match="cannot be None",
    ):
        SchwabMarketDataClient(
            None
        )


def test_quotes_normalize_symbols_and_fields():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    result = client.get_quotes(
        ["spy", "qqq"],
        fields=[
            "quote",
            "fundamental",
        ],
        indicative=False,
    )

    assert set(result) == {
        "SPY",
        "QQQ",
    }

    assert broker.last_call == (
        "get_quotes",
        ["SPY", "QQQ"],
        [
            broker.Quote.Fields.QUOTE,
            broker.Quote.Fields.FUNDAMENTAL,
        ],
        False,
    )


def test_quotes_reject_duplicates():
    client = SchwabMarketDataClient(
        FakeSchwabClient()
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        client.get_quotes(
            ["SPY", "spy"]
        )


def test_price_history_maps_enum_values():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    start = datetime(
        2026,
        10,
        1,
    )
    end = datetime(
        2026,
        10,
        8,
    )

    result = client.get_price_history(
        "spy",
        period_type="year",
        period=20,
        frequency_type="daily",
        frequency=1,
        start_datetime=start,
        end_datetime=end,
        need_extended_hours_data=False,
        need_previous_close=True,
    )

    assert result["symbol"] == "SPY"

    assert broker.last_call == (
        "get_price_history",
        "SPY",
        broker.PriceHistory.PeriodType.YEAR,
        broker.PriceHistory.Period.TWENTY_YEARS,
        broker.PriceHistory.FrequencyType.DAILY,
        broker.PriceHistory.Frequency.EVERY_MINUTE,
        start,
        end,
        False,
        True,
    )


def test_option_chain_maps_research_parameters():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    from_date = date(
        2026,
        10,
        9,
    )
    to_date = date(
        2026,
        12,
        31,
    )

    client.get_option_chain(
        "spy",
        contract_type="ALL",
        strike_count=20,
        include_underlying_quote=True,
        strategy="SINGLE",
        strike_range="NTM",
        from_date=from_date,
        to_date=to_date,
        exp_month="ALL",
        option_type="ALL",
        entitlement="NP",
    )

    name, symbol, kwargs = (
        broker.last_call
    )

    assert name == "get_option_chain"
    assert symbol == "SPY"

    assert (
        kwargs["contract_type"]
        == broker.Options.ContractType.ALL
    )

    assert (
        kwargs["strategy"]
        == broker.Options.Strategy.SINGLE
    )

    assert (
        kwargs["strike_range"]
        == broker.Options.StrikeRange.NEAR_THE_MONEY
    )

    assert (
        kwargs["exp_month"]
        == broker.Options.ExpirationMonth.ALL
    )

    assert (
        kwargs["option_type"]
        == broker.Options.Type.ALL
    )

    assert (
        kwargs["entitlement"]
        == broker.Options.Entitlement.NON_PRO
    )


def test_option_chain_rejects_reverse_date_range():
    client = SchwabMarketDataClient(
        FakeSchwabClient()
    )

    with pytest.raises(
        ValueError,
        match="from_date",
    ):
        client.get_option_chain(
            "SPY",
            from_date=date(
                2026,
                12,
                1,
            ),
            to_date=date(
                2026,
                11,
                1,
            ),
        )


def test_option_expiration_chain_normalizes_symbol():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    client.get_option_expiration_chain(
        "spy"
    )

    assert broker.last_call == (
        "get_option_expiration_chain",
        "SPY",
    )


def test_movers_maps_enum_values():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    client.get_movers(
        "$SPX",
        sort_order="PERCENT_CHANGE_UP",
        frequency=5,
    )

    assert broker.last_call == (
        "get_movers",
        broker.Movers.Index.SPX,
        broker.Movers.SortOrder.PERCENT_CHANGE_UP,
        broker.Movers.Frequency.FIVE,
    )


def test_market_hours_maps_markets():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    target_date = date(
        2026,
        10,
        9,
    )

    client.get_market_hours(
        [
            "equity",
            "option",
        ],
        date_value=target_date,
    )

    assert broker.last_call == (
        "get_market_hours",
        [
            broker.MarketHours.Market.EQUITY,
            broker.MarketHours.Market.OPTION,
        ],
        target_date,
    )


def test_instrument_lookup_preserves_search_text():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    client.get_instruments(
        "technology",
        projection="desc-search",
    )

    assert broker.last_call == (
        "get_instruments",
        ["technology"],
        broker.Instrument.Projection.DESCRIPTION_SEARCH,
    )


def test_cusip_lookup_requires_string_and_preserves_leading_zeroes():
    broker = FakeSchwabClient()
    client = SchwabMarketDataClient(
        broker
    )

    client.get_instrument_by_cusip(
        "037833100"
    )

    assert broker.last_call == (
        "get_instrument_by_cusip",
        "037833100",
    )

    with pytest.raises(
        ValueError,
        match="str",
    ):
        client.get_instrument_by_cusip(
            37833100
        )


def test_failed_market_data_request_rejected_without_body():
    broker = FakeSchwabClient()
    broker.status_code = 429

    client = SchwabMarketDataClient(
        broker
    )

    with pytest.raises(
        RuntimeError,
        match="HTTP 429",
    ):
        client.get_quotes(
            ["SPY"]
        )


def test_wrapper_exposes_no_account_or_order_methods():
    client = SchwabMarketDataClient(
        FakeSchwabClient()
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
            client,
            method_name,
        )
