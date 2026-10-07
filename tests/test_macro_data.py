import pandas as pd
import pytest

from src.data.macro_data import (
    DEFAULT_MACRO_SERIES,
    MacroSeriesSpec,
    apply_availability_lag,
    build_fred_csv_url,
    validate_macro_specs,
)


def test_default_macro_specs():
    validate_macro_specs(
        DEFAULT_MACRO_SERIES
    )

    names = {
        spec.column_name
        for spec
        in DEFAULT_MACRO_SERIES
    }

    expected = {
        "yield_2y",
        "yield_10y",
        "yield_30y",
        "fed_funds",
        "breakeven_10y",
        "real_yield_10y",
        "unemployment",
        "initial_claims",
        "payrolls",
        "retail_sales",
    }

    assert expected.issubset(
        names
    )


def test_fred_url():
    url = build_fred_csv_url(
        series_id="DGS10",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert "DGS10" in url
    assert "2020-01-01" in url
    assert "2020-12-31" in url


def test_availability_lag():
    spec = MacroSeriesSpec(
        series_id="TEST",
        column_name="test_value",
        availability_lag_days=5,
        frequency="monthly",
    )

    series = pd.Series(
        [100.0],
        index=pd.to_datetime(
            [
                "2026-08-01",
            ]
        ),
    )

    result = apply_availability_lag(
        series,
        spec,
    )

    assert result.index[
        0
    ] == pd.Timestamp(
        "2026-08-06"
    )


def test_negative_lag_rejected():
    specs = (
        MacroSeriesSpec(
            series_id="TEST",
            column_name="test",
            availability_lag_days=-1,
            frequency="daily",
        ),
    )

    with pytest.raises(
        ValueError
    ):
        validate_macro_specs(
            specs
        )
