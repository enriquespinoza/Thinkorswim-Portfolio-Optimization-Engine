from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.schwab_option_features import (
    OptionFeatureConfig,
    build_schwab_option_features,
    eligibility_summary,
    prepare_schwab_option_contracts,
)


CAPTURE_TIME = "2026-10-09T17:00:00+00:00"
QUOTE_TIME = int(
    datetime(
        2026,
        10,
        9,
        16,
        59,
        tzinfo=timezone.utc,
    ).timestamp()
    * 1000
)


def make_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "quote__mark": 500.0,
            },
            {
                "symbol": "TLT",
                "quote__mark": 100.0,
            },
        ]
    )


def make_options() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "underlying_symbol": "SPY",
                "put_call": "CALL",
                "strike": 500.0,
                "expiration_date": "2026-11-20",
                "expiration_dte": 42,
                "bid": 10.0,
                "ask": 10.5,
                "mark": 10.25,
                "volatility": 20.0,
                "delta": 0.50,
                "gamma": 0.02,
                "theta": -0.10,
                "vega": 0.30,
                "rho": 0.10,
                "openInterest": 100,
                "totalVolume": 20,
                "quoteTimeInLong": QUOTE_TIME,
                "raw_captured_at_utc": CAPTURE_TIME,
                "percentChange": 99999.0,
                "markPercentChange": 99999.0,
                "intrinsicValue": -1.0,
                "extrinsicValue": -1.0,
                "theoreticalVolatility": 29.0,
            },
            {
                "underlying_symbol": "SPY",
                "put_call": "PUT",
                "strike": 500.0,
                "expiration_date": "2026-11-20",
                "expiration_dte": 42,
                "bid": 11.0,
                "ask": 11.5,
                "mark": 11.25,
                "volatility": 22.0,
                "delta": -0.50,
                "gamma": 0.02,
                "theta": -0.11,
                "vega": 0.31,
                "rho": -0.10,
                "openInterest": 150,
                "totalVolume": 30,
                "quoteTimeInLong": QUOTE_TIME,
                "raw_captured_at_utc": CAPTURE_TIME,
                "percentChange": 99999.0,
                "markPercentChange": 99999.0,
                "intrinsicValue": -2.0,
                "extrinsicValue": -2.0,
                "theoreticalVolatility": 29.0,
            },
            {
                "underlying_symbol": "TLT",
                "put_call": "CALL",
                "strike": 95.0,
                "expiration_date": "2026-10-09",
                "expiration_dte": 0,
                "bid": 0.0,
                "ask": 1.0,
                "mark": 0.5,
                "volatility": 18.0,
                "delta": 0.80,
                "gamma": 0.03,
                "theta": -0.15,
                "vega": 0.20,
                "rho": 0.05,
                "openInterest": 10,
                "totalVolume": 0,
                "quoteTimeInLong": QUOTE_TIME,
                "raw_captured_at_utc": CAPTURE_TIME,
                "percentChange": 0.0,
                "markPercentChange": 0.0,
                "intrinsicValue": 5.0,
                "extrinsicValue": -4.5,
                "theoreticalVolatility": 29.0,
            },
        ]
    )


def test_prepare_contracts_builds_canonical_values_and_eligibility():
    prepared = prepare_schwab_option_contracts(
        make_options(),
        make_quotes(),
    )

    spy_call = prepared.iloc[0]
    spy_put = prepared.iloc[1]
    tlt_call = prepared.iloc[2]

    assert spy_call["spot"] == 500.0
    assert spy_call["canonical_intrinsic_value"] == 0.0
    assert spy_call["canonical_extrinsic_value"] == 10.25
    assert spy_call["signed_moneyness_pct"] == 0.0
    assert bool(spy_call["eligible_contract"])

    assert spy_put["canonical_intrinsic_value"] == 0.0
    assert spy_put["canonical_extrinsic_value"] == 11.25
    assert bool(spy_put["eligible_contract"])

    assert tlt_call["canonical_intrinsic_value"] == 5.0
    assert not bool(tlt_call["eligible_contract"])
    assert "bid" in tlt_call["ineligibility_reasons"]
    assert "dte" in tlt_call["ineligibility_reasons"]


def test_provider_intrinsic_and_percent_change_do_not_drive_features():
    frame = make_options()
    features = build_schwab_option_features(
        frame,
        make_quotes(),
    )

    spy = features.loc[
        features["underlying_symbol"] == "SPY"
    ].iloc[0]

    assert spy["contract_count_eligible"] == 2
    assert spy["atm_call_iv_median"] == 20.0
    assert spy["atm_put_iv_median"] == 22.0
    assert spy["atm_contract_count"] == 2
    assert spy["atm_contract_count_dte_31_90"] == 2
    assert spy["atm_put_call_iv_skew_matched_median"] == 2.0
    assert spy["atm_expiration_pair_count"] == 1
    assert spy["atm_put_call_iv_skew_dte_31_90"] == 2.0
    assert spy["atm_skew_pair_count_dte_31_90"] == 1
    assert spy["delta_25_put_call_iv_skew_matched_median"] == 2.0
    assert spy["delta_25_expiration_pair_count"] == 1
    assert spy["delta_25_put_call_iv_skew_dte_31_90"] == 2.0
    assert spy["delta_25_pair_count_dte_31_90"] == 1
    assert spy["put_call_open_interest_ratio"] == 1.5
    assert spy["put_call_volume_ratio"] == 1.5
    assert spy["canonical_extrinsic_value_median"] == 10.75

    assert "percentChange" not in features.columns
    assert "markPercentChange" not in features.columns
    assert "intrinsicValue" not in features.columns
    assert "extrinsicValue" not in features.columns
    assert "theoreticalVolatility" not in features.columns



def test_matched_expiration_skew_does_not_mix_maturities():
    frame = make_options().iloc[:2].copy()

    call_90 = frame.iloc[0].copy()
    call_90["expiration_date"] = "2027-01-08"
    call_90["expiration_dte"] = 91
    call_90["volatility"] = 30.0
    call_90["delta"] = 0.25

    put_90 = frame.iloc[1].copy()
    put_90["expiration_date"] = "2027-01-08"
    put_90["expiration_dte"] = 91
    put_90["volatility"] = 34.0
    put_90["delta"] = -0.25

    frame.loc[0, "delta"] = 0.25
    frame.loc[0, "volatility"] = 20.0
    frame.loc[1, "delta"] = -0.25
    frame.loc[1, "volatility"] = 22.0

    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    call_90,
                    put_90,
                ]
            ),
        ],
        ignore_index=True,
    )

    features = build_schwab_option_features(
        frame,
        make_quotes(),
    )

    spy = features.loc[
        features["underlying_symbol"] == "SPY"
    ].iloc[0]

    assert spy["delta_25_expiration_pair_count"] == 2
    assert spy["delta_25_put_call_iv_skew_matched_median"] == 3.0
    assert spy["delta_25_pair_count_dte_31_90"] == 1
    assert spy["delta_25_put_call_iv_skew_dte_31_90"] == 2.0
    assert spy["delta_25_pair_count_dte_91_365"] == 1
    assert spy["delta_25_put_call_iv_skew_dte_91_365"] == 4.0


def test_zero_dte_can_be_included_by_configuration():
    config = OptionFeatureConfig(
        require_positive_bid=False,
        exclude_zero_dte=False,
        max_relative_spread_pct=250.0,
    )

    prepared = prepare_schwab_option_contracts(
        make_options(),
        make_quotes(),
        config=config,
    )

    tlt = prepared.loc[
        prepared["underlying_symbol"] == "TLT"
    ].iloc[0]

    assert bool(tlt["eligible_contract"])


def test_eligibility_summary_counts_reasons():
    prepared = prepare_schwab_option_contracts(
        make_options(),
        make_quotes(),
    )

    summary = eligibility_summary(
        prepared
    )

    tlt = summary.loc[
        summary["underlying_symbol"] == "TLT"
    ].iloc[0]

    assert tlt["contract_count_total"] == 1
    assert tlt["contract_count_eligible"] == 0
    assert tlt["ineligible_contract_count"] == 1
    assert '"bid":1' in tlt["ineligibility_reason_counts"]
    assert '"dte":1' in tlt["ineligibility_reason_counts"]


def test_missing_quote_symbol_makes_contract_ineligible():
    quotes = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "quote__mark": 500.0,
            }
        ]
    )

    prepared = prepare_schwab_option_contracts(
        make_options(),
        quotes,
    )

    tlt = prepared.loc[
        prepared["underlying_symbol"] == "TLT"
    ].iloc[0]

    assert np.isnan(
        tlt["spot"]
    )
    assert not bool(
        tlt["eligible_contract"]
    )
    assert "spot" in (
        tlt[
            "ineligibility_reasons"
        ]
    )
