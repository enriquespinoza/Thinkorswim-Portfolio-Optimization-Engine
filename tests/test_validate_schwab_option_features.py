from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validate_schwab_option_features import (
    QUALITY_POLICY_VERSION,
    classify_symbol_quality,
    validate_option_feature_frame,
)


def _base_row(
    symbol: str,
) -> dict:
    row = {
        "feature_schema_version": "1.2.0",
        "underlying_symbol": symbol,
        "spot": 100.0,
        "contract_count_total": 200,
        "contract_count_eligible": 180,
        "eligible_contract_ratio": 0.90,
        "zero_dte_contract_count": 20,
        "zero_dte_contract_ratio": 0.10,
        "eligible_call_count": 90,
        "eligible_put_count": 90,
        "eligible_expiration_count": 12,
        "eligible_dte_min": 3.0,
        "eligible_dte_max": 365.0,
        "relative_spread_pct_median": 1.0,
        "atm_relative_spread_pct_median": 1.0,
        "iv_median": 20.0,
        "atm_iv_median": 20.0,
        "atm_call_iv_median": 19.0,
        "atm_put_iv_median": 21.0,
        "put_call_open_interest_ratio": 1.2,
        "put_call_volume_ratio": 1.1,
        "open_interest_total": 10000.0,
        "option_volume_total": 2000.0,
        "abs_delta_median": 0.5,
        "gamma_median": 0.01,
        "theta_median": -0.1,
        "vega_median": 0.2,
        "canonical_extrinsic_value_median": 5.0,
        "quote_age_minutes_median": 0.1,
        "atm_contract_count": 100,
        "atm_expiration_count": 12,
        "delta_25_put_call_iv_skew_matched_median": -1.0,
        "delta_25_expiration_pair_count": 12,
        "atm_put_call_iv_skew_matched_median": 0.5,
        "atm_expiration_pair_count": 12,
        "atm_iv_term_spread_91_365_minus_8_30": 2.0,
    }

    buckets = (
        "dte_1_7",
        "dte_8_30",
        "dte_31_90",
        "dte_91_365",
        "dte_366_plus",
    )

    for bucket in buckets:
        row[
            f"atm_iv_median_{bucket}"
        ] = 20.0

        row[
            f"atm_contract_count_{bucket}"
        ] = 25

        row[
            f"atm_expiration_count_{bucket}"
        ] = 5

        row[
            f"delta_25_put_call_iv_skew_{bucket}"
        ] = -1.0

        row[
            f"delta_25_pair_count_{bucket}"
        ] = 5

        row[
            f"atm_put_call_iv_skew_{bucket}"
        ] = 0.5

        row[
            f"atm_skew_pair_count_{bucket}"
        ] = 5

    return row


def test_symbol_quality_thresholds():
    status, reasons = classify_symbol_quality(
        _base_row(
            "SPY"
        )
    )

    assert status == "PASS"
    assert reasons == []

    caution = _base_row(
        "TLT"
    )
    caution[
        "eligible_contract_ratio"
    ] = 0.70
    caution[
        "contract_count_eligible"
    ] = 75
    caution[
        "eligible_expiration_count"
    ] = 7

    status, reasons = classify_symbol_quality(
        caution
    )

    assert status == "CAUTION"
    assert (
        "eligible_contract_ratio_below_pass"
        in reasons
    )

    suppress = _base_row(
        "SCHD"
    )
    suppress[
        "eligible_contract_ratio"
    ] = 0.15
    suppress[
        "contract_count_eligible"
    ] = 38

    status, _ = classify_symbol_quality(
        suppress
    )

    assert status == "SUPPRESS"


def test_feature_support_masks_only_unsupported_feature():
    row = _base_row(
        "SPY"
    )

    row[
        "delta_25_pair_count_dte_8_30"
    ] = 4

    frame = pd.DataFrame(
        [
            row
        ]
    )

    validated, quality = (
        validate_option_feature_frame(
            frame
        )
    )

    result = validated.iloc[
        0
    ]

    assert (
        result[
            "quality_symbol_status"
        ]
        == "PASS"
    )

    assert np.isnan(
        result[
            "delta_25_put_call_iv_skew_dte_8_30"
        ]
    )

    assert (
        result[
            "delta_25_put_call_iv_skew_dte_31_90"
        ]
        == -1.0
    )

    quality_row = quality.loc[
        quality[
            "feature_name"
        ]
        == "delta_25_put_call_iv_skew_dte_8_30"
    ].iloc[
        0
    ]

    assert not bool(
        quality_row[
            "usable"
        ]
    )

    assert (
        quality_row[
            "minimum_support"
        ]
        == 5
    )


def test_term_structure_requires_both_supported_buckets():
    row = _base_row(
        "QQQ"
    )

    row[
        "atm_contract_count_dte_8_30"
    ] = 19

    frame = pd.DataFrame(
        [
            row
        ]
    )

    validated, quality = (
        validate_option_feature_frame(
            frame
        )
    )

    result = validated.iloc[
        0
    ]

    assert np.isnan(
        result[
            "atm_iv_median_dte_8_30"
        ]
    )

    assert np.isnan(
        result[
            "atm_iv_term_spread_91_365_minus_8_30"
        ]
    )

    term_quality = quality.loc[
        quality[
            "feature_name"
        ]
        == "atm_iv_term_spread_91_365_minus_8_30"
    ].iloc[
        0
    ]

    assert not bool(
        term_quality[
            "usable"
        ]
    )


def test_suppressed_symbol_retains_support_and_identity():
    row = _base_row(
        "SCHD"
    )

    row[
        "eligible_contract_ratio"
    ] = 0.158333
    row[
        "contract_count_eligible"
    ] = 38
    row[
        "eligible_expiration_count"
    ] = 11

    frame = pd.DataFrame(
        [
            row
        ]
    )

    validated, _ = (
        validate_option_feature_frame(
            frame
        )
    )

    result = validated.iloc[
        0
    ]

    assert (
        result[
            "underlying_symbol"
        ]
        == "SCHD"
    )

    assert (
        result[
            "contract_count_eligible"
        ]
        == 38
    )

    assert (
        result[
            "eligible_expiration_count"
        ]
        == 11
    )

    assert (
        result[
            "quality_symbol_status"
        ]
        == "SUPPRESS"
    )

    assert np.isnan(
        result[
            "iv_median"
        ]
    )

    assert np.isnan(
        result[
            "put_call_volume_ratio"
        ]
    )

    assert (
        result[
            "quality_policy_version"
        ]
        == QUALITY_POLICY_VERSION
    )
