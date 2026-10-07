from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forward.meta_allocation_v1_weights import (
    append_weight_record,
    build_target_weights,
    month_distance,
    validate_target_weights,
    is_quarterly_decision_month,
    validate_information_date,
    calculate_effective_from,
    expected_trading_month_end,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2025-01-01",
        periods=400,
    )

    data = rng.normal(
        loc=0.0003,
        scale=0.01,
        size=(400, 5),
    )

    return pd.DataFrame(
        data,
        index=dates,
        columns=[
            "SPY",
            "QQQ",
            "TLT",
            "GLD",
            "SCHD",
        ],
    )


def test_equal_weight_target(
    sample_returns,
):
    weights, stats = build_target_weights(
        allocation_state="EQUAL_WEIGHT",
        historical_returns=sample_returns,
        max_weight=0.40,
    )

    assert np.allclose(
        weights.values,
        np.full(5, 0.20),
        atol=1e-12,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    assert np.isnan(
        stats["expected_return"]
    )


def test_month_distance():
    assert (
        month_distance(
            pd.Timestamp("2026-10-30"),
            pd.Timestamp("2026-10-30"),
        )
        == 0
    )

    assert (
        month_distance(
            pd.Timestamp("2026-10-30"),
            pd.Timestamp("2026-11-30"),
        )
        == 1
    )

    assert (
        month_distance(
            pd.Timestamp("2026-10-30"),
            pd.Timestamp("2026-12-31"),
        )
        == 2
    )

    assert (
        month_distance(
            pd.Timestamp("2026-10-30"),
            pd.Timestamp("2027-01-29"),
        )
        == 3
    )


def test_maximum_sharpe_respects_cap(
    sample_returns,
):
    weights, stats = build_target_weights(
        allocation_state="MAXIMUM_SHARPE",
        historical_returns=sample_returns,
        max_weight=0.40,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
        atol=1e-8,
    )

    assert (
        weights
        >= -1e-10
    ).all()

    assert (
        weights
        <= 0.40 + 1e-8
    ).all()

    assert np.isfinite(
        stats["expected_return"]
    )

    assert np.isfinite(
        stats["volatility"]
    )

    assert np.isfinite(
        stats["sharpe"]
    )


def test_validate_target_weights_rejects_cap():
    weights = pd.Series(
        {
            "SPY": 0.50,
            "QQQ": 0.20,
            "TLT": 0.10,
            "GLD": 0.10,
            "SCHD": 0.10,
        }
    )

    with pytest.raises(
        ValueError
    ):
        validate_target_weights(
            weights=weights,
            max_weight=0.40,
        )


def sample_weight_record():
    return {
        "information_date":
            "2026-10-30",

        "decision_signal_date":
            "2026-10-30",

        "effective_from":
            "2026-11-02",

        "is_quarterly_decision":
            True,

        "state_age_months":
            0,

        "model_version":
            "1.0.0",

        "allocation_state":
            "EQUAL_WEIGHT",

        "predicted_ms_excess_3m":
            0.01,

        "input_market_data_end":
            "2026-10-30",

        "input_market_data_hash":
            "market123",

        "target_weights_hash":
            "weights123",

        "portfolio_expected_return":
            np.nan,

        "portfolio_expected_volatility":
            np.nan,

        "portfolio_expected_sharpe":
            np.nan,

        "generated_at_utc":
            "2026-10-30T20:00:00+00:00",

        "weight_SPY":
            0.20,

        "weight_QQQ":
            0.20,

        "weight_TLT":
            0.20,

        "weight_GLD":
            0.20,

        "weight_SCHD":
            0.20,
    }

def test_append_weight_record_is_idempotent(
    tmp_path: Path,
):
    ledger = (
        tmp_path
        / "weights.csv"
    )

    record = sample_weight_record()

    first = append_weight_record(
        record=record,
        ledger_path=ledger,
    )

    second = append_weight_record(
        record=record,
        ledger_path=ledger,
    )

    assert first == "APPENDED"
    assert second == "EXISTS"

    stored = pd.read_csv(
        ledger
    )

    assert len(stored) == 1


def test_append_weight_record_rejects_rewrite(
    tmp_path: Path,
):
    ledger = (
        tmp_path
        / "weights.csv"
    )

    record = sample_weight_record()

    append_weight_record(
        record=record,
        ledger_path=ledger,
    )

    changed = record.copy()

    changed[
        "target_weights_hash"
    ] = "different"

    with pytest.raises(
        RuntimeError
    ):
        append_weight_record(
            record=changed,
            ledger_path=ledger,
        )

def test_maximum_sharpe_matches_historical_engine():
    from src.backtest.walk_forward import (
        _build_model_weights,
    )

    from src.forward.meta_allocation_v1_signal import (
        load_market_returns,
    )

    from src.forward.meta_allocation_v1_weights import (
        FROZEN_ASSETS,
        build_target_weights,
        get_max_weight,
        load_model_spec,
    )

    returns, _ = (
        load_market_returns()
    )

    training_returns = (
        returns.loc[
            :"2026-09-30",
            FROZEN_ASSETS,
        ]
        .dropna()
        .copy()
    )

    model_spec = (
        load_model_spec()
    )

    max_weight = (
        get_max_weight(
            model_spec
        )
    )

    forward_weights, _ = (
        build_target_weights(
            allocation_state=
                "MAXIMUM_SHARPE",

            historical_returns=
                training_returns,

            max_weight=
                max_weight,
        )
    )

    historical_models = (
        _build_model_weights(
            training_returns=
                training_returns,

            max_weight=
                max_weight,

            risk_free_rate=
                0.0,

            ewma_span=
                126,
        )
    )

    historical_weights = (
        historical_models[
            "maximum_sharpe"
        ]
        .reindex(
            FROZEN_ASSETS
        )
    )

    assert np.allclose(
        forward_weights
        .reindex(
            FROZEN_ASSETS
        )
        .to_numpy(),

        historical_weights
        .to_numpy(),

        rtol=0.0,
        atol=1e-6,
    )

from src.forward.meta_allocation_v1_weights import (
    calculate_effective_from,
    is_quarterly_decision_month,
    validate_information_date,
)


def test_october_2026_is_quarterly_decision_month():
    information_date = pd.Timestamp(
        "2026-10-30"
    )

    assert (
        is_quarterly_decision_month(
            information_date
        )
        is True
    )


def test_november_2026_is_not_quarterly_decision_month():
    information_date = pd.Timestamp(
        "2026-11-30"
    )

    assert (
        is_quarterly_decision_month(
            information_date
        )
        is False
    )


def test_october_target_effective_november_2():
    information_date = pd.Timestamp(
        "2026-10-30"
    )

    effective_from = (
        calculate_effective_from(
            information_date
        )
    )

    assert (
        effective_from
        == pd.Timestamp(
            "2026-11-02"
        )
    )


def test_november_target_effective_december_1():
    information_date = pd.Timestamp(
        "2026-11-30"
    )

    effective_from = (
        calculate_effective_from(
            information_date
        )
    )

    assert (
        effective_from
        == pd.Timestamp(
            "2026-12-01"
        )
    )


def test_october_signal_age_is_one_month_in_november():
    signal_date = pd.Timestamp(
        "2026-10-30"
    )

    information_date = pd.Timestamp(
        "2026-11-30"
    )

    assert (
        month_distance(
            signal_date,
            information_date,
        )
        == 1
    )


def test_october_signal_age_is_two_months_in_december():
    signal_date = pd.Timestamp(
        "2026-10-30"
    )

    information_date = pd.Timestamp(
        "2026-12-31"
    )

    assert (
        month_distance(
            signal_date,
            information_date,
        )
        == 2
    )


def test_october_signal_is_stale_by_january():
    signal_date = pd.Timestamp(
        "2026-10-30"
    )

    information_date = pd.Timestamp(
        "2027-01-29"
    )

    assert (
        month_distance(
            signal_date,
            information_date,
        )
        == 3
    )


def test_rejects_incomplete_month_information_date():
    with pytest.raises(
        RuntimeError
    ):
        validate_information_date(
            pd.Timestamp(
                "2026-10-06"
            )
        )

def test_december_target_skips_new_years_day():
    information_date = pd.Timestamp(
        "2026-12-31"
    )

    effective_from = (
        calculate_effective_from(
            information_date
        )
    )

    assert (
        effective_from
        == pd.Timestamp(
            "2027-01-04"
        )
    )
def test_october_target_effective_november_2():
    assert (
        calculate_effective_from(
            pd.Timestamp(
                "2026-10-30"
            )
        )
        == pd.Timestamp(
            "2026-11-02"
        )
    )


def test_november_target_effective_december_1():
    assert (
        calculate_effective_from(
            pd.Timestamp(
                "2026-11-30"
            )
        )
        == pd.Timestamp(
            "2026-12-01"
        )
    )


def test_january_2027_month_end_is_january_29():
    assert (
        expected_trading_month_end(
            pd.Timestamp(
                "2027-01-15"
            )
        )
        == pd.Timestamp(
            "2027-01-29"
        )
    )