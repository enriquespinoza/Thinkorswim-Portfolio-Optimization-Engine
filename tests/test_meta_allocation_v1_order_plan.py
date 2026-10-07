import numpy as np
import pandas as pd
import pytest

from src.forward.meta_allocation_v1_portfolio_snapshot import (
    build_portfolio_snapshot,
)

from src.forward.meta_allocation_v1_order_plan import (
    EXECUTION_MODE,
    build_order_plan,
    build_order_plan_from_snapshot,
    build_order_plan_from_weight_record,
)


ASSETS = [
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "SCHD",
]


def equal_weights():
    return pd.Series(
        {
            asset: 0.20
            for asset in ASSETS
        }
    )


def sample_prices():
    return {
        "SPY": 500.0,
        "QQQ": 600.0,
        "TLT": 100.0,
        "GLD": 200.0,
        "SCHD": 80.0,
    }


def sample_holdings():
    return {
        "SPY": 100,
        "QQQ": 50,
        "TLT": 0,
        "GLD": 0,
        "SCHD": 0,
    }


def sample_weight_record():
    return {
        "information_date":
            "2026-10-30",

        "decision_signal_date":
            "2026-10-30",

        "effective_from":
            "2026-11-02",

        "allocation_state":
            "EQUAL_WEIGHT",

        "model_version":
            "1.0.0",

        "target_weights_hash":
            "target123",

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


def test_portfolio_value_is_correct():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            0.0,
    )

    assert np.isclose(
        result.portfolio_value_before,
        100_000.0,
    )


def test_whole_share_targets():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            0.0,
    )

    orders = (
        result.orders
        .set_index(
            "asset"
        )
    )

    assert (
        orders.loc[
            "SPY",
            "target_shares",
        ]
        == 40
    )

    assert (
        orders.loc[
            "QQQ",
            "target_shares",
        ]
        == 33
    )

    assert (
        orders.loc[
            "TLT",
            "target_shares",
        ]
        == 200
    )

    assert (
        orders.loc[
            "GLD",
            "target_shares",
        ]
        == 100
    )

    assert (
        orders.loc[
            "SCHD",
            "target_shares",
        ]
        == 250
    )


def test_trade_directions_are_correct():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            0.0,
    )

    orders = (
        result.orders
        .set_index(
            "asset"
        )
    )

    assert (
        orders.loc[
            "SPY",
            "side",
        ]
        == "SELL"
    )

    assert (
        orders.loc[
            "QQQ",
            "side",
        ]
        == "SELL"
    )

    assert (
        orders.loc[
            "TLT",
            "side",
        ]
        == "BUY"
    )

    assert (
        orders.loc[
            "GLD",
            "side",
        ]
        == "BUY"
    )

    assert (
        orders.loc[
            "SCHD",
            "side",
        ]
        == "BUY"
    )


def test_trade_share_counts():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            0.0,
    )

    orders = (
        result.orders
        .set_index(
            "asset"
        )
    )

    assert (
        orders.loc[
            "SPY",
            "trade_shares",
        ]
        == -60
    )

    assert (
        orders.loc[
            "QQQ",
            "trade_shares",
        ]
        == -17
    )

    assert (
        orders.loc[
            "TLT",
            "trade_shares",
        ]
        == 200
    )

    assert (
        orders.loc[
            "GLD",
            "trade_shares",
        ]
        == 100
    )

    assert (
        orders.loc[
            "SCHD",
            "trade_shares",
        ]
        == 250
    )


def test_zero_cost_residual_cash():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            0.0,
    )

    assert np.isclose(
        result.cash_after,
        200.0,
    )


def test_transaction_cost_is_estimated():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            5.0,
    )

    assert (
        result.estimated_transaction_cost
        > 0.0
    )

    assert (
        result.cash_after
        >= 0.0
    )


def test_no_margin_is_used():
    result = build_order_plan(
        target_weights=
            equal_weights(),

        current_shares=
            sample_holdings(),

        current_cash=
            0.0,

        reference_prices=
            sample_prices(),

        information_date=
            "2026-10-30",

        effective_from=
            "2026-11-02",

        allocation_state=
            "EQUAL_WEIGHT",

        model_version=
            "1.0.0",

        target_weights_hash=
            "target123",

        transaction_cost_bps=
            5.0,
    )

    assert (
        result.cash_after
        >= -1e-8
    )


def test_short_positions_rejected():
    holdings = (
        sample_holdings()
    )

    holdings[
        "SPY"
    ] = -10

    with pytest.raises(
        ValueError
    ):
        build_order_plan(
            target_weights=
                equal_weights(),

            current_shares=
                holdings,

            current_cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            information_date=
                "2026-10-30",

            effective_from=
                "2026-11-02",

            allocation_state=
                "EQUAL_WEIGHT",

            model_version=
                "1.0.0",

            target_weights_hash=
                "target123",
        )


def test_weight_record_adapter():
    result = (
        build_order_plan_from_weight_record(
            weight_record=
                sample_weight_record(),

            current_shares=
                sample_holdings(),

            current_cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            transaction_cost_bps=
                0.0,
        )
    )

    assert (
        result.information_date
        == pd.Timestamp(
            "2026-10-30"
        )
    )

    assert (
        result.effective_from
        == pd.Timestamp(
            "2026-11-02"
        )
    )

    assert (
        result.allocation_state
        == "EQUAL_WEIGHT"
    )


def test_execution_mode_is_always_dry_run():
    result = (
        build_order_plan_from_weight_record(
            weight_record=
                sample_weight_record(),

            current_shares=
                sample_holdings(),

            current_cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            transaction_cost_bps=
                0.0,
        )
    )

    assert (
        result.execution_mode
        == EXECUTION_MODE
    )

    assert (
        result.execution_mode
        == "DRY_RUN"
    )


def test_order_plan_hash_is_deterministic():
    kwargs = {
        "weight_record":
            sample_weight_record(),

        "current_shares":
            sample_holdings(),

        "current_cash":
            20_000.0,

        "reference_prices":
            sample_prices(),

        "transaction_cost_bps":
            0.0,
    }

    first = (
        build_order_plan_from_weight_record(
            **kwargs
        )
    )

    second = (
        build_order_plan_from_weight_record(
            **kwargs
        )
    )

    assert (
        first.order_plan_hash
        == second.order_plan_hash
    )
def sample_portfolio_snapshot():
    return build_portfolio_snapshot(
        holdings=
            sample_holdings(),

        cash=
            20_000.0,

        reference_prices=
            sample_prices(),

        snapshot_timestamp=
            "2026-11-02T14:30:00+00:00",

        price_as_of=
            "2026-10-30T20:00:00+00:00",

        account_mode=
            "PAPERMONEY",
    )


def test_build_order_plan_from_snapshot():
    snapshot = (
        sample_portfolio_snapshot()
    )

    result = (
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                snapshot,

            transaction_cost_bps=
                5.0,
        )
    )

    assert np.isclose(
        result.portfolio_value_before,
        snapshot.portfolio_value,
    )

    assert np.isclose(
        result.cash_before,
        snapshot.cash,
    )

    assert (
        result.holdings_snapshot_hash
        == snapshot.holdings_snapshot_hash
    )

    assert (
        result.price_snapshot_hash
        == snapshot.price_snapshot_hash
    )


def test_snapshot_adapter_matches_direct_order_plan():
    snapshot = (
        sample_portfolio_snapshot()
    )

    from_snapshot = (
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                snapshot,

            transaction_cost_bps=
                5.0,
        )
    )

    direct = (
        build_order_plan_from_weight_record(
            weight_record=
                sample_weight_record(),

            current_shares=
                sample_holdings(),

            current_cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            transaction_cost_bps=
                5.0,
        )
    )

    assert (
        from_snapshot.order_plan_hash
        == direct.order_plan_hash
    )

    pd.testing.assert_frame_equal(
        from_snapshot.orders,
        direct.orders,
    )


def test_snapshot_before_effective_date_rejected():
    snapshot = (
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-01T20:00:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",

            account_mode=
                "PAPERMONEY",
        )
    )

    with pytest.raises(
        ValueError,
        match="predates",
    ):
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                snapshot,
        )


def test_snapshot_adapter_remains_dry_run():
    result = (
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                sample_portfolio_snapshot(),

            transaction_cost_bps=
                5.0,
        )
    )

    assert (
        result.execution_mode
        == "DRY_RUN"
    )


def test_snapshot_adapter_preserves_trade_plan():
    result = (
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                sample_portfolio_snapshot(),

            transaction_cost_bps=
                0.0,
        )
    )

    orders = (
        result.orders
        .set_index(
            "asset"
        )
    )

    assert (
        orders.loc[
            "SPY",
            "trade_shares",
        ]
        == -60
    )

    assert (
        orders.loc[
            "QQQ",
            "trade_shares",
        ]
        == -17
    )

    assert (
        orders.loc[
            "TLT",
            "trade_shares",
        ]
        == 200
    )

    assert (
        orders.loc[
            "GLD",
            "trade_shares",
        ]
        == 100
    )

    assert (
        orders.loc[
            "SCHD",
            "trade_shares",
        ]
        == 250
    )
