import numpy as np
import pandas as pd
import pytest

from src.forward.meta_allocation_v1_portfolio_snapshot import (
    build_portfolio_snapshot,
)


def sample_holdings():
    return {
        "SPY": 100,
        "QQQ": 50,
        "TLT": 0,
        "GLD": 0,
        "SCHD": 0,
    }


def sample_prices():
    return {
        "SPY": 500.0,
        "QQQ": 600.0,
        "TLT": 100.0,
        "GLD": 200.0,
        "SCHD": 80.0,
    }


def build_sample_snapshot():
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


def test_snapshot_portfolio_value():
    snapshot = (
        build_sample_snapshot()
    )

    assert np.isclose(
        snapshot.holdings_value,
        80_000.0,
    )

    assert np.isclose(
        snapshot.portfolio_value,
        100_000.0,
    )


def test_snapshot_preserves_frozen_asset_order():
    snapshot = (
        build_sample_snapshot()
    )

    assert list(
        snapshot.current_shares.index
    ) == [
        "SPY",
        "QQQ",
        "TLT",
        "GLD",
        "SCHD",
    ]

    assert list(
        snapshot.reference_prices.index
    ) == [
        "SPY",
        "QQQ",
        "TLT",
        "GLD",
        "SCHD",
    ]


def test_snapshot_hash_is_deterministic():
    first = (
        build_sample_snapshot()
    )

    second = (
        build_sample_snapshot()
    )

    assert (
        first.portfolio_snapshot_hash
        == second.portfolio_snapshot_hash
    )

    assert (
        first.holdings_snapshot_hash
        == second.holdings_snapshot_hash
    )

    assert (
        first.price_snapshot_hash
        == second.price_snapshot_hash
    )


def test_snapshot_hash_changes_when_cash_changes():
    first = (
        build_sample_snapshot()
    )

    second = (
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_001.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",

            account_mode=
                "PAPERMONEY",
        )
    )

    assert (
        first.portfolio_snapshot_hash
        != second.portfolio_snapshot_hash
    )


def test_holdings_hash_changes_when_shares_change():
    changed = sample_holdings()
    changed["SPY"] = 101

    first = (
        build_sample_snapshot()
    )

    second = (
        build_portfolio_snapshot(
            holdings=
                changed,

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
    )

    assert (
        first.holdings_snapshot_hash
        != second.holdings_snapshot_hash
    )


def test_price_hash_changes_when_price_changes():
    changed = sample_prices()
    changed["SPY"] = 501.0

    first = (
        build_sample_snapshot()
    )

    second = (
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                changed,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",

            account_mode=
                "PAPERMONEY",
        )
    )

    assert (
        first.price_snapshot_hash
        != second.price_snapshot_hash
    )


def test_negative_cash_rejected():
    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                -1.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_short_position_rejected():
    holdings = sample_holdings()
    holdings["SPY"] = -1

    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                holdings,

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_fractional_position_rejected():
    holdings = sample_holdings()
    holdings["SPY"] = 100.5

    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                holdings,

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_missing_price_rejected():
    prices = sample_prices()
    del prices["TLT"]

    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                prices,

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_unsupported_asset_rejected():
    holdings = sample_holdings()
    holdings["TMF"] = 100

    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                holdings,

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_duplicate_holdings_rejected():
    holdings = pd.DataFrame(
        {
            "asset": [
                "SPY",
                "SPY",
                "QQQ",
            ],

            "shares": [
                50,
                50,
                50,
            ],
        }
    )

    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                holdings,

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_stale_price_snapshot_rejected():
    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-15T14:30:00+00:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",

            max_price_age=
                "7D",
        )


def test_future_price_timestamp_rejected():
    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02T14:30:00+00:00",

            price_as_of=
                "2026-11-02T15:00:00+00:00",
        )


def test_naive_timestamp_rejected():
    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
            holdings=
                sample_holdings(),

            cash=
                20_000.0,

            reference_prices=
                sample_prices(),

            snapshot_timestamp=
                "2026-11-02 14:30:00",

            price_as_of=
                "2026-10-30T20:00:00+00:00",
        )


def test_live_account_mode_rejected():
    with pytest.raises(
        ValueError
    ):
        build_portfolio_snapshot(
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
                "LIVE",
        )