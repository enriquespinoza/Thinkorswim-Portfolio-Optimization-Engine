from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.portfolio_blend_forward_validation import (
    build_forward_blend_table,
    forward_price_returns,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-19", periods=12)
    x = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 * np.cumprod(1.0 + 0.002 + 0.001 * np.sin(x)),
            "BBB": 80.0 * np.cumprod(1.0 + 0.001 + 0.001 * np.cos(x)),
        },
        index=dates,
    )


def test_forward_returns_start_after_freeze_session() -> None:
    prices = _prices()
    returns = forward_price_returns(
        prices,
        symbols=["AAA", "BBB"],
        freeze_session="2026-10-22",
        evaluation_session="2026-10-30",
    )

    assert (returns.index > pd.Timestamp("2026-10-22")).all()
    assert (returns.index <= pd.Timestamp("2026-10-30")).all()


def test_forward_returns_reject_nonforward_evaluation() -> None:
    with pytest.raises(ValueError, match="after freeze_session"):
        forward_price_returns(
            _prices(),
            symbols=["AAA"],
            freeze_session="2026-10-22",
            evaluation_session="2026-10-22",
        )


def test_forward_table_preserves_frozen_blends() -> None:
    returns = forward_price_returns(
        _prices(),
        symbols=["AAA", "BBB"],
        freeze_session="2026-10-22",
        evaluation_session="2026-10-30",
    )
    personal = pd.Series({"AAA": 1.0})
    optimizer = pd.Series({"BBB": 1.0})

    table, daily = build_forward_blend_table(
        returns=returns,
        personal_weights=personal,
        optimizer_weights=optimizer,
        optimizer_blend_weights=(0.0, 0.2, 0.3, 1.0),
    )

    assert table["optimizer_blend_weight"].tolist() == [
        0.0,
        0.2,
        0.3,
        1.0,
    ]
    assert list(daily.columns) == [
        "optimizer_000",
        "optimizer_020",
        "optimizer_030",
        "optimizer_100",
    ]
    assert len(daily) == len(returns)
