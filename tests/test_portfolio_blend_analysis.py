from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.portfolio_blend_analysis import (
    blend_weights,
    build_blend_table,
    turnover_from_personal,
    validate_blend_weights,
)


def _returns() -> pd.DataFrame:
    dates = pd.bdate_range("2025-10-01", periods=260)
    x = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "AAA": 0.0004 + 0.010 * np.sin(x / 9.0),
            "BBB": 0.0003 + 0.008 * np.cos(x / 11.0),
            "CCC": 0.0005 + 0.012 * np.sin(x / 13.0),
        },
        index=dates,
    )


def test_validate_blend_weights_sorts_and_deduplicates() -> None:
    assert validate_blend_weights([0.3, 0.0, 0.3, 1.0]) == [
        0.0,
        0.3,
        1.0,
    ]


def test_validate_blend_weights_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="\[0, 1\]"):
        validate_blend_weights([0.0, 1.1])


def test_blend_endpoints_match_source_portfolios() -> None:
    personal = pd.Series({"AAA": 0.7, "BBB": 0.3})
    optimizer = pd.Series({"BBB": 0.2, "CCC": 0.8})

    zero = blend_weights(
        personal,
        optimizer,
        optimizer_weight=0.0,
    )
    full = blend_weights(
        personal,
        optimizer,
        optimizer_weight=1.0,
    )

    assert zero["AAA"] == pytest.approx(0.7)
    assert zero["BBB"] == pytest.approx(0.3)
    assert zero["CCC"] == pytest.approx(0.0)

    assert full["AAA"] == pytest.approx(0.0)
    assert full["BBB"] == pytest.approx(0.2)
    assert full["CCC"] == pytest.approx(0.8)


def test_turnover_scales_with_optimizer_blend_when_no_overlap() -> None:
    personal = pd.Series({"AAA": 1.0})
    optimizer = pd.Series({"BBB": 1.0})
    blended = blend_weights(
        personal,
        optimizer,
        optimizer_weight=0.3,
    )

    assert turnover_from_personal(
        personal,
        blended,
    ) == pytest.approx(0.3)


def test_build_blend_table_contains_expected_endpoints() -> None:
    returns = _returns()
    personal = pd.Series({"AAA": 0.7, "BBB": 0.3})
    optimizer = pd.Series({"BBB": 0.2, "CCC": 0.8})

    table = build_blend_table(
        returns=returns,
        personal_weights=personal,
        optimizer_weights=optimizer,
        optimizer_blend_weights=(0.0, 0.1, 0.3, 1.0),
    )

    assert table["optimizer_blend_weight"].tolist() == [
        0.0,
        0.1,
        0.3,
        1.0,
    ]
    assert table.loc[0, "turnover_from_personal"] == pytest.approx(0.0)
    assert table.loc[3, "turnover_from_personal"] > table.loc[1, "turnover_from_personal"]
    assert "worst_rolling_20d_return" in table.columns
    assert "worst_rolling_60d_return" in table.columns
    assert table["concentration"].gt(0).all()
