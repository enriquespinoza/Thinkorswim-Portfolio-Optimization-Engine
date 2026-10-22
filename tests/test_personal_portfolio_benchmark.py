from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.portfolio_comparison import (
    build_summary,
    compare_weights,
    concentration,
    risk_contribution,
)


def _returns() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=260)
    x = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "AAA": 0.0005 + 0.01 * np.sin(x / 11.0),
            "BBB": 0.0003 + 0.008 * np.cos(x / 9.0),
            "SPY": 0.0004 + 0.007 * np.sin(x / 13.0),
        },
        index=dates,
    )


def test_weight_comparison_metrics() -> None:
    personal = pd.Series({"AAA": 0.75, "BBB": 0.25})
    optimizer = pd.Series({"AAA": 0.50, "BBB": 0.50})

    metrics = compare_weights(personal, optimizer)

    assert metrics["portfolio_overlap"] == pytest.approx(0.75)
    assert metrics["weight_difference"] == pytest.approx(0.50)
    assert metrics["turnover"] == pytest.approx(0.25)


def test_concentration_detects_more_concentrated_portfolio() -> None:
    concentrated = pd.Series({"AAA": 0.90, "BBB": 0.10})
    balanced = pd.Series({"AAA": 0.50, "BBB": 0.50})

    assert concentration(concentrated) > concentration(balanced)


def test_risk_contributions_sum_to_one() -> None:
    returns = _returns()[["AAA", "BBB"]]
    weights = pd.Series({"AAA": 0.60, "BBB": 0.40})

    contribution = risk_contribution(returns, weights)

    assert contribution.sum() == pytest.approx(1.0)


def test_build_summary_is_read_only() -> None:
    returns = _returns()
    personal = pd.Series({"AAA": 0.75, "BBB": 0.25})
    optimizer = pd.Series({"AAA": 0.50, "BBB": 0.50})

    summary = build_summary(
        as_of_session="2026-10-16",
        personal_weights=personal,
        optimizer_weights=optimizer,
        returns=returns,
        benchmark_symbol="SPY",
    )

    assert summary["benchmark_version"] == "personal-portfolio-benchmark-v1"
    assert summary["read_only_research"] is True
    assert summary["order_submission_permitted"] is False
    assert 0.0 <= summary["portfolio_overlap"] <= 1.0
    assert summary["tracking_error"] >= 0.0
