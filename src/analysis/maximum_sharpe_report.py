from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.forecasts.expected_returns import (
    estimate_expected_returns,
)

from src.optimizers.maximum_sharpe import (
    calculate_portfolio_expected_return,
    optimize_maximum_sharpe,
)

from src.optimizers.minimum_variance import (
    build_equal_weight_portfolio,
)

from src.portfolio.returns import (
    build_return_matrix,
)

from src.risk.covariance import (
    calculate_ledoit_wolf_covariance,
)

from src.risk.portfolio_risk import (
    calculate_percentage_risk_contribution,
    calculate_portfolio_volatility,
    risk_concentration_index,
)


def calculate_sharpe(
    expected_return: float,
    volatility: float,
    risk_free_rate: float,
) -> float:
    return (
        expected_return
        - risk_free_rate
    ) / volatility


def main() -> None:
    prices = pd.read_csv(
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = (
        build_return_matrix(
            prices
        )
    )

    covariance, shrinkage = (
        calculate_ledoit_wolf_covariance(
            returns
        )
    )

    forecasts = (
        estimate_expected_returns(
            returns,
            ewma_span=126,
        )
    )

    expected_returns = (
        forecasts.robust
    )

    # V1 uses zero so we can inspect the optimizer mechanics
    # independently of a changing Treasury/cash rate.
    # Later this becomes a market-data input.
    risk_free_rate = 0.0

    result = (
        optimize_maximum_sharpe(
            expected_returns=
                expected_returns,
            covariance=
                covariance,
            risk_free_rate=
                risk_free_rate,
            max_weight=0.40,
        )
    )

    equal_weights = (
        build_equal_weight_portfolio(
            covariance
        )
    )

    equal_return = (
        calculate_portfolio_expected_return(
            equal_weights,
            expected_returns,
        )
    )

    equal_volatility = (
        calculate_portfolio_volatility(
            equal_weights,
            covariance,
        )
    )

    equal_sharpe = (
        calculate_sharpe(
            expected_return=
                equal_return,
            volatility=
                equal_volatility,
            risk_free_rate=
                risk_free_rate,
        )
    )

    max_sharpe_risk = (
        calculate_percentage_risk_contribution(
            result.weights,
            covariance,
        )
    )

    concentration = (
        risk_concentration_index(
            result.weights,
            covariance,
        )
    )

    report = pd.DataFrame(
        {
            "robust_expected_return_pct":
                expected_returns * 100,

            "max_sharpe_weight_pct":
                result.weights * 100,

            "risk_contribution_pct":
                max_sharpe_risk * 100,
        }
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    print()
    print("=" * 80)
    print("MAXIMUM-SHARPE PORTFOLIO")
    print("=" * 80)

    print()
    print(
        "Ledoit-Wolf shrinkage:",
        round(
            shrinkage,
            6,
        ),
    )

    print(
        "Risk-free rate:",
        f"{risk_free_rate:.2%}",
    )

    print()
    print("ASSET ALLOCATION")
    print("-" * 80)

    print(
        report
        .sort_values(
            "max_sharpe_weight_pct",
            ascending=False,
        )
        .round(2)
    )

    print()
    print("MAXIMUM-SHARPE METRICS")
    print("-" * 80)

    print(
        "Expected return:",
        f"{result.expected_return:.2%}",
    )

    print(
        "Annualized volatility:",
        f"{result.portfolio_volatility:.2%}",
    )

    print(
        "Expected Sharpe:",
        round(
            result.sharpe_ratio,
            3,
        ),
    )

    print(
        "Risk concentration:",
        round(
            concentration,
            4,
        ),
    )

    print()
    print("EQUAL-WEIGHT COMPARISON")
    print("-" * 80)

    print(
        "Expected return:",
        f"{equal_return:.2%}",
    )

    print(
        "Annualized volatility:",
        f"{equal_volatility:.2%}",
    )

    print(
        "Expected Sharpe:",
        round(
            equal_sharpe,
            3,
        ),
    )

    print()
    print("OPTIMIZER DIAGNOSTICS")
    print("-" * 80)

    print(
        "Iterations:",
        result.iterations,
    )

    print(
        "Success:",
        result.success,
    )

    print()


if __name__ == "__main__":
    main()
