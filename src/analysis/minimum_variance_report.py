from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR
from src.optimizers.minimum_variance import (
    build_equal_weight_portfolio,
    optimize_minimum_variance,
)
from src.portfolio.returns import (
    build_return_matrix,
    summarize_returns,
)
from src.risk.covariance import (
    calculate_ledoit_wolf_covariance,
    calculate_sample_covariance,
    covariance_comparison_table,
)
from src.risk.portfolio_risk import (
    calculate_portfolio_volatility,
    risk_concentration_index,
    risk_contribution_table,
)


def main() -> None:
    price_path = (
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv"
    )

    prices = pd.read_csv(
        price_path,
        index_col=0,
        parse_dates=True,
    )

    returns = build_return_matrix(
        prices
    )

    asset_summary = summarize_returns(
        prices
    )

    sample_covariance = (
        calculate_sample_covariance(
            returns
        )
    )

    ledoit_wolf_covariance, shrinkage = (
        calculate_ledoit_wolf_covariance(
            returns
        )
    )

    equal_weights = (
        build_equal_weight_portfolio(
            ledoit_wolf_covariance
        )
    )

    # Practical V1 constraint:
    # no individual holding may exceed 40%.
    sample_result = (
        optimize_minimum_variance(
            covariance=sample_covariance,
            max_weight=0.40,
        )
    )

    lw_result = (
        optimize_minimum_variance(
            covariance=ledoit_wolf_covariance,
            max_weight=0.40,
        )
    )

    equal_volatility = (
        calculate_portfolio_volatility(
            equal_weights,
            ledoit_wolf_covariance,
        )
    )

    allocations = pd.DataFrame(
        {
            "equal_weight": equal_weights,
            "sample_min_variance":
                sample_result.weights,
            "ledoit_wolf_min_variance":
                lw_result.weights,
        }
    )

    lw_risk = (
        risk_contribution_table(
            lw_result.weights,
            ledoit_wolf_covariance,
        )
    )

    risk_comparison = pd.DataFrame(
        {
            "weight_pct":
                lw_result.weights * 100,
            "risk_contribution_pct":
                (
                    lw_risk[
                        "percentage_risk_contribution"
                    ]
                    * 100
                ),
            "asset_volatility_pct":
                (
                    lw_risk[
                        "asset_volatility"
                    ]
                    * 100
                ),
        }
    )

    covariance_diagnostics = (
        covariance_comparison_table(
            returns
        )
    )

    print()
    print("=" * 72)
    print("PORTFOLIO OPTIMIZATION ENGINE")
    print("MINIMUM-VARIANCE ANALYSIS")
    print("=" * 72)

    print()
    print("DATASET")
    print("-" * 72)
    print(
        f"Start date:   "
        f"{returns.index.min().date()}"
    )
    print(
        f"End date:     "
        f"{returns.index.max().date()}"
    )
    print(
        f"Observations: "
        f"{len(returns):,}"
    )
    print(
        f"Assets:       "
        f"{', '.join(returns.columns)}"
    )

    print()
    print("ASSET STATISTICS")
    print("-" * 72)

    summary_display = (
        asset_summary.copy()
    )

    summary_display[
        "total_return"
    ] *= 100

    summary_display[
        "annualized_return"
    ] *= 100

    summary_display[
        "annualized_volatility"
    ] *= 100

    print(
        summary_display.round(2)
    )

    print()
    print("COVARIANCE ESTIMATOR DIAGNOSTICS")
    print("-" * 72)

    print(
        covariance_diagnostics.round(6)
    )

    print()
    print(
        "Ledoit-Wolf shrinkage:",
        round(shrinkage, 6),
    )

    print()
    print("PORTFOLIO ALLOCATIONS (%)")
    print("-" * 72)

    print(
        (
            allocations * 100
        ).round(2)
    )

    print()
    print("ANNUALIZED PORTFOLIO VOLATILITY")
    print("-" * 72)

    volatility_table = pd.Series(
        {
            "Equal Weight":
                equal_volatility * 100,

            "Sample Minimum Variance":
                sample_result.portfolio_volatility
                * 100,

            "Ledoit-Wolf Minimum Variance":
                lw_result.portfolio_volatility
                * 100,
        },
        name="volatility_pct",
    )

    print(
        volatility_table.round(2)
    )

    print()
    print(
        "LEDOIT-WOLF MINIMUM-VARIANCE "
        "RISK DECOMPOSITION"
    )
    print("-" * 72)

    print(
        risk_comparison.round(2)
    )

    print()
    print("RISK CONCENTRATION")
    print("-" * 72)

    concentration = (
        risk_concentration_index(
            lw_result.weights,
            ledoit_wolf_covariance,
        )
    )

    print(
        round(concentration, 6)
    )

    print()
    print("OPTIMIZER DIAGNOSTICS")
    print("-" * 72)

    print(
        "Sample covariance:"
    )
    print(
        f"  Success:     "
        f"{sample_result.success}"
    )
    print(
        f"  Iterations:  "
        f"{sample_result.iterations}"
    )
    print(
        f"  Message:     "
        f"{sample_result.message}"
    )

    print()

    print(
        "Ledoit-Wolf:"
    )
    print(
        f"  Success:     "
        f"{lw_result.success}"
    )
    print(
        f"  Iterations:  "
        f"{lw_result.iterations}"
    )
    print(
        f"  Message:     "
        f"{lw_result.message}"
    )


if __name__ == "__main__":
    main()
