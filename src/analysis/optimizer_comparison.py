from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.optimizers.minimum_variance import (
    build_equal_weight_portfolio,
    optimize_minimum_variance,
)

from src.optimizers.risk_parity import (
    optimize_risk_parity,
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

from src.optimizers.hrp import (
    optimize_hrp,
)

def effective_risk_positions(
    concentration: float,
) -> float:
    """
    Convert the risk concentration index into the
    effective number of equally important risk positions.
    """
    return 1.0 / concentration


def main() -> None:
    prices = pd.read_csv(
        RAW_DATA_DIR
        / "portfolio_universe_daily.csv",
        index_col=0,
        parse_dates=True,
    )

    returns = build_return_matrix(
        prices
    )

    covariance, shrinkage = (
        calculate_ledoit_wolf_covariance(
            returns
        )
    )

    equal_weights = (
        build_equal_weight_portfolio(
            covariance
        )
    )

    minimum_variance = (
        optimize_minimum_variance(
            covariance=covariance,
            max_weight=0.40,
        )
    )

    risk_parity = (
        optimize_risk_parity(
            covariance=covariance,
            max_weight=0.40,
        )
    )
    hrp = optimize_hrp(
        covariance=covariance
    )

    allocations = pd.DataFrame(
        {
            "equal_weight":
                equal_weights,

            "minimum_variance":
                minimum_variance.weights,

            "risk_parity":
                risk_parity.weights,

            "hrp":
                hrp.weights,
        }
    )

    equal_volatility = (
        calculate_portfolio_volatility(
            equal_weights,
            covariance,
        )
    )

    equal_concentration = (
        risk_concentration_index(
            equal_weights,
            covariance,
        )
    )

    minimum_concentration = (
        risk_concentration_index(
            minimum_variance.weights,
            covariance,
        )
    )

    risk_parity_concentration = (
        risk_concentration_index(
            risk_parity.weights,
            covariance,
        )
    )

    hrp_concentration = (
        risk_concentration_index(
            hrp.weights,
            covariance,
        )
    )

    metrics = pd.DataFrame(
        {
            "equal_weight": {
                "volatility_pct":
                    equal_volatility * 100,

                "risk_concentration":
                    equal_concentration,

                "effective_risk_positions":
                    effective_risk_positions(
                        equal_concentration
                    ),
            },

            "minimum_variance": {
                "volatility_pct":
                    minimum_variance.portfolio_volatility
                    * 100,

                "risk_concentration":
                    minimum_concentration,

                "effective_risk_positions":
                    effective_risk_positions(
                        minimum_concentration
                    ),
            },

            "risk_parity": {
                "volatility_pct":
                    risk_parity.portfolio_volatility
                    * 100,

                "risk_concentration":
                    risk_parity_concentration,

                "effective_risk_positions":
                    effective_risk_positions(
                        risk_parity_concentration
                    ),
            },

            "hrp": {
                "volatility_pct":
                    hrp.portfolio_volatility * 100,

                "risk_concentration":
                    hrp_concentration,

                "effective_risk_positions":
                    effective_risk_positions(
                        hrp_concentration
                    ),
            },
        }
    )

    equal_risk = (
        calculate_percentage_risk_contribution(
            equal_weights,
            covariance,
        )
    )

    minimum_risk = (
        calculate_percentage_risk_contribution(
            minimum_variance.weights,
            covariance,
        )
    )

    risk_table = pd.DataFrame(
        {
            "equal_weight_risk_pct":
                equal_risk * 100,

            "minimum_variance_risk_pct":
                minimum_risk * 100,

            "risk_parity_risk_pct":
                (
                    risk_parity
                    .percentage_risk_contribution
                    * 100
                ),

            "hrp_risk_pct":
                (
                    hrp
                    .percentage_risk_contribution
                    * 100
                ),
        }
    )

    print()
    print("=" * 72)
    print("PORTFOLIO OPTIMIZER COMPARISON")
    print("=" * 72)

    print()
    print(
        "Ledoit-Wolf shrinkage:",
        round(
            shrinkage,
            6,
        ),
    )

    print()
    print("ALLOCATIONS (%)")
    print("-" * 72)

    print(
        (
            allocations * 100
        ).round(2)
    )

    print()
    print("PORTFOLIO METRICS")
    print("-" * 72)

    print(
        metrics.round(4)
    )

    print()
    print("RISK CONTRIBUTIONS (%)")
    print("-" * 72)

    print(
        risk_table.round(2)
    )

    print()
    print("RISK PARITY DIAGNOSTICS")
    print("-" * 72)

    print(
        "Target risk contribution:",
        f"{risk_parity.target_risk_contribution:.2%}",
    )

    print(
        "Risk contribution RMSE:",
        round(
            risk_parity.risk_contribution_error,
            8,
        ),
    )

    print()
    print("HRP CLUSTER ORDER")
    print("-" * 72)

    print(
        " -> ".join(
            hrp.ordered_assets
        )
    )

    print(
        "Iterations:",
        risk_parity.iterations,
    )

    print(
        "Success:",
        risk_parity.success,
    )

    print(
        "Message:",
        risk_parity.message,
    )


if __name__ == "__main__":
    main()
