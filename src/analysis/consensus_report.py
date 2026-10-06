from __future__ import annotations

import pandas as pd

from config.settings import RAW_DATA_DIR

from src.ensemble.consensus import (
    build_consensus_portfolio,
)

from src.optimizers.hrp import (
    optimize_hrp,
)

from src.optimizers.minimum_variance import (
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


def effective_risk_positions(
    concentration: float,
) -> float:
    return (
        1.0 / concentration
    )


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

    model_weights = pd.DataFrame(
        {
            "minimum_variance":
                minimum_variance.weights,

            "risk_parity":
                risk_parity.weights,

            "hrp":
                hrp.weights,
        }
    )

    consensus = (
        build_consensus_portfolio(
            model_weights
        )
    )

    consensus_volatility = (
        calculate_portfolio_volatility(
            consensus.consensus_weights,
            covariance,
        )
    )

    consensus_risk = (
        calculate_percentage_risk_contribution(
            consensus.consensus_weights,
            covariance,
        )
    )

    concentration = (
        risk_concentration_index(
            consensus.consensus_weights,
            covariance,
        )
    )

    report = pd.DataFrame(
        {
            "minimum_variance_pct":
                minimum_variance.weights
                * 100,

            "risk_parity_pct":
                risk_parity.weights
                * 100,

            "hrp_pct":
                hrp.weights
                * 100,

            "consensus_pct":
                consensus.consensus_weights
                * 100,

            "allocation_std_pct":
                consensus.allocation_std
                * 100,

            "allocation_range_pct":
                consensus.allocation_range
                * 100,

            "consensus_risk_pct":
                consensus_risk
                * 100,
        }
    )

    report[
        "model_agreement"
    ] = pd.cut(
        consensus.disagreement_score,
        bins=[
            -float("inf"),
            0.50,
            1.00,
            float("inf"),
        ],
        labels=[
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
    )

    print()
    print("=" * 80)
    print(
        "PORTFOLIO MODEL CONSENSUS"
    )
    print("=" * 80)

    print()
    print(
        "Ledoit-Wolf shrinkage:",
        round(
            shrinkage,
            6,
        ),
    )

    print()
    print(
        "MODEL CONSENSUS TABLE"
    )
    print("-" * 80)

    print(
        report.round(2)
    )

    print()
    print(
        "CONSENSUS PORTFOLIO"
    )
    print("-" * 80)

    print(
        (
            consensus
            .consensus_weights
            * 100
        )
        .sort_values(
            ascending=False
        )
        .round(2)
    )

    print()
    print(
        "CONSENSUS PORTFOLIO METRICS"
    )
    print("-" * 80)

    print(
        "Annualized volatility:",
        f"{consensus_volatility:.2%}",
    )

    print(
        "Risk concentration:",
        round(
            concentration,
            4,
        ),
    )

    print(
        "Effective risk positions:",
        round(
            effective_risk_positions(
                concentration
            ),
            2,
        ),
    )

    print(
        "Overall model disagreement:",
        round(
            consensus
            .overall_disagreement
            * 100,
            2,
        ),
        "percentage points",
    )

    print()
    print(
        "MODEL DISAGREEMENT BY ASSET"
    )
    print("-" * 80)

    disagreement = pd.DataFrame(
        {
            "allocation_range_pct":
                consensus
                .allocation_range
                * 100,

            "allocation_std_pct":
                consensus
                .allocation_std
                * 100,

            "relative_disagreement":
                consensus
                .disagreement_score,
        }
    )

    print(
        disagreement
        .sort_values(
            "allocation_range_pct",
            ascending=False,
        )
        .round(3)
    )


if __name__ == "__main__":
    main()
