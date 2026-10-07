from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from config.settings import DEFAULT_UNIVERSE


FROZEN_ASSETS = list(DEFAULT_UNIVERSE)

DEFAULT_TRANSACTION_COST_BPS = 5.0

EXECUTION_MODE = "DRY_RUN"


@dataclass(frozen=True)
class OrderPlanResult:
    information_date: pd.Timestamp
    effective_from: pd.Timestamp
    allocation_state: str
    model_version: str

    orders: pd.DataFrame

    portfolio_value_before: float
    cash_before: float
    cash_after: float

    gross_trade_notional: float
    turnover_ratio: float
    estimated_transaction_cost: float
    transaction_cost_bps: float

    target_weights_hash: str
    holdings_snapshot_hash: str
    price_snapshot_hash: str
    order_plan_hash: str

    execution_mode: str = EXECUTION_MODE

def _as_series(
    values: dict | pd.Series,
    name: str,
) -> pd.Series:
    series = pd.Series(
        values,
        dtype=float,
        name=name,
    )

    return series


def extract_target_weights(
    weight_record: dict,
) -> pd.Series:
    missing = [
        asset
        for asset in FROZEN_ASSETS
        if f"weight_{asset}"
        not in weight_record
    ]

    if missing:
        raise ValueError(
            "Weight record missing target weights "
            f"for: {missing}"
        )

    weights = pd.Series(
        {
            asset:
                float(
                    weight_record[
                        f"weight_{asset}"
                    ]
                )
            for asset in FROZEN_ASSETS
        },
        dtype=float,
        name="target_weight",
    )

    validate_target_weights(
        weights
    )

    return weights


def validate_target_weights(
    weights: pd.Series,
) -> None:
    weights = weights.reindex(
        FROZEN_ASSETS
    )

    if weights.isna().any():
        raise ValueError(
            "Target weights are missing frozen assets."
        )

    if not np.isfinite(
        weights.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Target weights contain non-finite values."
        )

    if (
        weights < -1e-12
    ).any():
        raise ValueError(
            "Negative target weights are not allowed."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "Target weights must sum to 1.0. "
            f"Actual={weights.sum():.12f}"
        )


def validate_weight_record(
    weight_record: dict,
) -> None:
    required = {
        "information_date",
        "decision_signal_date",
        "effective_from",
        "allocation_state",
        "model_version",
        "target_weights_hash",
    }

    missing = (
        required
        - set(weight_record)
    )

    if missing:
        raise ValueError(
            "Weight record missing required fields: "
            f"{sorted(missing)}"
        )

    information_date = pd.Timestamp(
        weight_record[
            "information_date"
        ]
    ).normalize()

    effective_from = pd.Timestamp(
        weight_record[
            "effective_from"
        ]
    ).normalize()

    if effective_from <= information_date:
        raise ValueError(
            "effective_from must be after "
            "information_date."
        )

    state = str(
        weight_record[
            "allocation_state"
        ]
    ).strip().upper()

    if state not in {
        "EQUAL_WEIGHT",
        "MAXIMUM_SHARPE",
    }:
        raise ValueError(
            f"Invalid allocation state: {state}"
        )

    extract_target_weights(
        weight_record
    )


def validate_portfolio_inputs(
    current_shares: pd.Series,
    current_cash: float,
    reference_prices: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    float,
]:
    shares = (
        current_shares
        .reindex(
            FROZEN_ASSETS
        )
        .fillna(0.0)
        .astype(float)
    )

    prices = (
        reference_prices
        .reindex(
            FROZEN_ASSETS
        )
        .astype(float)
    )

    if prices.isna().any():
        missing = (
            prices[
                prices.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Missing reference prices for: "
            f"{missing}"
        )

    if not np.isfinite(
        shares.to_numpy()
    ).all():
        raise ValueError(
            "Current shares contain non-finite values."
        )

    if not np.isfinite(
        prices.to_numpy()
    ).all():
        raise ValueError(
            "Reference prices contain non-finite values."
        )

    if (
        shares < 0
    ).any():
        raise ValueError(
            "Short positions are not supported in V1."
        )

    if not np.allclose(
        shares.to_numpy(),
        np.floor(
            shares.to_numpy()
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "V1 requires whole-share holdings."
        )

    if (
        prices <= 0
    ).any():
        raise ValueError(
            "Reference prices must be positive."
        )

    current_cash = float(
        current_cash
    )

    if (
        not np.isfinite(
            current_cash
        )
        or current_cash < 0
    ):
        raise ValueError(
            "Current cash must be finite "
            "and non-negative."
        )

    return (
        shares.astype(int),
        prices,
        current_cash,
    )


def calculate_portfolio_value(
    current_shares: pd.Series,
    current_cash: float,
    reference_prices: pd.Series,
) -> float:
    security_value = float(
        (
            current_shares
            * reference_prices
        ).sum()
    )

    portfolio_value = (
        security_value
        + current_cash
    )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be positive."
        )

    return float(
        portfolio_value
    )


def estimate_transaction_cost(
    trade_shares: pd.Series,
    reference_prices: pd.Series,
    transaction_cost_bps: float,
) -> float:
    if (
        not np.isfinite(
            transaction_cost_bps
        )
        or transaction_cost_bps < 0
    ):
        raise ValueError(
            "transaction_cost_bps must be "
            "finite and non-negative."
        )

    gross_notional = float(
        (
            trade_shares.abs()
            * reference_prices
        ).sum()
    )

    return (
        gross_notional
        * transaction_cost_bps
        / 10000.0
    )


def calculate_cash_after(
    current_cash: float,
    trade_shares: pd.Series,
    reference_prices: pd.Series,
    transaction_cost_bps: float,
) -> tuple[
    float,
    float,
]:
    """
    Positive trade_shares = BUY.
    Negative trade_shares = SELL.
    """
    signed_trade_value = float(
        (
            trade_shares
            * reference_prices
        ).sum()
    )

    estimated_cost = (
        estimate_transaction_cost(
            trade_shares=
                trade_shares,
            reference_prices=
                reference_prices,
            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    cash_after = (
        current_cash
        - signed_trade_value
        - estimated_cost
    )

    return (
        float(cash_after),
        float(estimated_cost),
    )


def _repair_cash_deficit(
    target_shares: pd.Series,
    current_shares: pd.Series,
    target_dollars: pd.Series,
    reference_prices: pd.Series,
    current_cash: float,
    transaction_cost_bps: float,
) -> pd.Series:
    """
    Ensure the proposed whole-share portfolio never
    requires margin.

    If estimated costs push residual cash below zero,
    remove BUY shares one at a time using the asset whose
    removal creates the smallest incremental target-dollar
    error.
    """
    target_shares = (
        target_shares
        .copy()
        .astype(int)
    )

    while True:
        trade_shares = (
            target_shares
            - current_shares
        )

        cash_after, _ = (
            calculate_cash_after(
                current_cash=
                    current_cash,
                trade_shares=
                    trade_shares,
                reference_prices=
                    reference_prices,
                transaction_cost_bps=
                    transaction_cost_bps,
            )
        )

        if cash_after >= -1e-8:
            break

        buy_assets = [
            asset
            for asset in FROZEN_ASSETS
            if (
                target_shares.loc[
                    asset
                ]
                > current_shares.loc[
                    asset
                ]
            )
        ]

        if not buy_assets:
            raise RuntimeError(
                "Unable to create a cash-feasible "
                "order plan without margin."
            )

        marginal_errors = {}

        for asset in buy_assets:
            price = float(
                reference_prices.loc[
                    asset
                ]
            )

            shares_now = int(
                target_shares.loc[
                    asset
                ]
            )

            current_value = (
                shares_now
                * price
            )

            reduced_value = (
                (
                    shares_now
                    - 1
                )
                * price
            )

            target_value = float(
                target_dollars.loc[
                    asset
                ]
            )

            current_error = abs(
                target_value
                - current_value
            )

            reduced_error = abs(
                target_value
                - reduced_value
            )

            marginal_errors[
                asset
            ] = (
                reduced_error
                - current_error
            )

        asset_to_reduce = min(
            marginal_errors,
            key=lambda asset: (
                marginal_errors[
                    asset
                ],
                asset,
            ),
        )

        target_shares.loc[
            asset_to_reduce
        ] -= 1

    return target_shares


def sha256_order_plan(
    orders: pd.DataFrame,
    information_date: pd.Timestamp,
    effective_from: pd.Timestamp,
    allocation_state: str,
    model_version: str,
    target_weights_hash: str,
    holdings_snapshot_hash: str,
    price_snapshot_hash: str,
    cash_before: float,
    transaction_cost_bps: float,
) -> str:
    order_columns = [
        "asset",
        "current_shares",
        "target_shares",
        "trade_shares",
        "side",
        "reference_price",
        "target_weight",
    ]

    payload = {
        "information_date":
            pd.Timestamp(
                information_date
            ).strftime(
                "%Y-%m-%d"
            ),

        "effective_from":
            pd.Timestamp(
                effective_from
            ).strftime(
                "%Y-%m-%d"
            ),

        "allocation_state":
            str(
                allocation_state
            ).upper(),

        "model_version":
            str(
                model_version
            ),

        "target_weights_hash":
            str(
                target_weights_hash
            ),

        "holdings_snapshot_hash":
            holdings_snapshot_hash,

        "price_snapshot_hash":
            price_snapshot_hash,

        "cash_before":
            float(
                cash_before
            ),

        "transaction_cost_bps":
            float(
                transaction_cost_bps
            ),

        "orders":
            (
                orders[
                    order_columns
                ]
                .copy()
                .sort_values(
                    "asset"
                )
                .to_dict(
                    orient="records"
                )
            ),
    }

    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()

def build_order_plan(
    target_weights: pd.Series,
    current_shares: dict | pd.Series,
    current_cash: float,
    reference_prices: dict | pd.Series,
    information_date: str | pd.Timestamp,
    effective_from: str | pd.Timestamp,
    allocation_state: str,
    model_version: str,
    target_weights_hash: str,
    transaction_cost_bps: float = (
        DEFAULT_TRANSACTION_COST_BPS
    ),
) -> OrderPlanResult:
    validate_target_weights(
        target_weights
    )

    shares = _as_series(
        current_shares,
        "current_shares",
    )

    prices = _as_series(
        reference_prices,
        "reference_price",
    )

    (
        shares,
        prices,
        current_cash,
    ) = validate_portfolio_inputs(
        current_shares=
            shares,

        current_cash=
            current_cash,

        reference_prices=
            prices,
    )

    information_date = pd.Timestamp(
        information_date
    ).normalize()

    effective_from = pd.Timestamp(
        effective_from
    ).normalize()

    if effective_from <= information_date:
        raise ValueError(
            "effective_from must occur after "
            "information_date."
        )

    allocation_state = (
        str(
            allocation_state
        )
        .strip()
        .upper()
    )

    if allocation_state not in {
        "EQUAL_WEIGHT",
        "MAXIMUM_SHARPE",
    }:
        raise ValueError(
            "allocation_state must be "
            "EQUAL_WEIGHT or MAXIMUM_SHARPE."
        )

    model_version = str(
        model_version
    ).strip()

    if not model_version:
        raise ValueError(
            "model_version cannot be empty."
        )

    target_weights_hash = str(
        target_weights_hash
    ).strip()

    if not target_weights_hash:
        raise ValueError(
            "target_weights_hash cannot be empty."
        )

    # ------------------------------------------------------------------
    # Immutable input snapshots
    # ------------------------------------------------------------------

    holdings_snapshot_hash = (
        sha256_series_snapshot(
            series=
                shares.astype(
                    float
                ),

            assets=
                FROZEN_ASSETS,
        )
    )

    price_snapshot_hash = (
        sha256_series_snapshot(
            series=
                prices,

            assets=
                FROZEN_ASSETS,
        )
    )

    # ------------------------------------------------------------------
    # Beginning portfolio state
    # ------------------------------------------------------------------

    portfolio_value = (
        calculate_portfolio_value(
            current_shares=
                shares,

            current_cash=
                current_cash,

            reference_prices=
                prices,
        )
    )

    current_values = (
        shares
        * prices
    )

    current_weights = (
        current_values
        / portfolio_value
    )

    # ------------------------------------------------------------------
    # Desired target
    # ------------------------------------------------------------------

    target_weights = (
        target_weights
        .reindex(
            FROZEN_ASSETS
        )
        .astype(float)
    )

    target_dollars = (
        target_weights
        * portfolio_value
    )

    preliminary_target_shares = (
        np.floor(
            target_dollars
            / prices
        )
        .astype(int)
    )

    # Whole-share rounding plus transaction costs can create
    # a small cash deficit. Repair the plan rather than using margin.
    target_shares = (
        _repair_cash_deficit(
            target_shares=
                preliminary_target_shares,

            current_shares=
                shares,

            target_dollars=
                target_dollars,

            reference_prices=
                prices,

            current_cash=
                current_cash,

            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    # ------------------------------------------------------------------
    # Proposed trades
    # ------------------------------------------------------------------

    trade_shares = (
        target_shares
        - shares
    ).astype(int)

    (
        cash_after,
        estimated_cost,
    ) = calculate_cash_after(
        current_cash=
            current_cash,

        trade_shares=
            trade_shares,

        reference_prices=
            prices,

        transaction_cost_bps=
            transaction_cost_bps,
    )

    if cash_after < -1e-8:
        raise RuntimeError(
            "Order plan would require margin."
        )

    post_trade_values = (
        target_shares
        * prices
    )

    post_trade_portfolio_value = (
        float(
            post_trade_values.sum()
        )
        + cash_after
    )

    # ------------------------------------------------------------------
    # Accounting invariant
    #
    # No market move is modeled between planning and hypothetical
    # execution. Therefore the only reduction in portfolio value
    # should be estimated transaction costs.
    # ------------------------------------------------------------------

    expected_ending_value = (
        portfolio_value
        - estimated_cost
    )

    actual_ending_value = (
        post_trade_portfolio_value
    )

    if not np.isclose(
        actual_ending_value,
        expected_ending_value,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Order-plan accounting invariant failed.\n"
            f"Beginning portfolio value: "
            f"{portfolio_value:.12f}\n"
            f"Estimated transaction cost: "
            f"{estimated_cost:.12f}\n"
            f"Expected ending value: "
            f"{expected_ending_value:.12f}\n"
            f"Actual ending value: "
            f"{actual_ending_value:.12f}"
        )

    sides = pd.Series(
        np.where(
            trade_shares > 0,
            "BUY",
            np.where(
                trade_shares < 0,
                "SELL",
                "HOLD",
            ),
        ),
        index=FROZEN_ASSETS,
    )

    trade_notional = (
        trade_shares.abs()
        * prices
    )

    signed_trade_value = (
        trade_shares
        * prices
    )

    gross_trade_notional = float(
        trade_notional.sum()
    )

    turnover_ratio = (
        gross_trade_notional
        / portfolio_value
    )

    post_trade_weights = (
        post_trade_values
        / post_trade_portfolio_value
    )

    # ------------------------------------------------------------------
    # Order detail table
    # ------------------------------------------------------------------

    orders = pd.DataFrame(
        {
            "asset":
                FROZEN_ASSETS,

            "current_shares":
                shares.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=int
                ),

            "reference_price":
                prices.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "current_value":
                current_values.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "current_weight":
                current_weights.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "target_weight":
                target_weights.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "target_dollars":
                target_dollars.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "target_shares":
                target_shares.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=int
                ),

            "trade_shares":
                trade_shares.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=int
                ),

            "side":
                sides.reindex(
                    FROZEN_ASSETS
                ).to_numpy(),

            "trade_notional":
                trade_notional.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "signed_trade_value":
                signed_trade_value.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "post_trade_value":
                post_trade_values.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),

            "post_trade_weight":
                post_trade_weights.reindex(
                    FROZEN_ASSETS
                ).to_numpy(
                    dtype=float
                ),
        }
    )

    # ------------------------------------------------------------------
    # Deterministic execution-plan hash
    # ------------------------------------------------------------------

    plan_hash = (
        sha256_order_plan(
            orders=
                orders,

            information_date=
                information_date,

            effective_from=
                effective_from,

            allocation_state=
                allocation_state,

            model_version=
                model_version,

            target_weights_hash=
                target_weights_hash,

            holdings_snapshot_hash=
                holdings_snapshot_hash,

            price_snapshot_hash=
                price_snapshot_hash,

            cash_before=
                current_cash,

            transaction_cost_bps=
                transaction_cost_bps,
        )
    )

    return OrderPlanResult(
        information_date=
            information_date,

        effective_from=
            effective_from,

        allocation_state=
            allocation_state,

        model_version=
            model_version,

        orders=
            orders,

        portfolio_value_before=
            float(
                portfolio_value
            ),

        cash_before=
            float(
                current_cash
            ),

        cash_after=
            float(
                cash_after
            ),

        gross_trade_notional=
            gross_trade_notional,

        turnover_ratio=
            float(
                turnover_ratio
            ),

        estimated_transaction_cost=
            float(
                estimated_cost
            ),

        transaction_cost_bps=
            float(
                transaction_cost_bps
            ),

        target_weights_hash=
            target_weights_hash,

        holdings_snapshot_hash=
            holdings_snapshot_hash,

        price_snapshot_hash=
            price_snapshot_hash,

        order_plan_hash=
            plan_hash,
    )

def build_order_plan_from_weight_record(
    weight_record: dict,
    current_shares: dict | pd.Series,
    current_cash: float,
    reference_prices: dict | pd.Series,
    transaction_cost_bps: float = (
        DEFAULT_TRANSACTION_COST_BPS
    ),
) -> OrderPlanResult:
    """
    Convert a validated Meta Allocation V1 target-weight
    record into a hypothetical whole-share rebalance plan.

    This function does not submit orders and has no broker
    connectivity.
    """
    validate_weight_record(
        weight_record
    )

    target_weights = (
        extract_target_weights(
            weight_record
        )
    )

    return build_order_plan(
        target_weights=
            target_weights,

        current_shares=
            current_shares,

        current_cash=
            current_cash,

        reference_prices=
            reference_prices,

        information_date=
            weight_record[
                "information_date"
            ],

        effective_from=
            weight_record[
                "effective_from"
            ],

        allocation_state=
            weight_record[
                "allocation_state"
            ],

        model_version=
            weight_record[
                "model_version"
            ],

        target_weights_hash=
            weight_record[
                "target_weights_hash"
            ],

        transaction_cost_bps=
            transaction_cost_bps,
    )
def sha256_series_snapshot(
    series: pd.Series,
    assets: list[str],
) -> str:
    payload = [
        {
            "asset": asset,
            "value": float(
                series.loc[
                    asset
                ]
            ),
        }
        for asset in assets
    ]

    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()

def test_plan_hash_changes_when_cash_changes():
    first = build_order_plan(
        target_weights=equal_weights(),
        current_shares=sample_holdings(),
        current_cash=20_000.0,
        reference_prices=sample_prices(),
        information_date="2026-10-30",
        effective_from="2026-11-02",
        allocation_state="EQUAL_WEIGHT",
        model_version="1.0.0",
        target_weights_hash="target123",
        transaction_cost_bps=0.0,
    )

    second = build_order_plan(
        target_weights=equal_weights(),
        current_shares=sample_holdings(),
        current_cash=20_001.0,
        reference_prices=sample_prices(),
        information_date="2026-10-30",
        effective_from="2026-11-02",
        allocation_state="EQUAL_WEIGHT",
        model_version="1.0.0",
        target_weights_hash="target123",
        transaction_cost_bps=0.0,
    )

    assert (
        first.order_plan_hash
        != second.order_plan_hash
    )


def test_plan_hash_changes_when_price_changes():
    changed_prices = sample_prices()
    changed_prices["SPY"] = 501.0

    first = build_order_plan(
        target_weights=equal_weights(),
        current_shares=sample_holdings(),
        current_cash=20_000.0,
        reference_prices=sample_prices(),
        information_date="2026-10-30",
        effective_from="2026-11-02",
        allocation_state="EQUAL_WEIGHT",
        model_version="1.0.0",
        target_weights_hash="target123",
        transaction_cost_bps=0.0,
    )

    second = build_order_plan(
        target_weights=equal_weights(),
        current_shares=sample_holdings(),
        current_cash=20_000.0,
        reference_prices=changed_prices,
        information_date="2026-10-30",
        effective_from="2026-11-02",
        allocation_state="EQUAL_WEIGHT",
        model_version="1.0.0",
        target_weights_hash="target123",
        transaction_cost_bps=0.0,
    )

    assert (
        first.order_plan_hash
        != second.order_plan_hash
    )


def test_portfolio_value_conservation_after_costs():
    result = build_order_plan(
        target_weights=equal_weights(),
        current_shares=sample_holdings(),
        current_cash=20_000.0,
        reference_prices=sample_prices(),
        information_date="2026-10-30",
        effective_from="2026-11-02",
        allocation_state="EQUAL_WEIGHT",
        model_version="1.0.0",
        target_weights_hash="target123",
        transaction_cost_bps=5.0,
    )

    ending_value = (
        result.orders[
            "post_trade_value"
        ].sum()
        + result.cash_after
    )

    assert np.isclose(
        ending_value,
        (
            result.portfolio_value_before
            - result.estimated_transaction_cost
        ),
        atol=1e-8,
    )