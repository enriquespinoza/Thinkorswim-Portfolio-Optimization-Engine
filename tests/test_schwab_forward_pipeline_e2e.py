from pathlib import Path

import pandas as pd

from src.integrations.schwab.auth_bootstrap import (
    SchwabAuthConfig,
    create_read_only_schwab_client,
)

from src.integrations.schwab.read_only_snapshot_adapter import (
    build_schwab_read_only_snapshot,
)

from src.forward.meta_allocation_v1_order_plan import (
    build_order_plan_from_snapshot,
)

from src.forward.meta_allocation_v1_order_plan_audit import (
    append_order_plan_audit,
)


class FakeResponse:
    def __init__(
        self,
        payload,
        status_code=200,
    ):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeRawSchwabClient:
    class Account:
        class Fields:
            POSITIONS = "positions"

    def __init__(self):
        quote_time = (
            pd.Timestamp.now(
                tz="UTC"
            )
            - pd.Timedelta(
                minutes=1
            )
        )

        quote_time_ms = int(
            quote_time.timestamp()
            * 1000
        )

        self.account_records = [
            {
                "accountNumber":
                    "11111111",

                "hashValue":
                    "HASH-ONE",
            }
        ]

        self.account_payload = {
            "securitiesAccount": {
                "currentBalances": {
                    "cashBalance":
                        20_000.0,
                },

                "positions": [
                    {
                        "instrument": {
                            "symbol":
                                "SPY",
                        },

                        "longQuantity":
                            100.0,

                        "shortQuantity":
                            0.0,
                    },

                    {
                        "instrument": {
                            "symbol":
                                "QQQ",
                        },

                        "longQuantity":
                            50.0,

                        "shortQuantity":
                            0.0,
                    },
                ],
            },
        }

        self.quote_payload = {
            "SPY": {
                "quote": {
                    "lastPrice":
                        500.0,

                    "quoteTime":
                        quote_time_ms,
                },
            },

            "QQQ": {
                "quote": {
                    "lastPrice":
                        600.0,

                    "quoteTime":
                        quote_time_ms,
                },
            },

            "TLT": {
                "quote": {
                    "lastPrice":
                        100.0,

                    "quoteTime":
                        quote_time_ms,
                },
            },

            "GLD": {
                "quote": {
                    "lastPrice":
                        200.0,

                    "quoteTime":
                        quote_time_ms,
                },
            },

            "SCHD": {
                "quote": {
                    "lastPrice":
                        80.0,

                    "quoteTime":
                        quote_time_ms,
                },
            },
        }

    def get_account_numbers(self):
        return FakeResponse(
            self.account_records
        )

    def get_account(
        self,
        account_hash,
        fields=None,
    ):
        assert (
            account_hash
            == "HASH-ONE"
        )

        assert fields == [
            "positions"
        ]

        return FakeResponse(
            self.account_payload
        )

    def get_quotes(
        self,
        symbols,
    ):
        requested = {
            symbol:
                self.quote_payload[
                    symbol
                ]
            for symbol in symbols
        }

        return FakeResponse(
            requested
        )


class FakeAuthFactory:
    def __init__(self):
        self.client = (
            FakeRawSchwabClient()
        )

        self.calls = []

    def __call__(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return self.client


def sample_weight_record():
    return {
        "information_date":
            "2026-09-30",

        "decision_signal_date":
            "2026-07-31",

        "effective_from":
            "2026-10-01",

        "allocation_state":
            "EQUAL_WEIGHT",

        "model_version":
            "1.0.0",

        "target_weights_hash":
            "synthetic-e2e-target",

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


def test_schwab_read_only_forward_pipeline_e2e(
    tmp_path: Path,
):
    # --------------------------------------------------
    # 1. Mock OAuth bootstrap
    # --------------------------------------------------

    config = SchwabAuthConfig(
        app_key=
            "TEST-APP-KEY",

        app_secret=
            "TEST-APP-SECRET",

        callback_url=
            "https://127.0.0.1:8182",

        token_path=
            tmp_path
            / "tokens"
            / "schwab_token.json",
    )

    factory = (
        FakeAuthFactory()
    )

    client = (
        create_read_only_schwab_client(
            config=
                config,

            auth_factory=
                factory,
        )
    )

    # --------------------------------------------------
    # 2. Read-only Schwab payload retrieval
    # --------------------------------------------------

    payloads = (
        client
        .fetch_meta_allocation_v1_payloads()
    )

    assert (
        payloads.account_hash
        == "HASH-ONE"
    )

    # --------------------------------------------------
    # 3. Broker-independent PortfolioSnapshot
    # --------------------------------------------------

    snapshot = (
        build_schwab_read_only_snapshot(
            payloads=
                payloads,

            max_price_age=
                "7D",
        )
    )

    assert (
        snapshot.account_mode
        == "SCHWAB_READ_ONLY"
    )

    assert (
        snapshot.cash
        == 20_000.0
    )

    assert (
        snapshot.portfolio_value
        == 100_000.0
    )

    # --------------------------------------------------
    # 4. Dry-run order plan
    # --------------------------------------------------

    plan = (
        build_order_plan_from_snapshot(
            weight_record=
                sample_weight_record(),

            snapshot=
                snapshot,

            transaction_cost_bps=
                5.0,
        )
    )

    assert (
        plan.execution_mode
        == "DRY_RUN"
    )

    assert (
        plan.holdings_snapshot_hash
        == snapshot.holdings_snapshot_hash
    )

    assert (
        plan.price_snapshot_hash
        == snapshot.price_snapshot_hash
    )

    # --------------------------------------------------
    # 5. Expected synthetic trade plan
    # --------------------------------------------------

    orders = (
        plan.orders
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

    # --------------------------------------------------
    # 6. Append-only audit
    # --------------------------------------------------

    summary_path = (
        tmp_path
        / "order_plan_summary.csv"
    )

    detail_path = (
        tmp_path
        / "order_plan_details.csv"
    )

    action = (
        append_order_plan_audit(
            result=
                plan,

            summary_path=
                summary_path,

            detail_path=
                detail_path,
        )
    )

    assert action == {
        "summary":
            "APPENDED",

        "details":
            "APPENDED",
    }

    summary = pd.read_csv(
        summary_path
    )

    details = pd.read_csv(
        detail_path
    )

    assert len(
        summary
    ) == 1

    assert len(
        details
    ) == 5

    assert (
        summary.loc[
            0,
            "execution_mode",
        ]
        == "DRY_RUN"
    )

    assert (
        summary.loc[
            0,
            "order_plan_hash",
        ]
        == plan.order_plan_hash
    )

    # --------------------------------------------------
    # 7. Architectural safety assertions
    # --------------------------------------------------

    assert not hasattr(
        client,
        "place_order",
    )

    assert not hasattr(
        client,
        "replace_order",
    )

    assert not hasattr(
        client,
        "cancel_order",
    )

    assert not hasattr(
        client,
        "preview_order",
    )

    # Mock auth must not have created a real token.
    assert not (
        config.token_path.exists()
    )