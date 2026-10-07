from __future__ import annotations

from src.integrations.schwab.auth_bootstrap import (
    create_read_only_schwab_client,
    load_schwab_auth_config,
)

from src.forward.meta_allocation_v1_portfolio_snapshot import (
    FROZEN_ASSETS,
)


def main() -> None:
    print()
    print("=" * 88)
    print("SCHWAB READ-ONLY CONNECTIVITY SMOKE TEST")
    print("=" * 88)

    # --------------------------------------------------
    # 1. Load local OAuth configuration
    # --------------------------------------------------

    config = load_schwab_auth_config()

    print("Configuration loaded: YES")
    print("Credentials printed: NO")
    print("Token printed: NO")

    # --------------------------------------------------
    # 2. Authenticate and immediately wrap
    #    raw client in read-only boundary
    # --------------------------------------------------

    client = create_read_only_schwab_client(
        config
    )

    print("Authenticated client created: YES")
    print("Read-only wrapper active: YES")

    # --------------------------------------------------
    # 3. Discover accessible accounts
    # --------------------------------------------------

    records = (
        client
        .get_account_number_records()
    )

    print(
        "Accessible accounts:",
        len(records),
    )

    # Do not print account numbers or hashes.

    if len(records) != 1:
        print()
        print(
            "STOP: multiple accounts are accessible."
        )
        print(
            "Explicit account selection must be "
            "implemented before continuing."
        )
        return

    account_hash = (
        records[0][
            "hashValue"
        ]
    )

    # --------------------------------------------------
    # 4. Retrieve balances + positions
    # --------------------------------------------------

    account_payload = (
        client
        .get_account_payload(
            account_hash
        )
    )

    account = (
        account_payload
        .get(
            "securitiesAccount",
            {}
        )
    )

    positions = (
        account.get(
            "positions",
            []
        )
        or []
    )

    symbols = []

    for position in positions:
        instrument = (
            position.get(
                "instrument",
                {}
            )
        )

        symbol = str(
            instrument.get(
                "symbol",
                "",
            )
        ).strip().upper()

        if symbol:
            symbols.append(
                symbol
            )

    print()
    print("ACCOUNT READ")
    print("-" * 88)

    print(
        "Positions retrieved:",
        len(positions),
    )

    print(
        "Symbols present:",
        ", ".join(
            sorted(
                set(symbols)
            )
        )
        if symbols
        else "NONE",
    )

    # --------------------------------------------------
    # 5. Retrieve only the five strategy quotes
    # --------------------------------------------------

    quotes = (
        client
        .get_quote_payload(
            FROZEN_ASSETS
        )
    )

    print()
    print("MARKET DATA READ")
    print("-" * 88)

    print(
        "Requested symbols:",
        ", ".join(
            FROZEN_ASSETS
        ),
    )

    print(
        "Quotes returned:",
        len(quotes),
    )

    print(
        "Quote symbols:",
        ", ".join(
            sorted(
                quotes.keys()
            )
        ),
    )

    # --------------------------------------------------
    # 6. Architectural safety checks
    # --------------------------------------------------

    print()
    print("SAFETY BOUNDARY")
    print("-" * 88)

    print(
        "place_order exposed:",
        hasattr(
            client,
            "place_order",
        ),
    )

    print(
        "replace_order exposed:",
        hasattr(
            client,
            "replace_order",
        ),
    )

    print(
        "cancel_order exposed:",
        hasattr(
            client,
            "cancel_order",
        ),
    )

    print(
        "preview_order exposed:",
        hasattr(
            client,
            "preview_order",
        ),
    )

    print()
    print("ORDERS GENERATED: NO")
    print("ORDERS SUBMITTED: NO")
    print("ACCOUNT HASH PRINTED: NO")
    print("ACCOUNT NUMBER PRINTED: NO")
    print("TOKEN PRINTED: NO")

    print()
    print("=" * 88)
    print("READ-ONLY CONNECTIVITY TEST COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()