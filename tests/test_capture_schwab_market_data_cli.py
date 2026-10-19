from datetime import datetime, timezone

from scripts.capture_schwab_market_data import (
    build_parser,
)


def test_explicit_price_history_window_parses_as_utc():
    args = build_parser().parse_args(
        [
            "--symbols",
            "LITE",
            "--datasets",
            "price_history",
            "--price-history-start-datetime",
            "2024-10-15T00:00:00+00:00",
            "--price-history-end-datetime",
            "2026-10-20T00:00:00+00:00",
        ]
    )

    assert args.price_history_start_datetime == datetime(
        2024, 10, 15, tzinfo=timezone.utc
    )

    assert args.price_history_end_datetime == datetime(
        2026, 10, 20, tzinfo=timezone.utc
    )
