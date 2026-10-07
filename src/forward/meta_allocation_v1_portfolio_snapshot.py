from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from config.settings import DEFAULT_UNIVERSE


FROZEN_ASSETS = list(DEFAULT_UNIVERSE)

ALLOWED_ACCOUNT_MODES = {
    "MANUAL",
    "PAPERMONEY",
    "SCHWAB_READ_ONLY",
}

DEFAULT_MAX_PRICE_AGE = pd.Timedelta(
    days=7
)


@dataclass(frozen=True)
class PortfolioSnapshot:
    snapshot_timestamp: pd.Timestamp
    price_as_of: pd.Timestamp

    account_mode: str

    cash: float
    holdings_value: float
    portfolio_value: float

    current_shares: pd.Series
    reference_prices: pd.Series

    holdings_snapshot_hash: str
    price_snapshot_hash: str
    portfolio_snapshot_hash: str


def _normalize_timestamp(
    value: str | pd.Timestamp,
    field_name: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:
        raise ValueError(
            f"{field_name} must be timezone-aware."
        )

    return timestamp.tz_convert(
        "UTC"
    )


def _validate_account_mode(
    account_mode: str,
) -> str:
    normalized = (
        str(
            account_mode
        )
        .strip()
        .upper()
    )

    if normalized not in ALLOWED_ACCOUNT_MODES:
        raise ValueError(
            "account_mode must be one of "
            f"{sorted(ALLOWED_ACCOUNT_MODES)}. "
            "LIVE brokerage mode is not supported."
        )

    return normalized


def _validate_asset_index(
    index: pd.Index,
    field_name: str,
) -> None:
    if index.duplicated().any():
        duplicates = (
            index[
                index.duplicated(
                    keep=False
                )
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{field_name} contains duplicate assets: "
            f"{sorted(set(duplicates))}"
        )

    unsupported = (
        set(
            index.astype(str)
        )
        - set(
            FROZEN_ASSETS
        )
    )

    if unsupported:
        raise ValueError(
            f"{field_name} contains unsupported assets: "
            f"{sorted(unsupported)}"
        )


def normalize_holdings(
    holdings: (
        dict
        | pd.Series
        | pd.DataFrame
    ),
) -> pd.Series:
    if isinstance(
        holdings,
        pd.DataFrame,
    ):
        required = {
            "asset",
            "shares",
        }

        missing = (
            required
            - set(
                holdings.columns
            )
        )

        if missing:
            raise ValueError(
                "Holdings DataFrame missing columns: "
                f"{sorted(missing)}"
            )

        if holdings[
            "asset"
        ].duplicated().any():
            duplicates = (
                holdings.loc[
                    holdings[
                        "asset"
                    ].duplicated(
                        keep=False
                    ),
                    "asset",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Holdings contain duplicate assets: "
                f"{sorted(set(duplicates))}"
            )

        series = (
            holdings
            .set_index(
                "asset"
            )[
                "shares"
            ]
            .copy()
        )

    elif isinstance(
        holdings,
        pd.Series,
    ):
        series = holdings.copy()

    elif isinstance(
        holdings,
        dict,
    ):
        series = pd.Series(
            holdings,
            dtype=float,
        )

    else:
        raise TypeError(
            "holdings must be a dict, "
            "pandas Series, or pandas DataFrame."
        )

    series.index = (
        series.index
        .astype(str)
    )

    _validate_asset_index(
        series.index,
        "holdings",
    )

    series = (
        series
        .astype(float)
        .reindex(
            FROZEN_ASSETS
        )
        .fillna(0.0)
    )

    values = series.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Holdings contain non-finite values."
        )

    if (
        values < 0
    ).any():
        raise ValueError(
            "Short positions are not supported "
            "by Meta Allocation V1."
        )

    if not np.allclose(
        values,
        np.floor(
            values
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Meta Allocation V1 requires "
            "whole-share holdings."
        )

    return series.astype(int)


def normalize_reference_prices(
    reference_prices: (
        dict
        | pd.Series
        | pd.DataFrame
    ),
) -> pd.Series:
    if isinstance(
        reference_prices,
        pd.DataFrame,
    ):
        required = {
            "asset",
            "reference_price",
        }

        missing = (
            required
            - set(
                reference_prices.columns
            )
        )

        if missing:
            raise ValueError(
                "Reference-price DataFrame "
                "missing columns: "
                f"{sorted(missing)}"
            )

        if reference_prices[
            "asset"
        ].duplicated().any():
            duplicates = (
                reference_prices.loc[
                    reference_prices[
                        "asset"
                    ].duplicated(
                        keep=False
                    ),
                    "asset",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Reference prices contain "
                "duplicate assets: "
                f"{sorted(set(duplicates))}"
            )

        series = (
            reference_prices
            .set_index(
                "asset"
            )[
                "reference_price"
            ]
            .copy()
        )

    elif isinstance(
        reference_prices,
        pd.Series,
    ):
        series = (
            reference_prices
            .copy()
        )

    elif isinstance(
        reference_prices,
        dict,
    ):
        series = pd.Series(
            reference_prices,
            dtype=float,
        )

    else:
        raise TypeError(
            "reference_prices must be a dict, "
            "pandas Series, or pandas DataFrame."
        )

    series.index = (
        series.index
        .astype(str)
    )

    _validate_asset_index(
        series.index,
        "reference_prices",
    )

    missing_assets = (
        set(
            FROZEN_ASSETS
        )
        - set(
            series.index
        )
    )

    if missing_assets:
        raise ValueError(
            "Missing reference prices for: "
            f"{sorted(missing_assets)}"
        )

    series = (
        series
        .astype(float)
        .reindex(
            FROZEN_ASSETS
        )
    )

    values = series.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Reference prices contain "
            "non-finite values."
        )

    if (
        values <= 0
    ).any():
        raise ValueError(
            "Reference prices must be positive."
        )

    return series


def validate_cash(
    cash: float,
) -> float:
    cash = float(
        cash
    )

    if (
        not np.isfinite(
            cash
        )
        or cash < 0
    ):
        raise ValueError(
            "Cash must be finite "
            "and non-negative."
        )

    return cash


def validate_price_freshness(
    snapshot_timestamp: pd.Timestamp,
    price_as_of: pd.Timestamp,
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> None:
    max_price_age = pd.Timedelta(
        max_price_age
    )

    if max_price_age < pd.Timedelta(0):
        raise ValueError(
            "max_price_age cannot be negative."
        )

    if price_as_of > snapshot_timestamp:
        raise ValueError(
            "price_as_of cannot be later than "
            "snapshot_timestamp."
        )

    age = (
        snapshot_timestamp
        - price_as_of
    )

    if age > max_price_age:
        raise ValueError(
            "Reference prices are stale.\n"
            f"Price age: {age}\n"
            f"Maximum allowed age: "
            f"{max_price_age}"
        )


def sha256_series_snapshot(
    series: pd.Series,
    assets: list[str],
) -> str:
    """
    Deterministic hash for an ordered asset/value
    snapshot.

    Keep this implementation shared with the
    order-plan layer.
    """
    payload = [
        {
            "asset":
                asset,

            "value":
                float(
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


def sha256_portfolio_snapshot(
    snapshot_timestamp: pd.Timestamp,
    price_as_of: pd.Timestamp,
    account_mode: str,
    cash: float,
    holdings_snapshot_hash: str,
    price_snapshot_hash: str,
) -> str:
    payload = {
        "snapshot_timestamp":
            snapshot_timestamp.isoformat(),

        "price_as_of":
            price_as_of.isoformat(),

        "account_mode":
            account_mode,

        "cash":
            float(
                cash
            ),

        "holdings_snapshot_hash":
            holdings_snapshot_hash,

        "price_snapshot_hash":
            price_snapshot_hash,
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


def build_portfolio_snapshot(
    holdings: (
        dict
        | pd.Series
        | pd.DataFrame
    ),
    cash: float,
    reference_prices: (
        dict
        | pd.Series
        | pd.DataFrame
    ),
    snapshot_timestamp: (
        str
        | pd.Timestamp
    ),
    price_as_of: (
        str
        | pd.Timestamp
    ),
    account_mode: str = "MANUAL",
    max_price_age: (
        str
        | pd.Timedelta
    ) = DEFAULT_MAX_PRICE_AGE,
) -> PortfolioSnapshot:
    account_mode = (
        _validate_account_mode(
            account_mode
        )
    )

    snapshot_timestamp = (
        _normalize_timestamp(
            snapshot_timestamp,
            "snapshot_timestamp",
        )
    )

    price_as_of = (
        _normalize_timestamp(
            price_as_of,
            "price_as_of",
        )
    )

    validate_price_freshness(
        snapshot_timestamp=
            snapshot_timestamp,

        price_as_of=
            price_as_of,

        max_price_age=
            max_price_age,
    )

    current_shares = (
        normalize_holdings(
            holdings
        )
    )

    prices = (
        normalize_reference_prices(
            reference_prices
        )
    )

    cash = validate_cash(
        cash
    )

    holdings_value = float(
        (
            current_shares
            * prices
        ).sum()
    )

    portfolio_value = (
        holdings_value
        + cash
    )

    if (
        not np.isfinite(
            portfolio_value
        )
        or portfolio_value <= 0
    ):
        raise ValueError(
            "Portfolio value must be "
            "positive and finite."
        )

    holdings_snapshot_hash = (
        sha256_series_snapshot(
            series=
                current_shares.astype(
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

    portfolio_snapshot_hash = (
        sha256_portfolio_snapshot(
            snapshot_timestamp=
                snapshot_timestamp,

            price_as_of=
                price_as_of,

            account_mode=
                account_mode,

            cash=
                cash,

            holdings_snapshot_hash=
                holdings_snapshot_hash,

            price_snapshot_hash=
                price_snapshot_hash,
        )
    )

    return PortfolioSnapshot(
        snapshot_timestamp=
            snapshot_timestamp,

        price_as_of=
            price_as_of,

        account_mode=
            account_mode,

        cash=
            cash,

        holdings_value=
            holdings_value,

        portfolio_value=
            portfolio_value,

        current_shares=
            current_shares,

        reference_prices=
            prices,

        holdings_snapshot_hash=
            holdings_snapshot_hash,

        price_snapshot_hash=
            price_snapshot_hash,

        portfolio_snapshot_hash=
            portfolio_snapshot_hash,
    )