from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "1.0.0"

DEFAULT_ATM_MONEYNESS_PCT = 2.0
DEFAULT_MAX_QUOTE_AGE_MINUTES = 30.0
DEFAULT_MAX_RELATIVE_SPREAD_PCT = 25.0

DTE_BUCKETS = (
    ("dte_1_7", 1, 7),
    ("dte_8_30", 8, 30),
    ("dte_31_90", 31, 90),
    ("dte_91_365", 91, 365),
    ("dte_366_plus", 366, None),
)

REQUIRED_OPTION_COLUMNS = {
    "underlying_symbol",
    "put_call",
    "strike",
    "expiration_dte",
    "bid",
    "ask",
    "mark",
    "volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "openInterest",
    "totalVolume",
    "quoteTimeInLong",
    "raw_captured_at_utc",
}

REQUIRED_QUOTE_COLUMNS = {
    "symbol",
}

MODEL_EXCLUDED_PROVIDER_FIELDS = {
    "percentChange",
    "markPercentChange",
    "intrinsicValue",
    "extrinsicValue",
    "theoreticalVolatility",
}


@dataclass(frozen=True)
class OptionFeatureConfig:
    """
    Configuration for Schwab option-contract eligibility
    and one-capture feature aggregation.

    The defaults mirror the semantic-quality gate used
    before V2 feature engineering.
    """

    max_relative_spread_pct: float = (
        DEFAULT_MAX_RELATIVE_SPREAD_PCT
    )

    max_quote_age_minutes: float = (
        DEFAULT_MAX_QUOTE_AGE_MINUTES
    )

    atm_moneyness_pct: float = (
        DEFAULT_ATM_MONEYNESS_PCT
    )

    require_positive_bid: bool = True
    exclude_zero_dte: bool = True

    def __post_init__(
        self,
    ) -> None:
        if self.max_relative_spread_pct < 0:
            raise ValueError(
                "max_relative_spread_pct must be "
                "non-negative."
            )

        if self.max_quote_age_minutes < 0:
            raise ValueError(
                "max_quote_age_minutes must be "
                "non-negative."
            )

        if self.atm_moneyness_pct < 0:
            raise ValueError(
                "atm_moneyness_pct must be "
                "non-negative."
            )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            f"{context} is missing required columns: "
            f"{missing}"
        )


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )


def _quote_price_column(
    quotes: pd.DataFrame,
) -> str:
    candidates = (
        "quote__mark",
        "quote__lastPrice",
        "regular__regularMarketLastPrice",
        "quote__closePrice",
    )

    for column in candidates:
        if column in quotes.columns:
            return column

    raise ValueError(
        "Schwab quote frame has no supported spot-price "
        f"column. Expected one of {candidates}."
    )


def build_spot_price_map(
    quotes: pd.DataFrame,
) -> dict[str, float]:
    """
    Build one positive spot price per symbol from a
    normalized Schwab quote snapshot.

    Preference order:
        quote__mark
        quote__lastPrice
        regular__regularMarketLastPrice
        quote__closePrice
    """
    _require_columns(
        quotes,
        REQUIRED_QUOTE_COLUMNS,
        context=
            "Schwab quote frame",
    )

    price_column = (
        _quote_price_column(
            quotes
        )
    )

    prices = pd.to_numeric(
        quotes[
            price_column
        ],
        errors="coerce",
    )

    result: dict[
        str,
        float,
    ] = {}

    for symbol, price in zip(
        quotes[
            "symbol"
        ],
        prices,
    ):
        symbol_text = str(
            symbol
        ).strip().upper()

        if (
            not symbol_text
            or pd.isna(
                price
            )
            or float(
                price
            )
            <= 0
        ):
            continue

        if symbol_text in result:
            raise ValueError(
                "Schwab quote frame contains duplicate "
                f"symbol {symbol_text!r}."
            )

        result[
            symbol_text
        ] = float(
            price
        )

    return result


def _epoch_millis_to_utc(
    values: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(
            values,
            errors="coerce",
        ),
        unit="ms",
        errors="coerce",
        utc=True,
    )


def _capture_times(
    values: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )


def _canonical_intrinsic_value(
    *,
    side: pd.Series,
    spot: pd.Series,
    strike: pd.Series,
) -> pd.Series:
    call = (
        spot
        - strike
    ).clip(
        lower=0.0
    )

    put = (
        strike
        - spot
    ).clip(
        lower=0.0
    )

    result = pd.Series(
        np.nan,
        index=
            side.index,
        dtype="float64",
    )

    result.loc[
        side
        == "CALL"
    ] = call.loc[
        side
        == "CALL"
    ]

    result.loc[
        side
        == "PUT"
    ] = put.loc[
        side
        == "PUT"
    ]

    return result


def _signed_moneyness_pct(
    *,
    side: pd.Series,
    spot: pd.Series,
    strike: pd.Series,
) -> pd.Series:
    """
    Positive values are in-the-money for both calls
    and puts; negative values are out-of-the-money.
    """
    result = pd.Series(
        np.nan,
        index=
            side.index,
        dtype="float64",
    )

    positive_spot = (
        spot
        > 0
    )

    call_mask = (
        (
            side
            == "CALL"
        )
        & positive_spot
    )

    put_mask = (
        (
            side
            == "PUT"
        )
        & positive_spot
    )

    result.loc[
        call_mask
    ] = (
        (
            spot.loc[
                call_mask
            ]
            - strike.loc[
                call_mask
            ]
        )
        / spot.loc[
            call_mask
        ]
        * 100.0
    )

    result.loc[
        put_mask
    ] = (
        (
            strike.loc[
                put_mask
            ]
            - spot.loc[
                put_mask
            ]
        )
        / spot.loc[
            put_mask
        ]
        * 100.0
    )

    return result


def _relative_spread_pct(
    *,
    bid: pd.Series,
    ask: pd.Series,
) -> pd.Series:
    midpoint = (
        bid
        + ask
    ) / 2.0

    result = pd.Series(
        np.nan,
        index=
            bid.index,
        dtype="float64",
    )

    valid = (
        bid.notna()
        & ask.notna()
        & (
            midpoint
            > 0
        )
    )

    result.loc[
        valid
    ] = (
        (
            ask.loc[
                valid
            ]
            - bid.loc[
                valid
            ]
        )
        / midpoint.loc[
            valid
        ]
        * 100.0
    )

    return result


def _eligibility_reasons(
    frame: pd.DataFrame,
) -> pd.Series:
    reason_columns = [
        column
        for column in frame.columns
        if column.startswith(
            "eligibility__"
        )
    ]

    def row_reasons(
        row: pd.Series,
    ) -> str:
        values = [
            column.removeprefix(
                "eligibility__"
            )
            for column in reason_columns
            if not bool(
                row[
                    column
                ]
            )
        ]

        return "|".join(
            values
        )

    return frame[
        reason_columns
    ].apply(
        row_reasons,
        axis=1,
    )


def prepare_schwab_option_contracts(
    options: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    config: OptionFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Add canonical contract-level fields and one explicit
    modeling-eligibility flag to normalized Schwab option
    contracts.

    Provider fields excluded by the V2 policy are retained
    in the returned frame only for provenance/diagnostics;
    they are not used to derive eligibility or aggregate
    features.
    """
    if config is None:
        config = (
            OptionFeatureConfig()
        )

    _require_columns(
        options,
        REQUIRED_OPTION_COLUMNS,
        context=
            "Schwab option frame",
    )

    spots = (
        build_spot_price_map(
            quotes
        )
    )

    frame = options.copy()

    frame[
        "underlying_symbol"
    ] = (
        frame[
            "underlying_symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame[
        "put_call"
    ] = (
        frame[
            "put_call"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame[
        "spot"
    ] = (
        frame[
            "underlying_symbol"
        ].map(
            spots
        )
    )

    for column in (
        "strike",
        "expiration_dte",
        "bid",
        "ask",
        "mark",
        "volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "openInterest",
        "totalVolume",
    ):
        frame[
            column
        ] = _numeric(
            frame,
            column,
        )

    frame[
        "canonical_intrinsic_value"
    ] = (
        _canonical_intrinsic_value(
            side=
                frame[
                    "put_call"
                ],

            spot=
                frame[
                    "spot"
                ],

            strike=
                frame[
                    "strike"
                ],
        )
    )

    frame[
        "canonical_extrinsic_value"
    ] = (
        frame[
            "mark"
        ]
        - frame[
            "canonical_intrinsic_value"
        ]
    )

    frame[
        "signed_moneyness_pct"
    ] = (
        _signed_moneyness_pct(
            side=
                frame[
                    "put_call"
                ],

            spot=
                frame[
                    "spot"
                ],

            strike=
                frame[
                    "strike"
                ],
        )
    )

    frame[
        "abs_moneyness_pct"
    ] = (
        frame[
            "signed_moneyness_pct"
        ].abs()
    )

    frame[
        "relative_spread_pct"
    ] = (
        _relative_spread_pct(
            bid=
                frame[
                    "bid"
                ],

            ask=
                frame[
                    "ask"
                ],
        )
    )

    quote_time = (
        _epoch_millis_to_utc(
            frame[
                "quoteTimeInLong"
            ]
        )
    )

    capture_time = (
        _capture_times(
            frame[
                "raw_captured_at_utc"
            ]
        )
    )

    frame[
        "quote_age_minutes"
    ] = (
        (
            capture_time
            - quote_time
        )
        .dt.total_seconds()
        / 60.0
    )

    frame[
        "is_atm"
    ] = (
        frame[
            "abs_moneyness_pct"
        ]
        <= config.atm_moneyness_pct
    )

    frame[
        "is_zero_dte"
    ] = (
        frame[
            "expiration_dte"
        ]
        == 0
    )

    frame[
        "eligibility__spot"
    ] = (
        frame[
            "spot"
        ].notna()
        & (
            frame[
                "spot"
            ]
            > 0
        )
    )

    frame[
        "eligibility__side"
    ] = (
        frame[
            "put_call"
        ].isin(
            [
                "CALL",
                "PUT",
            ]
        )
    )

    if config.require_positive_bid:
        frame[
            "eligibility__bid"
        ] = (
            frame[
                "bid"
            ].notna()
            & (
                frame[
                    "bid"
                ]
                > 0
            )
        )

    else:
        frame[
            "eligibility__bid"
        ] = (
            frame[
                "bid"
            ].notna()
            & (
                frame[
                    "bid"
                ]
                >= 0
            )
        )

    frame[
        "eligibility__ask"
    ] = (
        frame[
            "ask"
        ].notna()
        & (
            frame[
                "ask"
            ]
            >= frame[
                "bid"
            ]
        )
    )

    frame[
        "eligibility__mark"
    ] = (
        frame[
            "mark"
        ].notna()
        & (
            frame[
                "mark"
            ]
            >= frame[
                "bid"
            ]
        )
        & (
            frame[
                "mark"
            ]
            <= frame[
                "ask"
            ]
        )
    )

    frame[
        "eligibility__spread"
    ] = (
        frame[
            "relative_spread_pct"
        ].notna()
        & (
            frame[
                "relative_spread_pct"
            ]
            <= config.max_relative_spread_pct
        )
    )

    frame[
        "eligibility__quote_age"
    ] = (
        frame[
            "quote_age_minutes"
        ].notna()
        & (
            frame[
                "quote_age_minutes"
            ]
            >= 0
        )
        & (
            frame[
                "quote_age_minutes"
            ]
            <= config.max_quote_age_minutes
        )
    )

    frame[
        "eligibility__dte"
    ] = (
        frame[
            "expiration_dte"
        ].notna()
        & (
            frame[
                "expiration_dte"
            ]
            >= (
                1
                if config.exclude_zero_dte
                else 0
            )
        )
    )

    frame[
        "eligibility__volatility"
    ] = (
        frame[
            "volatility"
        ].notna()
        & (
            frame[
                "volatility"
            ]
            > 0
        )
    )

    frame[
        "eligibility__delta"
    ] = (
        frame[
            "delta"
        ].notna()
        & (
            frame[
                "delta"
            ]
            >= -1.0
        )
        & (
            frame[
                "delta"
            ]
            <= 1.0
        )
    )

    frame[
        "eligibility__gamma"
    ] = (
        frame[
            "gamma"
        ].notna()
        & (
            frame[
                "gamma"
            ]
            >= 0
        )
    )

    frame[
        "eligibility__vega"
    ] = (
        frame[
            "vega"
        ].notna()
        & (
            frame[
                "vega"
            ]
            >= 0
        )
    )

    eligibility_columns = [
        column
        for column in frame.columns
        if column.startswith(
            "eligibility__"
        )
    ]

    frame[
        "eligible_contract"
    ] = (
        frame[
            eligibility_columns
        ].all(
            axis=1
        )
    )

    frame[
        "ineligibility_reasons"
    ] = (
        _eligibility_reasons(
            frame
        )
    )

    return frame


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if (
        pd.isna(
            denominator
        )
        or denominator
        <= 0
    ):
        return np.nan

    return float(
        numerator
        / denominator
    )


def _median(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(
        numeric.median()
    )


def _mean(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(
        numeric.mean()
    )


def _sum(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return 0.0

    return float(
        numeric.sum()
    )


def _dte_mask(
    values: pd.Series,
    *,
    lower: int,
    upper: int | None,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    mask = (
        numeric
        >= lower
    )

    if upper is not None:
        mask &= (
            numeric
            <= upper
        )

    return mask


def _atm_iv_by_bucket(
    eligible: pd.DataFrame,
) -> dict[str, float]:
    result: dict[
        str,
        float,
    ] = {}

    atm = eligible.loc[
        eligible[
            "is_atm"
        ]
    ]

    for name, lower, upper in (
        DTE_BUCKETS
    ):
        mask = _dte_mask(
            atm[
                "expiration_dte"
            ],
            lower=
                lower,
            upper=
                upper,
        )

        result[
            f"atm_iv_median_{name}"
        ] = _median(
            atm.loc[
                mask,
                "volatility",
            ]
        )

    return result


def _nearest_delta_iv(
    frame: pd.DataFrame,
    *,
    side: str,
    target_delta: float,
) -> float:
    subset = frame.loc[
        frame[
            "put_call"
        ]
        == side
    ].copy()

    if subset.empty:
        return np.nan

    distance = (
        subset[
            "delta"
        ]
        - target_delta
    ).abs()

    valid = (
        distance.notna()
        & subset[
            "volatility"
        ].notna()
    )

    if not valid.any():
        return np.nan

    nearest_index = (
        distance.loc[
            valid
        ].idxmin()
    )

    return float(
        subset.loc[
            nearest_index,
            "volatility",
        ]
    )


def _aggregate_symbol_features(
    symbol: str,
    all_contracts: pd.DataFrame,
) -> dict[str, Any]:
    eligible = all_contracts.loc[
        all_contracts[
            "eligible_contract"
        ]
    ].copy()

    total_count = int(
        len(
            all_contracts
        )
    )

    eligible_count = int(
        len(
            eligible
        )
    )

    zero_dte_count = int(
        all_contracts[
            "is_zero_dte"
        ].sum()
    )

    call = eligible.loc[
        eligible[
            "put_call"
        ]
        == "CALL"
    ]

    put = eligible.loc[
        eligible[
            "put_call"
        ]
        == "PUT"
    ]

    atm = eligible.loc[
        eligible[
            "is_atm"
        ]
    ]

    atm_call = atm.loc[
        atm[
            "put_call"
        ]
        == "CALL"
    ]

    atm_put = atm.loc[
        atm[
            "put_call"
        ]
        == "PUT"
    ]

    call_oi = _sum(
        call[
            "openInterest"
        ]
    )

    put_oi = _sum(
        put[
            "openInterest"
        ]
    )

    call_volume = _sum(
        call[
            "totalVolume"
        ]
    )

    put_volume = _sum(
        put[
            "totalVolume"
        ]
    )

    atm_call_iv = _median(
        atm_call[
            "volatility"
        ]
    )

    atm_put_iv = _median(
        atm_put[
            "volatility"
        ]
    )

    delta_25_call_iv = (
        _nearest_delta_iv(
            eligible,
            side="CALL",
            target_delta=0.25,
        )
    )

    delta_25_put_iv = (
        _nearest_delta_iv(
            eligible,
            side="PUT",
            target_delta=-0.25,
        )
    )

    features: dict[
        str,
        Any,
    ] = {
        "feature_schema_version":
            FEATURE_SCHEMA_VERSION,

        "underlying_symbol":
            symbol,

        "spot":
            _median(
                all_contracts[
                    "spot"
                ]
            ),

        "contract_count_total":
            total_count,

        "contract_count_eligible":
            eligible_count,

        "eligible_contract_ratio":
            (
                float(
                    eligible_count
                    / total_count
                )
                if total_count
                else np.nan
            ),

        "zero_dte_contract_count":
            zero_dte_count,

        "zero_dte_contract_ratio":
            (
                float(
                    zero_dte_count
                    / total_count
                )
                if total_count
                else np.nan
            ),

        "eligible_call_count":
            int(
                len(
                    call
                )
            ),

        "eligible_put_count":
            int(
                len(
                    put
                )
            ),

        "eligible_expiration_count":
            int(
                eligible[
                    "expiration_date"
                ].nunique(
                    dropna=True
                )
            )
            if "expiration_date"
            in eligible.columns
            else 0,

        "eligible_dte_min":
            (
                float(
                    eligible[
                        "expiration_dte"
                    ].min()
                )
                if not eligible.empty
                else np.nan
            ),

        "eligible_dte_max":
            (
                float(
                    eligible[
                        "expiration_dte"
                    ].max()
                )
                if not eligible.empty
                else np.nan
            ),

        "relative_spread_pct_median":
            _median(
                eligible[
                    "relative_spread_pct"
                ]
            ),

        "atm_relative_spread_pct_median":
            _median(
                atm[
                    "relative_spread_pct"
                ]
            ),

        "iv_median":
            _median(
                eligible[
                    "volatility"
                ]
            ),

        "atm_iv_median":
            _median(
                atm[
                    "volatility"
                ]
            ),

        "atm_call_iv_median":
            atm_call_iv,

        "atm_put_iv_median":
            atm_put_iv,

        "atm_put_call_iv_skew":
            (
                atm_put_iv
                - atm_call_iv
            ),

        "delta_25_call_iv":
            delta_25_call_iv,

        "delta_25_put_iv":
            delta_25_put_iv,

        "delta_25_put_call_iv_skew":
            (
                delta_25_put_iv
                - delta_25_call_iv
            ),

        "put_call_open_interest_ratio":
            _safe_ratio(
                put_oi,
                call_oi,
            ),

        "put_call_volume_ratio":
            _safe_ratio(
                put_volume,
                call_volume,
            ),

        "open_interest_total":
            (
                call_oi
                + put_oi
            ),

        "option_volume_total":
            (
                call_volume
                + put_volume
            ),

        "abs_delta_median":
            _median(
                eligible[
                    "delta"
                ].abs()
            ),

        "gamma_median":
            _median(
                eligible[
                    "gamma"
                ]
            ),

        "theta_median":
            _median(
                eligible[
                    "theta"
                ]
            ),

        "vega_median":
            _median(
                eligible[
                    "vega"
                ]
            ),

        "canonical_extrinsic_value_median":
            _median(
                eligible[
                    "canonical_extrinsic_value"
                ]
            ),

        "quote_age_minutes_median":
            _median(
                eligible[
                    "quote_age_minutes"
                ]
            ),
    }

    features.update(
        _atm_iv_by_bucket(
            eligible
        )
    )

    near = features[
        "atm_iv_median_dte_8_30"
    ]

    back = features[
        "atm_iv_median_dte_91_365"
    ]

    features[
        "atm_iv_term_spread_91_365_minus_8_30"
    ] = (
        back
        - near
    )

    return features


def build_schwab_option_features(
    options: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    config: OptionFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Build one deterministic V2 option-feature row per
    underlying symbol from one normalized Schwab capture.

    Only contracts passing prepare_schwab_option_contracts
    eligibility are used in aggregate option features.

    Provider fields deliberately excluded from initial V2
    modeling are not exposed as aggregate features:
      - percentChange
      - markPercentChange
      - intrinsicValue
      - extrinsicValue
      - theoreticalVolatility
    """
    prepared = (
        prepare_schwab_option_contracts(
            options,
            quotes,
            config=
                config,
        )
    )

    symbols = sorted(
        prepared[
            "underlying_symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    rows = [
        _aggregate_symbol_features(
            symbol,
            prepared.loc[
                prepared[
                    "underlying_symbol"
                ]
                == symbol
            ],
        )
        for symbol in symbols
    ]

    return pd.DataFrame(
        rows
    )


def eligibility_summary(
    prepared_contracts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize contract eligibility by underlying symbol
    and ineligibility reason for diagnostics.
    """
    required = {
        "underlying_symbol",
        "eligible_contract",
        "ineligibility_reasons",
    }

    _require_columns(
        prepared_contracts,
        required,
        context=
            "Prepared Schwab option frame",
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for symbol, group in (
        prepared_contracts.groupby(
            "underlying_symbol",
            sort=True,
        )
    ):
        total = int(
            len(
                group
            )
        )

        eligible = int(
            group[
                "eligible_contract"
            ].sum()
        )

        reason_counts: dict[
            str,
            int,
        ] = {}

        for value in group.loc[
            ~group[
                "eligible_contract"
            ],
            "ineligibility_reasons",
        ]:
            for reason in str(
                value
            ).split(
                "|"
            ):
                if not reason:
                    continue

                reason_counts[
                    reason
                ] = (
                    reason_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

        rows.append(
            {
                "underlying_symbol":
                    str(
                        symbol
                    ),

                "contract_count_total":
                    total,

                "contract_count_eligible":
                    eligible,

                "eligible_contract_ratio":
                    (
                        float(
                            eligible
                            / total
                        )
                        if total
                        else np.nan
                    ),

                "ineligible_contract_count":
                    (
                        total
                        - eligible
                    ),

                "ineligibility_reason_counts":
                    json.dumps(
                        reason_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )
