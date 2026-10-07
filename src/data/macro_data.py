from __future__ import annotations

import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import RAW_DATA_DIR


load_dotenv()


FRED_GRAPH_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
)

FRED_API_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


@dataclass(frozen=True)
class MacroSeriesSpec:
    """
    Definition of one macroeconomic series.

    availability_lag_days approximates the delay between the
    economic observation date and when that observation would
    have been available to the portfolio engine.

    This protects against obvious observation-date look-ahead,
    but does not provide vintage/revision-safe data.
    """

    series_id: str
    column_name: str
    availability_lag_days: int
    frequency: str


DEFAULT_MACRO_SERIES = (
    MacroSeriesSpec(
        series_id="DGS2",
        column_name="yield_2y",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="DGS10",
        column_name="yield_10y",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="DGS30",
        column_name="yield_30y",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="DFF",
        column_name="fed_funds",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="T10YIE",
        column_name="breakeven_10y",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="DFII10",
        column_name="real_yield_10y",
        availability_lag_days=1,
        frequency="daily",
    ),
    MacroSeriesSpec(
        series_id="UNRATE",
        column_name="unemployment",
        availability_lag_days=40,
        frequency="monthly",
    ),
    MacroSeriesSpec(
        series_id="ICSA",
        column_name="initial_claims",
        availability_lag_days=6,
        frequency="weekly",
    ),
    MacroSeriesSpec(
        series_id="PAYEMS",
        column_name="payrolls",
        availability_lag_days=40,
        frequency="monthly",
    ),
    MacroSeriesSpec(
        series_id="RSAFS",
        column_name="retail_sales",
        availability_lag_days=45,
        frequency="monthly",
    ),
)


def validate_macro_specs(
    specs: tuple[MacroSeriesSpec, ...],
) -> None:
    """
    Validate macro-series definitions.
    """
    if not specs:
        raise ValueError(
            "At least one macro series is required."
        )

    series_ids = [
        spec.series_id
        for spec in specs
    ]

    column_names = [
        spec.column_name
        for spec in specs
    ]

    if len(series_ids) != len(
        set(series_ids)
    ):
        raise ValueError(
            "Macro series IDs must be unique."
        )

    if len(column_names) != len(
        set(column_names)
    ):
        raise ValueError(
            "Macro column names must be unique."
        )

    for spec in specs:
        if (
            spec.availability_lag_days
            < 0
        ):
            raise ValueError(
                "availability_lag_days "
                "cannot be negative."
            )


def build_fred_csv_url(
    series_id: str,
    start: str,
    end: str | None = None,
) -> str:
    """
    Build a human-readable public FRED CSV URL.

    The downloader itself uses requests parameters rather
    than relying on this function.
    """
    url = (
        f"{FRED_GRAPH_CSV_URL}"
        f"?id={series_id}"
        f"&cosd={start}"
    )

    if end is not None:
        url += (
            f"&coed={end}"
        )

    return url


def build_http_session() -> requests.Session:
    """
    Build an HTTP session with retry behavior for transient
    network/FRED errors.
    """
    retry = Retry(
        total=2,
        connect=2,
        read=1,
        status=5,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=frozenset(
            [
                "GET",
            ]
        ),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = (
        requests.Session()
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent":
                (
                    "Thinkorswim-Portfolio-"
                    "Optimization-Engine/1.0"
                ),

            "Accept":
                (
                    "text/csv,"
                    "application/json;q=0.9,"
                    "*/*;q=0.8"
                ),

            "Connection":
                "close",
        }
    )

    return session


def _series_from_frame(
    frame: pd.DataFrame,
    spec: MacroSeriesSpec,
    date_column: str,
    value_column: str,
) -> pd.Series:
    """
    Convert downloaded FRED data into a clean numeric Series.
    """
    dates = pd.to_datetime(
        frame[
            date_column
        ],
        errors="coerce",
    )

    values = pd.to_numeric(
        frame[
            value_column
        ].replace(
            ".",
            np.nan,
        ),
        errors="coerce",
    )

    series = pd.Series(
        values.to_numpy(),
        index=dates,
        name=spec.column_name,
        dtype=float,
    )

    valid_index = (
        series.index.notna()
    )

    series = (
        series[
            valid_index
        ]
        .dropna()
        .sort_index()
    )

    if series.empty:
        raise ValueError(
            f"No usable observations returned "
            f"for {spec.series_id}."
        )

    return series


def _download_fred_graph_csv(
    spec: MacroSeriesSpec,
    start: str,
    end: str | None,
    session: requests.Session,
    timeout: int,
) -> pd.Series:
    """
    Download one FRED series from the public graph CSV
    endpoint.
    """
    params = {
        "id":
            spec.series_id,

        "cosd":
            start,
    }

    if end is not None:
        params[
            "coed"
        ] = end

    response = session.get(
        FRED_GRAPH_CSV_URL,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    frame = pd.read_csv(
        StringIO(
            response.text
        ),
        na_values=[
            ".",
            "",
        ],
    )

    if frame.shape[
        1
    ] < 2:
        raise ValueError(
            f"Unexpected FRED CSV response "
            f"for {spec.series_id}."
        )

    date_column = (
        frame.columns[
            0
        ]
    )

    value_column = (
        frame.columns[
            1
        ]
    )

    return _series_from_frame(
        frame=frame,
        spec=spec,
        date_column=
            date_column,
        value_column=
            value_column,
    )


def _download_fred_api(
    spec: MacroSeriesSpec,
    start: str,
    end: str | None,
    api_key: str,
    session: requests.Session,
    timeout: int,
) -> pd.Series:
    """
    Download one FRED series through the official JSON API.
    """
    params = {
        "series_id":
            spec.series_id,

        "api_key":
            api_key,

        "file_type":
            "json",

        "observation_start":
            start,
    }

    if end is not None:
        params[
            "observation_end"
        ] = end

    response = session.get(
        FRED_API_OBSERVATIONS_URL,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    payload = (
        response.json()
    )

    observations = (
        payload.get(
            "observations",
            []
        )
    )

    if not observations:
        raise ValueError(
            f"No FRED API observations "
            f"returned for {spec.series_id}."
        )

    frame = pd.DataFrame(
        observations
    )

    if (
        "date"
        not in frame.columns
        or "value"
        not in frame.columns
    ):
        raise ValueError(
            f"Unexpected FRED API response "
            f"for {spec.series_id}."
        )

    return _series_from_frame(
        frame=frame,
        spec=spec,
        date_column="date",
        value_column="value",
    )


def download_fred_series(
    spec: MacroSeriesSpec,
    start: str,
    end: str | None = None,
    timeout: int = 20,
    session: requests.Session | None = None,
) -> pd.Series:
    """
    Download one FRED series.

    Strategy
    --------
    1. If FRED_API_KEY exists, use the official FRED API first.
    2. If the API fails, fall back to the public graph CSV endpoint.
    3. If no API key exists, use the public CSV endpoint.
    """
    owns_session = (
        session is None
    )

    if session is None:
        session = (
            build_http_session()
        )

    api_key = os.getenv(
        "FRED_API_KEY"
    )

    api_error: Exception | None = None
    csv_error: Exception | None = None

    try:
        # Preferred path: official FRED API.
        if api_key:
            try:
                return (
                    _download_fred_api(
                        spec=spec,
                        start=start,
                        end=end,
                        api_key=api_key,
                        session=session,
                        timeout=timeout,
                    )
                )

            except Exception as exc:
                api_error = exc

                print(
                    f"Official FRED API failed for "
                    f"{spec.series_id}; "
                    f"trying CSV fallback..."
                )

        # Fallback path: public graph CSV endpoint.
        try:
            return (
                _download_fred_graph_csv(
                    spec=spec,
                    start=start,
                    end=end,
                    session=session,
                    timeout=timeout,
                )
            )

        except Exception as exc:
            csv_error = exc

        if api_key:
            raise RuntimeError(
                f"Unable to download FRED series "
                f"{spec.series_id}.\n"
                f"Official API error: {api_error}\n"
                f"CSV fallback error: {csv_error}"
            ) from csv_error

        raise RuntimeError(
            f"Unable to download FRED series "
            f"{spec.series_id} through the public "
            f"CSV endpoint.\n"
            f"Error: {csv_error}\n\n"
            "No FRED_API_KEY is configured."
        ) from csv_error

    finally:
        if owns_session:
            session.close()


def apply_availability_lag(
    series: pd.Series,
    spec: MacroSeriesSpec,
) -> pd.Series:
    """
    Shift an economic observation to its approximate
    information-availability date.
    """
    if not isinstance(
        series.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "series must use a DatetimeIndex."
        )

    lagged = (
        series.copy()
    )

    lagged.index = (
        lagged.index
        + pd.to_timedelta(
            spec.availability_lag_days,
            unit="D",
        )
    )

    lagged.index.name = (
        "available_date"
    )

    lagged.name = (
        spec.column_name
    )

    return (
        lagged.sort_index()
    )


def download_macro_dataset(
    start: str = "2014-01-01",
    end: str | None = None,
    specs: tuple[
        MacroSeriesSpec,
        ...
    ] = DEFAULT_MACRO_SERIES,
) -> pd.DataFrame:
    """
    Download and combine all macroeconomic series.

    Each series is shifted to its approximate information
    availability date before being merged.

    One retry-enabled session is reused for all downloads.
    """
    validate_macro_specs(
        specs
    )

    series_collection = []

    session = (
        build_http_session()
    )

    try:
        for spec in specs:
            print(
                f"Downloading "
                f"{spec.series_id} "
                f"-> "
                f"{spec.column_name}..."
            )

            raw_series = (
                download_fred_series(
                    spec=spec,
                    start=start,
                    end=end,
                    session=session,
                )
            )

            available_series = (
                apply_availability_lag(
                    raw_series,
                    spec,
                )
            )

            series_collection.append(
                available_series
            )

    finally:
        session.close()

    if not series_collection:
        raise ValueError(
            "No macroeconomic series were downloaded."
        )

    data = pd.concat(
        series_collection,
        axis=1,
        join="outer",
    )

    data = (
        data
        .sort_index()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    data.index.name = (
        "available_date"
    )

    return data


def save_macro_data(
    data: pd.DataFrame,
    filename: str = "portfolio_macro_fred.csv",
    directory: Path = RAW_DATA_DIR,
) -> Path:
    """
    Save downloaded macroeconomic data.
    """
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / filename
    )

    data.to_csv(
        path
    )

    return path
