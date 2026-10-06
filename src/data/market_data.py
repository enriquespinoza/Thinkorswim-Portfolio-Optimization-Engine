from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from config.settings import RAW_DATA_DIR


def download_daily_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start:
        Start date in YYYY-MM-DD format.
    end:
        Optional end date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per ticker.
    """

    tickers = list(tickers)

    if not tickers:
        raise ValueError("At least one ticker is required.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No market data was returned.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    return prices


def save_prices(
    prices: pd.DataFrame,
    filename: str = "daily_prices.csv",
    directory: Path = RAW_DATA_DIR,
) -> Path:
    """
    Save price data to the raw data directory.
    """

    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / filename
    prices.to_csv(output_path)

    return output_path
