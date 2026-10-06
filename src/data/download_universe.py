from config.settings import DEFAULT_UNIVERSE
from src.data.market_data import download_daily_prices, save_prices


def main():
    prices = download_daily_prices(
        tickers=DEFAULT_UNIVERSE,
        start="2015-01-01",
    )

    output_path = save_prices(
        prices,
        filename="portfolio_universe_daily.csv",
    )

    print(f"Downloaded {len(prices):,} trading days.")
    print(f"Assets: {list(prices.columns)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
