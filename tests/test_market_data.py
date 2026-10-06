import pandas as pd

from src.data.market_data import save_prices


def test_save_prices(tmp_path):
    prices = pd.DataFrame(
        {
            "SPY": [500.0, 505.0],
            "TLT": [90.0, 91.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    output_path = save_prices(
        prices=prices,
        filename="test_prices.csv",
        directory=tmp_path,
    )

    assert output_path.exists()

    saved = pd.read_csv(output_path, index_col=0)

    assert "SPY" in saved.columns
    assert "TLT" in saved.columns
    assert len(saved) == 2
