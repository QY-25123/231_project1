"""Data loading utilities."""

import pandas as pd


def load_prices(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset and return a wide-format price matrix.

    Returns
    -------
    pd.DataFrame
        Index: trading dates (DatetimeIndex, sorted ascending)
        Columns: ticker symbols
        Values: daily closing prices
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    prices = df.pivot(index="date", columns="ticker", values="close")
    prices.columns.name = None
    prices.sort_index(inplace=True)
    return prices
