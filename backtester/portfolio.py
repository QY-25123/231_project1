"""Portfolio construction helpers: convert signals to weight vectors."""

import numpy as np
import pandas as pd


def uniform_weights(signals: np.ndarray) -> np.ndarray:
    """
    Equal weight across all stocks with a positive signal.
    Cash (weight=0) for the remainder.
    """
    selected = signals > 0
    n = int(selected.sum())
    weights = np.zeros(len(signals))
    if n > 0:
        weights[selected] = 1.0 / n
    return weights


def risk_adjusted_weights(
    signals: np.ndarray,
    prices: pd.DataFrame,
    vol_lookback: int = 20,
) -> np.ndarray:
    """
    Risk-adjusted weighting across selected stocks.

    Parameters
    ----------
    signals   : binary or scored signal array (length = n_tickers)
    prices    : historical price matrix (dates × tickers), full history up to today
    vol_lookback : rolling window for volatility estimation
    """
    selected = signals > 0
    if not selected.any():
        return np.zeros(len(signals))

    n_rows = len(prices)
    window = min(vol_lookback, n_rows - 1)
    if window < 2:
        return uniform_weights(signals)

    # Use fill_method=None + skipna std to handle sparse tickers (e.g. recent IPOs)
    # that have NaN values in the lookback window.
    daily_rets = prices.iloc[-window:].pct_change(fill_method=None)
    vols = daily_rets.std().values                     # skipna=True by default

    # Restrict to selected stocks with valid, positive volatility
    valid = selected & np.isfinite(vols) & (vols > 0)
    if not valid.any():
        return uniform_weights(signals)

    weights = np.zeros(len(signals))
    sel_vols = vols[valid]
    risk_adj = 1.0 / sel_vols
    weights[valid] = risk_adj / risk_adj.sum()
    return weights
