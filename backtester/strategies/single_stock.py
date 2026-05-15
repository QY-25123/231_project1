"""
Single-stock trading strategies (Deliverable 3).

Each strategy targets one ticker; all other weights remain 0.
"""

from typing import List
import numpy as np
import pandas as pd

from .base import Strategy


class _SingleStockBase(Strategy):
    """Shared plumbing for single-stock strategies."""

    def __init__(self, ticker: str, tickers: List[str]):
        if ticker not in tickers:
            raise ValueError(f"Ticker {ticker!r} not found in universe.")
        self.ticker = ticker
        self.tickers = tickers
        self._idx = tickers.index(ticker)

    def _zero(self) -> np.ndarray:
        return np.zeros(len(self.tickers))

    def _full(self) -> np.ndarray:
        w = self._zero()
        w[self._idx] = 1.0
        return w

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}({self.ticker})"


# ---------------------------------------------------------------------------
# 1. Momentum
# ---------------------------------------------------------------------------
class MomentumStrategy(_SingleStockBase):
    """
    Hold when the trailing return over `lookback` days is positive; else cash.
    """

    def __init__(self, ticker: str, tickers: List[str], lookback: int = 20):
        super().__init__(ticker, tickers)
        self.lookback = lookback

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        col = prices[self.ticker].dropna()
        if len(col) < self.lookback + 1:
            return self._zero()
        ret = col.iloc[-1] / col.iloc[-self.lookback - 1] - 1
        return self._full() if ret > 0 else self._zero()

    @property
    def name(self) -> str:
        return f"Momentum-{self.lookback}d({self.ticker})"


# ---------------------------------------------------------------------------
# 2. Mean Reversion
# ---------------------------------------------------------------------------
class MeanReversionStrategy(_SingleStockBase):
    """
    Hold when the price is below its rolling mean by more than z_threshold
    standard deviations (oversold → expect reversion upward); else cash.
    """

    def __init__(
        self,
        ticker: str,
        tickers: List[str],
        lookback: int = 20,
        z_threshold: float = 1.0,
    ):
        super().__init__(ticker, tickers)
        self.lookback = lookback
        self.z_threshold = z_threshold

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        col = prices[self.ticker].dropna()
        if len(col) < self.lookback:
            return self._zero()
        window = col.iloc[-self.lookback:]
        mean = window.mean()
        std = window.std()
        if std == 0:
            return self._zero()
        z = (col.iloc[-1] - mean) / std
        return self._full() if z < -self.z_threshold else self._zero()

    @property
    def name(self) -> str:
        return f"MeanReversion-{self.lookback}d/{self.z_threshold}σ({self.ticker})"


# ---------------------------------------------------------------------------
# 3. RSI (Relative Strength Index)
# ---------------------------------------------------------------------------
class RSIStrategy(_SingleStockBase):
    """
    Hold when RSI < oversold_threshold (mean-reversion buy signal).
    Exit (cash) when RSI >= oversold_threshold.
    """

    def __init__(
        self,
        ticker: str,
        tickers: List[str],
        period: int = 14,
        oversold: float = 40.0,
    ):
        super().__init__(ticker, tickers)
        self.period = period
        self.oversold = oversold

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> float:
        delta = series.diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if pd.isna(avg_gain) or pd.isna(avg_loss):
            return 50.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        col = prices[self.ticker].dropna()
        if len(col) < self.period + 1:
            return self._zero()
        rsi = self._rsi(col, self.period)
        return self._full() if rsi < self.oversold else self._zero()

    @property
    def name(self) -> str:
        return f"RSI-{self.period}d/{self.oversold}({self.ticker})"


# ---------------------------------------------------------------------------
# 4. Bollinger Bands (mean reversion)
# ---------------------------------------------------------------------------
class BollingerBandsStrategy(_SingleStockBase):
    """
    Buy when price crosses below the lower Bollinger Band (mean - n_std * σ).
    Exit when price is above the middle band (rolling mean).
    """

    def __init__(
        self,
        ticker: str,
        tickers: List[str],
        lookback: int = 20,
        n_std: float = 2.0,
    ):
        super().__init__(ticker, tickers)
        self.lookback = lookback
        self.n_std = n_std
        self._in_trade = False

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        col = prices[self.ticker].dropna()
        if len(col) < self.lookback:
            return self._zero()
        window = col.iloc[-self.lookback:]
        mean = window.mean()
        std = window.std()
        lower = mean - self.n_std * std
        current = col.iloc[-1]

        if current < lower:
            self._in_trade = True
        elif current > mean:
            self._in_trade = False

        return self._full() if self._in_trade else self._zero()

    @property
    def name(self) -> str:
        return f"BollingerBands-{self.lookback}d/{self.n_std}σ({self.ticker})"


# ---------------------------------------------------------------------------
# 5. SMA Crossover (trend-following, single stock)
# ---------------------------------------------------------------------------
class SMACrossoverStrategy(_SingleStockBase):
    """
    Hold when the short-window SMA is above the long-window SMA (golden cross);
    cash when short SMA is below long SMA (death cross).
    """

    def __init__(
        self,
        ticker: str,
        tickers: List[str],
        short_window: int = 10,
        long_window: int = 30,
    ):
        super().__init__(ticker, tickers)
        self.short_window = short_window
        self.long_window = long_window

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        col = prices[self.ticker].dropna()
        if len(col) < self.long_window:
            return self._zero()
        sma_short = col.iloc[-self.short_window:].mean()
        sma_long = col.iloc[-self.long_window:].mean()
        return self._full() if sma_short > sma_long else self._zero()

    @property
    def name(self) -> str:
        return f"SMACross-{self.short_window}/{self.long_window}({self.ticker})"
