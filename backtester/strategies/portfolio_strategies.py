"""
Portfolio-level trading strategies (Deliverables 4 & 5).

Each strategy returns a weight vector across all 101 tickers.
"""

import numpy as np
import pandas as pd

from .base import Strategy
from ..portfolio import uniform_weights, risk_adjusted_weights


# ---------------------------------------------------------------------------
# Benchmark 1 — SMA Crossover (Deliverable 5, Benchmark 1)
# ---------------------------------------------------------------------------
class SMAcrossoverPortfolio(Strategy):
    """
    Select stocks whose short-window SMA is above the long-window SMA.
    Portfolio construction: uniform equal-weight.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        if len(prices) < self.long_window:
            return np.zeros(len(prices.columns))
        sma_short = prices.iloc[-self.short_window:].mean()
        sma_long = prices.iloc[-self.long_window:].mean()
        signals = (sma_short > sma_long).astype(float).values
        return uniform_weights(signals)

    @property
    def name(self) -> str:
        return f"SMAcross-{self.short_window}/{self.long_window}(equal-weight)"


# ---------------------------------------------------------------------------
# Benchmark 2 — Top-K Trailing Momentum (Deliverable 5, Benchmark 2)
# ---------------------------------------------------------------------------
class TopKMomentumPortfolio(Strategy):
    """
    Rank all stocks by trailing return over `lookback` days; select top-K.
    Portfolio construction: uniform equal-weight.
    """

    def __init__(self, lookback: int = 30, top_k: int = 10):
        self.lookback = lookback
        self.top_k = top_k

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        n = len(prices.columns)
        if len(prices) < self.lookback + 1:
            return np.zeros(n)
        trailing_ret = prices.iloc[-1] / prices.iloc[-self.lookback - 1] - 1
        trailing_ret = trailing_ret.fillna(-np.inf)
        signals = np.zeros(n)
        top_idx = trailing_ret.nlargest(self.top_k).index
        col_idx = [prices.columns.get_loc(t) for t in top_idx]
        signals[col_idx] = 1.0
        return uniform_weights(signals)

    @property
    def name(self) -> str:
        return f"TopK-Momentum-{self.lookback}d/K{self.top_k}(equal-weight)"


# ---------------------------------------------------------------------------
# Deliverable 4 — SMA Crossover with Risk-Adjusted Weighting
# ---------------------------------------------------------------------------
class SMAcrossoverRiskAdjusted(Strategy):
    """SMA crossover signal + risk-adjusted portfolio weights."""

    def __init__(
        self, short_window: int = 20, long_window: int = 50, vol_lookback: int = 20
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        if len(prices) < self.long_window:
            return np.zeros(len(prices.columns))
        sma_short = prices.iloc[-self.short_window:].mean()
        sma_long = prices.iloc[-self.long_window:].mean()
        signals = (sma_short > sma_long).astype(float).values
        return risk_adjusted_weights(signals, prices, self.vol_lookback)

    @property
    def name(self) -> str:
        return f"SMAcross-{self.short_window}/{self.long_window}(risk-adj)"


# ---------------------------------------------------------------------------
# Deliverable 4 — Top-K Momentum with Risk-Adjusted Weighting
# ---------------------------------------------------------------------------
class TopKMomentumRiskAdjusted(Strategy):
    """Top-K momentum signal + risk-adjusted portfolio weights."""

    def __init__(
        self, lookback: int = 30, top_k: int = 10, vol_lookback: int = 20
    ):
        self.lookback = lookback
        self.top_k = top_k
        self.vol_lookback = vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        n = len(prices.columns)
        if len(prices) < self.lookback + 1:
            return np.zeros(n)
        trailing_ret = prices.iloc[-1] / prices.iloc[-self.lookback - 1] - 1
        trailing_ret = trailing_ret.fillna(-np.inf)
        signals = np.zeros(n)
        top_idx = trailing_ret.nlargest(self.top_k).index
        col_idx = [prices.columns.get_loc(t) for t in top_idx]
        signals[col_idx] = 1.0
        return risk_adjusted_weights(signals, prices, self.vol_lookback)

    @property
    def name(self) -> str:
        return f"TopK-Momentum-{self.lookback}d/K{self.top_k}(risk-adj)"


# ---------------------------------------------------------------------------
# New Strategy 1 — Dual Momentum (Absolute + Relative, Equal-Weight)
# ---------------------------------------------------------------------------
class DualMomentumPortfolio(Strategy):
    """
    Two-gate momentum strategy:
      • Absolute gate : stock's 90-day return must be positive (uptrend
        confirmation; avoids holding stocks that are in a structural downtrend).
      • Relative gate : among those that pass, select the top-K by 30-day return.
      • Portfolio      : equal-weight.

    Rationale: combining absolute and relative momentum avoids the common
    pitfall of pure cross-sectional strategies — investing in the "least bad"
    losers during broad bear markets (e.g. 2022).  By requiring positive
    90-day performance, the strategy shifts to cash during sustained downturns,
    reducing drawdowns and improving Sharpe.
    """

    def __init__(
        self,
        abs_lookback: int = 90,
        rel_lookback: int = 30,
        top_k: int = 10,
        min_stocks: int = 5,
    ):
        self.abs_lookback = abs_lookback
        self.rel_lookback = rel_lookback
        self.top_k = top_k
        self.min_stocks = min_stocks

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        n = len(prices.columns)
        min_rows = self.abs_lookback + 1
        if len(prices) < min_rows:
            return np.zeros(n)

        # Absolute momentum filter: 90-day return > 0
        ret_abs = prices.iloc[-1] / prices.iloc[-self.abs_lookback - 1] - 1
        abs_ok = ret_abs.fillna(-np.inf) > 0

        if abs_ok.sum() < self.min_stocks:
            return np.zeros(n)  # not enough qualified stocks → cash

        # Relative momentum: top-K by 30-day return among qualified
        ret_rel = prices.iloc[-1] / prices.iloc[-self.rel_lookback - 1] - 1
        ret_rel = ret_rel.where(abs_ok, -np.inf)
        top_idx = ret_rel.nlargest(self.top_k).index

        signals = np.zeros(n)
        col_idx = [prices.columns.get_loc(t) for t in top_idx]
        signals[col_idx] = 1.0
        return uniform_weights(signals)

    @property
    def name(self) -> str:
        return (
            f"DualMomentum-abs{self.abs_lookback}d"
            f"+rel{self.rel_lookback}d/K{self.top_k}(equal-wt)"
        )


# ---------------------------------------------------------------------------
# New Strategy 2 — Risk-Adjusted Momentum Score (Equal-Weight)
# ---------------------------------------------------------------------------
class RiskAdjustedMomentumPortfolio(Strategy):
    """
    Select stocks by their recent risk-adjusted return (a mini-Sharpe over the
    lookback window) rather than raw return rank.

    Score_i = Return_i(lookback) / Volatility_i(lookback)

    Top-K stocks by this score are held with equal weight.

    Rationale: pure return ranking (Benchmark 2) systematically selects
    high-volatility stocks.  Dividing by recent volatility penalises noisy
    momentum and rewards smoother, more reliable trends.  In practice this
    reduces concentration in high-beta names during corrections, lowering
    portfolio volatility while preserving upside capture.
    """

    def __init__(
        self,
        lookback: int = 30,
        top_k: int = 10,
    ):
        self.lookback = lookback
        self.top_k = top_k

    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        n = len(prices.columns)
        if len(prices) < self.lookback + 1:
            return np.zeros(n)

        # Recent return
        ret = prices.iloc[-1] / prices.iloc[-self.lookback - 1] - 1

        # Recent realised volatility (annualised doesn't matter; we only rank)
        daily_rets = prices.iloc[-self.lookback:].pct_change(fill_method=None)
        vol = daily_rets.std()

        # Risk-adjusted score; avoid division by zero or NaN
        score = ret / vol.replace(0, np.nan)
        score = score.fillna(-np.inf)

        top_idx = score.nlargest(self.top_k).index
        signals = np.zeros(n)
        col_idx = [prices.columns.get_loc(t) for t in top_idx]
        signals[col_idx] = 1.0
        return uniform_weights(signals)

    @property
    def name(self) -> str:
        return f"RiskAdjMomentumScore-{self.lookback}d/K{self.top_k}(equal-wt)"
