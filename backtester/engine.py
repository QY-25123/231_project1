"""Core backtesting engine."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from .strategies.base import Strategy


@dataclass
class BacktestResult:
    nav: pd.Series                    # portfolio NAV indexed by date
    weights_log: List[dict] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    strategy_name: str = ""

    def to_weights_df(self) -> pd.DataFrame:
        """Return a DataFrame of daily weights (dates × tickers)."""
        if not self.weights_log:
            return pd.DataFrame()
        dates = [r["date"] for r in self.weights_log]
        arr = np.vstack([r["weights"] for r in self.weights_log])
        return pd.DataFrame(arr, index=dates, columns=self.tickers)


class BacktestEngine:
    """
    Daily-close backtesting engine.

    Workflow per trading day t:
      1. Strategy receives all closing prices up to and including day t.
      2. Strategy outputs a target weight vector (summing to ≤ 1).
      3. Portfolio value evolves from day t close to day t+1 close.
    """

    def __init__(self, prices: pd.DataFrame, initial_capital: float = 1_000_000.0):
        self.prices = prices.copy()
        self.tickers = prices.columns.tolist()
        self.initial_capital = initial_capital

    def run(
        self,
        strategy: Strategy,
        start_date=None,
        end_date=None,
    ) -> BacktestResult:
        """
        Run one strategy over the full (or filtered) price history.

        Parameters
        ----------
        strategy   : any Strategy subclass
        start_date : optional start date (str or Timestamp)
        end_date   : optional end date   (str or Timestamp)
        """
        prices = self.prices
        if start_date is not None:
            prices = prices[prices.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            prices = prices[prices.index <= pd.Timestamp(end_date)]

        dates = prices.index
        n = len(dates)
        if n < 2:
            raise ValueError("Not enough trading days in the selected range.")

        nav_values = {dates[0]: self.initial_capital}
        weights_log = []
        portfolio_value = self.initial_capital

        for i in range(n - 1):
            t = dates[i]
            t1 = dates[i + 1]

            # Give the strategy all history up to and including day t
            hist = self.prices.loc[: t]

            raw_w = strategy.generate_weights(hist)
            raw_w = np.clip(raw_w, 0.0, None)          # enforce no shorting
            total = raw_w.sum()
            if total > 1.0:
                raw_w = raw_w / total                   # enforce no leverage

            weights_log.append({"date": t, "weights": raw_w.copy()})

            # Daily returns from t to t+1 (only within the window slice)
            p_t  = prices.loc[t].values
            p_t1 = prices.loc[t1].values

            with np.errstate(divide="ignore", invalid="ignore"):
                day_ret = np.where(p_t > 0, (p_t1 - p_t) / p_t, 0.0)
            day_ret = np.nan_to_num(day_ret, nan=0.0)

            portfolio_value *= 1.0 + float((raw_w * day_ret).sum())
            nav_values[t1] = portfolio_value

        nav = pd.Series(nav_values)
        return BacktestResult(
            nav=nav,
            weights_log=weights_log,
            tickers=self.tickers,
            strategy_name=strategy.name,
        )
