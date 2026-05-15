"""Performance metrics for backtesting results."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_return(nav: pd.Series) -> float:
    return nav.iloc[-1] / nav.iloc[0] - 1


def annualized_return(nav: pd.Series) -> float:
    total = cumulative_return(nav)
    n_years = (len(nav) - 1) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def annualized_volatility(nav: pd.Series) -> float:
    daily_rets = nav.pct_change().dropna()
    return float(daily_rets.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(nav: pd.Series, risk_free_rate: float = 0.04) -> float:
    ann_ret = annualized_return(nav)
    ann_vol = annualized_volatility(nav)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(nav: pd.Series) -> float:
    rolling_max = nav.cummax()
    dd = (nav - rolling_max) / rolling_max
    return float(dd.min())


def win_rate(nav: pd.Series) -> float:
    daily_rets = nav.pct_change().dropna()
    return float((daily_rets > 0).mean())


def compute_metrics(nav: pd.Series, risk_free_rate: float = 0.04) -> dict:
    """Return a dict of all standard performance metrics."""
    return {
        "Cumulative Return":   f"{cumulative_return(nav):.2%}",
        "Annualized Return":   f"{annualized_return(nav):.2%}",
        "Annualized Volatility": f"{annualized_volatility(nav):.2%}",
        "Sharpe Ratio":        f"{sharpe_ratio(nav, risk_free_rate):.3f}",
        "Max Drawdown":        f"{max_drawdown(nav):.2%}",
        "Win Rate":            f"{win_rate(nav):.2%}",
    }


def sharpe_float(nav: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Return Sharpe ratio as a float (for programmatic comparison)."""
    return sharpe_ratio(nav, risk_free_rate)
