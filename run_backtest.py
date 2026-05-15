"""
INDENG 231 Project 1 — Backtesting Simulation Entry Point
=========================================================
Usage:
    python run_backtest.py

All plots and tables are saved under ./results/.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all environments)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from backtester.data_loader import load_prices
from backtester.engine import BacktestEngine
import backtester.metrics as M
from backtester.strategies.single_stock import (
    MomentumStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    SMACrossoverStrategy,
)
from backtester.strategies.portfolio_strategies import (
    SMAcrossoverPortfolio,
    TopKMomentumPortfolio,
    SMAcrossoverRiskAdjusted,
    TopKMomentumRiskAdjusted,
    DualMomentumPortfolio,
    RiskAdjustedMomentumPortfolio,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = HERE / "nasdaq100_daily_5y.csv"
RESULTS_DIR     = HERE / "results"
INITIAL_CAPITAL = 1_000_000.0
SINGLE_TICKER   = "NVDA"
RISK_FREE_RATE  = 0.04

RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_all(engine: BacktestEngine, strategies: dict) -> dict:
    """Run each strategy and collect NAV series."""
    navs = {}
    for label, strat in strategies.items():
        print(f"    [{label}] ...", end=" ", flush=True)
        result = engine.run(strat)
        navs[label] = result.nav
        print("done")
    return navs


def _metrics_table(navs: dict, title: str, save_csv: bool = True) -> pd.DataFrame:
    rows = []
    for label, nav in navs.items():
        row = M.compute_metrics(nav, RISK_FREE_RATE)
        row["Strategy"] = label
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Strategy")
    cols = ["Cumulative Return", "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Max Drawdown", "Win Rate"]
    df = df[cols]
    print(f"\n  {title}")
    print("  " + "─" * 78)
    print(df.to_string())
    if save_csv:
        fname = title.replace(" ", "_").replace("/", "-") + ".csv"
        df.to_csv(RESULTS_DIR / fname)
    return df


def _plot_navs(navs: dict, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, nav in navs.items():
        ax.plot(nav.index, nav / nav.iloc[0], label=label, linewidth=1.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised NAV (start = 1.0)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot → {path}")


def _save_log(navs: dict, filename: str):
    """Save NAV series to CSV for reproducibility."""
    df = pd.DataFrame(navs)
    df.index.name = "date"
    df.to_csv(RESULTS_DIR / filename)
    print(f"  Saved NAV log → {RESULTS_DIR / filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  INDENG 231 Project 1: Trading Strategy Backtesting Simulation")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\n[1] Loading data from {DATA_PATH.name} ...")
    prices = load_prices(str(DATA_PATH))
    tickers = prices.columns.tolist()
    print(f"  {len(prices)} trading days  |  {len(tickers)} tickers")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    engine = BacktestEngine(prices, initial_capital=INITIAL_CAPITAL)

    # ------------------------------------------------------------------
    # Deliverable 3: Single-Stock Strategies (NVDA)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[DELIVERABLE 3]  Single-Stock Strategies  ({SINGLE_TICKER})")
    print("=" * 60)

    single_strategies = {
        f"Momentum-20d":              MomentumStrategy(SINGLE_TICKER, tickers, lookback=20),
        f"MeanReversion-20d/1σ":      MeanReversionStrategy(SINGLE_TICKER, tickers, lookback=20, z_threshold=1.0),
        f"RSI-14d(oversold<40)":      RSIStrategy(SINGLE_TICKER, tickers, period=14, oversold=40),
        f"BollingerBands-20d/2σ":     BollingerBandsStrategy(SINGLE_TICKER, tickers, lookback=20, n_std=2.0),
        f"SMACross-10/30d":           SMACrossoverStrategy(SINGLE_TICKER, tickers, short_window=10, long_window=30),
    }

    single_navs = _run_all(engine, single_strategies)

    # Add buy-and-hold benchmark (full 5-year range)
    bh_prices = prices[SINGLE_TICKER].dropna()
    single_navs["Buy&Hold"] = (bh_prices / bh_prices.iloc[0]) * INITIAL_CAPITAL

    _metrics_table(single_navs, f"D3 Single-Stock Strategies ({SINGLE_TICKER})")
    _plot_navs(single_navs,
               f"Deliverable 3 — Single-Stock Strategies: {SINGLE_TICKER}",
               "d3_single_stock_nav.png")
    _save_log(single_navs, "d3_single_stock_nav.csv")

    # ------------------------------------------------------------------
    # Deliverable 4: Portfolio Backtesting
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[DELIVERABLE 4]  Portfolio Backtesting")
    print("=" * 60)

    portfolio_strategies = {
        "SMAcross-20/50(equal-wt)":   SMAcrossoverPortfolio(short_window=20, long_window=50),
        "SMAcross-20/50(risk-adj)":   SMAcrossoverRiskAdjusted(short_window=20, long_window=50),
        "TopK-Momentum-30d/K10(equal-wt)": TopKMomentumPortfolio(lookback=30, top_k=10),
        "TopK-Momentum-30d/K10(risk-adj)": TopKMomentumRiskAdjusted(lookback=30, top_k=10),
    }

    port_navs = _run_all(engine, portfolio_strategies)

    # Nasdaq-100 equal-weight buy-and-hold as reference
    ew_nav = prices.pct_change().fillna(0).mean(axis=1).add(1).cumprod() * INITIAL_CAPITAL
    port_navs["NDX100-EqualWeight-BH"] = ew_nav

    _metrics_table(port_navs, "D4 Portfolio Construction Methods")
    _plot_navs(port_navs,
               "Deliverable 4 — Portfolio Backtesting: Uniform vs Risk-Adjusted",
               "d4_portfolio_nav.png")
    _save_log(port_navs, "d4_portfolio_nav.csv")

    # ------------------------------------------------------------------
    # Deliverable 5: Benchmark vs New Strategies
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[DELIVERABLE 5]  Benchmark vs New Strategies")
    print("=" * 60)

    d5_strategies = {
        "Benchmark1-SMAcross-20/50":          SMAcrossoverPortfolio(short_window=20, long_window=50),
        "Benchmark2-TopK-Momentum-30d/K10":   TopKMomentumPortfolio(lookback=30, top_k=10),
        "New1-DualMomentum(abs90+rel30)":     DualMomentumPortfolio(abs_lookback=90, rel_lookback=30, top_k=10),
        "New2-RiskAdjMomentumScore-30d":      RiskAdjustedMomentumPortfolio(lookback=30, top_k=10),
    }

    d5_navs = _run_all(engine, d5_strategies)
    df5 = _metrics_table(d5_navs, "D5 New Strategies vs Benchmarks")
    _plot_navs(d5_navs,
               "Deliverable 5 — Benchmark Strategies vs New Strategies",
               "d5_new_vs_benchmark_nav.png")
    _save_log(d5_navs, "d5_new_vs_benchmark_nav.csv")

    # Summary: Sharpe comparison
    print("\n  Sharpe Ratio Comparison (D5):")
    for label, nav in d5_navs.items():
        sr = M.sharpe_float(nav, RISK_FREE_RATE)
        print(f"    {label:<50s}  {sr:+.3f}")

    # ------------------------------------------------------------------
    # Combined overview plot
    # ------------------------------------------------------------------
    all_navs = {}
    all_navs.update(d5_navs)
    all_navs["NDX100-EqualWeight-BH"] = ew_nav
    _plot_navs(all_navs,
               "All Portfolio Strategies — NAV Overview",
               "all_strategies_nav.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"  All results saved to  {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
