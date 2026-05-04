#!/usr/bin/env python3
"""Run a 10-year CAN SLIM proxy vs Minervini portfolio comparison."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.research import (
    BacktestConfig,
    CANSLIMConfig,
    CANSLIMScreener,
    MarketDataWarehouse,
    MinerviniConfig,
    MinerviniScreener,
    PortfolioCANSLIMBacktester,
    PortfolioMinerviniBacktester,
    resolve_universe,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CAN SLIM proxy vs Minervini over 10 years")
    parser.add_argument("--db", default=str(ROOT / "research_data" / "canslim_eval.duckdb"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "canslim"))
    parser.add_argument("--universe", default="combined", help="growth | combined")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", default="2015-05-01", help="Warmup start date YYYY-MM-DD")
    parser.add_argument("--trade-start", default="2016-05-01", help="Trading start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-fundamentals", action="store_true")
    return parser.parse_args()


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_shared_backtest_config() -> BacktestConfig:
    return BacktestConfig(
        max_position_pct=0.12,
        risk_per_trade=0.012,
        stop_loss_pct=0.08,
        trail_stop_pct=0.12,
        max_hold_days=90,
        min_template_score=7,
        require_volume_surge=False,
        require_market_regime=False,
        progressive_entries=True,
        initial_entry_fraction=0.50,
        add_on_trigger_pct_1=0.025,
        add_on_trigger_pct_2=0.05,
        add_on_fraction_1=0.30,
        add_on_fraction_2=0.20,
        breakeven_trigger_pct=0.05,
        trailing_lock_trigger_pct_1=0.12,
        trailing_lock_floor_pct_1=0.03,
        trailing_lock_trigger_pct_2=0.20,
        trailing_lock_floor_pct_2=0.08,
        partial_profit_trigger_pct=0.12,
        partial_profit_fraction=0.33,
        use_ema21_exit=True,
        use_close_range_filter=True,
        min_close_range_pct=0.55,
        scale_exposure_in_weak_market=True,
        weak_market_position_scale=0.60,
        target_exposure_confirmed_uptrend=1.00,
        target_exposure_uptrend_under_pressure=0.60,
        target_exposure_market_correction=0.00,
        allow_new_entries_in_correction=False,
        max_positions=6,
    )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_universe(args.universe)
    required = sorted(set(symbols + [args.benchmark]))
    stamp = args.end

    warehouse = MarketDataWarehouse(args.db)
    try:
        available = set(warehouse.available_symbols())
        if args.refresh_data or not set(required).issubset(available):
            warehouse.fetch_and_store_daily_bars(required, args.start, args.end)

        if args.refresh_fundamentals:
            warehouse.fetch_and_store_quarterly_fundamentals(symbols)
            warehouse.fetch_and_store_fundamentals(symbols)

        data_by_symbol = {
            symbol: warehouse.get_daily_bars(symbol, args.start, args.end)
            for symbol in symbols
        }
        data_by_symbol = {symbol: df for symbol, df in data_by_symbol.items() if not df.empty}
        benchmark_df = warehouse.get_daily_bars(args.benchmark, args.start, args.end)
        quarterly_df = warehouse.get_quarterly_fundamentals(symbols)
        quarterly_by_symbol = {
            symbol: frame.copy()
            for symbol, frame in quarterly_df.groupby("symbol")
        } if not quarterly_df.empty else {}

        shared_cfg = build_shared_backtest_config()

        minervini = PortfolioMinerviniBacktester(
            screener=MinerviniScreener(
                MinerviniConfig(
                    require_fundamentals=False,
                    require_market_uptrend=False,
                    max_stage_number=3,
                    max_buy_zone_pct=0.07,
                    pivot_buffer_pct=0.0,
                )
            ),
            config=shared_cfg,
        )
        canslim = PortfolioCANSLIMBacktester(
            screener=CANSLIMScreener(
                CANSLIMConfig(
                    require_fundamentals=False,
                    require_market_uptrend=False,
                    near_52w_high_pct=0.15,
                    min_current_eps_growth=0.25,
                    min_current_revenue_growth=0.20,
                    min_annual_eps_growth=0.20,
                    min_annual_revenue_growth=0.15,
                    min_close_range_pct=0.50,
                    volume_surge_multiple=1.2,
                )
            ),
            config=shared_cfg,
            quarterly_by_symbol=quarterly_by_symbol,
        )

        minervini_result = minervini.backtest_portfolio(
            data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=args.trade_start,
        )
        canslim_result = canslim.backtest_portfolio(
            data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=args.trade_start,
        )

        comparison_rows = [
            {"strategy": "minervini_baseline", **minervini_result.summary},
            {"strategy": "canslim_proxy", **canslim_result.summary},
        ]
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["alpha_vs_spy"] = (
            comparison_df["total_return"] - comparison_df["benchmark_return"]
        ).round(4)
        comparison_df["return_over_drawdown"] = (
            comparison_df["total_return"] / comparison_df["max_drawdown"].clip(lower=0.01)
        ).round(4)

        prefix = f"canslim_vs_minervini_{args.universe}_{stamp}"
        save(comparison_df, results_dir / f"{prefix}_comparison.csv")
        save(pd.DataFrame([minervini_result.summary]), results_dir / f"{prefix}_minervini_metrics.csv")
        save(pd.DataFrame([canslim_result.summary]), results_dir / f"{prefix}_canslim_metrics.csv")
        save(minervini_result.trades, results_dir / f"{prefix}_minervini_trades.csv")
        save(canslim_result.trades, results_dir / f"{prefix}_canslim_trades.csv")
        save(minervini_result.daily_state, results_dir / f"{prefix}_minervini_daily_state.csv")
        save(canslim_result.daily_state, results_dir / f"{prefix}_canslim_daily_state.csv")
        save(minervini_result.equity_curve, results_dir / f"{prefix}_minervini_equity_curve.csv")
        save(canslim_result.equity_curve, results_dir / f"{prefix}_canslim_equity_curve.csv")
        save(minervini_result.symbol_summary, results_dir / f"{prefix}_minervini_symbol_summary.csv")
        save(canslim_result.symbol_summary, results_dir / f"{prefix}_canslim_symbol_summary.csv")

        notes_df = pd.DataFrame(
            [
                {
                    "note": (
                        "CAN SLIM 10y result is a price/volume-led proxy. "
                        "Quarterly fundamentals stored locally only begin around 2024-06, "
                        "so full point-in-time CAN SLIM fundamentals are not available across the entire 10-year window."
                    )
                }
            ]
        )
        save(notes_df, results_dir / f"{prefix}_notes.csv")

        print(comparison_df.to_string(index=False))
        print(f"\nSaved outputs with prefix {prefix}")
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()
