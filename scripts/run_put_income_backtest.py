#!/usr/bin/env python3
"""Run synthetic cash-secured put income backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.options import PutIncomeBacktester, PutIncomeParams


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _annual_returns(equity: pd.Series) -> dict[str, float]:
    yearly = equity.resample("YE").last().pct_change().dropna()
    return {str(idx.year): float(value) for idx, value in yearly.items()}


def _benchmark_curve(symbol: str, start: str, end: str | None, initial_equity: float) -> pd.Series:
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return initial_equity * df["Close"] / float(df["Close"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Quality Put Income strategy")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,META,COST,JPM,V,MA,ORCL")
    parser.add_argument("--target-delta", type=float, default=0.18)
    parser.add_argument("--min-iv", type=float, default=0.22)
    parser.add_argument("--max-total-notional-pct", type=float, default=0.65)
    parser.add_argument("--max-symbol-notional-pct", type=float, default=0.20)
    parser.add_argument("--cash-yield", type=float, default=0.0)
    parser.add_argument("--output-dir", default="results/put_income")
    args = parser.parse_args()

    params = PutIncomeParams(
        initial_equity=args.equity,
        symbols=tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip()),
        start=args.start,
        end=args.end,
        target_delta=args.target_delta,
        min_iv=args.min_iv,
        max_total_notional_pct=args.max_total_notional_pct,
        max_symbol_notional_pct=args.max_symbol_notional_pct,
        cash_yield=args.cash_yield,
    )
    backtester = PutIncomeBacktester(params)
    result = backtester.run()
    spy = backtester.benchmark_stats("SPY")
    qqq = backtester.benchmark_stats("QQQ")
    smh = backtester.benchmark_stats("SMH")
    spy_curve = _benchmark_curve("SPY", args.start, args.end, args.equity)
    qqq_curve = _benchmark_curve("QQQ", args.start, args.end, args.equity)
    smh_curve = _benchmark_curve("SMH", args.start, args.end, args.equity)
    annual = {
        "strategy": _annual_returns(result["equity_curve"]["equity"]),
        "SPY": _annual_returns(spy_curve) if not spy_curve.empty else {},
        "QQQ": _annual_returns(qqq_curve) if not qqq_curve.empty else {},
        "SMH": _annual_returns(smh_curve) if not smh_curve.empty else {},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["equity_curve"].to_csv(out_dir / "put_income_equity_curve.csv")
    (out_dir / "put_income_trades.json").write_text(
        json.dumps(result["trades"], indent=2, default=str),
        encoding="utf-8",
    )
    report = {
        "strategy": "quality_put_income_v1",
        "data_note": "Synthetic option prices from daily underlying bars and realized-volatility proxy; not historical option quotes.",
        "stats": result["stats"],
        "benchmarks": {"SPY": spy, "QQQ": qqq, "SMH": smh},
        "annual_returns": annual,
        "trade_count": len(result["trades"]),
        "open_positions": result["open_positions"],
        "params": result["params"],
    }
    (out_dir / "put_income_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("Quality Put Income v1")
    print("=" * 72)
    print(f"Window: {result['stats']['start']} -> {result['stats']['end']}")
    print(f"Trades: {len(result['trades'])}")
    print()
    print("Strategy")
    for key in ["total_return", "cagr", "max_drawdown", "volatility", "sharpe", "calmar", "yearly_win_rate"]:
        value = result["stats"][key]
        print(f"  {key:18}: {_pct(value) if key not in {'sharpe', 'calmar'} else f'{value:.2f}'}")
    print()
    for name, stats in [("SPY", spy), ("QQQ", qqq), ("SMH", smh)]:
        print(name)
        for key in ["total_return", "cagr", "max_drawdown", "volatility", "sharpe", "calmar", "yearly_win_rate"]:
            value = stats[key]
            print(f"  {key:18}: {_pct(value) if key not in {'sharpe', 'calmar'} else f'{value:.2f}'}")
        print()
    print("Annual returns")
    for year in sorted(annual["strategy"].keys()):
        print(
            f"  {year}: strategy {_pct(annual['strategy'].get(year, 0.0))} | "
            f"SPY {_pct(annual['SPY'].get(year, 0.0))} | "
            f"QQQ {_pct(annual['QQQ'].get(year, 0.0))} | "
            f"SMH {_pct(annual['SMH'].get(year, 0.0))}"
        )
    print(f"Saved: {out_dir / 'put_income_report.json'}")


if __name__ == "__main__":
    main()
