#!/usr/bin/env python3
"""Scan current option chains for Quality Put Income candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.options import PutIncomeParams, PutIncomeScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan current cash-secured put candidates")
    parser.add_argument(
        "--symbols",
        default=(
            "SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,META,COST,JPM,V,MA,ORCL,SNOW,"
            "AMD,INTC,CSCO,GILD,MRK,PFE,KO,BAC,C,WMT,UBER,DIS,NKE,PYPL"
        ),
    )
    parser.add_argument("--min-iv", type=float, default=0.22)
    parser.add_argument("--target-delta", type=float, default=0.18)
    parser.add_argument("--output-dir", default="results/put_income")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    params = PutIncomeParams(
        symbols=tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip()),
        min_iv=args.min_iv,
        target_delta=args.target_delta,
    )
    rows = PutIncomeScanner(params).scan()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "put_income_candidates.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "put_income_candidates.csv", index=False)

    print(f"Candidates: {len(rows)}")
    print("=" * 120)
    for row in rows[: args.limit]:
        print(
            f"{row['symbol']:5} {row['expiry']} {row['dte']:2}D "
            f"P{row['strike']:<8.2f} spot={row['spot']:<8.2f} "
            f"mid={row['mid']:<6.2f} delta={row['delta_est']:<6.3f} "
            f"iv={row['iv']:<5.2f} yld={row['premium_yield']:<6.2%} "
            f"OI={row['open_interest']:<6} vol={row['volume']:<6} {row['contract']}"
        )
    print(f"Saved: {out_dir / 'put_income_candidates.json'}")


if __name__ == "__main__":
    main()
