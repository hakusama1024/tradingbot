# paper_put_income

Isolated paper simulation profile for `Quality Put Income v1`.

This profile is for validating a cash-secured short-put income system before
broker-side options execution is added.

Default rules:

- Universe: liquid ETFs and quality large caps, with `SNOW` included only as a capped high-IV test name.
- Entry: 30-60 DTE option chains, default around 45 DTE.
- Delta: target 0.25, allowed 0.16-0.30.
- Trend filter: underlying must trade above its 200-day average.
- IV filter: implied volatility must be at least 20%.
- Exit: buy to close at 50% profit, manage at 21 DTE, or risk-stop at 2.5x entry premium.
- Sizing: max 20% secured notional per symbol, max 85% total secured notional.
- High-IV cap: symbols with IV above 55% are capped to 10% secured notional.

Run research:

```bash
python scripts/run_put_income_backtest.py \
  --start 2016-01-01 \
  --target-delta 0.30 \
  --min-iv 0.20 \
  --max-total-notional-pct 0.85 \
  --cash-yield 0.035 \
  --output-dir results/put_income/quality_put_income_v1
```

Run current scan:

```bash
python scripts/run_put_income_scan.py \
  --target-delta 0.25 \
  --min-iv 0.20 \
  --output-dir runtime/paper_put_income/results
```

Run one paper simulation pass:

```bash
python scripts/run_put_income_paper.py --profile paper_put_income
```

Important caveat:

The 10-year backtest uses synthetic option prices from underlying daily bars and
realized volatility. It validates broad rule behavior, but it is not equivalent
to historical option quote backtesting.
