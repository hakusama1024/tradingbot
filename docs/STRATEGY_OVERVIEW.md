# TradingBot Strategy Overview

This document summarizes the current strategy stack by role. The system is not
one single strategy; it is a set of isolated profiles that can be run, compared,
and disabled independently.

## Current Strategy Map

| Category | Profile | Role | Trading Style | Cadence |
| --- | --- | --- | --- | --- |
| Core rotation | `paper` | Main paper account core | SPY/QQQ/SMH tactical rotation | launchd checks every 15 min; strategy rebalances about every 15 trading days |
| Stock opportunity scanner | `paper_stockscan` | Satellite entries for the main paper account | Minervini / leader-continuation stock breakouts | intraday scan every 10 min |
| CAN SLIM experiment | `paper_canslim` | A/B test account | CAN SLIM selection + Minervini execution | intraday scan every 10 min |
| TriCore test | `paper_spy_alpha` | Isolated ETF-rotation test account | SPY/QQQ/SMH adaptive momentum | launchd checks every 15 min; strategy rebalances about every 15 trading days |
| Put-income research | `paper_put_income` | Options income simulator | cash-secured put scanning and local paper simulation | manual or scheduled separately |
| Live trading | `live` | Real-money profile | disabled by default in current setup | not installed unless explicitly enabled |

## 1. Core Rotation: `paper`

The main paper profile currently uses the SPY-alpha / TriCore tactical rotation
logic. It rotates among `SPY`, `QQQ`, and `SMH` using momentum, trend, and
volatility targeting.

Core idea:

- Rank `SPY`, `QQQ`, and `SMH` by 161-trading-day momentum.
- Require market breadth/trend confirmation using the 200-day moving average.
- Hold the top ETF when leadership is clear.
- Hold top two ETFs when leadership is close.
- Scale exposure by realized volatility, targeting about 17% annualized vol.
- Rebalance slowly to reduce churn.

Why it exists:

- It provides the account's lower-frequency core exposure.
- It avoids forcing trades every day.
- It is designed to participate in strong market/sector trends while reducing
  drawdown compared with fully concentrated high-beta exposure.

Main tradeoff:

- It can miss fast individual-stock breakouts.
- It is not meant to be a high-frequency stock picker.

## 2. Stock Opportunity Scanner: `paper_stockscan`

`paper_stockscan` was added to restore the 10-minute broad-market stock scan
without replacing the core TriCore strategy. It uses the same paper Alpaca
account as `paper`, but writes to its own runtime database and logs.

Core idea:

- Scan a broad universe of liquid common stocks.
- Apply Minervini-style trend/template filtering.
- Look for actionable leader-continuation or breakout setups.
- Submit protected entries with stop protection when rules allow.
- Run intraday from 10:00 to 15:59 ET every 10 minutes.

Why it exists:

- It catches stock opportunities that the slow ETF rotation would miss.
- It keeps aggressive entries isolated from the core rotation logic.
- It allows faster reaction while still keeping rule-based gating.

Current important guardrails:

- `OVERLAY_ENABLED=0` for this profile, so it does not duplicate SMH exposure.
- Existing positions are managed for stop/profit protection.
- New entries are still gated by market regime, setup quality, and breakout
  readiness.

## 3. CAN SLIM Experiment: `paper_canslim`

`paper_canslim` is an A/B test profile. It combines CAN SLIM-style selection
with Minervini-style execution.

Core idea:

- CAN SLIM is used to decide "which stocks are worth attention".
- Minervini-style trend, base, pivot, and stop logic decides "when to enter and
  how to manage risk".
- The profile runs separately from the main paper account so performance can be
  compared.

Why it exists:

- CAN SLIM emphasizes earnings/revenue growth, leadership, new highs, and
  institutional-style momentum.
- It is useful as a discovery engine even if it is not the best live executor.

Current observation:

- It has underperformed the main paper strategy recently.
- It should remain an experiment until its live/paper results improve.

## 4. TriCore Momentum Test: `paper_spy_alpha`

`paper_spy_alpha` is an isolated ETF rotation test account. It uses the same
TriCore logic family as the main `paper` profile, but remains separated for
research and comparison.

Core idea:

- Use SPY/QQQ/SMH adaptive momentum.
- Apply 200-day trend filters.
- Use volatility-targeted exposure.
- Rebalance slowly instead of chasing every intraday move.

Why it exists:

- It allows ETF-rotation research without contaminating stock-scanner results.
- It is useful for comparing "simple ETF alpha" versus stock-picking systems.

## 5. Put-Income Research: `paper_put_income`

`paper_put_income` is a local paper simulator for cash-secured put income. It
does not submit Alpaca options orders yet.

Core idea:

- Sell cash-secured puts only; no naked short options.
- Focus on liquid ETFs and quality large caps.
- Use 30-60 DTE, default around 45 DTE.
- Target roughly 0.16-0.30 delta.
- Require the underlying to be above its 200-day moving average.
- Require minimum implied volatility and liquidity.
- Buy to close at 50% profit.
- Manage around 21 DTE.
- Cap notional exposure per symbol and portfolio.

Why it exists:

- It is a passive-income/risk-controlled strategy research path.
- It is not designed to beat SPY/QQQ/SMH in strong bull markets.
- It can be useful as a separate "income sleeve" if options execution is added
  later.

Important caveat:

- The current 10-year backtest uses synthetic option prices from daily
  underlying bars and a realized-volatility proxy. It is useful for rule
  validation, but it is not equivalent to historical option quote backtesting.

## 6. Notifications And News

The system supports ntfy notifications for:

- Order alerts.
- Morning scan summaries.
- Daily summaries.
- Weekly summaries.
- Social/RSS monitoring.
- Market news radar, when enabled.

The live profile is currently disabled. The public repo should not contain real
notification topics, API keys, broker keys, or account IDs.

## Recommended Operating Model

Use separate profiles for separate jobs:

- `paper` should stay as the low-frequency core ETF rotation.
- `paper_stockscan` should handle faster stock-opportunity scanning.
- `paper_canslim` should stay isolated as an experiment.
- `paper_spy_alpha` should remain an ETF-rotation research/control account.
- `paper_put_income` should remain paper-only until options execution and quote
  quality are validated.
- `live` should stay disabled unless intentionally turned on with real account
  risk controls.

## Safety Rules

- Do not commit `.env` files.
- Do not commit Alpaca keys, OpenAI keys, ntfy topics, account IDs, or database
  files.
- Treat all generated trading signals as experimental.
- Paper performance does not guarantee live results.
- Use profile isolation before changing strategy behavior.

