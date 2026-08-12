# paper_stockscan

Satellite stock scanner for the main paper Alpaca account.

This profile exists because the main `paper` launch agent now runs the low
frequency TriCore/SMH rotation script directly. `paper_stockscan` restores the
old 10-minute broad-market stock scan without stopping the TriCore core.

Runtime behavior:

- Same paper Alpaca account as `paper`.
- Separate runtime folder, database, logs, and launchd label.
- Runs `run_trading.py --profile paper_stockscan schedule --mode both`.
- Intraday scan runs every 10 minutes from 10:00 to 15:59 ET.
- Uses broad-market Minervini/leader-continuation stock selection.
- Keeps SMH overlay management enabled so stock opportunities can release idle
  overlay capital when rules allow.

Service label:

```bash
com.tradingagents.scheduler.paper_stockscan
```

Install or refresh:

```bash
python run_trading.py --profile paper_stockscan install-service --mode both
```

Check:

```bash
python run_trading.py --profile paper_stockscan status
```
