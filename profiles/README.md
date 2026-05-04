# Runtime Profiles

Use profiles to keep `paper` and `live` fully separated while sharing the same codebase.

Directory layout:

- `profiles/paper/.env`
- `profiles/live/.env`
- `runtime/paper/trading.db`
- `runtime/paper/results/`
- `runtime/paper/trading.log`
- `runtime/live/trading.db`
- `runtime/live/results/`
- `runtime/live/trading.log`

Rules:

- Put shared keys such as `OPENAI_API_KEY` in the repo-root `.env` if you want both profiles to use them.
- Put account-specific keys such as `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the profile `.env`.
- Never commit `profiles/*/.env`.
- Run commands with `--profile paper` or `--profile live`.

Examples:

```bash
python run_trading.py --profile paper status
python run_trading.py --profile live status
python run_trading.py --profile paper install-service --mode both
python run_trading.py --profile live install-service --mode both
```
