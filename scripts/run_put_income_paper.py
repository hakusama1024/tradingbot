#!/usr/bin/env python3
"""Paper simulator for the Quality Put Income strategy.

This script records simulated short-put trades from live option chains. It does
not submit Alpaca option orders. Use it to validate the rules and notifications
before wiring broker-side options execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.automation.config import build_config
from tradingagents.automation.notifier import NtfyNotifier
from tradingagents.options import PutIncomeParams, PutIncomeScanner


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_profile(profile: str) -> dict:
    load_dotenv(ROOT / ".env")
    profile_dir = ROOT / "profiles" / profile
    load_dotenv(profile_dir / ".env", override=True)
    os.environ["TRADING_PROFILE"] = profile
    os.environ.setdefault("TRADING_PROFILE_DIR", str(profile_dir))
    os.environ.setdefault("TRADING_RUNTIME_ROOT", str(ROOT / "runtime" / profile))
    return build_config({"profile_name": profile})


def _strategy_params() -> PutIncomeParams:
    symbols = tuple(
        item.strip().upper()
        for item in os.getenv(
            "PUT_INCOME_SYMBOLS",
            (
                "SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,META,COST,JPM,V,MA,ORCL,SNOW,"
                "AMD,INTC,CSCO,GILD,MRK,PFE,KO,BAC,C,WMT,UBER,DIS,NKE,PYPL"
            ),
        ).split(",")
        if item.strip()
    )
    target_delta = float(os.getenv("PUT_INCOME_TARGET_DELTA", "0.25"))
    return PutIncomeParams(
        symbols=symbols,
        entry_dte=int(os.getenv("PUT_INCOME_ENTRY_DTE", "45")),
        manage_dte=int(os.getenv("PUT_INCOME_MANAGE_DTE", "21")),
        target_delta=target_delta,
        min_delta=float(os.getenv("PUT_INCOME_MIN_DELTA", "0.16")),
        max_delta=float(os.getenv("PUT_INCOME_MAX_DELTA", "0.30")),
        min_iv=float(os.getenv("PUT_INCOME_MIN_IV", "0.20")),
        min_premium_yield=float(os.getenv("PUT_INCOME_MIN_PREMIUM_YIELD", "0.010")),
        close_profit_pct=float(os.getenv("PUT_INCOME_CLOSE_PROFIT_PCT", "0.50")),
        max_symbol_notional_pct=float(os.getenv("PUT_INCOME_MAX_SYMBOL_NOTIONAL_PCT", "0.20")),
        max_total_notional_pct=float(os.getenv("PUT_INCOME_MAX_TOTAL_NOTIONAL_PCT", "0.85")),
        max_open_positions=int(os.getenv("PUT_INCOME_MAX_OPEN_POSITIONS", "5")),
        max_contracts_per_symbol=int(os.getenv("PUT_INCOME_MAX_CONTRACTS_PER_SYMBOL", "2")),
        trend_sma_days=int(os.getenv("PUT_INCOME_TREND_SMA_DAYS", "200")),
        min_open_interest=int(os.getenv("PUT_INCOME_MIN_OPEN_INTEREST", "100")),
        min_option_volume=int(os.getenv("PUT_INCOME_MIN_OPTION_VOLUME", "0")),
        cash_yield=float(os.getenv("PUT_INCOME_CASH_YIELD", "0.035")),
    )


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "cash": float(os.getenv("PUT_INCOME_STARTING_EQUITY", "100000")),
            "positions": [],
            "trades": [],
            "last_run": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "cash": float(os.getenv("PUT_INCOME_STARTING_EQUITY", "100000")),
            "positions": [],
            "trades": [],
            "last_run": None,
        }


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _date_now() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _mark_from_candidates(position: dict, rows: list[dict]) -> dict:
    for row in rows:
        if (
            row["symbol"] == position["symbol"]
            and row["expiry"] == position["expiry"]
            and abs(float(row["strike"]) - float(position["strike"])) < 0.01
        ):
            return {
                "mid": float(row["mid"]),
                "spot": float(row["spot"]),
                "dte": int(row["dte"]),
                "source": "chain",
            }
    return {
        "mid": float(position["entry_premium"]),
        "spot": None,
        "dte": 999,
        "source": "stale",
    }


def _close_position(state: dict, position: dict, mark: dict, reason: str) -> dict:
    contracts = int(position["contracts"])
    debit = float(mark["mid"])
    cash_delta = -debit * 100.0 * contracts
    state["cash"] += cash_delta
    trade = {
        "date": _date_now(),
        "action": "BUY_TO_CLOSE",
        "symbol": position["symbol"],
        "expiry": position["expiry"],
        "strike": position["strike"],
        "contracts": contracts,
        "debit": round(debit, 2),
        "profit": round((float(position["entry_premium"]) - debit) * 100.0 * contracts, 2),
        "profit_pct": round(1.0 - debit / max(float(position["entry_premium"]), 0.01), 3),
        "reason": reason,
    }
    state["trades"].append(trade)
    return trade


def _manage_positions(state: dict, params: PutIncomeParams, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    closed = []
    keep = []
    max_loss_multiple = float(os.getenv("PUT_INCOME_MAX_LOSS_MULTIPLE", "2.50"))
    for position in state["positions"]:
        mark = _mark_from_candidates(position, rows)
        entry = float(position["entry_premium"])
        profit_pct = 1.0 - float(mark["mid"]) / max(entry, 0.01)
        reason = None
        if mark["source"] == "chain" and profit_pct >= params.close_profit_pct:
            reason = "50% profit target"
        elif mark["source"] == "chain" and int(mark["dte"]) <= params.manage_dte:
            reason = "21 DTE management"
        elif mark["source"] == "chain" and float(mark["mid"]) >= entry * max_loss_multiple:
            reason = f"risk stop {max_loss_multiple:.1f}x premium"

        position["last_mark"] = mark
        position["unrealized_profit"] = round((entry - float(mark["mid"])) * 100.0 * int(position["contracts"]), 2)
        position["profit_pct"] = round(profit_pct, 3)
        if reason:
            closed.append(_close_position(state, position, mark, reason))
        else:
            keep.append(position)
    state["positions"] = keep
    return closed, keep


def _open_candidates(state: dict, params: PutIncomeParams, rows: list[dict], equity: float) -> list[dict]:
    opened = []
    existing_symbols = {p["symbol"] for p in state["positions"]}
    current_notional = sum(float(p["strike"]) * 100.0 * int(p["contracts"]) for p in state["positions"])
    high_iv_symbol_cap = float(os.getenv("PUT_INCOME_HIGH_IV_SYMBOL_CAP_PCT", "0.10"))
    high_iv_threshold = float(os.getenv("PUT_INCOME_HIGH_IV_THRESHOLD", "0.55"))

    for row in rows:
        symbol = row["symbol"]
        if symbol in existing_symbols:
            continue
        if len(state["positions"]) + len(opened) >= params.max_open_positions:
            break
        symbol_cap = params.max_symbol_notional_pct
        if float(row["iv"]) >= high_iv_threshold:
            symbol_cap = min(symbol_cap, high_iv_symbol_cap)
        max_symbol_notional = equity * symbol_cap
        contracts = min(params.max_contracts_per_symbol, int(max_symbol_notional // (float(row["strike"]) * 100.0)))
        if contracts < 1:
            continue
        notional = float(row["strike"]) * 100.0 * contracts
        if current_notional + notional > equity * params.max_total_notional_pct:
            continue
        premium = float(row["mid"])
        state["cash"] += premium * 100.0 * contracts
        position = {
            "opened_at": _date_now(),
            "symbol": symbol,
            "expiry": row["expiry"],
            "strike": row["strike"],
            "contracts": contracts,
            "entry_premium": round(premium, 2),
            "entry_spot": row["spot"],
            "delta_est": row["delta_est"],
            "iv": row["iv"],
            "premium_yield": row["premium_yield"],
            "contract": row["contract"],
        }
        trade = {
            "date": _date_now(),
            "action": "SELL_PUT",
            **position,
            "notional": round(notional, 2),
            "premium_credit": round(premium * 100.0 * contracts, 2),
        }
        state["positions"].append(position)
        state["trades"].append(trade)
        opened.append(trade)
        existing_symbols.add(symbol)
        current_notional += notional
    return opened


def _send_notification(notifier: NtfyNotifier, profile: str, opened: list[dict], closed: list[dict], state: dict) -> None:
    positions = state["positions"]
    notional = sum(float(p["strike"]) * 100.0 * int(p["contracts"]) for p in positions)
    realized = sum(float(t.get("profit", 0.0)) for t in state["trades"] if t.get("action") == "BUY_TO_CLOSE")
    lines = [
        f"Profile: {profile}",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"新开仓: {len(opened)}",
        f"平仓: {len(closed)}",
        f"当前模拟仓位: {len(positions)}",
        f"担保名义金额: ${notional:,.0f}",
        f"已实现模拟收益: ${realized:,.2f}",
    ]
    if opened:
        lines.append("开仓明细:")
        lines.extend(
            f"SELL PUT {t['symbol']} {t['expiry']} P{float(t['strike']):.2f} x{t['contracts']} "
            f"@ {float(t['entry_premium']):.2f}, delta {float(t['delta_est']):.2f}"
            for t in opened
        )
    if closed:
        lines.append("平仓明细:")
        lines.extend(
            f"BTC {t['symbol']} {t['expiry']} P{float(t['strike']):.2f} x{t['contracts']} "
            f"@ {float(t['debit']):.2f}, P/L ${float(t['profit']):,.2f}, {t['reason']}"
            for t in closed
        )
    if not opened and not closed and positions:
        lines.append("持仓:")
        lines.extend(
            f"{p['symbol']} {p['expiry']} P{float(p['strike']):.2f} x{p['contracts']} "
            f"浮动P/L ${float(p.get('unrealized_profit', 0.0)):,.2f}"
            for p in positions[:5]
        )
    notifier.send(
        "Quality Put Income 模拟运行",
        "\n".join(lines),
        priority="high" if opened or closed else "default",
        tags=["moneybag"],
        dedupe_key=f"put_income:{profile}:{_date_now()}:{len(opened)}:{len(closed)}:{len(positions)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Quality Put Income paper simulation pass.")
    parser.add_argument("--profile", default="paper_put_income")
    parser.add_argument("--no-open", action="store_true", help="Manage/mark only; do not open new simulated positions")
    parser.add_argument("--notify", action="store_true", help="Force notification even when no trades changed")
    args = parser.parse_args()

    config = _load_profile(args.profile)
    params = _strategy_params()
    runtime_root = Path(config.get("runtime_root") or ROOT / "runtime" / args.profile)
    state_path = runtime_root / "put_income_state.json"
    candidates_path = runtime_root / "results" / "put_income_candidates.json"
    state = _load_state(state_path)
    scanner = PutIncomeScanner(params)
    rows = scanner.scan()
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    closed, _ = _manage_positions(state, params, rows)
    equity = float(os.getenv("PUT_INCOME_STARTING_EQUITY", str(state.get("cash", 100000))))
    opened = [] if args.no_open else _open_candidates(state, params, rows, equity)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_state(state_path, state)

    print("Quality Put Income paper simulator")
    print("=" * 80)
    print(f"Profile: {args.profile}")
    print(f"Candidates: {len(rows)}")
    print(f"Opened: {len(opened)}")
    print(f"Closed: {len(closed)}")
    print(f"Open positions: {len(state['positions'])}")
    print(f"State: {state_path}")
    for trade in opened[:10]:
        print(
            f"SELL PUT {trade['symbol']} {trade['expiry']} P{float(trade['strike']):.2f} "
            f"x{trade['contracts']} @ {float(trade['entry_premium']):.2f}"
        )
    for trade in closed[:10]:
        print(
            f"BTC {trade['symbol']} {trade['expiry']} P{float(trade['strike']):.2f} "
            f"x{trade['contracts']} @ {float(trade['debit']):.2f} P/L ${float(trade['profit']):,.2f}"
        )

    if _env_bool("PUT_INCOME_NTFY_ENABLED", True) or args.notify:
        notifier = NtfyNotifier(config)
        _send_notification(notifier, args.profile, opened, closed, state)


if __name__ == "__main__":
    main()
