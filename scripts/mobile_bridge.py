#!/usr/bin/env python3
"""Read-only mobile bridge for TradingAgents.

Designed for OpenClaw/mobile use. It exposes concise summaries and drafting
helpers without giving the mobile agent a reason to call write-side commands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from tradingagents.automation.config import build_config
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.storage.database import TradingDatabase


def quiet_logging() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "alpaca",
        "tradingagents",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def pct(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "0.00%"
    return f"{number:.2%}"


def compact_pct(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "0.00%"
    return f"{number:.2f}%"


def smart_pct(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "0.00%"
    if abs(number) <= 1:
        return f"{number:.2%}"
    return f"{number:.2f}%"


def short_list(items: List[str], limit: int = 5) -> str:
    cleaned = [str(item) for item in items if item]
    if not cleaned:
        return "(none)"
    if len(cleaned) <= limit:
        return ", ".join(cleaned)
    return ", ".join(cleaned[:limit]) + f" +{len(cleaned) - limit}"


def build_runtime() -> tuple[Dict[str, Any], Orchestrator, TradingDatabase]:
    config = build_config()
    orch = Orchestrator(config)
    db = TradingDatabase(config["db_path"])
    return config, orch, db


def resolve_report(
    orch: Orchestrator, report_date: Optional[str] = None
) -> Dict[str, Any]:
    return orch._safe_generate_daily_report(save=False, report_date=report_date)


def resolve_status(orch: Orchestrator) -> Optional[Dict[str, Any]]:
    try:
        return orch.get_status()
    except Exception:
        return None


def position_rows_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    positions = report.get("positions") or []
    normalized: List[Dict[str, Any]] = []
    for row in positions:
        normalized.append(
            {
                "symbol": row.get("symbol"),
                "qty": float(row.get("qty") or row.get("quantity") or 0.0),
                "entry": float(row.get("avg_entry_price") or row.get("entry_price") or 0.0),
                "current": float(row.get("current_price") or row.get("market_price") or 0.0),
                "pl": float(row.get("unrealized_pl") or row.get("pl") or 0.0),
                "pl_pct": float(
                    row.get("unrealized_pl_pct")
                    or row.get("unrealized_plpc")
                    or row.get("pl_pct_decimal")
                    or 0.0
                ),
            }
        )
    return normalized


def top_winners_and_losers(
    positions: List[Dict[str, Any]], limit: int = 3
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(positions, key=lambda row: float(row.get("pl") or 0.0), reverse=True)
    winners = ordered[:limit]
    losers = sorted(positions, key=lambda row: float(row.get("pl") or 0.0))[:limit]
    return winners, losers


def approved_and_watch(setups: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    approved = [row for row in setups if bool(row.get("selected_for_analysis"))]
    approved_symbols = {row.get("symbol") for row in approved}
    watch = [
        row for row in setups
        if bool(row.get("rule_watch_candidate")) and row.get("symbol") not in approved_symbols
    ]
    approved.sort(key=lambda row: float(row.get("rs_percentile") or -1), reverse=True)
    watch.sort(key=lambda row: float(row.get("rs_percentile") or -1), reverse=True)
    return approved, watch


def format_setup_line(row: Dict[str, Any]) -> str:
    parts = [str(row.get("symbol") or "?")]
    status = row.get("candidate_status")
    if status:
        parts.append(str(status))
    rs = row.get("rs_percentile")
    if rs is not None:
        parts.append(f"RS {float(rs):.1f}")
    distance = row.get("distance_to_buy_point_pct")
    if distance is not None:
        parts.append(f"{float(distance):.1%} from buy point")
    pivot = row.get("buy_point") or row.get("pivot_price")
    if pivot is not None:
        parts.append(f"buy {float(pivot):.2f}")
    return " | ".join(parts)


def cmd_status(args: argparse.Namespace) -> int:
    _, orch, _ = build_runtime()
    status = resolve_status(orch)
    if status is None:
        report = resolve_report(orch)
        account = report.get("account", {})
        screening = report.get("screening_batch", {}) or {}
        print("交易状态（fallback）")
        print(f"日期: {report.get('date')}")
        print(f"权益: {money(account.get('equity'))}")
        print(f"现金: {money(account.get('cash'))}")
        print(f"当日盈亏: {money(account.get('daily_pl'))}")
        print(f"Regime: {screening.get('market_regime') or 'unknown'}")
        print(f"Approved: {short_list(screening.get('approved_symbols') or [])}")
        return 0

    account = status["account"]
    market = status["market"]
    screening = status.get("screening", {}) or {}
    today = status.get("today", {}) or {}
    trade_summary = today.get("trade_summary", {}) or {}
    positions = sorted(status.get("positions") or [], key=lambda row: float(row.get("pl") or 0.0), reverse=True)

    print("交易状态")
    print(f"市场: {'OPEN' if market.get('is_open') else 'CLOSED'}")
    print(f"权益: {money(account.get('equity'))}")
    print(f"现金: {money(account.get('cash'))}")
    print(f"购买力: {money(account.get('buying_power'))}")
    print(f"当日盈亏: {money(account.get('daily_pl'))} ({account.get('daily_pl_pct')})")
    print(f"持仓数: {len(status.get('positions') or [])}")
    print(f"今日订单: {trade_summary.get('total_orders', 0)}")
    print(f"今日成交: {trade_summary.get('filled_orders', 0)}")
    print(f"Screen regime: {screening.get('market_regime') or 'unknown'}")
    print(f"Approved: {short_list(screening.get('approved_symbols') or [], args.limit)}")

    if positions:
        print("持仓前排:")
        for row in positions[: args.limit]:
            print(
                f"- {row['symbol']}: {money(row.get('pl'))} ({row.get('pl_pct')})"
            )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    _, orch, _ = build_runtime()
    report = resolve_report(orch, args.date)
    account = report.get("account", {})
    trade_summary = report.get("trade_summary", {}) or {}
    screening = report.get("screening_batch", {}) or {}
    setups = report.get("setups", []) or []
    positions = position_rows_from_report(report)
    winners, losers = top_winners_and_losers(positions, args.limit)
    approved, watch = approved_and_watch(setups)

    print("日报")
    print(f"日期: {report.get('date')}")
    print(f"权益: {money(account.get('equity'))}")
    print(f"当日盈亏: {money(account.get('daily_pl'))} ({smart_pct(account.get('daily_pl_pct') or 0.0)})")
    print(f"订单/成交: {trade_summary.get('total_orders', 0)}/{trade_summary.get('filled_orders', 0)}")
    print(f"Regime: {screening.get('market_regime') or 'unknown'}")
    print(f"Approved: {short_list([row.get('symbol') for row in approved], args.limit)}")
    print(f"Watch: {short_list([row.get('symbol') for row in watch], args.limit)}")

    if winners:
        print("最强持仓:")
        for row in winners:
            print(f"- {row['symbol']}: {money(row['pl'])} ({pct(row['pl_pct'])})")
    if losers:
        print("回撤持仓:")
        for row in losers:
            print(f"- {row['symbol']}: {money(row['pl'])} ({pct(row['pl_pct'])})")

    trades = report.get("trades") or []
    if trades:
        print("今日交易:")
        for row in trades[: args.limit]:
            filled_qty = row.get("filled_qty") or row.get("qty") or 0
            filled_price = row.get("filled_price") or 0.0
            print(
                f"- {row.get('side', '?').upper()} {row.get('symbol', '?')} "
                f"{float(filled_qty):.0f} @ {money(filled_price)} [{row.get('status', 'unknown')}]"
            )
    else:
        print("今日交易: 无")
    return 0


def cmd_setups(args: argparse.Namespace) -> int:
    config, _, db = build_runtime()
    batch = (
        db.get_screening_batch_on_date(args.date)
        if args.date
        else db.get_latest_screening_batch()
    ) or {}
    rows = (
        db.get_setup_candidates_on_date(args.date)
        if args.date
        else db.get_latest_setup_candidates()
    )
    approved, watch = approved_and_watch(rows)
    approved = approved[: args.limit]
    watch = [row for row in watch if row not in approved][: args.limit]

    print("候选池")
    print(f"日期: {batch.get('screen_date') or args.date or date.today().isoformat()}")
    print(f"Regime: {batch.get('market_regime') or 'unknown'}")
    print(f"Universe: {config.get('trading_universe')}")
    print(f"Approved: {len(batch.get('approved_symbols') or [])}")
    print(f"Setups: {batch.get('row_count') or len(rows)}")

    if approved:
        print("可执行候选:")
        for row in approved:
            print(f"- {format_setup_line(row)}")
    else:
        print("可执行候选: 无")

    if watch:
        print("观察名单:")
        for row in watch:
            print(f"- {format_setup_line(row)}")
    elif rows:
        print("观察名单: 无")
    return 0


def cmd_weekly(args: argparse.Namespace) -> int:
    config = build_config()
    db = TradingDatabase(config["db_path"])
    end_day = datetime.fromisoformat(args.end_date).date() if args.end_date else date.today()
    start_day = end_day - timedelta(days=end_day.weekday())
    start_date = start_day.isoformat()
    end_date = end_day.isoformat()
    snapshots = db.get_snapshots_between(start_date, end_date)
    summary = db.get_trade_summary_between(start_date, end_date)

    week_pl = sum(float(row.get("daily_pl") or 0.0) for row in snapshots)
    week_return = 0.0
    ending_equity = 0.0
    if snapshots:
        first_equity = float(snapshots[0].get("equity") or 0.0)
        first_daily_pl = float(snapshots[0].get("daily_pl") or 0.0)
        start_equity = first_equity - first_daily_pl
        ending_equity = float(snapshots[-1].get("equity") or 0.0)
        if start_equity > 0:
            week_return = (ending_equity - start_equity) / start_equity
    best_day = max(snapshots, key=lambda row: float(row.get("daily_pl") or 0.0), default=None)
    worst_day = min(snapshots, key=lambda row: float(row.get("daily_pl") or 0.0), default=None)

    print("周总结")
    print(f"区间: {start_date} -> {end_date}")
    print(f"周盈亏: {money(week_pl)}")
    print(f"周收益率: {pct(week_return)}")
    print(f"交易日: {len(snapshots)}")
    print(f"订单/成交: {summary.get('total_orders', 0)}/{summary.get('filled_orders', 0)}")
    print(f"交易股票: {short_list(summary.get('symbols') or [], args.limit)}")
    print(f"期末权益: {money(ending_equity)}")
    if best_day:
        print(f"最好一天: {best_day.get('date')} {money(best_day.get('daily_pl'))}")
    if worst_day:
        print(f"最差一天: {worst_day.get('date')} {money(worst_day.get('daily_pl'))}")
    return 0


def cmd_xiaohongshu(args: argparse.Namespace) -> int:
    _, orch, _ = build_runtime()
    report = resolve_report(orch, args.date)
    positions = position_rows_from_report(report)
    winners, losers = top_winners_and_losers(positions, 3)
    screening = report.get("screening_batch", {}) or {}
    account = report.get("account", {}) or {}
    trades = report.get("trades") or []
    setups = report.get("setups") or []
    approved, _ = approved_and_watch(setups)

    report_date = report.get("date") or date.today().isoformat()
    dt = datetime.fromisoformat(report_date)
    title = (
        f"{dt.month}月{dt.day}日美股Swing交易日记｜"
        f"{'今天有新开仓' if trades else '今天没新开仓'}，"
        f"账户{smart_pct(account.get('daily_pl_pct') or 0.0)}"
    )

    best_lines = [
        f"{row['symbol']}：{pct(row['pl_pct'])}" for row in winners if row.get("symbol")
    ]
    weak_lines = [
        f"{row['symbol']}：{pct(row['pl_pct'])}" for row in losers if row.get("symbol")
    ]

    if trades:
        trade_lines = []
        for row in trades[:5]:
            filled_qty = row.get("filled_qty") or row.get("qty") or 0
            trade_lines.append(
                f"{row.get('side', '?').upper()} {row.get('symbol', '?')} "
                f"{float(filled_qty):.0f}股"
            )
        trade_reason = "今天有新的高质量信号进入买点区间，所以系统执行了新单。"
    else:
        trade_lines = ["今天没有新开仓，收益主要来自已有持仓波动。"]
        trade_reason = "今天虽然在扫描市场，但没有新的标的同时满足市场环境、买点和风控条件，所以没有强行追单。"

    approved_names = [row.get("symbol") for row in approved[:5] if row.get("symbol")]
    if not approved_names:
        approved_names = screening.get("approved_symbols") or []

    lines = [
        f"标题：{title}",
        "",
        "正文：",
        "今天继续记录一套低人工干预的美股 swing 交易系统。",
        "",
        "1. 今日结果",
        f"账户当日收益：{smart_pct(account.get('daily_pl_pct') or 0.0)}",
        f"收盘权益：{money(account.get('equity'))}",
        f"当日盈亏：{money(account.get('daily_pl'))}",
        "",
        "2. 当前持仓表现",
        ("今天表现最强的几只：" + "、".join(best_lines)) if best_lines else "今天表现最强的几只：暂无",
        ("拖累比较明显的：" + "、".join(weak_lines)) if weak_lines else "拖累比较明显的：暂无",
        "",
        "3. 今天的交易",
        *trade_lines,
        "",
        "4. 这套自动化策略根据什么买",
        "这套系统不是看到涨就追，而是先做动态 broad 扫描，再筛相对强度、趋势结构、买点位置和风控条件。",
        f"今天的市场状态是 {screening.get('market_regime') or 'unknown'}，系统盘中每10分钟扫描一次。",
        trade_reason,
        "",
        "5. 为什么之前会买入这些",
        "核心不是猜涨跌，而是等强势股进入系统定义的 breakout 或 leader continuation 区间，再按仓位和止损规则执行。",
        "",
        "6. 明天继续观察",
        (
            "重点看：" + "、".join(approved_names)
            if approved_names
            else "重点看：明天继续等新的高质量候选。"
        ),
        "",
        "以上只是个人交易日志，不构成任何投资建议。投资有风险，入市需谨慎。",
    ]
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only mobile bridge for TradingAgents"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Concise live status")
    status_parser.add_argument("--limit", type=int, default=5, help="Rows to show")
    status_parser.set_defaults(func=cmd_status)

    report_parser = subparsers.add_parser("report", help="Concise daily report")
    report_parser.add_argument("--date", help="Report date YYYY-MM-DD")
    report_parser.add_argument("--limit", type=int, default=5, help="Rows to show")
    report_parser.set_defaults(func=cmd_report)

    setups_parser = subparsers.add_parser("setups", help="Approved/watch setups")
    setups_parser.add_argument("--date", help="Screen date YYYY-MM-DD")
    setups_parser.add_argument("--limit", type=int, default=5, help="Rows to show")
    setups_parser.set_defaults(func=cmd_setups)

    weekly_parser = subparsers.add_parser("weekly", help="Weekly summary")
    weekly_parser.add_argument("--end-date", help="Week end date YYYY-MM-DD")
    weekly_parser.add_argument("--limit", type=int, default=8, help="Rows to show")
    weekly_parser.set_defaults(func=cmd_weekly)

    xhs_parser = subparsers.add_parser("xiaohongshu", help="Daily Xiaohongshu draft")
    xhs_parser.add_argument("--date", help="Report date YYYY-MM-DD")
    xhs_parser.set_defaults(func=cmd_xiaohongshu)

    return parser


def main() -> int:
    quiet_logging()
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
