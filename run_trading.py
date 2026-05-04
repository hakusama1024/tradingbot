#!/usr/bin/env python3
"""Entry point for the TradingAgents automated trading platform.

Usage:
    # Run analysis immediately (manual trigger)
    python run_trading.py run

    # Start the scheduler (runs daily on autopilot)
    python run_trading.py schedule

    # Check current status (account, positions, performance)
    python run_trading.py status

    # Emergency: close all positions
    python run_trading.py close-all

    # Run with custom watchlist
    python run_trading.py run --symbols AAPL,NVDA,TSLA
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _detect_profile(argv: list[str]) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--profile" and idx + 1 < len(argv):
            return argv[idx + 1].strip().lower()
        if arg.startswith("--profile="):
            return arg.split("=", 1)[1].strip().lower()
    return os.getenv("TRADING_PROFILE", "default").strip().lower()


def _load_runtime_env() -> str:
    repo_root = _repo_root()
    profile = _detect_profile(sys.argv[1:])
    load_dotenv(repo_root / ".env")
    profile_dir = repo_root / "profiles" / profile
    profile_env = profile_dir / ".env"
    if profile_env.exists():
        load_dotenv(profile_env, override=True)
    os.environ["TRADING_PROFILE"] = profile
    os.environ.setdefault("TRADING_PROFILE_DIR", str(profile_dir))
    os.environ.setdefault(
        "TRADING_RUNTIME_ROOT",
        str(repo_root / "runtime" / profile),
    )
    return profile


CURRENT_PROFILE = _load_runtime_env()

from tradingagents.automation.config import build_config
from tradingagents.automation.launchd import install_launch_agent, uninstall_launch_agent
from tradingagents.automation.notifier import NtfyNotifier
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.automation.scheduler import TradingScheduler
from tradingagents.automation.social_monitor import SocialFeedMonitor


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    log_path = Path(os.getenv("TRADING_LOG_PATH", "trading.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)


def _runtime_overrides(args) -> dict:
    profile = getattr(args, "profile", None) or CURRENT_PROFILE
    return {"profile_name": profile}


def cmd_run(args):
    """Run analysis and trading immediately."""
    config = build_config(_runtime_overrides(args))
    if args.symbols:
        config["watchlist"] = [s.strip().upper() for s in args.symbols.split(",")]
    if args.model:
        config["deep_think_llm"] = args.model
        config["quick_think_llm"] = args.model
    if args.disable_minervini:
        config["minervini_enabled"] = False

    orch = Orchestrator(config)
    print(f"\nRunning analysis for: {config['watchlist']}")
    print(f"Paper mode: {config['paper_trading']}")
    print(f"LLM: {config['deep_think_llm']}")
    print("-" * 50)

    results = orch.run_daily_analysis()

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for symbol, result in results.items():
        if "error" in result:
            print(f"  {symbol}: ERROR - {result['error']}")
        elif result.get("traded"):
            print(f"  {symbol}: {result['side'].upper()} {result['qty']} shares ({result['status']})")
        elif result.get("screen_rejected"):
            print(f"  {symbol}: SKIP - {result['screen_rejected']}")
        elif result.get("risk_rejected"):
            print(f"  {symbol}: {result['action']} - BLOCKED by risk: {result['risk_rejected']}")
        else:
            print(f"  {symbol}: {result['action']} - no trade needed")
    print()


def cmd_schedule(args):
    """Start the automated scheduler."""
    config = build_config(_runtime_overrides(args))
    if args.symbols:
        config["watchlist"] = [s.strip().upper() for s in args.symbols.split(",")]
    if args.mode:
        config["trading_mode"] = args.mode
    if args.disable_minervini:
        config["minervini_enabled"] = False

    print(f"Starting TradingAgents Scheduler")
    print(f"  Mode: {config['trading_mode']}")
    print(f"  Paper: {config['paper_trading']}")
    if config.get("broad_market_enabled"):
        print("  Watchlist: dynamic broad-market universe")
    else:
        print(f"  Watchlist: {config['watchlist']}")
    print(f"  Swing time: {config['swing_analysis_time']} ET")
    print(f"  Minervini gate: {config['minervini_enabled']}")
    print(f"  Selection: {config.get('selection_strategy', 'minervini')}")
    print(f"  Overlay: {config.get('overlay_enabled', False)}")
    print(f"  Press Ctrl+C to stop")
    print()

    scheduler = TradingScheduler(config)
    scheduler.start()


def cmd_status(args):
    """Show current status."""
    config = build_config(_runtime_overrides(args))
    orch = Orchestrator(config)
    status = orch.get_status()

    print("\n" + "=" * 50)
    print("TRADING SYSTEM STATUS")
    print("=" * 50)
    print(f"\n  Profile: {config.get('profile_name', 'default')}")
    print(f"  Selection: {config.get('selection_strategy', 'minervini')}")

    acct = status["account"]
    print(f"\n  Account:")
    print(f"    Equity:       ${acct['equity']:>12,.2f}")
    print(f"    Cash:         ${acct['cash']:>12,.2f}")
    print(f"    Buying Power: ${acct['buying_power']:>12,.2f}")
    print(f"    Daily P&L:    ${acct['daily_pl']:>12,.2f}  ({acct['daily_pl_pct']})")

    print(f"\n  Market: {'OPEN' if status['market']['is_open'] else 'CLOSED'}")
    print(f"    Next open:  {status['market']['next_open']}")
    print(f"    Next close: {status['market']['next_close']}")

    print(f"\n  Positions ({len(status['positions'])}):")
    if status["positions"]:
        for p in status["positions"]:
            print(f"    {p['symbol']:>6}: {p['qty']:>6.0f} shares @ ${p['entry']:.2f} "
                  f"-> ${p['current']:.2f}  P&L: ${p['pl']:>8,.2f} ({p['pl_pct']})")
    else:
        print("    (none)")

    perf = status["performance"]
    if perf.get("current_equity"):
        print(f"\n  Performance:")
        print(f"    Total return:  {perf.get('total_return_pct', 'N/A')}")
        print(f"    Max drawdown:  {perf.get('max_drawdown_pct', 'N/A')}")
        print(f"    Sharpe ratio:  {perf.get('sharpe_ratio', 'N/A')}")

    today = status["today"]
    trade_summary = today["trade_summary"]
    print(f"\n  Today:")
    print(f"    Orders:        {trade_summary['total_orders']}")
    print(f"    Filled:        {trade_summary['filled_orders']}")
    print(f"    Symbols:       {', '.join(trade_summary['symbols']) if trade_summary['symbols'] else '(none)'}")
    print(f"    Open P&L:      ${today['unrealized_pl']:>12,.2f}")

    screening = status.get("screening", {})
    if screening.get("screen_date"):
        print(f"\n  Screening:")
        print(f"    Screen date:   {screening['screen_date']}")
        print(f"    Regime:        {screening['market_regime']}")
        print(f"    Uptrend:       {screening['confirmed_uptrend']}")
        print(f"    Approved:      {', '.join(screening['approved_symbols']) if screening['approved_symbols'] else '(none)'}")
        print(f"    Setups:        {screening['setup_count']}")

    overlay = status.get("overlay", {})
    if overlay.get("enabled"):
        print(f"\n  Overlay:")
        print(f"    Symbol:        {overlay['symbol']}")
        print(f"    Trigger:       {overlay['trigger']}")
        print(f"    Regime:        {overlay['market_regime']}")
        print(f"    Score:         {overlay['market_score'] if overlay['market_score'] is not None else 'n/a'}")
        print(f"    Allowed:       {overlay['overlay_allowed']}")
        print(f"    Position Qty:  {overlay['position_qty']:.0f}")
        print(f"    Position Val:  ${overlay['position_value']:>12,.2f}")

    notifications = status.get("notifications", {})
    if notifications.get("enabled"):
        print(f"\n  Notifications:")
        print(f"    Provider:      {notifications['provider']}")
        print(f"    Topic:         {notifications['topic']}")
        print(f"    Server:        {notifications['server']}")

    if config.get("social_monitor_enabled"):
        print(f"\n  Social Monitor:")
        print(f"    Users:         {', '.join(config.get('social_monitor_usernames', [])) or '(none)'}")
        print(f"    Topic:         {config.get('social_ntfy_topic') or '(unset)'}")
        print(f"    Feed:          {config.get('social_feed_url_template')}")

    print(f"\n  Watchlist: {status['watchlist']}")
    if config.get("broad_market_enabled"):
        print("  Universe mode: dynamic broad-market coarse scan")
    print(f"  Paper mode: {status['paper_mode']}")
    print()


def cmd_close_all(args):
    """Emergency close all positions."""
    config = build_config(_runtime_overrides(args))
    orch = Orchestrator(config)

    if not args.confirm:
        print("WARNING: This will close ALL open positions.")
        print("Run with --confirm to proceed.")
        return

    print("Closing all positions...")
    results = orch.emergency_close_all()
    print(f"Closed {len(results)} positions.")


def cmd_trades(args):
    """Show recent trades."""
    config = build_config(_runtime_overrides(args))
    from tradingagents.storage.database import TradingDatabase
    db = TradingDatabase(config["db_path"])
    trades = db.get_recent_trades(limit=args.limit)

    print(f"\nRecent Trades (last {args.limit}):")
    print("-" * 80)
    for t in trades:
        print(f"  {t['timestamp']}  {t['side']:>4} {t['qty'] or 0:>6.0f} "
              f"{t['symbol']:>6} @ ${t['filled_price'] or 0:>8.2f}  [{t['status']}]")
    if not trades:
        print("  (no trades yet)")
    print()


def cmd_report(args):
    """Show and optionally save the current daily report."""
    config = build_config(_runtime_overrides(args))
    orch = Orchestrator(config)
    report = orch.generate_daily_report(save=not args.no_save)

    acct = report["account"]
    trade_summary = report["trade_summary"]
    perf = report["performance"]
    position_summary = report["position_summary"]
    screening_batch = report.get("screening_batch")
    attribution = report.get("attribution", {}) or {}

    print("\n" + "=" * 50)
    print("DAILY REPORT")
    print("=" * 50)
    print(f"\n  Date: {report['date']}")
    print(f"  Paper mode: {report['paper_mode']}")
    print(f"  Watchlist: {report['watchlist']}")

    print(f"\n  Account:")
    print(f"    Equity:       ${acct['equity']:>12,.2f}")
    print(f"    Cash:         ${acct['cash']:>12,.2f}")
    print(f"    Buying Power: ${acct['buying_power']:>12,.2f}")
    print(f"    Daily P&L:    ${acct['daily_pl']:>12,.2f}  ({acct['daily_pl_pct']:.2%})")

    print(f"\n  Trades Today:")
    print(f"    Orders:        {trade_summary['total_orders']}")
    print(f"    Filled:        {trade_summary['filled_orders']}")
    print(f"    Buy orders:    {trade_summary['buy_orders']}")
    print(f"    Sell orders:   {trade_summary['sell_orders']}")
    print(f"    Gross notional:${trade_summary['gross_filled_notional']:>12,.2f}")
    print(f"    Symbols:       {', '.join(trade_summary['symbols']) if trade_summary['symbols'] else '(none)'}")

    print(f"\n  Open Positions:")
    print(f"    Count:         {position_summary['open_positions']}")
    print(f"    Unrealized P&L:${position_summary['total_unrealized_pl']:>12,.2f}")
    for pos in report["positions"]:
        print(
            f"    {pos['symbol']:>6}: {pos['qty']:>6.0f} shares @ ${pos['entry']:.2f} "
            f"-> ${pos['current']:.2f}  P&L: ${pos['unrealized_pl']:>8,.2f} "
            f"({pos['unrealized_pl_pct']:.2%})"
        )
    if not report["positions"]:
        print("    (none)")

    if perf.get("current_equity"):
        print(f"\n  Performance:")
        print(f"    Total return:  {perf.get('total_return_pct', 'N/A')}")
        print(f"    Max drawdown:  {perf.get('max_drawdown_pct', 'N/A')}")
        print(f"    Sharpe ratio:  {perf.get('sharpe_ratio', 'N/A')}")

    benchmark_window = attribution.get("benchmark_window", {}) or {}
    snapshot_attr = attribution.get("snapshot_attribution", {}) or {}
    benchmarks = benchmark_window.get("benchmarks", {}) or {}
    if benchmark_window.get("strategy_return") is not None:
        print(f"\n  Benchmark Window ({benchmark_window.get('period', 'n/a')}):")
        print(f"    Start:         {benchmark_window.get('start_date', 'n/a')}")
        print(f"    End:           {benchmark_window.get('end_date', 'n/a')}")
        print(f"    Strategy:      {float(benchmark_window.get('strategy_return', 0.0)):.2%}")
        for symbol in ("SPY", "QQQ", "SMH"):
            if symbol in benchmarks:
                print(f"    {symbol}:          {float(benchmarks[symbol]):.2%}")
        if "alpha_vs_spy" in benchmark_window:
            print(f"    Alpha vs SPY:  {float(benchmark_window.get('alpha_vs_spy', 0.0)):.2%}")
        if "alpha_vs_qqq" in benchmark_window:
            print(f"    Alpha vs QQQ:  {float(benchmark_window.get('alpha_vs_qqq', 0.0)):.2%}")

    if snapshot_attr.get("available"):
        print(f"\n  Return Attribution:")
        print(f"    Window:        {snapshot_attr.get('start_date')} -> {snapshot_attr.get('end_date')}")
        print(f"    Benchmark:     {snapshot_attr.get('benchmark_symbol')}")
        print(f"    Strategy:      {float(snapshot_attr.get('strategy_return', 0.0)):.2%}")
        print(f"    Overlay:       ${float(snapshot_attr.get('overlay_pnl', 0.0)):>12,.2f}  ({float(snapshot_attr.get('overlay_return_equiv', 0.0)):.2%})")
        print(f"    Beta:          ${float(snapshot_attr.get('beta_pnl', 0.0)):>12,.2f}  ({float(snapshot_attr.get('beta_return_equiv', 0.0)):.2%})")
        print(f"    Selection:     ${float(snapshot_attr.get('selection_pnl', 0.0)):>12,.2f}  ({float(snapshot_attr.get('selection_return_equiv', 0.0)):.2%})")
        print(f"    Avg overlay:   {float(snapshot_attr.get('avg_overlay_weight', 0.0)):.2%}")
        print(f"    Avg stock:     {float(snapshot_attr.get('avg_stock_weight', 0.0)):.2%}")
    elif snapshot_attr:
        print(f"\n  Return Attribution:")
        print(f"    Status:        {snapshot_attr.get('reason') or snapshot_attr.get('error') or 'Unavailable'}")

    if screening_batch:
        print(f"\n  Screening Batch:")
        print(f"    Screen date:   {screening_batch['screen_date']}")
        print(f"    Regime:        {screening_batch['market_regime']}")
        print(f"    Uptrend:       {bool(screening_batch['market_confirmed_uptrend'])}")
        print(f"    Approved:      {', '.join(screening_batch['approved_symbols']) if screening_batch['approved_symbols'] else '(none)'}")
        print(f"    Rows:          {screening_batch['row_count']}")

    print(f"\n  Setups ({len(report['setups'])} rows):")
    for setup in report["setups"][: args.limit]:
        approved = "YES" if setup["selected_for_analysis"] else "NO"
        print(
            f"    {setup['symbol']:>6}  approved={approved:<3} "
            f"regime={setup['market_regime'] or 'n/a':<22} "
            f"rs={setup['rs_percentile'] or 0:>5.1f}  "
            f"base={setup['base_label'] or 'n/a':<12} "
            f"pivot={setup['pivot_price'] or 0:>8.2f}"
        )
    if not report["setups"]:
        print("    (no setups saved yet)")

    print(f"\n  Trade Log ({len(report['trades'])} entries):")
    for trade in report["trades"][: args.limit]:
        qty = trade["filled_qty"] or trade["qty"] or 0
        print(
            f"    {trade['timestamp']}  {trade['side']:>4} {qty:>6.0f} "
            f"{trade['symbol']:>6} @ ${trade['filled_price'] or 0:>8.2f}  [{trade['status']}]"
        )
    if not report["trades"]:
        print("    (no trades today)")

    if report.get("report_path"):
        print(f"\n  Saved: {report['report_path']}")
    print()


def cmd_setups(args):
    """Show the latest Minervini setup candidates saved by automation."""
    config = build_config(_runtime_overrides(args))
    from tradingagents.storage.database import TradingDatabase

    db = TradingDatabase(config["db_path"])
    latest_batch = db.get_latest_screening_batch()
    if args.date:
        setups = db.get_setup_candidates_on_date(args.date)
    elif latest_batch:
        setups = db.get_setup_candidates_on_date(latest_batch["screen_date"])
    else:
        setups = db.get_latest_setup_candidates()

    print("\n" + "=" * 70)
    print("MINERVINI SETUPS")
    print("=" * 70)
    if latest_batch:
        print(f"\n  Screen date: {latest_batch['screen_date']}")
        print(f"  Market regime: {latest_batch['market_regime']}")
        print(f"  Confirmed uptrend: {bool(latest_batch['market_confirmed_uptrend'])}")
        print(f"  Approved symbols: {', '.join(latest_batch['approved_symbols']) if latest_batch['approved_symbols'] else '(none)'}")
        print(f"  Setup rows: {latest_batch['row_count']}")

    if not setups:
        print("\n  (no setup rows saved for this batch)\n")
        return

    print(f"\n  Rows ({min(len(setups), args.limit)} shown):")
    for setup in setups[: args.limit]:
        approved = "YES" if setup["selected_for_analysis"] else "NO"
        print(
            f"    {setup['symbol']:>6}  approved={approved:<3} "
            f"passed={bool(setup['passed_template'])!s:<5} "
            f"rs={setup['rs_percentile'] or 0:>5.1f}  "
            f"rev={setup['revenue_growth'] or 0:>6.2f}  "
            f"eps={setup['eps_growth'] or 0:>6.2f}  "
            f"base={setup['base_label'] or 'n/a':<12} "
            f"pivot={setup['pivot_price'] or 0:>8.2f}"
        )
    print()


def cmd_install_service(args):
    """Install the scheduler as a macOS launch agent."""
    repo_root = Path(__file__).resolve().parent
    project_python = repo_root / ".venv" / "bin" / "python"
    python_bin = project_python if project_python.exists() else Path(sys.executable)
    config = build_config(_runtime_overrides(args))
    label = args.label or config.get("service_label", "com.tradingagents.scheduler")
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    plist_path = install_launch_agent(
        repo_root=str(repo_root),
        python_bin=str(python_bin),
        mode=args.mode,
        symbols=symbols,
        label=label,
        log_dir=str(Path(config["results_dir"]) / "service_logs"),
        environment_variables={
            "TRADING_PROFILE": config.get("profile_name", "default"),
            "TRADING_PROFILE_DIR": os.getenv("TRADING_PROFILE_DIR", ""),
            "TRADING_RUNTIME_ROOT": config.get("runtime_root", ""),
            "TRADING_SERVICE_LABEL": label,
        },
    )

    print("Launch agent installed.")
    print(f"  Profile: {config.get('profile_name', 'default')}")
    print(f"  Label: {label}")
    print(f"  Plist: {plist_path}")
    print(f"  Python: {python_bin}")
    print(f"  Mode: {args.mode}")
    print(f"  Symbols: {symbols or 'config watchlist'}")
    print(f"  Logs: {Path(config['results_dir']) / 'service_logs'}")
    print()


def cmd_remove_service(args):
    """Remove the macOS launch agent."""
    config = build_config(_runtime_overrides(args))
    label = args.label or config.get("service_label", "com.tradingagents.scheduler")
    plist_path = uninstall_launch_agent(label=label)
    print("Launch agent removed.")
    print(f"  Profile: {config.get('profile_name', 'default')}")
    print(f"  Label: {label}")
    print(f"  Plist: {plist_path}")
    print()


def cmd_dashboard(args):
    """Launch the local Streamlit dashboard."""
    repo_root = Path(__file__).resolve().parent
    project_python = repo_root / ".venv" / "bin" / "python"
    python_bin = project_python if project_python.exists() else Path(sys.executable)
    streamlit_app = repo_root / "tradingagents" / "dashboard" / "app.py"
    port = str(args.port)

    command = [
        str(python_bin),
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        port,
        "--server.headless",
        "true",
    ]
    if args.address:
        command.extend(["--server.address", args.address])

    print("Starting dashboard...")
    print(f"  URL: http://{args.address}:{port}")
    print(f"  App: {streamlit_app}")
    print()
    subprocess.run(command, check=True, cwd=repo_root)


def cmd_notify_test(args):
    """Send a test ntfy notification."""
    config = build_config(_runtime_overrides(args))
    notifier = NtfyNotifier(config)
    if not notifier.enabled:
        print("ntfy is not enabled. Set NTFY_ENABLED=1 and NTFY_TOPIC first.")
        return

    ok = notifier.send(
        args.title,
        args.message,
        priority=args.priority,
        tags=["test_tube", "iphone"],
    )
    if ok:
        print("Test notification sent.")
        print(f"  Topic: {notifier.topic}")
        print(f"  Server: {notifier.server}")
    else:
        print("Test notification failed. Check your topic/server settings and logs.")


def cmd_social_check(args):
    """Poll configured X RSS feeds once."""
    config = build_config(_runtime_overrides(args))
    monitor = SocialFeedMonitor(config)
    result = monitor.check_once()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_social_test(args):
    """Send a test social notification."""
    config = build_config(_runtime_overrides(args))
    monitor = SocialFeedMonitor(config)
    if not monitor.enabled:
        print("social monitor is not enabled. Set SOCIAL_MONITOR_ENABLED=1 and SOCIAL_NTFY_TOPIC first.")
        return
    ok = monitor.send_test_notification()
    if ok:
        print("Social test notification sent.")
        print(f"  Topic: {config.get('social_ntfy_topic')}")
        print(f"  Users: {', '.join(config.get('social_monitor_usernames', []))}")
    else:
        print("Social test notification failed.")


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents Automated Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--profile",
        type=str,
        default=CURRENT_PROFILE,
        help="Runtime profile (for example: paper, live)",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = sub.add_parser("run", help="Run analysis and trading now")
    p_run.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. AAPL,NVDA)")
    p_run.add_argument("--model", type=str, help="LLM model to use")
    p_run.add_argument("--disable-minervini", action="store_true", help="Disable Minervini entry gating")

    # schedule
    p_sched = sub.add_parser("schedule", help="Start automated scheduler")
    p_sched.add_argument("--symbols", type=str, help="Comma-separated symbols")
    p_sched.add_argument("--mode", choices=["swing", "day", "both"], help="Trading mode")
    p_sched.add_argument("--disable-minervini", action="store_true", help="Disable Minervini entry gating")

    # status
    sub.add_parser("status", help="Show current status")

    # close-all
    p_close = sub.add_parser("close-all", help="Emergency close all positions")
    p_close.add_argument("--confirm", action="store_true", help="Confirm close all")

    # trades
    p_trades = sub.add_parser("trades", help="Show recent trades")
    p_trades.add_argument("--limit", type=int, default=20, help="Number of trades")

    # report
    p_report = sub.add_parser("report", help="Show today's trade and P&L report")
    p_report.add_argument("--limit", type=int, default=20, help="Number of trade rows to show")
    p_report.add_argument("--no-save", action="store_true", help="Do not save the JSON report")

    # setups
    p_setups = sub.add_parser("setups", help="Show the latest Minervini setup candidates")
    p_setups.add_argument("--date", type=str, help="Specific screen date (YYYY-MM-DD)")
    p_setups.add_argument("--limit", type=int, default=20, help="Number of rows to show")

    # install-service
    p_install = sub.add_parser(
        "install-service",
        help="Install the scheduler as a macOS launch agent",
    )
    p_install.add_argument(
        "--mode",
        choices=["swing", "day", "both"],
        default="both",
        help="Trading mode for the background scheduler",
    )
    p_install.add_argument("--symbols", type=str, help="Optional comma-separated symbols")
    p_install.add_argument(
        "--label",
        type=str,
        default=None,
        help="launchd label to install (defaults to the profile label)",
    )

    # remove-service
    p_remove = sub.add_parser(
        "remove-service",
        help="Remove the macOS launch agent",
    )
    p_remove.add_argument(
        "--label",
        type=str,
        default=None,
        help="launchd label to remove (defaults to the profile label)",
    )

    # dashboard
    p_dashboard = sub.add_parser(
        "dashboard",
        help="Launch the local monitoring dashboard",
    )
    p_dashboard.add_argument("--port", type=int, default=8501, help="Dashboard port")
    p_dashboard.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="Dashboard bind address",
    )

    # notify-test
    p_notify = sub.add_parser(
        "notify-test",
        help="Send a test phone notification through ntfy",
    )
    p_notify.add_argument("--title", type=str, default="TradingAgents Test")
    p_notify.add_argument(
        "--message",
        type=str,
        default="ntfy is connected. Future setup/order alerts will arrive here.",
    )
    p_notify.add_argument(
        "--priority",
        type=str,
        default="default",
        help="ntfy priority: min, low, default, high, max, urgent",
    )

    p_social_check = sub.add_parser(
        "social-check",
        help="Poll the configured X RSS sources once",
    )

    p_social_test = sub.add_parser(
        "social-test",
        help="Send a test alert to the social ntfy topic",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.verbose)

    commands = {
        "run": cmd_run,
        "schedule": cmd_schedule,
        "status": cmd_status,
        "close-all": cmd_close_all,
        "trades": cmd_trades,
        "report": cmd_report,
        "setups": cmd_setups,
        "install-service": cmd_install_service,
        "remove-service": cmd_remove_service,
        "dashboard": cmd_dashboard,
        "notify-test": cmd_notify_test,
        "social-check": cmd_social_check,
        "social-test": cmd_social_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
