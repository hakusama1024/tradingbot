"""Streamlit dashboard for monitoring the automated trading system."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_runtime_env():
    repo_root = _repo_root()
    profile = os.getenv("TRADING_PROFILE", "default").strip().lower()
    load_dotenv(repo_root / ".env")
    profile_env = repo_root / "profiles" / profile / ".env"
    if profile_env.exists():
        load_dotenv(profile_env, override=True)
    os.environ["TRADING_PROFILE"] = profile
    os.environ.setdefault("TRADING_PROFILE_DIR", str(repo_root / "profiles" / profile))
    os.environ.setdefault("TRADING_RUNTIME_ROOT", str(repo_root / "runtime" / profile))


_load_runtime_env()

from tradingagents.automation.config import build_config
from tradingagents.automation.orchestrator import Orchestrator
from tradingagents.storage.database import TradingDatabase


def _log_paths(results_dir: Path) -> tuple[Path, Path]:
    log_dir = results_dir / "service_logs"
    return (
        log_dir / "automation_service.out.log",
        log_dir / "automation_service.err.log",
    )


@st.cache_resource(show_spinner=False)
def _load_runtime():
    config = build_config()
    db = TradingDatabase(config["db_path"])
    orchestrator = Orchestrator(config)
    return config, db, orchestrator


def _launchctl_status(label: str) -> dict:
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{_uid()}/{label}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"state": "unknown", "detail": str(exc)}

    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return {"state": "not_loaded", "detail": text.strip()}

    state = "unknown"
    pid = None
    last_exit_code = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("state = "):
            state = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("pid = "):
            pid = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("last exit code = "):
            last_exit_code = stripped.split("=", 1)[1].strip()
    return {
        "state": state,
        "pid": pid,
        "last_exit_code": last_exit_code,
        "detail": text.strip(),
    }


def _uid() -> str:
    return subprocess.check_output(["id", "-u"], text=True).strip()


def _tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(log file not found)"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:]).strip() or "(empty)"


def _snapshots_frame(db: TradingDatabase) -> pd.DataFrame:
    snapshots = db.get_snapshots(days=60)
    if not snapshots:
        return pd.DataFrame()
    frame = pd.DataFrame(list(reversed(snapshots)))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _recent_trades_frame(db: TradingDatabase, limit: int = 50) -> pd.DataFrame:
    frame = pd.DataFrame(db.get_recent_trades(limit=limit))
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _setups_frame(db: TradingDatabase) -> pd.DataFrame:
    frame = pd.DataFrame(db.get_latest_setup_candidates())
    if frame.empty:
        return frame
    ordered = [
        "screen_date",
        "symbol",
        "selected_for_analysis",
        "passed_template",
        "market_regime",
        "rs_percentile",
        "revenue_growth",
        "eps_growth",
        "stage_number",
        "base_label",
        "candidate_status",
        "pivot_price",
        "buy_point",
        "buy_limit_price",
        "initial_stop_price",
        "earnings_days_away",
    ]
    available = [column for column in ordered if column in frame.columns]
    return frame[available]


def _latest_report(repo_root: Path) -> tuple[dict | None, Path | None]:
    config = build_config()
    report_dir = Path(config.get("results_dir", repo_root / "results")) / "daily_reports"
    reports = sorted(report_dir.glob("*.json"))
    if not reports:
        return None, None
    latest = reports[-1]
    with latest.open("r", encoding="utf-8") as handle:
        return json.load(handle), latest


def main():
    st.set_page_config(
        page_title="TradingAgents Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TradingAgents Dashboard")
    st.caption("Live monitor for Alpaca paper trading, Minervini screening, and automation service.")

    config, db, orchestrator = _load_runtime()
    repo_root = _repo_root()
    results_dir = Path(config.get("results_dir", repo_root / "results"))
    stdout_log, stderr_log = _log_paths(results_dir)
    status = orchestrator.get_status()
    service = _launchctl_status(config.get("service_label", "com.tradingagents.scheduler"))
    snapshots = _snapshots_frame(db)
    recent_trades = _recent_trades_frame(db)
    latest_setups = _setups_frame(db)
    report, report_path = _latest_report(repo_root)

    with st.sidebar:
        st.subheader("System")
        st.write(f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
        st.write(f"Profile: `{config.get('profile_name', 'default')}`")
        st.write(f"Universe: `{config.get('trading_universe', 'n/a')}`")
        st.write(f"Paper mode: `{config.get('paper_trading', True)}`")
        st.write(f"Service: `{service.get('state', 'unknown')}`")
        if service.get("pid"):
            st.write(f"PID: `{service['pid']}`")
        if service.get("last_exit_code"):
            st.write(f"Last exit: `{service['last_exit_code']}`")
        if st.button("Refresh"):
            st.rerun()

    acct = status["account"]
    today = status["today"]
    screening = status.get("screening", {})
    overlay = status.get("overlay", {})

    metric_cols = st.columns(6)
    metric_cols[0].metric("Equity", f"${acct['equity']:,.2f}")
    metric_cols[1].metric("Cash", f"${acct['cash']:,.2f}")
    metric_cols[2].metric("Daily P&L", f"${acct['daily_pl']:,.2f}", acct["daily_pl_pct"])
    metric_cols[3].metric("Open Positions", str(len(status["positions"])))
    metric_cols[4].metric("Today Orders", str(today["trade_summary"]["total_orders"]))
    metric_cols[5].metric("Approved Setups", str(len(screening.get("approved_symbols", []))))

    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader("Equity Curve")
        if snapshots.empty:
            st.info("No daily snapshots saved yet.")
        else:
            chart = snapshots.set_index("date")[["equity", "cash"]]
            st.line_chart(chart)
            st.dataframe(
                snapshots[["date", "equity", "cash", "daily_pl", "daily_pl_pct"]].sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Open Positions")
        positions = pd.DataFrame(status["positions"])
        if positions.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions, use_container_width=True, hide_index=True)

        st.subheader("Recent Trades")
        if recent_trades.empty:
            st.info("No trades logged yet.")
        else:
            st.dataframe(recent_trades, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Screening")
        st.write(
            {
                "screen_date": screening.get("screen_date"),
                "market_regime": screening.get("market_regime"),
                "confirmed_uptrend": screening.get("confirmed_uptrend"),
                "approved_symbols": screening.get("approved_symbols", []),
                "setup_count": screening.get("setup_count"),
            }
        )
        if latest_setups.empty:
            st.info("No setup rows saved yet.")
        else:
            st.dataframe(latest_setups.head(25), use_container_width=True, hide_index=True)

        st.subheader("Overlay")
        if not overlay.get("enabled"):
            st.info("Overlay is disabled.")
        else:
            st.write(overlay)

        st.subheader("Latest Daily Report")
        if report is None:
            st.info("No daily report saved yet.")
        else:
            st.write(
                {
                    "report_file": str(report_path),
                    "date": report.get("date"),
                    "gross_filled_notional": report.get("trade_summary", {}).get("gross_filled_notional"),
                    "open_positions": report.get("position_summary", {}).get("open_positions"),
                }
            )

        st.subheader("Service Logs")
        out_tab, err_tab = st.tabs(["stdout", "stderr"])
        with out_tab:
            st.code(_tail_text(stdout_log), language="text")
        with err_tab:
            st.code(_tail_text(stderr_log), language="text")


if __name__ == "__main__":
    main()
