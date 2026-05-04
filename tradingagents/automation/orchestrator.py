"""Core orchestrator — the heart of the automated trading platform.

Ties together: AI analysis → signal extraction → position sizing →
risk checks → order execution → logging.
"""

import logging
import os
import traceback
import json
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, List, Dict, Optional

import pandas as pd
from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import Account, Position
from tradingagents.risk.risk_engine import RiskEngine
from tradingagents.portfolio.position_sizer import PositionSizer
from tradingagents.portfolio.portfolio_tracker import PortfolioTracker
from tradingagents.automation.prescreener import MinerviniPreScreener
from tradingagents.broker.models import OrderRequest
from tradingagents.automation.notifier import NtfyNotifier
from tradingagents.storage.database import TradingDatabase
from tradingagents.storage.memory_store import PersistentMemory
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.research import MarketDataWarehouse, build_market_context

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main controller: AI analysis → sizing → risk → execution → logging."""

    def __init__(self, config: dict):
        self.config = config
        self.execution_enabled = bool(config.get("execution_enabled", True))
        self.watchlist = config.get("watchlist", [])
        self._latest_analysis_states: Dict[str, Dict] = {}
        self._latest_minervini_preflight = None
        self._active_universe: list[str] = list(self.watchlist)
        self._latest_overlay_context: Optional[Dict] = None
        self._latest_market_context: Optional[Dict] = None
        self._latest_earnings_context: Dict[str, Dict[str, Any]] = {}

        # Broker
        self.broker = None
        if self.execution_enabled:
            self.broker = AlpacaBroker(
                api_key=config["alpaca_api_key"],
                secret_key=config["alpaca_secret_key"],
                paper=config.get("paper_trading", True),
            )

        # Database
        self.db = TradingDatabase(config.get("db_path", "trading.db"))
        self.notifier = NtfyNotifier(config)

        # Risk engine
        starting_equity = self.db.get_starting_equity()
        risk_config = {**config, "starting_equity": starting_equity}
        self.risk_engine = RiskEngine(risk_config)

        # Position sizer
        self.sizer = PositionSizer(config)

        # Portfolio tracker
        self.tracker = (
            PortfolioTracker(self.broker, self.db, self.config)
            if self.broker is not None
            else None
        )

        # AI analysis engine (lazy init to avoid heavy startup cost)
        self._ta: Optional[TradingAgentsGraph] = None

    def _get_ai_engine(self) -> TradingAgentsGraph:
        """Lazy-initialize the AI analysis engine."""
        if self._ta is None:
            logger.info("Initializing AI analysis engine...")
            ai_config = self.config.copy()
            self._ta = TradingAgentsGraph(
                selected_analysts=["market", "social", "news", "fundamentals"],
                debug=False,
                config=ai_config,
            )
            self._load_persistent_memories()
        return self._ta

    def _load_persistent_memories(self):
        """Load persistent memories into the AI engine."""
        if self._ta is None:
            return
        for name, mem in [
            ("bull_memory", self._ta.bull_memory),
            ("bear_memory", self._ta.bear_memory),
            ("trader_memory", self._ta.trader_memory),
            ("invest_judge_memory", self._ta.invest_judge_memory),
            ("risk_manager_memory", self._ta.risk_manager_memory),
        ]:
            stored = self.db.load_memories(name)
            if stored:
                mem.add_situations(stored)
                logger.info(f"Loaded {len(stored)} memories for {name}")

    def _save_persistent_memories(self):
        """Save AI engine memories to database."""
        if self._ta is None:
            return
        for name, mem in [
            ("bull_memory", self._ta.bull_memory),
            ("bear_memory", self._ta.bear_memory),
            ("trader_memory", self._ta.trader_memory),
            ("invest_judge_memory", self._ta.invest_judge_memory),
            ("risk_manager_memory", self._ta.risk_manager_memory),
        ]:
            pairs = list(zip(mem.documents, mem.recommendations))
            if pairs:
                self.db.conn.execute(
                    "DELETE FROM agent_memories WHERE memory_name = ?", (name,)
                )
                self.db.save_memories(name, pairs)

    # ── Main Analysis Flow ───────────────────────────────────────────

    def _run_minervini_preflight(self):
        """Refresh Minervini screen results used to gate new entries."""
        if not self.config.get("minervini_enabled", False):
            return None

        today = date.today().isoformat()
        if (
            self._latest_minervini_preflight is not None
            and self._latest_minervini_preflight.trade_date == today
        ):
            return self._latest_minervini_preflight

        logger.info("Running Minervini preflight screen for %s", self.watchlist)
        preflight = MinerviniPreScreener(self.config).run(self.watchlist)
        screen_rows = (
            preflight.screen_df.to_dict("records")
            if preflight.screen_df is not None and not preflight.screen_df.empty
            else []
        )
        self.db.save_setup_candidates(
            screen_rows,
            screen_date=preflight.trade_date,
            selected_symbols=preflight.approved_symbols,
        )
        self.db.save_screening_batch(
            screen_date=preflight.trade_date,
            market_regime=preflight.market_regime,
            confirmed_uptrend=preflight.confirmed_uptrend,
            approved_symbols=preflight.approved_symbols,
            row_count=len(screen_rows),
        )
        self._latest_minervini_preflight = preflight
        logger.info(
            "Minervini preflight done: regime=%s, confirmed_uptrend=%s, approved=%s",
            preflight.market_regime,
            preflight.confirmed_uptrend,
            preflight.approved_symbols,
        )
        self._notify_preflight(preflight)
        return preflight

    def _notify_preflight(self, preflight):
        if (
            not self.notifier.enabled
            or preflight is None
            or preflight.screen_df is None
            or preflight.screen_df.empty
        ):
            return

        screen_df = preflight.screen_df.copy()
        actionable = screen_df[screen_df.get("rule_entry_candidate", False) == True]  # noqa: E712
        watch = screen_df[screen_df.get("rule_watch_candidate", False) == True]  # noqa: E712

        if not actionable.empty:
            rows = actionable.head(5).to_dict("records")
            symbols = [row["symbol"] for row in rows]
            details = []
            for row in rows:
                buy_point = self._to_float(row.get("buy_point"))
                rs = self._to_float(row.get("rs_percentile"))
                parts = [row["symbol"]]
                if buy_point is not None:
                    parts.append(f"buy {buy_point:.2f}")
                if rs is not None:
                    parts.append(f"RS {rs:.1f}")
                details.append(" | ".join(parts))
            message = "\n".join(
                [
                    f"{preflight.trade_date} actionable setups",
                    f"Regime: {preflight.market_regime}",
                    f"Symbols: {', '.join(symbols)}",
                    *details,
                ]
            )
            self.notifier.send(
                "TradingAgents Actionable Setups",
                message,
                priority="high",
                tags=["moneybag", "chart_with_upwards_trend"],
                dedupe_key=f"setup-actionable:{preflight.trade_date}:{','.join(symbols)}",
            )
            return

        if watch.empty:
            return

        rows = watch.head(5).to_dict("records")
        symbols = [row["symbol"] for row in rows]
        details = []
        for row in rows:
            distance = self._to_float(row.get("distance_to_buy_point_pct"))
            status = row.get("candidate_status") or "watch"
            detail = f"{row['symbol']} | {status}"
            if distance is not None:
                detail += f" | {distance:.1%} below buy point"
            details.append(detail)
        message = "\n".join(
            [
                f"{preflight.trade_date} setup watchlist",
                f"Regime: {preflight.market_regime}",
                f"Watch: {', '.join(symbols)}",
                *details,
            ]
        )
        self.notifier.send(
            "TradingAgents Setup Watch",
            message,
            priority="default",
            tags=["eyes", "chart_with_upwards_trend"],
            dedupe_key=f"setup-watch:{preflight.trade_date}:{','.join(symbols)}",
        )

    def _notify_morning_scan(self, preflight, snapshot: Optional[Dict] = None):
        if (
            not self.notifier.enabled
            or not self.config.get("ntfy_morning_scan_enabled", True)
            or preflight is None
            or preflight.screen_df is None
            or preflight.screen_df.empty
        ):
            return

        screen_df = preflight.screen_df.copy()
        top_n = max(int(self.config.get("ntfy_morning_scan_top_n", 5)), 1)
        actionable = (
            screen_df[screen_df["rule_entry_candidate"].fillna(False).astype(bool)]
            if "rule_entry_candidate" in screen_df.columns
            else screen_df.iloc[0:0]
        )
        watch = (
            screen_df[screen_df["rule_watch_candidate"].fillna(False).astype(bool)]
            if "rule_watch_candidate" in screen_df.columns
            else screen_df.iloc[0:0]
        )
        ranked = screen_df.sort_values(
            by=["rs_percentile", "distance_to_buy_point_pct"],
            ascending=[False, True],
            na_position="last",
        ).head(top_n)
        canslim_selected = (
            screen_df[screen_df["canslim_selected"].fillna(False).astype(bool)]
            if "canslim_selected" in screen_df.columns
            else screen_df.iloc[0:0]
        )

        lines = [
            f"Date: {preflight.trade_date}",
            f"Regime: {preflight.market_regime}",
            f"Approved: {len(preflight.approved_symbols)}",
            f"Watch: {len(watch)}",
        ]
        if "canslim_selected" in screen_df.columns:
            lines.extend(
                [
                    f"CAN SLIM通过: {len(canslim_selected)}",
                    f"Minervini观察: {len(watch)}",
                    f"最终可下单: {len(actionable)}",
                ]
            )
        if snapshot is not None:
            daily_pl = self._to_float(snapshot.get("daily_pl"))
            if daily_pl is not None:
                lines.append(f"Account daily P/L: ${daily_pl:,.2f}")

        if preflight.approved_symbols:
            lines.append(
                "Approved symbols: " + ", ".join(preflight.approved_symbols[:top_n])
            )
        elif not watch.empty:
            lines.append(
                "No approved setups yet. Watching: "
                + ", ".join(watch["symbol"].head(top_n).tolist())
            )
        else:
            lines.append("No approved or watch candidates this morning.")

        if "canslim_selected" in screen_df.columns and not canslim_selected.empty:
            lines.append(
                "CAN SLIM名单: " + ", ".join(canslim_selected["symbol"].head(top_n).tolist())
            )
        if not actionable.empty:
            lines.append(
                "可下单名单: " + ", ".join(actionable["symbol"].head(top_n).tolist())
            )

        rotation_lines = self._build_rotation_summary_lines(screen_df, top_n=top_n)
        lines.extend(rotation_lines)

        if not ranked.empty:
            lines.append("Top names:")
            for row in ranked.to_dict("records"):
                rs = self._to_float(row.get("rs_percentile"))
                distance = self._to_float(row.get("distance_to_buy_point_pct"))
                parts = [str(row.get("symbol", ""))]
                if row.get("candidate_status"):
                    parts.append(str(row["candidate_status"]))
                if rs is not None:
                    parts.append(f"RS {rs:.1f}")
                if distance is not None:
                    parts.append(f"{distance:.1%} from buy point")
                lines.append(" | ".join(parts))

        self.notifier.send(
            "TradingAgents Morning Scan",
            "\n".join(lines),
            priority="high" if preflight.approved_symbols else "default",
            tags=["sunrise", "chart_with_upwards_trend"],
            dedupe_key=f"morning-scan:{preflight.trade_date}",
        )

    def _build_rotation_summary_lines(
        self, screen_df: pd.DataFrame, top_n: int = 5
    ) -> list[str]:
        if (
            not self.config.get("ntfy_morning_scan_rotation_enabled", True)
            or screen_df is None
            or screen_df.empty
        ):
            return []

        focus_mask = pd.Series(False, index=screen_df.index)
        for column in (
            "rule_entry_candidate",
            "rule_watch_candidate",
            "selected_for_analysis",
            "passed_template",
        ):
            if column in screen_df.columns:
                focus_mask = focus_mask | screen_df[column].fillna(False).astype(bool)

        focus_df = screen_df[focus_mask].copy()
        if focus_df.empty:
            sort_cols = [
                col
                for col in ("rs_percentile", "distance_to_buy_point_pct")
                if col in screen_df.columns
            ]
            if sort_cols:
                ascending = [False, True][: len(sort_cols)]
                focus_df = screen_df.sort_values(
                    by=sort_cols,
                    ascending=ascending,
                    na_position="last",
                ).head(max(top_n * 4, 12)).copy()
            else:
                focus_df = screen_df.head(max(top_n * 4, 12)).copy()

        sector_summary = self._summarize_rotation_groups(
            focus_df,
            group_field="sector",
            top_groups=int(self.config.get("ntfy_morning_scan_rotation_top_groups", 3)),
        )
        industry_summary = self._summarize_rotation_groups(
            focus_df,
            group_field="industry",
            top_groups=int(self.config.get("ntfy_morning_scan_rotation_top_groups", 3)),
        )

        focus_groups = []
        if industry_summary:
            focus_groups = [entry["name"] for entry in industry_summary[:2]]
        elif sector_summary:
            focus_groups = [entry["name"] for entry in sector_summary[:2]]

        lines: list[str] = []
        if sector_summary:
            lines.append(
                "板块轮动："
                + "; ".join(
                    f"{entry['name']} x{entry['count']}"
                    for entry in sector_summary
                )
            )
        if industry_summary:
            lines.append(
                "行业组："
                + "; ".join(
                    f"{entry['name']} x{entry['count']} ({entry['symbols']})"
                    for entry in industry_summary
                )
            )
        if focus_groups:
            lines.append("重点关注：" + ", ".join(focus_groups))
        return lines

    def _summarize_rotation_groups(
        self,
        screen_df: pd.DataFrame,
        group_field: str,
        top_groups: int = 3,
    ) -> list[Dict[str, Any]]:
        if group_field not in screen_df.columns or screen_df.empty:
            return []

        valid = screen_df.copy()
        valid[group_field] = (
            valid[group_field]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown"})
        )
        if "symbol" not in valid.columns:
            return []

        summaries: list[Dict[str, Any]] = []
        for name, group in valid.groupby(group_field):
            rank_cols = [
                col
                for col in ("rule_entry_candidate", "rs_percentile")
                if col in group.columns
            ]
            if rank_cols:
                ranked = group.sort_values(
                    by=rank_cols,
                    ascending=[False, False][: len(rank_cols)],
                    na_position="last",
                )
            else:
                ranked = group
            symbols = ", ".join(ranked["symbol"].astype(str).head(3).tolist())
            avg_rs = self._to_float(group.get("rs_percentile", pd.Series(dtype=float)).mean())
            summaries.append(
                {
                    "name": name,
                    "count": int(len(group)),
                    "avg_rs": avg_rs if avg_rs is not None else -999.0,
                    "symbols": symbols,
                }
            )

        summaries.sort(key=lambda item: (-item["count"], -item["avg_rs"], item["name"]))
        return summaries[: max(top_groups, 1)]

    def _notify_daily_summary(self, report: Dict):
        if (
            not self.notifier.enabled
            or not self.config.get("ntfy_daily_summary_enabled", True)
            or not report
        ):
            return

        trade_summary = report.get("trade_summary", {}) or {}
        screening_batch = report.get("screening_batch", {}) or {}
        position_summary = report.get("position_summary", {}) or {}
        account = report.get("account", {}) or {}
        attribution = report.get("attribution", {}) or {}
        benchmark_window = attribution.get("benchmark_window", {}) or {}
        snapshot_attr = attribution.get("snapshot_attribution", {}) or {}

        symbols = trade_summary.get("symbols") or []
        lines = [
            f"Date: {report.get('date')}",
            f"Orders: {trade_summary.get('total_orders', 0)}",
            f"Filled: {trade_summary.get('filled_orders', 0)}",
            f"Regime: {screening_batch.get('market_regime') or 'unknown'}",
            f"Approved setups: {screening_batch.get('selected_count', 0)}",
            f"Daily P/L: ${float(account.get('daily_pl', 0.0)):,.2f}",
            f"Open positions: {position_summary.get('open_positions', 0)}",
            f"Open unrealized P/L: ${float(position_summary.get('total_unrealized_pl', 0.0)):,.2f}",
        ]
        benchmarks = benchmark_window.get("benchmarks", {}) or {}
        if benchmark_window.get("strategy_return") is not None and benchmarks:
            strategy_return = float(benchmark_window.get("strategy_return", 0.0))
            comp_parts = [f"1M Strategy {strategy_return:.1%}"]
            for symbol in ("SPY", "QQQ", "SMH"):
                if symbol in benchmarks:
                    comp_parts.append(f"{symbol} {float(benchmarks[symbol]):.1%}")
            lines.append(" | ".join(comp_parts))
        if snapshot_attr.get("available"):
            lines.append(
                "Attribution: "
                f"Overlay {float(snapshot_attr.get('overlay_return_equiv', 0.0)):.1%} | "
                f"Beta {float(snapshot_attr.get('beta_return_equiv', 0.0)):.1%} | "
                f"Selection {float(snapshot_attr.get('selection_return_equiv', 0.0)):.1%}"
            )
        if symbols:
            lines.append("Symbols traded: " + ", ".join(symbols))
        else:
            lines.append("No trades executed today.")

        self.notifier.send(
            "TradingAgents Daily Summary",
            "\n".join(lines),
            priority="default",
            tags=["memo", "moneybag"],
            dedupe_key=f"daily-summary:{report.get('date')}",
        )

    def _build_miss_review(self, report: Dict) -> Dict:
        setups = report.get("setups", []) or []
        trade_summary = report.get("trade_summary", {}) or {}
        screening_batch = report.get("screening_batch", {}) or {}
        traded_symbols = set(trade_summary.get("symbols") or [])
        top_n = max(int(self.config.get("ntfy_miss_review_top_n", 5)), 1)
        near_buy_threshold = float(
            self.config.get("ntfy_miss_review_near_buy_threshold_pct", 0.12)
        )

        status_counts = Counter(str(row.get("candidate_status") or "unknown") for row in setups)
        approved = [row for row in setups if bool(row.get("selected_for_analysis"))]
        watch = [row for row in setups if bool(row.get("rule_watch_candidate"))]
        passed_template = [row for row in setups if bool(row.get("passed_template"))]
        breakout = [row for row in setups if bool(row.get("breakout_signal"))]

        def sort_key(row: Dict):
            distance = self._to_float(row.get("distance_to_buy_point_pct"))
            if distance is None:
                distance = 999.0
            rs = self._to_float(row.get("rs_percentile"))
            if rs is None:
                rs = -999.0
            return (
                0 if bool(row.get("selected_for_analysis")) else 1,
                0 if bool(row.get("rule_watch_candidate")) else 1,
                0 if bool(row.get("passed_template")) else 1,
                distance,
                -rs,
            )

        near_candidates = [
            row
            for row in setups
            if row.get("symbol") not in traded_symbols
            and (
                bool(row.get("selected_for_analysis"))
                or bool(row.get("rule_watch_candidate"))
                or bool(row.get("passed_template"))
                or (
                    self._to_float(row.get("distance_to_buy_point_pct")) is not None
                    and self._to_float(row.get("distance_to_buy_point_pct")) <= near_buy_threshold
                )
                or str(row.get("candidate_status") or "")
                in {"building_base", "near_pivot", "breakout_ready"}
            )
        ]
        if not near_candidates:
            near_candidates = [
                row
                for row in setups
                if row.get("symbol") not in traded_symbols
                and str(row.get("candidate_status") or "") != "no_base"
            ]

        near_candidates = sorted(near_candidates, key=sort_key)[:top_n]

        if screening_batch.get("market_regime") == "market_correction" and not approved:
            primary_reason = "Market correction blocked new swing entries."
        elif status_counts.get("no_base", 0) == len(setups) and setups:
            primary_reason = "All screened names were still no_base."
        elif watch and not approved:
            primary_reason = "A few names were on watch, but none were actionable."
        elif approved and not trade_summary.get("total_orders", 0):
            primary_reason = "Some setups were approved, but none triggered a trade."
        else:
            primary_reason = "No obvious missed trade from today's screen."

        return {
            "primary_reason": primary_reason,
            "market_regime": screening_batch.get("market_regime"),
            "approved_count": len(approved),
            "watch_count": len(watch),
            "passed_template_count": len(passed_template),
            "breakout_count": len(breakout),
            "status_counts": dict(status_counts),
            "near_candidates": near_candidates,
        }

    def _notify_miss_review(self, report: Dict):
        if (
            not self.notifier.enabled
            or not self.config.get("ntfy_miss_review_enabled", True)
            or not report
        ):
            return

        review = report.get("miss_review") or self._build_miss_review(report)
        trade_summary = report.get("trade_summary", {}) or {}

        lines = [
            f"Date: {report.get('date')}",
            f"Reason: {review.get('primary_reason')}",
            f"Regime: {review.get('market_regime') or 'unknown'}",
            f"Approved: {review.get('approved_count', 0)}",
            f"Watch: {review.get('watch_count', 0)}",
            f"Breakouts: {review.get('breakout_count', 0)}",
            f"Orders: {trade_summary.get('total_orders', 0)}",
        ]

        near_candidates = review.get("near_candidates") or []
        if near_candidates:
            lines.append("Closest names:")
            for row in near_candidates:
                symbol = row.get("symbol") or "?"
                status = row.get("candidate_status") or "unknown"
                rs = self._to_float(row.get("rs_percentile"))
                distance = self._to_float(row.get("distance_to_buy_point_pct"))
                parts = [symbol, status]
                if rs is not None:
                    parts.append(f"RS {rs:.1f}")
                if distance is not None:
                    parts.append(f"{distance:.1%} from buy point")
                lines.append(" | ".join(parts))
        else:
            lines.append("No close candidates worth flagging.")

        self.notifier.send(
            "TradingAgents Miss Review",
            "\n".join(lines),
            priority="default",
            tags=["mag", "bar_chart"],
            dedupe_key=f"miss-review:{report.get('date')}",
        )

    def _build_fallback_daily_report(self, report_date: Optional[str] = None) -> Dict:
        report_day = report_date or date.today().isoformat()
        snapshot = self.db.get_snapshot_on_date(report_day) or {}
        screening_batch = self.db.get_screening_batch_on_date(report_day) or {}
        setups = self.db.get_setup_candidates_on_date(report_day)
        trades = self.db.get_trades_on_date(report_day)
        trade_summary = self.db.get_trade_summary(report_day)

        positions = []
        raw_positions = snapshot.get("positions_json")
        if raw_positions:
            try:
                positions = json.loads(raw_positions)
            except Exception:
                positions = []

        total_unrealized_pl = sum(
            float((row or {}).get("unrealized_pl") or 0.0) for row in positions
        )

        report = {
            "date": report_day,
            "account": {
                "equity": float(snapshot.get("equity") or 0.0),
                "cash": float(snapshot.get("cash") or 0.0),
                "buying_power": float(snapshot.get("buying_power") or 0.0),
                "portfolio_value": float(snapshot.get("portfolio_value") or 0.0),
                "daily_pl": float(snapshot.get("daily_pl") or 0.0),
                "daily_pl_pct": float(snapshot.get("daily_pl_pct") or 0.0),
            },
            "trade_summary": trade_summary,
            "trades": trades,
            "setups": setups,
            "screening_batch": screening_batch,
            "positions": positions,
            "position_summary": {
                "open_positions": len(positions),
                "total_unrealized_pl": total_unrealized_pl,
            },
            "performance": self.tracker.get_performance_summary(),
            "attribution": {
                "available": False,
                "benchmark_window": {},
                "snapshot_attribution": {
                    "available": False,
                    "reason": "Fallback report",
                },
            },
            "paper_mode": self.config.get("paper_trading", True),
            "watchlist": self.watchlist,
            "fallback_used": True,
        }
        report["miss_review"] = self._build_miss_review(report)
        return report

    def _safe_generate_daily_report(
        self, save: bool = True, report_date: Optional[str] = None
    ) -> Dict:
        try:
            return self.generate_daily_report(save=save, report_date=report_date)
        except Exception as exc:
            logger.warning(
                "Falling back to DB-backed daily report for %s: %s",
                report_date or date.today().isoformat(),
                exc,
                exc_info=True,
            )
            report = self._build_fallback_daily_report(report_date)
            report["fallback_reason"] = str(exc)
            if save:
                output_dir = os.path.join(
                    self.config.get("results_dir", "./results"),
                    "daily_reports",
                )
                report_path = self.tracker.save_daily_report(report, output_dir)
                report["report_path"] = str(report_path)
                logger.info("Fallback daily report saved to %s", report_path)
            return report

    def send_daily_notifications(self, report_date: Optional[str] = None) -> Dict:
        report = self._safe_generate_daily_report(save=True, report_date=report_date)
        self._notify_daily_summary(report)
        self._notify_miss_review(report)
        return {
            "date": report.get("date"),
            "report_path": report.get("report_path"),
            "fallback_used": bool(report.get("fallback_used")),
        }

    def send_weekly_summary(self, week_end_date: Optional[str] = None) -> Dict:
        if (
            not self.notifier.enabled
            or not self.config.get("ntfy_weekly_summary_enabled", True)
        ):
            return {"enabled": False}

        end_day = datetime.fromisoformat(
            week_end_date or date.today().isoformat()
        ).date()
        start_day = end_day - timedelta(days=end_day.weekday())
        start_date = start_day.isoformat()
        end_date = end_day.isoformat()

        snapshots = self.db.get_snapshots_between(start_date, end_date)
        trade_summary = self.db.get_trade_summary_between(start_date, end_date)

        week_pl = sum(float(row.get("daily_pl") or 0.0) for row in snapshots)
        positive_days = sum(1 for row in snapshots if float(row.get("daily_pl") or 0.0) > 0)
        negative_days = sum(1 for row in snapshots if float(row.get("daily_pl") or 0.0) < 0)

        week_return = 0.0
        if snapshots:
            first_equity = float(snapshots[0].get("equity") or 0.0)
            first_daily_pl = float(snapshots[0].get("daily_pl") or 0.0)
            start_equity = first_equity - first_daily_pl
            end_equity = float(snapshots[-1].get("equity") or 0.0)
            if start_equity > 0:
                week_return = (end_equity - start_equity) / start_equity
        else:
            end_equity = 0.0

        best_day = max(
            snapshots,
            key=lambda row: float(row.get("daily_pl") or 0.0),
            default=None,
        )
        worst_day = min(
            snapshots,
            key=lambda row: float(row.get("daily_pl") or 0.0),
            default=None,
        )

        lines = [
            f"Week: {start_date} to {end_date}",
            f"Week P/L: ${week_pl:,.2f}",
            f"Week return: {week_return:.2%}",
            f"Trading days: {len(snapshots)}",
            f"Up days: {positive_days}",
            f"Down days: {negative_days}",
            f"Orders: {trade_summary.get('total_orders', 0)}",
            f"Filled: {trade_summary.get('filled_orders', 0)}",
        ]

        symbols = trade_summary.get("symbols") or []
        if symbols:
            lines.append("Symbols traded: " + ", ".join(symbols[:12]))
        if best_day is not None:
            lines.append(
                "Best day: "
                f"{best_day.get('date')} (${float(best_day.get('daily_pl') or 0.0):,.2f})"
            )
        if worst_day is not None:
            lines.append(
                "Worst day: "
                f"{worst_day.get('date')} (${float(worst_day.get('daily_pl') or 0.0):,.2f})"
            )
        lines.append(f"Ending equity: ${end_equity:,.2f}")

        self.notifier.send(
            "TradingAgents Weekly Summary",
            "\n".join(lines),
            priority="default",
            tags=["calendar", "bar_chart", "moneybag"],
            dedupe_key=f"weekly-summary:{end_date}",
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "week_pl": week_pl,
            "week_return": week_return,
            "orders": trade_summary.get("total_orders", 0),
        }

    def _build_screen_rejection(self, symbol: str, preflight) -> Dict:
        base = {
            "symbol": symbol,
            "action": "SKIP",
            "traded": False,
            "screen_rejected": "Not approved by Minervini preflight",
        }
        if preflight is None or preflight.screen_df is None or preflight.screen_df.empty:
            return base

        row_match = preflight.screen_df[preflight.screen_df["symbol"] == symbol]
        if row_match.empty:
            if not preflight.confirmed_uptrend:
                base["screen_rejected"] = (
                    f"Market regime is {preflight.market_regime}; new swing entries disabled"
                )
            return base

        row = row_match.iloc[0]
        regime = row.get("market_regime") or preflight.market_regime
        row_dict = row.to_dict()
        if not self._entries_allowed_for_setup(row_dict, regime):
            base["screen_rejected"] = (
                f"Market regime is {regime}; new swing entries disabled"
            )
        else:
            base["screen_rejected"] = (
                f"Template score {row['template_score']} with base={row['base_label']} "
                f"stage={row.get('stage_number')} status={row.get('candidate_status')} "
                f"and breakout_ready={row.get('breakout_ready')}"
            )
        base["market_regime"] = row.get("market_regime")
        base["rs_percentile"] = row.get("rs_percentile")
        base["pivot_price"] = row.get("pivot_price")
        base["buy_point"] = row.get("buy_point")
        base["candidate_status"] = row.get("candidate_status")
        return base

    def run_daily_analysis(self) -> Dict:
        """Run full analysis → trade cycle for all watchlist symbols."""
        logger.info("=" * 60)
        logger.info(f"Starting daily analysis at {datetime.now()}")
        logger.info(f"Watchlist: {self.watchlist}")
        logger.info("=" * 60)
        self._latest_analysis_states = {}

        if self.execution_enabled and self.broker is not None and not self.broker.is_market_open():
            logger.info("Market is closed. Skipping analysis.")
            return {"status": "market_closed"}

        preflight = None
        preflight_error = None
        try:
            preflight = self._run_minervini_preflight()
        except Exception as e:
            preflight_error = str(e)
            logger.error("Minervini preflight failed: %s", e, exc_info=True)

        if not self.execution_enabled or self.broker is None:
            analysis_universe = [
                symbol for symbol in self._analysis_universe(preflight)
                if not self._is_overlay_symbol(symbol)
            ]
            self._active_universe = list(analysis_universe)
            results = {}
            if preflight_error:
                for symbol in analysis_universe:
                    results[symbol] = {
                        "symbol": symbol,
                        "action": "ALERT_ONLY",
                        "traded": False,
                        "screen_rejected": f"Minervini preflight failed: {preflight_error}",
                    }
                return results

            setup_rows = {}
            if preflight is not None and preflight.screen_df is not None and not preflight.screen_df.empty:
                setup_rows = {
                    row["symbol"]: row.to_dict()
                    for _, row in preflight.screen_df.iterrows()
                }
                approved = set(preflight.approved_symbols)
                for symbol in analysis_universe:
                    if symbol in approved:
                        row = setup_rows.get(symbol, {})
                        results[symbol] = {
                            "symbol": symbol,
                            "action": "ALERT_ONLY",
                            "traded": False,
                            "candidate_status": row.get("candidate_status"),
                            "buy_point": row.get("buy_point"),
                            "buy_limit_price": row.get("buy_limit_price"),
                            "market_regime": row.get("market_regime"),
                            "screen_rejected": "Execution disabled for this profile; alert only",
                        }
                    else:
                        results[symbol] = self._build_screen_rejection(symbol, preflight)
            logger.info("Alerts-only analysis complete. Results: %s", list(results.keys()))
            return results

        overlay_context = self._get_overlay_context()
        analysis_universe = [
            symbol for symbol in self._analysis_universe(preflight)
            if not self._is_overlay_symbol(symbol)
        ]
        self._active_universe = list(analysis_universe)
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        stock_positions = self._stock_positions(positions)
        results = {}

        # First check existing positions for SELL signals
        for pos in stock_positions:
            try:
                result = self._manage_existing_position(pos, account, stock_positions)
                results[pos.symbol] = result
            except Exception as e:
                logger.error(f"Error analyzing {pos.symbol}: {e}")
                results[pos.symbol] = {"error": str(e)}

        # Then analyze watchlist for new BUY opportunities
        candidate_symbols = list(analysis_universe)
        if preflight_error:
            candidate_symbols = []
            for symbol in analysis_universe:
                if symbol in results:
                    continue
                results[symbol] = {
                    "symbol": symbol,
                    "action": "SKIP",
                    "traded": False,
                    "screen_rejected": f"Minervini preflight failed: {preflight_error}",
                }
        elif preflight is not None:
            candidate_symbols = [
                symbol for symbol in analysis_universe if symbol in set(preflight.approved_symbols)
            ]
            for symbol in analysis_universe:
                if symbol in results or symbol in candidate_symbols:
                    continue
                results[symbol] = self._build_screen_rejection(symbol, preflight)
        setup_rows = {}
        if preflight is not None and preflight.screen_df is not None and not preflight.screen_df.empty:
            setup_rows = {
                row["symbol"]: row.to_dict()
                for _, row in preflight.screen_df.iterrows()
            }

        if self._overlay_enabled():
            try:
                overlay_step_aside = self._maybe_free_overlay_for_stock_candidates(
                    account=account,
                    positions=positions,
                    setup_rows=setup_rows,
                    analysis_universe=analysis_universe,
                )
                if overlay_step_aside is not None:
                    results[overlay_step_aside["symbol"]] = overlay_step_aside
                    account = self.broker.get_account()
                    positions = self.broker.get_positions()
                    stock_positions = self._stock_positions(positions)
            except Exception as e:
                logger.error("Overlay capital release failed: %s", e, exc_info=True)

        for symbol in analysis_universe:
            if preflight is not None and symbol not in candidate_symbols and symbol not in results:
                results[symbol] = self._build_screen_rejection(symbol, preflight)
                continue
            if symbol in results:
                continue
            try:
                if preflight is not None:
                    result = self._trade_rule_based_setup(
                        setup_rows.get(symbol, {"symbol": symbol}),
                        account,
                        stock_positions,
                    )
                else:
                    result = self._analyze_and_trade(symbol, account, stock_positions)
                results[symbol] = result
                # Refresh account and positions after each trade
                if result.get("traded"):
                    account = self.broker.get_account()
                    positions = self.broker.get_positions()
                    stock_positions = self._stock_positions(positions)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}\n{traceback.format_exc()}")
                results[symbol] = {"error": str(e)}

        if self._overlay_enabled():
            try:
                overlay_result = self._manage_overlay_position(
                    account=account,
                    positions=positions,
                    context=overlay_context,
                )
                if overlay_result is not None:
                    results[overlay_result["symbol"]] = overlay_result
            except Exception as e:
                logger.error("Overlay management failed: %s", e, exc_info=True)

        # Take daily snapshot
        self.tracker.take_daily_snapshot()

        logger.info(f"Daily analysis complete. Results: {list(results.keys())}")
        return results

    def _analyze_and_trade(
        self, symbol: str, account: Account, positions: List[Position]
    ) -> Dict:
        """Analyze a single symbol and execute trade if warranted."""

        logger.info(f"--- Analyzing {symbol} ---")

        # 1. Run AI analysis
        ta = self._get_ai_engine()
        today = date.today().isoformat()
        state, simple_signal = ta.propagate(symbol, today)
        self._latest_analysis_states[symbol] = state

        # 2. Extract structured signal
        structured = ta.signal_processor.process_signal_structured(
            state["final_trade_decision"], symbol=symbol
        )
        logger.info(
            f"{symbol}: action={structured['action']} "
            f"confidence={structured['confidence']:.2f} "
            f"reasoning={structured['reasoning'][:100]}"
        )
        return self._execute_structured_signal(
            symbol=symbol,
            structured=structured,
            account=account,
            positions=positions,
            full_analysis=state.get("final_trade_decision", ""),
        )

    def _execute_structured_signal(
        self,
        symbol: str,
        structured: Dict,
        account: Account,
        positions: List[Position],
        full_analysis: str = "",
    ) -> Dict:
        structured = {**structured, "symbol": symbol}

        # 3. Log signal to database
        signal_id = self.db.log_signal(
            symbol=symbol,
            action=structured["action"],
            confidence=structured["confidence"],
            reasoning=structured["reasoning"],
            stop_loss=structured.get("stop_loss_pct"),
            take_profit=structured.get("take_profit_pct"),
            timeframe=structured.get("timeframe", "swing"),
            full_analysis=full_analysis,
        )

        # 4. Calculate position size
        current_position = self._find_position(positions, symbol)
        current_price = self.broker.get_latest_price(symbol)
        total_pos_value = sum(p.market_value for p in positions)

        # Add stop_loss to signal for risk-based sizing
        if structured.get("stop_loss") is None and structured.get("stop_loss_pct"):
            structured["stop_loss"] = current_price * (1 - structured["stop_loss_pct"])

        order_request = self.sizer.calculate(
            signal=structured,
            account=account,
            current_price=current_price,
            current_position=current_position,
            total_position_value=total_pos_value,
        )

        if order_request is None:
            logger.info(f"{symbol}: No trade needed (action={structured['action']})")
            return {"symbol": symbol, "action": structured["action"], "traded": False}

        existing_open_order = self._find_existing_open_order(symbol, order_request.side)
        if existing_open_order is not None:
            reason = (
                f"Existing open {order_request.side} order "
                f"{existing_open_order.order_id} [{existing_open_order.status}]"
            )
            logger.info(f"{symbol}: {reason}")
            self.db.mark_signal_rejected(signal_id, reason)
            return {
                "symbol": symbol,
                "action": structured["action"],
                "traded": False,
                "screen_rejected": reason,
            }

        # 5. Risk check
        risk_result = self.risk_engine.check_order(
            order_request, account, positions, current_price
        )
        if not risk_result.passed:
            logger.warning(f"{symbol}: Risk check FAILED: {risk_result.reason}")
            self.db.mark_signal_rejected(signal_id, risk_result.reason)
            return {
                "symbol": symbol, "action": structured["action"],
                "traded": False, "risk_rejected": risk_result.reason
            }

        # 6. Execute order — use bracket orders for buys (auto SL + TP)
        logger.info(f"{symbol}: Submitting {order_request.side} {order_request.qty} shares")

        if order_request.side == "buy":
            explicit_stop = structured.get("stop_loss")
            explicit_take_profit = structured.get("take_profit")
            if explicit_stop and explicit_stop < current_price:
                sl_price = round(float(explicit_stop), 2)
            else:
                sl_price = self.risk_engine.get_stop_loss_price(
                    current_price, structured.get("stop_loss_pct")
                )
            if explicit_take_profit and explicit_take_profit > current_price:
                tp_price = round(float(explicit_take_profit), 2)
            else:
                tp_price = self.risk_engine.get_take_profit_price(
                    current_price, structured.get("take_profit_pct")
                )
            if self.config.get("minervini_use_stop_only_entries", True):
                tp_price = None
            order_result = self.broker.submit_bracket_order(
                order_request, stop_loss_price=sl_price, take_profit_price=tp_price
            )
            if tp_price is None:
                logger.info(f"{symbol}: Protected entry order SL=${sl_price:.2f} (no fixed TP)")
            else:
                logger.info(f"{symbol}: Bracket order SL=${sl_price:.2f} TP=${tp_price:.2f}")
        else:
            order_result = self.broker.submit_order(order_request)

        # Track trading frequency
        self.risk_engine.record_trade()

        # 7. Log trade
        self.db.log_trade(
            symbol=symbol,
            side=order_request.side,
            qty=order_request.qty,
            order_type="bracket" if order_request.side == "buy" else order_request.order_type,
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=structured["reasoning"],
        )
        self.db.mark_signal_executed(signal_id)

        logger.info(
            f"{symbol}: Order {order_result.status} — "
            f"{order_request.side} {order_request.qty} shares"
        )
        self._notify_order(
            symbol=symbol,
            side=order_request.side,
            qty=float(order_request.qty or 0),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=structured["reasoning"],
            source=structured.get("source", "ai"),
        )

        return {
            "symbol": symbol,
            "action": structured["action"],
            "traded": True,
            "side": order_request.side,
            "qty": order_request.qty,
            "order_id": order_result.order_id,
            "status": order_result.status,
        }

    def _manage_existing_position(
        self,
        position: Position,
        account: Account,
        positions: List[Position],
    ) -> Dict:
        if self._is_overlay_symbol(position.symbol):
            return {
                "symbol": position.symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": "Overlay position managed separately",
            }

        if self.config.get("minervini_live_exit_enabled", True):
            state = self._build_minervini_position_state(position)
            if state is None:
                return {
                    "symbol": position.symbol,
                    "action": "HOLD",
                    "traded": False,
                    "screen_rejected": "Missing exit-history context for Minervini management",
                }

            management_state = self._load_position_management_state(position.symbol, state)

            exit_result = self._trade_minervini_position_exit(position, state=state)
            if exit_result is not None:
                if exit_result.get("traded"):
                    self.db.delete_position_management_state(position.symbol)
                return exit_result

            earnings_result = self._trade_minervini_earnings_risk(
                position=position,
                state=state,
                management_state=management_state,
            )
            if earnings_result is not None:
                if earnings_result.get("fully_closed"):
                    self.db.delete_position_management_state(position.symbol)
                    return earnings_result
                if earnings_result.get("traded"):
                    return earnings_result
                self._persist_position_management_state(position.symbol, management_state)
                self._sync_minervini_protective_stop(position, state)
                return earnings_result

            deferred_reason = None

            partial_result = self._trade_minervini_partial_profit(
                position=position,
                state=state,
                management_state=management_state,
            )
            if partial_result is not None:
                if partial_result.get("traded"):
                    return partial_result
                deferred_reason = partial_result.get("screen_rejected") or deferred_reason

            add_on_result = self._trade_minervini_add_on(
                position=position,
                state=state,
                management_state=management_state,
                account=account,
                positions=positions,
            )
            if add_on_result is not None:
                if add_on_result.get("traded"):
                    return add_on_result
                deferred_reason = add_on_result.get("screen_rejected") or deferred_reason

            self._persist_position_management_state(position.symbol, management_state)
            self._sync_minervini_protective_stop(position, state)
            return {
                "symbol": position.symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": deferred_reason
                or "Holding above Minervini profit-protection levels",
            }

        return self._analyze_and_trade(position.symbol, account, positions)

    def _trade_minervini_position_exit(
        self,
        position: Position,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        state = state or self._build_minervini_position_state(position)
        if state is None:
            return None

        exit_signal = self._evaluate_minervini_position_exit(position, state)
        if exit_signal is None:
            return None

        symbol = position.symbol
        signal_id = self.db.log_signal(
            symbol=symbol,
            action="SELL",
            confidence=0.85,
            reasoning=exit_signal["reason"],
            timeframe="swing",
            full_analysis=json.dumps(exit_signal, default=str),
        )

        try:
            canceled = self._cancel_open_orders(symbol, side="sell")
            if canceled:
                logger.info("%s: canceled %s existing sell order(s) before exit", symbol, len(canceled))
            order_result = self.broker.close_position(symbol)
        except Exception as exc:
            self.db.mark_signal_rejected(signal_id, str(exc))
            logger.error("Rule-based exit failed for %s: %s", symbol, exc, exc_info=True)
            return {
                "symbol": symbol,
                "action": "SELL",
                "traded": False,
                "risk_rejected": str(exc),
            }

        self.db.log_trade(
            symbol=symbol,
            side="sell",
            qty=float(position.qty),
            order_type="market",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=exit_signal["reason"],
        )
        self.db.mark_signal_executed(signal_id)
        self._notify_order(
            symbol=symbol,
            side="sell",
            qty=float(position.qty),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=exit_signal["reason"],
            source="minervini_exit",
        )
        return {
            "symbol": symbol,
            "action": "SELL",
            "traded": True,
            "side": "sell",
            "qty": float(position.qty),
            "order_id": order_result.order_id,
            "status": order_result.status,
            "rule_exit": exit_signal["trigger"],
        }

    def _trade_minervini_earnings_risk(
        self,
        *,
        position: Position,
        state: Dict[str, Any],
        management_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.get("minervini_earnings_management_enabled", True):
            return None

        earnings_context = self._get_position_earnings_context(position.symbol)
        if not self._earnings_window_active(earnings_context):
            return None

        symbol = position.symbol
        event_date = str(earnings_context.get("earnings_event_date") or "")
        prior_event_date = str(management_state.get("earnings_event_date") or "")
        prior_action = str(management_state.get("earnings_action") or "")
        if event_date and prior_event_date == event_date and prior_action:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    f"Earnings plan already applied for {event_date} ({prior_action})"
                ),
                "earnings_managed": True,
            }

        current_gain_pct = float(state.get("current_gain_pct") or 0.0)
        flat_below_gain_pct = float(
            self.config.get("minervini_earnings_flat_below_gain_pct", 0.05)
        )
        trim_below_gain_pct = float(
            self.config.get("minervini_earnings_trim_below_gain_pct", 0.10)
        )
        trim_fraction = float(
            self.config.get("minervini_earnings_trim_fraction", 0.50)
        )
        core_trim_fraction = float(
            self.config.get("minervini_earnings_core_trim_fraction", 0.33)
        )

        total_qty = int(float(position.qty or 0))
        days_away = self._to_float(earnings_context.get("earnings_days_away"))
        days_label = (
            f"{days_away:.1f} day(s)"
            if days_away is not None
            else "the next few days"
        )

        if current_gain_pct < flat_below_gain_pct or total_qty <= 1:
            reason = (
                f"Minervini earnings risk exit: {symbol} reports in {days_label} and only has "
                f"a {current_gain_pct:.1%} profit cushion; exiting instead of carrying earnings risk."
            )
            signal_id = self.db.log_signal(
                symbol=symbol,
                action="SELL",
                confidence=0.82,
                reasoning=reason,
                timeframe="swing",
                full_analysis=json.dumps(
                    {
                        "symbol": symbol,
                        "trigger": "earnings_flat",
                        "earnings_context": earnings_context,
                        "current_gain_pct": current_gain_pct,
                    },
                    default=str,
                ),
            )
            try:
                canceled = self._cancel_open_orders(symbol, side="sell")
                if canceled:
                    logger.info(
                        "%s: canceled %s existing sell order(s) before earnings exit",
                        symbol,
                        len(canceled),
                    )
                order_result = self.broker.close_position(symbol)
            except Exception as exc:
                self.db.mark_signal_rejected(signal_id, str(exc))
                logger.error(
                    "Earnings-risk exit failed for %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )
                return {
                    "symbol": symbol,
                    "action": "SELL",
                    "traded": False,
                    "risk_rejected": str(exc),
                }

            self.db.log_trade(
                symbol=symbol,
                side="sell",
                qty=float(position.qty),
                order_type="market",
                status=order_result.status,
                filled_qty=order_result.filled_qty,
                filled_price=order_result.filled_avg_price,
                order_id=order_result.order_id,
                signal_id=signal_id,
                reasoning=reason,
            )
            self.db.mark_signal_executed(signal_id)
            self._notify_order(
                symbol=symbol,
                side="sell",
                qty=float(position.qty),
                status=str(order_result.status),
                order_id=order_result.order_id,
                filled_price=order_result.filled_avg_price,
                reasoning=reason,
                source="minervini_earnings_flat",
            )
            return {
                "symbol": symbol,
                "action": "SELL",
                "traded": True,
                "side": "sell",
                "qty": float(position.qty),
                "order_id": order_result.order_id,
                "status": order_result.status,
                "rule_exit": "earnings_flat",
                "fully_closed": True,
            }

        trim_action = "earnings_trim"
        sell_fraction = trim_fraction
        if current_gain_pct >= trim_below_gain_pct:
            trim_action = "earnings_hold_core"
            sell_fraction = core_trim_fraction

        sell_qty = min(total_qty - 1, max(1, int(total_qty * sell_fraction)))
        if sell_qty <= 0:
            management_state["earnings_event_date"] = event_date or None
            management_state["earnings_action"] = "hold_core"
            self._persist_position_management_state(symbol, management_state)
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    f"Holding reduced core into earnings for {event_date or 'upcoming report'}"
                ),
                "earnings_managed": True,
            }

        if trim_action == "earnings_hold_core":
            reason = (
                f"Minervini earnings trim: {symbol} reports in {days_label} and already has "
                f"a {current_gain_pct:.1%} cushion; trimming {sell_qty} shares to carry only a "
                "reduced core position into the event."
            )
        else:
            reason = (
                f"Minervini earnings trim: {symbol} reports in {days_label} and only has "
                f"a {current_gain_pct:.1%} cushion; cutting {sell_qty} shares ahead of the report."
            )

        open_sell_orders = self._get_open_orders(symbol, side="sell")
        if open_sell_orders:
            canceled = self._cancel_open_orders(symbol, side="sell")
            wait_reason = (
                f"Refreshing {len(canceled)} protective sell order(s) before earnings trim"
                if canceled
                else "Waiting for existing protective sell orders to clear before earnings trim"
            )
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": wait_reason,
                "earnings_managed": True,
            }

        signal_id = self.db.log_signal(
            symbol=symbol,
            action="SELL",
            confidence=0.8,
            reasoning=reason,
            timeframe="swing",
            full_analysis=json.dumps(
                {
                    "symbol": symbol,
                    "trigger": trim_action,
                    "earnings_context": earnings_context,
                    "current_gain_pct": current_gain_pct,
                    "sell_qty": sell_qty,
                },
                default=str,
            ),
        )

        try:
            order_result = self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side="sell",
                    qty=float(sell_qty),
                    order_type="market",
                )
            )
        except Exception as exc:
            self.db.mark_signal_rejected(signal_id, str(exc))
            logger.error(
                "Earnings-risk trim failed for %s: %s",
                symbol,
                exc,
                exc_info=True,
            )
            return {
                "symbol": symbol,
                "action": "SELL",
                "traded": False,
                "risk_rejected": str(exc),
            }

        self.db.log_trade(
            symbol=symbol,
            side="sell",
            qty=float(sell_qty),
            order_type="market",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=reason,
        )
        self.db.mark_signal_executed(signal_id)
        management_state["earnings_event_date"] = event_date or None
        management_state["earnings_action"] = (
            "hold_core" if trim_action == "earnings_hold_core" else "trim"
        )
        self._persist_position_management_state(symbol, management_state)
        self._notify_order(
            symbol=symbol,
            side="sell",
            qty=float(sell_qty),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=reason,
            source=trim_action,
        )
        return {
            "symbol": symbol,
            "action": "SELL",
            "traded": True,
            "side": "sell",
            "qty": float(sell_qty),
            "order_id": order_result.order_id,
            "status": order_result.status,
            "rule_exit": trim_action,
            "earnings_managed": True,
        }

    def _trade_minervini_partial_profit(
        self,
        *,
        position: Position,
        state: Dict[str, Any],
        management_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.get("minervini_partial_profit_enabled", True):
            return None
        if management_state.get("partial_profit_taken"):
            return None

        trigger_pct = float(self.config.get("minervini_partial_profit_trigger_pct", 0.12))
        fraction = float(self.config.get("minervini_partial_profit_fraction", 0.33))
        current_gain_pct = float(state.get("current_gain_pct") or 0.0)
        if current_gain_pct < trigger_pct:
            return None

        total_qty = int(float(position.qty or 0))
        if total_qty <= 1:
            return None
        sell_qty = min(total_qty - 1, max(1, int(total_qty * fraction)))
        if sell_qty <= 0:
            return None

        symbol = position.symbol
        open_sell_orders = self._get_open_orders(symbol, side="sell")
        if open_sell_orders:
            canceled = self._cancel_open_orders(symbol, side="sell")
            reason = (
                f"Refreshing {len(canceled)} protective sell order(s) before partial profit"
                if canceled
                else "Waiting for existing protective sell orders to clear before partial profit"
            )
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": reason,
            }

        reason = (
            f"Minervini partial profit: {symbol} is up {current_gain_pct:.1%}; "
            f"locking in {sell_qty} shares while keeping the core position."
        )
        signal_id = self.db.log_signal(
            symbol=symbol,
            action="SELL",
            confidence=0.8,
            reasoning=reason,
            timeframe="swing",
            full_analysis=json.dumps(
                {
                    "symbol": symbol,
                    "trigger": "partial_profit",
                    "current_gain_pct": current_gain_pct,
                    "sell_qty": sell_qty,
                },
                default=str,
            ),
        )

        try:
            order_result = self.broker.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side="sell",
                    qty=float(sell_qty),
                    order_type="market",
                )
            )
        except Exception as exc:
            self.db.mark_signal_rejected(signal_id, str(exc))
            logger.error("Partial-profit sell failed for %s: %s", symbol, exc, exc_info=True)
            return {
                "symbol": symbol,
                "action": "SELL",
                "traded": False,
                "risk_rejected": str(exc),
            }

        self.db.log_trade(
            symbol=symbol,
            side="sell",
            qty=float(sell_qty),
            order_type="market",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=reason,
        )
        self.db.mark_signal_executed(signal_id)
        management_state["partial_profit_taken"] = True
        self._persist_position_management_state(symbol, management_state)
        self._notify_order(
            symbol=symbol,
            side="sell",
            qty=float(sell_qty),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=reason,
            source="minervini_partial_profit",
        )
        return {
            "symbol": symbol,
            "action": "SELL",
            "traded": True,
            "side": "sell",
            "qty": float(sell_qty),
            "order_id": order_result.order_id,
            "status": order_result.status,
            "rule_exit": "partial_profit",
        }

    def _trade_minervini_add_on(
        self,
        *,
        position: Position,
        state: Dict[str, Any],
        management_state: Dict[str, Any],
        account: Account,
        positions: List[Position],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.get("minervini_add_on_enabled", True):
            return None
        if management_state.get("partial_profit_taken"):
            return {
                "symbol": position.symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": "Add-on disabled after partial-profit lock",
            }
        market_context = self._get_market_context()
        if self._market_extended_for_add_on(market_context):
            return {
                "symbol": position.symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    "Add-on blocked because QQQ is stretched above the 21EMA or running too fast"
                ),
            }

        symbol = position.symbol
        setup = self._get_latest_setup_for_symbol(symbol)
        if not self._setup_supports_pyramiding(setup, state.get("current_price")):
            return None

        current_price = self._to_float(state.get("current_price"))
        entry_price = self._to_float(state.get("entry_price"))
        if current_price is None or entry_price is None or current_price <= entry_price:
            return None

        add_on_level = None
        add_fraction = None
        first_trigger = float(self.config.get("minervini_add_on_trigger_pct_1", 0.025))
        second_trigger = float(self.config.get("minervini_add_on_trigger_pct_2", 0.05))
        if (
            not management_state.get("add_on_1_done")
            and current_price >= entry_price * (1.0 + first_trigger)
        ):
            add_on_level = 1
            add_fraction = float(self.config.get("minervini_add_on_fraction_1", 0.30))
        elif (
            not management_state.get("add_on_2_done")
            and current_price >= entry_price * (1.0 + second_trigger)
        ):
            add_on_level = 2
            add_fraction = float(self.config.get("minervini_add_on_fraction_2", 0.20))

        if add_on_level is None or add_fraction is None:
            return None

        market_regime = setup.get("market_regime") or (
            self._latest_minervini_preflight.market_regime
            if self._latest_minervini_preflight is not None
            else "unknown"
        )
        if not self._entries_allowed_for_setup(setup, market_regime):
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": f"Add-on blocked in market regime {market_regime}",
            }

        current_exposure = self._current_exposure(account, positions)
        target_exposure = self._target_exposure_for_setup(setup, market_regime)
        if current_exposure >= target_exposure:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    f"Exposure {current_exposure:.2%} already at add-on target {target_exposure:.2%}"
                ),
            }

        existing_open_buy_order = self._find_existing_open_order(symbol, side="buy")
        if existing_open_buy_order is not None:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    f"Existing open buy order {existing_open_buy_order.order_id} "
                    f"[{existing_open_buy_order.status}]"
                ),
            }

        open_sell_orders = self._get_open_orders(symbol, side="sell")
        order_type_names = {
            self._order_type_name(getattr(order, "order_type", "")) for order in open_sell_orders
        }
        if "limit" in order_type_names:
            canceled = self._cancel_open_orders(symbol, side="sell")
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": (
                    f"Refreshing {len(canceled)} legacy sell order(s) before add-on"
                    if canceled
                    else "Waiting for legacy sell orders to clear before add-on"
                ),
            }

        stop_price = max(
            self._to_float(setup.get("initial_stop_price")) or 0.0,
            self._to_float(state.get("protective_stop_price")) or 0.0,
        )
        if stop_price <= 0 or stop_price >= current_price:
            return None

        qty = self._calculate_add_on_qty(
            account=account,
            positions=positions,
            position=position,
            price=current_price,
            stop_price=stop_price,
            add_fraction=add_fraction,
        )
        if qty <= 0:
            return None

        reasoning = (
            f"Minervini add-on #{add_on_level}: {symbol} remains in a strong continuation "
            f"setup at {current_price:.2f}; adding {qty} shares with protection at {stop_price:.2f}."
        )
        signal_id = self.db.log_signal(
            symbol=symbol,
            action="BUY",
            confidence=0.82,
            reasoning=reasoning,
            stop_loss=stop_price,
            timeframe="swing",
            full_analysis=json.dumps(setup, default=str),
        )

        order_request = OrderRequest(symbol=symbol, side="buy", qty=float(qty))
        risk_result = self.risk_engine.check_order(order_request, account, positions, current_price)
        if not risk_result.passed:
            self.db.mark_signal_rejected(signal_id, risk_result.reason)
            return {
                "symbol": symbol,
                "action": "BUY",
                "traded": False,
                "risk_rejected": risk_result.reason,
            }

        try:
            order_result = self.broker.submit_bracket_order(
                order_request,
                stop_loss_price=round(stop_price, 2),
                take_profit_price=None,
            )
        except Exception as exc:
            self.db.mark_signal_rejected(signal_id, str(exc))
            logger.error("Add-on buy failed for %s: %s", symbol, exc, exc_info=True)
            return {
                "symbol": symbol,
                "action": "BUY",
                "traded": False,
                "risk_rejected": str(exc),
            }

        self.risk_engine.record_trade()
        self.db.log_trade(
            symbol=symbol,
            side="buy",
            qty=float(qty),
            order_type="oto",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=reasoning,
        )
        self.db.mark_signal_executed(signal_id)
        management_state[f"add_on_{add_on_level}_done"] = True
        self._persist_position_management_state(symbol, management_state)
        self._notify_order(
            symbol=symbol,
            side="buy",
            qty=float(qty),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=reasoning,
            source=f"minervini_add_on_{add_on_level}",
        )
        return {
            "symbol": symbol,
            "action": "BUY",
            "traded": True,
            "side": "buy",
            "qty": float(qty),
            "order_id": order_result.order_id,
            "status": order_result.status,
            "rule_entry": f"add_on_{add_on_level}",
        }

    def _evaluate_minervini_position_exit(
        self,
        position: Position,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        entry_price = state["entry_price"]
        current_price = state["current_price"]
        max_gain_pct = state["max_gain_pct"]
        current_gain_pct = state["current_gain_pct"]
        ema21 = state.get("ema21")

        breakeven_trigger = float(self.config.get("minervini_breakeven_trigger_pct", 0.10))
        trailing_lock_trigger_1 = float(
            self.config.get("minervini_trailing_lock_trigger_pct_1", 0.12)
        )
        trailing_lock_floor_1 = float(
            self.config.get("minervini_trailing_lock_floor_pct_1", 0.03)
        )
        trailing_lock_trigger_2 = float(
            self.config.get("minervini_trailing_lock_trigger_pct_2", 0.20)
        )
        trailing_lock_floor_2 = float(
            self.config.get("minervini_trailing_lock_floor_pct_2", 0.08)
        )
        ema21_floor = float(self.config.get("minervini_ema21_profit_floor_pct", 0.10))
        ema21_break_buffer = float(self.config.get("minervini_ema21_break_buffer_pct", 0.0025))
        protective_stop_price = self._to_float(state.get("protective_stop_price"))

        if max_gain_pct >= breakeven_trigger and current_price <= state["breakeven_floor_price"]:
            return {
                "symbol": position.symbol,
                "trigger": "breakeven_protect",
                "reason": (
                    f"Minervini profit protection: {position.symbol} reached "
                    f"{max_gain_pct:.1%} max gain and round-tripped to {current_gain_pct:.1%}; "
                    "exit before a winner turns into a loser."
                ),
                "entry_price": entry_price,
                "current_price": current_price,
                "max_gain_pct": max_gain_pct,
                "current_gain_pct": current_gain_pct,
            }

        trailing_floor_price = None
        trailing_trigger = None
        if max_gain_pct >= trailing_lock_trigger_2:
            trailing_floor_price = entry_price * (1.0 + trailing_lock_floor_2)
            trailing_trigger = trailing_lock_trigger_2
        elif max_gain_pct >= trailing_lock_trigger_1:
            trailing_floor_price = entry_price * (1.0 + trailing_lock_floor_1)
            trailing_trigger = trailing_lock_trigger_1

        if (
            trailing_floor_price is not None
            and protective_stop_price is not None
            and current_price <= protective_stop_price
        ):
            return {
                "symbol": position.symbol,
                "trigger": "trailing_profit_stop",
                "reason": (
                    f"Minervini trailing stop: {position.symbol} reached "
                    f"{max_gain_pct:.1%} max gain, activated the "
                    f"{trailing_trigger:.0%} profit-lock ladder, and fell back to "
                    f"the protective stop ({current_price:.2f} <= {protective_stop_price:.2f})."
                ),
                "entry_price": entry_price,
                "current_price": current_price,
                "protective_stop_price": protective_stop_price,
                "max_gain_pct": max_gain_pct,
                "current_gain_pct": current_gain_pct,
            }

        if (
            ema21 is not None
            and max_gain_pct >= ema21_floor
            and current_price < ema21 * (1.0 - ema21_break_buffer)
        ):
            return {
                "symbol": position.symbol,
                "trigger": "lost_21ema",
                "reason": (
                    f"Minervini profit protection: {position.symbol} reached "
                    f"{max_gain_pct:.1%} max gain and is now below the 21EMA "
                    f"({current_price:.2f} < {ema21:.2f})."
                ),
                "entry_price": entry_price,
                "current_price": current_price,
                "ema21": float(ema21),
                "max_gain_pct": max_gain_pct,
                "current_gain_pct": current_gain_pct,
            }

        return None

    def _build_minervini_position_state(self, position: Position) -> Optional[Dict[str, Any]]:
        entry_price = float(position.avg_entry_price or 0.0)
        current_price = float(position.current_price or 0.0)
        if entry_price <= 0 or current_price <= 0:
            return None

        history = self._load_position_exit_history(position.symbol)
        if history.empty:
            return None

        entry_trade_date = self._get_position_entry_date(position.symbol)
        if entry_trade_date is None:
            entry_trade_date = history.index[-1].date() - timedelta(days=30)

        bars_since_entry = history[history.index.date >= entry_trade_date]
        if bars_since_entry.empty:
            bars_since_entry = history.tail(40)

        ema21 = history["close"].ewm(span=21, adjust=False).mean().iloc[-1] if len(history) >= 21 else None
        current_gain_pct = (current_price / entry_price) - 1.0
        max_price_since_entry = max(float(bars_since_entry["high"].max()), current_price)
        max_gain_pct = (max_price_since_entry / entry_price) - 1.0

        breakeven_buffer = float(self.config.get("minervini_breakeven_buffer_pct", 0.003))
        breakeven_trigger = float(self.config.get("minervini_breakeven_trigger_pct", 0.08))
        trailing_lock_trigger_1 = float(
            self.config.get("minervini_trailing_lock_trigger_pct_1", 0.12)
        )
        trailing_lock_floor_1 = float(
            self.config.get("minervini_trailing_lock_floor_pct_1", 0.03)
        )
        trailing_lock_trigger_2 = float(
            self.config.get("minervini_trailing_lock_trigger_pct_2", 0.20)
        )
        trailing_lock_floor_2 = float(
            self.config.get("minervini_trailing_lock_floor_pct_2", 0.08)
        )
        ema21_floor = float(self.config.get("minervini_ema21_profit_floor_pct", 0.10))
        ema21_break_buffer = float(self.config.get("minervini_ema21_break_buffer_pct", 0.0025))
        base_stop_pct = min(
            float(self.config.get("default_stop_loss_pct", 0.08)),
            float(self.config.get("leader_continuation_stop_loss_pct", 0.06)),
        )
        protective_stop_price = entry_price * (1.0 - base_stop_pct)
        breakeven_floor_price = entry_price * (1.0 + breakeven_buffer)
        trailing_floor_price = None
        if max_gain_pct >= trailing_lock_trigger_2:
            trailing_floor_price = entry_price * (1.0 + trailing_lock_floor_2)
        elif max_gain_pct >= trailing_lock_trigger_1:
            trailing_floor_price = entry_price * (1.0 + trailing_lock_floor_1)
        if max_gain_pct >= breakeven_trigger:
            protective_stop_price = max(protective_stop_price, breakeven_floor_price)
        if trailing_floor_price is not None:
            protective_stop_price = max(protective_stop_price, trailing_floor_price)
        if ema21 is not None and max_gain_pct >= ema21_floor:
            protective_stop_price = max(
                protective_stop_price,
                ema21 * (1.0 - ema21_break_buffer),
            )
        protective_stop_price = min(protective_stop_price, current_price * 0.995)

        return {
            "symbol": position.symbol,
            "entry_price": entry_price,
            "current_price": current_price,
            "ema21": float(ema21) if ema21 is not None else None,
            "max_gain_pct": max_gain_pct,
            "current_gain_pct": current_gain_pct,
            "breakeven_floor_price": breakeven_floor_price,
            "trailing_floor_price": trailing_floor_price,
            "protective_stop_price": round(protective_stop_price, 2),
            "entry_trade_date": entry_trade_date.isoformat(),
        }

    def _sync_minervini_protective_stop(
        self,
        position: Position,
        state: Dict[str, Any],
    ) -> None:
        if not self.config.get("minervini_use_stop_only_entries", True):
            return

        symbol = position.symbol
        desired_stop = self._to_float(state.get("protective_stop_price"))
        current_price = self._to_float(state.get("current_price"))
        if desired_stop is None or current_price is None or desired_stop >= current_price:
            return

        open_sell_orders = self._get_open_orders(symbol, side="sell")
        order_type_names = {
            self._order_type_name(getattr(order, "order_type", "")) for order in open_sell_orders
        }
        has_limit_child = "limit" in order_type_names
        stop_children = [
            order
            for order in open_sell_orders
            if self._order_type_name(getattr(order, "order_type", "")) in {"stop", "stop_limit", "trailing_stop"}
        ]
        has_stop_child = bool(stop_children)

        if has_stop_child and not has_limit_child:
            exact_stop = any(
                abs((self._to_float(getattr(order, "stop_price", None)) or -1.0) - desired_stop) < 0.01
                and abs((self._to_float(getattr(order, "qty", None)) or 0.0) - float(position.qty)) < 0.01
                for order in stop_children
            )
            if exact_stop:
                return

        if open_sell_orders:
            canceled = self._cancel_open_orders(symbol, side="sell")
            if canceled:
                logger.info(
                    "%s: refreshing %s existing sell order(s) for updated stop-only protection",
                    symbol,
                    len(canceled),
                )

        stop_order = OrderRequest(
            symbol=symbol,
            side="sell",
            qty=float(position.qty),
            order_type="stop",
            stop_price=round(desired_stop, 2),
            time_in_force="gtc",
        )
        try:
            result = self.broker.submit_order(stop_order)
            logger.info(
                "%s: protective stop synced at %.2f -> %s",
                symbol,
                desired_stop,
                result.status,
            )
        except Exception as exc:
            logger.warning(
                "%s: could not sync protective stop %.2f: %s",
                symbol,
                desired_stop,
                exc,
            )

    def _get_position_entry_date(self, symbol: str) -> Optional[date]:
        inferred = self._infer_position_management_from_trades(symbol)
        return inferred.get("cycle_entry_date")

    def _infer_position_management_from_trades(self, symbol: str) -> Dict[str, Any]:
        trades = self.db.get_trades_for_symbol(symbol)
        if not trades:
            return {
                "cycle_entry_date": None,
                "partial_profit_taken": False,
                "add_on_1_done": False,
                "add_on_2_done": False,
            }

        def _parse_trade_date(value: Any) -> Optional[date]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value)).date()
            except ValueError:
                try:
                    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").date()
                except ValueError:
                    return None

        ordered_trades = sorted(
            trades,
            key=lambda trade: str(trade.get("timestamp") or ""),
        )
        running_qty = 0.0
        cycle_entry_date: Optional[date] = None
        cycle_buy_legs = 0
        partial_profit_taken = False

        for trade in ordered_trades:
            qty = float(trade.get("filled_qty") or trade.get("qty") or 0.0)
            if qty <= 0:
                continue
            side = str(trade.get("side") or "").lower()
            trade_date = _parse_trade_date(trade.get("timestamp"))
            if trade_date is None:
                continue

            if side == "buy":
                if running_qty <= 0:
                    cycle_entry_date = trade_date
                    cycle_buy_legs = 0
                    partial_profit_taken = False
                running_qty += qty
                cycle_buy_legs += 1
                continue

            if side == "sell":
                if running_qty > 0:
                    partial_profit_taken = True
                running_qty = max(0.0, running_qty - qty)
                if running_qty <= 0:
                    cycle_entry_date = None
                    cycle_buy_legs = 0
                    partial_profit_taken = False

        return {
            "cycle_entry_date": cycle_entry_date,
            "partial_profit_taken": partial_profit_taken if running_qty > 0 else False,
            "add_on_1_done": cycle_buy_legs >= 2 if running_qty > 0 else False,
            "add_on_2_done": cycle_buy_legs >= 3 if running_qty > 0 else False,
        }

    def _load_position_exit_history(self, symbol: str) -> pd.DataFrame:
        end_date = date.today().isoformat()
        history_days = max(int(self.config.get("minervini_exit_history_days", 180)), 60)
        start_date = (date.today() - timedelta(days=history_days)).isoformat()
        db_path = self.config.get("minervini_db_path", "research_data/market_data.duckdb")

        def _read_frame(read_only: bool) -> pd.DataFrame:
            warehouse = MarketDataWarehouse(db_path, read_only=read_only)
            try:
                return warehouse.get_daily_bars(symbol, start_date, end_date)
            finally:
                warehouse.close()

        try:
            frame = _read_frame(read_only=True)
        except Exception:
            frame = pd.DataFrame()

        stale = frame.empty
        if not frame.empty:
            freshest = frame.index[-1].date()
            stale = (date.today() - freshest).days > 5

        if stale:
            try:
                warehouse = MarketDataWarehouse(db_path)
                try:
                    warehouse.fetch_and_store_daily_bars([symbol], start_date, end_date)
                    frame = warehouse.get_daily_bars(symbol, start_date, end_date)
                finally:
                    warehouse.close()
            except Exception as exc:
                logger.warning("Could not refresh exit history for %s: %s", symbol, exc)

        return frame

    def _trade_rule_based_setup(
        self,
        setup: Dict,
        account: Account,
        positions: List[Position],
    ) -> Dict:
        symbol = setup.get("symbol", "")
        if not symbol:
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": "Missing setup payload",
            }

        current_position = self._find_position(positions, symbol)
        if current_position and current_position.qty > 0:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "traded": False,
                "screen_rejected": "Already holding position",
            }

        current_price = self.broker.get_latest_price(symbol)
        buy_point = self._to_float(setup.get("buy_point"))
        buy_limit_price = self._to_float(setup.get("buy_limit_price"))
        initial_stop_price = self._to_float(setup.get("initial_stop_price"))
        initial_stop_pct = self._to_float(setup.get("initial_stop_pct"))
        stage_number = self._to_float(setup.get("stage_number"))
        close_range_pct = self._to_float(setup.get("close_range_pct"))
        max_stage_number = float(self.config.get("minervini_max_stage_number", 3))
        market_regime = setup.get("market_regime") or (
            self._latest_minervini_preflight.market_regime
            if self._latest_minervini_preflight is not None
            else "unknown"
        )
        continuation_setup = self._is_leader_continuation_setup(setup)
        if not self._entries_allowed_for_setup(setup, market_regime):
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": f"Market regime is {market_regime}; new swing entries disabled",
                "market_regime": market_regime,
            }

        current_exposure = self._current_exposure(account, positions)
        target_exposure = self._target_exposure_for_setup(setup, market_regime)
        if current_exposure >= target_exposure:
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": (
                    f"Current exposure {current_exposure:.2%} already at regime target {target_exposure:.2%}"
                ),
                "market_regime": market_regime,
            }

        if buy_point is None or buy_limit_price is None:
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": "Setup has no buy point",
            }

        if (
            not continuation_setup
            and stage_number is not None
            and stage_number > max_stage_number
        ):
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": f"Late-stage base ({stage_number:.0f})",
                "candidate_status": setup.get("candidate_status"),
            }

        min_close_range_pct = float(
            self.config.get(
                "leader_continuation_min_close_range_pct",
                0.15,
            )
            if continuation_setup
            else self.config.get("minervini_min_close_range_pct", 0.55)
        )

        if (
            self.config.get("minervini_use_close_range_filter", True)
            and close_range_pct is not None
            and close_range_pct < min_close_range_pct
        ):
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": (
                    f"Close-range quality too weak ({close_range_pct:.2f} < "
                    f"{min_close_range_pct:.2f})"
                ),
                "candidate_status": setup.get("candidate_status"),
            }

        if current_price < buy_point:
            return {
                "symbol": symbol,
                "action": "WAIT",
                "traded": False,
                "screen_rejected": f"Waiting for breakout above {buy_point:.2f}",
                "current_price": round(current_price, 2),
                "buy_point": round(buy_point, 2),
                "buy_limit_price": round(buy_limit_price, 2),
                "candidate_status": setup.get("candidate_status"),
            }

        if current_price > buy_limit_price:
            return {
                "symbol": symbol,
                "action": "SKIP",
                "traded": False,
                "screen_rejected": f"Extended above buy zone ({buy_limit_price:.2f})",
                "current_price": round(current_price, 2),
                "buy_point": round(buy_point, 2),
                "buy_limit_price": round(buy_limit_price, 2),
                "candidate_status": "extended",
            }

        stop_pct = initial_stop_pct
        if stop_pct is None and initial_stop_price is not None and current_price > initial_stop_price:
            stop_pct = (current_price - initial_stop_price) / current_price
        if continuation_setup:
            continuation_stop_pct = float(
                self.config.get("leader_continuation_stop_loss_pct", 0.06)
            )
            stop_pct = min(
                max(stop_pct or continuation_stop_pct, 0.04),
                continuation_stop_pct,
            )
        else:
            stop_pct = min(
                max(stop_pct or self.config.get("default_stop_loss_pct", 0.05), 0.03),
                0.08,
            )
        take_profit_pct = max(
            float(self.config.get("default_take_profit_pct", 0.15)),
            round(stop_pct * 3, 4),
        )
        confidence = min(
            0.55
            + (self._to_float(setup.get("template_score")) or 0.0) / 25.0
            + (self._to_float(setup.get("rs_percentile")) or 0.0) / 250.0,
            0.95,
        )
        reasoning = (
            f"{'Leader continuation' if continuation_setup else 'Minervini rule entry'}: "
            f"base={setup.get('base_label')} "
            f"stage={int(stage_number) if stage_number is not None else 'n/a'} "
            f"status={setup.get('candidate_status')} "
            f"buy_point={buy_point:.2f} live={current_price:.2f} "
            f"buy_limit={buy_limit_price:.2f} rs={self._to_float(setup.get('rs_percentile')) or 0.0:.1f} "
            f"regime={market_regime} exposure={current_exposure:.2%}/{target_exposure:.2%}"
        )
        structured = {
            "symbol": symbol,
            "action": "BUY",
            "confidence": confidence,
            "reasoning": reasoning,
            "stop_loss_pct": round(stop_pct, 4),
            "take_profit_pct": round(take_profit_pct, 4),
            "stop_loss": initial_stop_price,
            "timeframe": "swing",
            "source": "minervini_rule",
        }
        return self._execute_structured_signal(
            symbol=symbol,
            structured=structured,
            account=account,
            positions=positions,
            full_analysis=json.dumps(setup, default=str),
        )

    # ── Reflection ───────────────────────────────────────────────────

    def take_market_snapshot(self) -> Dict:
        """Capture the current account and portfolio state."""
        logger.info("Capturing scheduled market snapshot...")
        preflight = None
        try:
            preflight = self._run_minervini_preflight()
        except Exception as e:
            logger.error("Minervini preflight failed during snapshot: %s", e, exc_info=True)
        if not self.execution_enabled or self.tracker is None:
            self._notify_morning_scan(preflight, None)
            return {
                "alerts_only": True,
                "trade_date": date.today().isoformat(),
                "approved_symbols": preflight.approved_symbols if preflight is not None else [],
                "market_regime": preflight.market_regime if preflight is not None else None,
            }

        snapshot = self.tracker.take_daily_snapshot()
        if self.config.get("minervini_live_exit_enabled", True):
            try:
                for position in self._stock_positions(self.broker.get_positions()):
                    state = self._build_minervini_position_state(position)
                    if state is not None:
                        self._sync_minervini_protective_stop(position, state)
            except Exception as exc:
                logger.warning("Protective-stop sync during snapshot failed: %s", exc, exc_info=True)
        self._notify_morning_scan(preflight, snapshot)
        return snapshot

    def run_daily_reflection(self) -> Dict:
        """After market close: reflect on today's trades and update memories."""
        if not self.execution_enabled or self.tracker is None:
            logger.info("Alerts-only mode: skipping daily reflection.")
            return {
                "alerts_only": True,
                "reflected": 0,
                "skipped": 0,
                "report_path": None,
                "fallback_used": False,
            }

        logger.info("Running daily reflection...")

        positions = self.broker.get_positions()
        ta = self._get_ai_engine()

        reflected = 0
        skipped = 0
        for pos in positions:
            state = self._latest_analysis_states.get(pos.symbol)
            if state is None:
                logger.warning(
                    "Skipping reflection for %s: no analysis state captured in this process",
                    pos.symbol,
                )
                skipped += 1
                continue
            try:
                ta.reflect_state_and_remember(state, pos.unrealized_pl)
                reflected += 1
            except Exception as e:
                logger.error(f"Reflection error for {pos.symbol}: {e}")

        self._save_persistent_memories()
        try:
            self.tracker.take_daily_snapshot()
        except Exception as exc:
            logger.warning("Daily snapshot failed during reflection: %s", exc, exc_info=True)
        summary_result = self.send_daily_notifications()

        logger.info(
            "Reflection complete. Reflected on %s positions, skipped %s.",
            reflected,
            skipped,
        )
        return {
            "reflected": reflected,
            "skipped": skipped,
            "report_path": summary_result.get("report_path"),
            "fallback_used": summary_result.get("fallback_used", False),
        }

    def generate_daily_report(
        self, save: bool = True, report_date: Optional[str] = None
    ) -> Dict:
        """Build a daily account/trade/P&L report and optionally save it."""
        if not self.execution_enabled or self.tracker is None:
            report_day = report_date or date.today().isoformat()
            report = {
                "date": report_day,
                "paper_mode": self.config.get("paper_trading", True),
                "watchlist": self.watchlist,
                "account": {},
                "trade_summary": self.db.get_trade_summary(report_day),
                "performance": {},
                "position_summary": {
                    "open_positions": 0,
                    "total_unrealized_pl": 0.0,
                },
                "positions": [],
                "screening_batch": self.db.get_screening_batch_on_date(report_day),
                "setups": self.db.get_setup_candidates_on_date(report_day),
                "trades": self.db.get_trades_on_date(report_day),
                "alerts_only": True,
            }
            report["miss_review"] = self._build_miss_review(report)
            return report

        report = self.tracker.build_daily_report(report_date)
        report["paper_mode"] = self.config.get("paper_trading", True)
        report["watchlist"] = self.watchlist
        report["miss_review"] = self._build_miss_review(report)

        if save:
            output_dir = os.path.join(
                self.config.get("results_dir", "./results"),
                "daily_reports",
            )
            report_path = self.tracker.save_daily_report(report, output_dir)
            report["report_path"] = str(report_path)
            logger.info("Daily report saved to %s", report_path)

        return report

    # ── Overlay Management ──────────────────────────────────────────

    def _overlay_enabled(self) -> bool:
        return bool(self.config.get("overlay_enabled", False))

    def _overlay_symbol(self) -> str:
        return str(self.config.get("overlay_symbol", "SMH")).upper()

    def _is_overlay_symbol(self, symbol: Optional[str]) -> bool:
        if not self._overlay_enabled() or not symbol:
            return False
        return str(symbol).upper() == self._overlay_symbol()

    def _stock_positions(self, positions: List[Position]) -> List[Position]:
        return [position for position in positions if not self._is_overlay_symbol(position.symbol)]

    def _overlay_context_allows_entry(self, context: Optional[Dict]) -> bool:
        if not context:
            return False
        trigger = str(self.config.get("overlay_trigger", "confirmed_uptrend")).lower()
        regime = str(context.get("market_regime", "")).lower()
        score = self._to_float(context.get("market_score"))
        if trigger == "confirmed_uptrend":
            return bool(context.get("confirmed_uptrend", False))
        if trigger == "not_correction":
            return regime != "market_correction"
        if trigger == "score_gte_5":
            return score is not None and score >= 5
        if trigger == "score_gte_6":
            return score is not None and score >= 6
        return bool(context.get("confirmed_uptrend", False))

    def _build_market_context_snapshot(self) -> Optional[Dict]:
        today = date.today().isoformat()
        lookback_days = max(int(self.config.get("minervini_lookback_days", 730)), 400)
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = today
        symbols = list(
            dict.fromkeys(
                list(self.config.get("overlay_context_symbols", ["SPY", "QQQ", "IWM", "SMH", "^VIX"]))
                + [self._overlay_symbol()]
            )
        )
        context: Optional[Dict] = None
        db_path = self.config.get("minervini_db_path", "research_data/market_data.duckdb")
        try:
            warehouse = MarketDataWarehouse(db_path)
            try:
                frames = {
                    symbol: warehouse.get_daily_bars(symbol, start_date, end_date)
                    for symbol in symbols
                }
                latest_dates = [
                    frame.index[-1].date()
                    for frame in frames.values()
                    if frame is not None and not frame.empty
                ]
                needs_refresh = len(latest_dates) != len(symbols)
                if not needs_refresh and latest_dates:
                    freshest = max(latest_dates)
                    stalest = min(latest_dates)
                    needs_refresh = (
                        (date.today() - freshest).days > 5
                        or (freshest - stalest).days > 3
                    )
                if needs_refresh:
                    warehouse.fetch_and_store_daily_bars(symbols, start_date, end_date)
                    frames = {
                        symbol: warehouse.get_daily_bars(symbol, start_date, end_date)
                        for symbol in symbols
                    }
            finally:
                warehouse.close()

            context_df = build_market_context(frames)
            if not context_df.empty:
                latest = context_df.iloc[-1]
                score = self._to_float(latest.get("market_score"))
                context = {
                    "trade_date": context_df.index[-1].date().isoformat(),
                    "market_score": int(score) if score is not None else None,
                    "market_regime": str(latest["market_regime"]),
                    "confirmed_uptrend": bool(latest["market_confirmed_uptrend"]),
                    "qqq_extension_pct": self._to_float(latest.get("qqq_extension_pct")),
                    "qqq_roc_5": self._to_float(latest.get("qqq_roc_5")),
                    "source": "market_context",
                }
        except Exception as exc:
            logger.warning("Market context refresh failed: %s", exc)

        if context is None:
            fallback_regime = (
                self._latest_minervini_preflight.market_regime
                if self._latest_minervini_preflight is not None
                else "unknown"
            )
            context = {
                "trade_date": today,
                "market_score": None,
                "market_regime": fallback_regime,
                "confirmed_uptrend": bool(
                    self._latest_minervini_preflight.confirmed_uptrend
                    if self._latest_minervini_preflight is not None
                    else False
                ),
                "qqq_extension_pct": None,
                "qqq_roc_5": None,
                "source": "preflight_fallback",
            }

        context["computed_on"] = today
        return context

    def _get_market_context(self) -> Optional[Dict]:
        today = date.today().isoformat()
        if (
            self._latest_market_context is not None
            and self._latest_market_context.get("computed_on") == today
        ):
            return self._latest_market_context
        context = self._build_market_context_snapshot()
        self._latest_market_context = context
        return context

    def _market_extended_for_add_on(self, context: Optional[Dict]) -> bool:
        if not self.config.get("market_extension_filter_enabled", True):
            return False
        if not context:
            return False
        extension_pct = self._to_float(context.get("qqq_extension_pct"))
        roc_5 = self._to_float(context.get("qqq_roc_5"))
        max_extension = float(self.config.get("market_extension_max_qqq_above_ema21_pct", 0.05))
        max_roc_5 = float(self.config.get("market_extension_max_qqq_roc_5", 0.05))
        extension_hot = extension_pct is not None and extension_pct >= max_extension
        momentum_hot = roc_5 is not None and roc_5 >= max_roc_5
        if extension_pct is not None and roc_5 is not None:
            return extension_hot and momentum_hot
        if extension_pct is not None:
            return extension_hot
        if roc_5 is not None:
            return momentum_hot
        return False

    def _get_overlay_context(self) -> Optional[Dict]:
        if not self._overlay_enabled():
            return None

        today = date.today().isoformat()
        if (
            self._latest_overlay_context is not None
            and self._latest_overlay_context.get("computed_on") == today
        ):
            return self._latest_overlay_context

        context = self._get_market_context() or {
            "trade_date": today,
            "market_score": None,
            "market_regime": "unknown",
            "confirmed_uptrend": False,
            "qqq_extension_pct": None,
            "qqq_roc_5": None,
            "source": "empty",
            "computed_on": today,
        }
        context = dict(context)
        context["symbol"] = self._overlay_symbol()
        context["overlay_allowed"] = self._overlay_context_allows_entry(context)
        self._latest_overlay_context = context
        return context

    def _setup_actionable_now(self, setup: Dict, positions: List[Position]) -> bool:
        symbol = setup.get("symbol", "")
        if not symbol or self._is_overlay_symbol(symbol):
            return False
        if self._find_position(positions, symbol) is not None:
            return False

        buy_point = self._to_float(setup.get("buy_point"))
        buy_limit_price = self._to_float(setup.get("buy_limit_price"))
        stage_number = self._to_float(setup.get("stage_number"))
        close_range_pct = self._to_float(setup.get("close_range_pct"))
        max_stage_number = float(self.config.get("minervini_max_stage_number", 3))
        market_regime = setup.get("market_regime") or (
            self._latest_minervini_preflight.market_regime
            if self._latest_minervini_preflight is not None
            else "unknown"
        )
        if not self._entries_allowed_for_regime(market_regime):
            return False
        if buy_point is None or buy_limit_price is None:
            return False
        if stage_number is not None and stage_number > max_stage_number:
            return False
        if (
            self.config.get("minervini_use_close_range_filter", True)
            and close_range_pct is not None
            and close_range_pct < float(self.config.get("minervini_min_close_range_pct", 0.55))
        ):
            return False
        current_price = self.broker.get_latest_price(symbol)
        return buy_point <= current_price <= buy_limit_price

    def _maybe_free_overlay_for_stock_candidates(
        self,
        account: Account,
        positions: List[Position],
        setup_rows: Dict[str, Dict],
        analysis_universe: List[str],
    ) -> Optional[Dict]:
        if not self._overlay_enabled():
            return None
        overlay_position = self._find_position(positions, self._overlay_symbol())
        if overlay_position is None or overlay_position.qty <= 0:
            return None

        stock_positions = self._stock_positions(positions)
        actionable = [
            symbol
            for symbol in analysis_universe
            if symbol in setup_rows and self._setup_actionable_now(setup_rows[symbol], stock_positions)
        ]
        if not actionable:
            return None

        reason = (
            "Releasing overlay capital for actionable stock setups: "
            f"{', '.join(actionable[:3])}"
        )
        return self._execute_overlay_order(
            symbol=self._overlay_symbol(),
            side="sell",
            qty=int(round(overlay_position.qty)),
            account=account,
            reasoning=reason,
            context=self._get_overlay_context(),
        )

    def _overlay_step_aside_executed_today(self) -> bool:
        if not self._overlay_enabled():
            return False
        overlay_symbol = self._overlay_symbol()
        for trade in self.db.get_today_trades():
            if str(trade.get("symbol") or "").upper() != overlay_symbol:
                continue
            if str(trade.get("side") or "").lower() != "sell":
                continue
            reasoning = str(trade.get("reasoning") or "")
            if "Releasing overlay capital for actionable stock setups" in reasoning:
                return True
        return False

    def _manage_overlay_position(
        self,
        account: Account,
        positions: List[Position],
        context: Optional[Dict],
    ) -> Optional[Dict]:
        if not self._overlay_enabled() or account.equity <= 0:
            return None

        overlay_symbol = self._overlay_symbol()
        overlay_position = self._find_position(positions, overlay_symbol)
        stock_positions = self._stock_positions(positions)
        stock_market_value = sum(position.market_value for position in stock_positions)
        max_total_exposure = max(float(self.config.get("overlay_max_total_exposure", 1.0)), 0.0)
        overlay_fraction = min(max(float(self.config.get("overlay_fraction", 1.0)), 0.0), 1.0)
        max_overlay_notional = max(0.0, (account.equity * max_total_exposure) - stock_market_value)
        desired_overlay_notional = (
            max_overlay_notional * overlay_fraction
            if context is not None and context.get("overlay_allowed")
            else 0.0
        )
        current_overlay_notional = overlay_position.market_value if overlay_position else 0.0
        min_notional = float(self.config.get("overlay_min_notional", 500.0))
        threshold = max(
            min_notional,
            account.equity * float(self.config.get("overlay_rebalance_threshold_pct", 0.03)),
        )

        if overlay_position is not None and desired_overlay_notional < min_notional:
            reason = (
                f"Overlay exit: market_regime={context.get('market_regime') if context else 'unknown'} "
                f"score={context.get('market_score') if context else 'n/a'}"
            )
            return self._execute_overlay_order(
                symbol=overlay_symbol,
                side="sell",
                qty=int(round(overlay_position.qty)),
                account=account,
                reasoning=reason,
                context=context,
            )

        if overlay_position is None and desired_overlay_notional < min_notional:
            return None

        if (
            overlay_position is None
            and desired_overlay_notional >= min_notional
            and self._overlay_step_aside_executed_today()
        ):
            return None

        delta_notional = desired_overlay_notional - current_overlay_notional
        if abs(delta_notional) < threshold:
            return None

        current_price = (
            overlay_position.current_price
            if overlay_position is not None and overlay_position.current_price > 0
            else self.broker.get_latest_price(overlay_symbol)
        )
        if current_price <= 0:
            return None

        if delta_notional > 0:
            budget = min(delta_notional, account.cash)
            qty = int(budget / current_price)
            if qty <= 0:
                return None
            reason = (
                f"Overlay buy: regime={context.get('market_regime') if context else 'unknown'} "
                f"score={context.get('market_score') if context else 'n/a'} "
                f"stock_exposure={self._current_exposure(account, stock_positions):.2%} "
                f"target_overlay=${desired_overlay_notional:,.0f}"
            )
            return self._execute_overlay_order(
                symbol=overlay_symbol,
                side="buy",
                qty=qty,
                account=account,
                reasoning=reason,
                context=context,
            )

        if overlay_position is None:
            return None

        qty = int(min(abs(delta_notional), overlay_position.market_value) / current_price)
        if desired_overlay_notional < min_notional or qty >= int(round(overlay_position.qty)):
            qty = int(round(overlay_position.qty))
        qty = min(max(qty, 1), int(round(overlay_position.qty)))
        reason = (
            f"Overlay trim: regime={context.get('market_regime') if context else 'unknown'} "
            f"score={context.get('market_score') if context else 'n/a'} "
            f"stock_exposure={self._current_exposure(account, stock_positions):.2%} "
            f"target_overlay=${desired_overlay_notional:,.0f}"
        )
        return self._execute_overlay_order(
            symbol=overlay_symbol,
            side="sell",
            qty=qty,
            account=account,
            reasoning=reason,
            context=context,
        )

    def _execute_overlay_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        account: Account,
        reasoning: str,
        context: Optional[Dict] = None,
    ) -> Optional[Dict]:
        qty = int(qty)
        if qty <= 0:
            return None

        action = side.upper()
        signal_id = self.db.log_signal(
            symbol=symbol,
            action=action,
            confidence=0.75 if side == "buy" else 0.70,
            reasoning=reasoning,
            timeframe="overlay",
            full_analysis=json.dumps({"overlay_context": context}, default=str),
        )
        try:
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                qty=float(qty),
                order_type="market",
            )
            existing_open_order = self._find_existing_open_order(symbol, side)
            if existing_open_order is not None:
                reason = (
                    f"Existing open {side} order "
                    f"{existing_open_order.order_id} [{existing_open_order.status}]"
                )
                self.db.mark_signal_rejected(signal_id, reason)
                return {
                    "symbol": symbol,
                    "action": action,
                    "traded": False,
                    "risk_rejected": reason,
                    "overlay_managed": True,
                }
            order_result = self.broker.submit_order(order_request)
        except Exception as exc:
            self.db.mark_signal_rejected(signal_id, str(exc))
            logger.error("Overlay order failed for %s: %s", symbol, exc, exc_info=True)
            return {
                "symbol": symbol,
                "action": action,
                "traded": False,
                "risk_rejected": str(exc),
                "overlay_managed": True,
            }

        self.db.log_trade(
            symbol=symbol,
            side=side,
            qty=float(qty),
            order_type="market",
            status=order_result.status,
            filled_qty=order_result.filled_qty,
            filled_price=order_result.filled_avg_price,
            order_id=order_result.order_id,
            signal_id=signal_id,
            reasoning=reasoning,
        )
        self.db.mark_signal_executed(signal_id)
        self._notify_order(
            symbol=symbol,
            side=side,
            qty=float(qty),
            status=str(order_result.status),
            order_id=order_result.order_id,
            filled_price=order_result.filled_avg_price,
            reasoning=reasoning,
            source="overlay",
        )
        return {
            "symbol": symbol,
            "action": action,
            "traded": True,
            "side": side,
            "qty": float(qty),
            "order_id": order_result.order_id,
            "status": order_result.status,
            "overlay_managed": True,
        }

    def _notify_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        status: str,
        order_id: Optional[str],
        filled_price: Optional[float],
        reasoning: str,
        source: str,
    ):
        if not self.notifier.enabled:
            return
        action = str(side).upper()
        price_text = (
            f"${float(filled_price):.2f}"
            if self._to_float(filled_price) is not None
            else "market"
        )
        message = "\n".join(
            [
                f"{action} {symbol}",
                f"Qty: {qty:.0f}",
                f"Status: {status}",
                f"Price: {price_text}",
                f"Source: {source}",
                f"Reason: {reasoning[:180]}",
            ]
        )
        self.notifier.send(
            f"TradingAgents Order {action}",
            message,
            priority="high" if action == "BUY" else "default",
            tags=["moneybag", "rotating_light"] if action == "BUY" else ["money_with_wings"],
            dedupe_key=f"order:{order_id or f'{symbol}:{action}:{qty}:{status}'}",
        )

    # ── Manual Controls ──────────────────────────────────────────────

    def emergency_close_all(self) -> List:
        """Close all positions immediately."""
        if not self.execution_enabled or self.broker is None:
            raise RuntimeError("Execution is disabled for this profile")
        logger.warning("EMERGENCY: Closing all positions!")
        results = self.broker.close_all_positions()
        for r in results:
            self.db.log_trade(
                symbol=r.symbol, side="sell", qty=r.qty,
                order_type="market", status=r.status,
                order_id=r.order_id, reasoning="Emergency close all",
            )
        return results

    def get_status(self) -> Dict:
        """Get current system status."""
        if not self.execution_enabled or self.broker is None:
            latest_setups = self.db.get_latest_setup_candidates()
            latest_batch = self.db.get_latest_screening_batch()
            watchlist = list(self.watchlist)
            if not watchlist and latest_setups:
                watchlist = [row["symbol"] for row in latest_setups if row.get("symbol")]
            screening = {
                "screen_date": latest_batch["screen_date"] if latest_batch else None,
                "market_regime": latest_batch["market_regime"] if latest_batch else None,
                "confirmed_uptrend": bool(latest_batch["market_confirmed_uptrend"]) if latest_batch else None,
                "approved_symbols": latest_batch["approved_symbols"] if latest_batch else [],
                "setup_count": latest_batch["row_count"] if latest_batch else len(latest_setups),
            }
            return {
                "account": {
                    "equity": 0.0,
                    "cash": 0.0,
                    "buying_power": 0.0,
                    "daily_pl": 0.0,
                    "daily_pl_pct": "0.00%",
                },
                "positions": [],
                "market": {
                    "is_open": False,
                    "next_open": "n/a",
                    "next_close": "n/a",
                },
                "performance": {},
                "today": {
                    "trade_summary": self.db.get_trade_summary(),
                    "unrealized_pl": 0.0,
                },
                "screening": screening,
                "overlay": {
                    "enabled": False,
                    "symbol": None,
                    "trigger": None,
                    "fraction": 0.0,
                    "market_regime": screening.get("market_regime"),
                    "market_score": None,
                    "confirmed_uptrend": screening.get("confirmed_uptrend"),
                    "overlay_allowed": False,
                    "position_qty": 0.0,
                    "position_value": 0.0,
                },
                "notifications": {
                    "enabled": self.notifier.enabled,
                    "provider": "ntfy" if self.notifier.enabled else None,
                    "topic": self.notifier.topic if self.notifier.enabled else None,
                    "server": self.notifier.server if self.notifier.enabled else None,
                },
                "watchlist": watchlist,
                "paper_mode": self.config.get("paper_trading", True),
            }

        account = self.broker.get_account()
        positions = self.broker.get_positions()
        clock = self.broker.get_clock()
        perf = self.tracker.get_performance_summary()
        overlay_context = self._get_overlay_context()
        overlay_position = (
            self._find_position(positions, self._overlay_symbol())
            if self._overlay_enabled()
            else None
        )
        latest_setups = self.db.get_latest_setup_candidates()
        latest_batch = self.db.get_latest_screening_batch()
        watchlist = list(self.watchlist)
        if not watchlist and latest_setups:
            watchlist = [row["symbol"] for row in latest_setups if row.get("symbol")]
        screening = {
            "screen_date": latest_batch["screen_date"] if latest_batch else None,
            "market_regime": latest_batch["market_regime"] if latest_batch else None,
            "confirmed_uptrend": bool(latest_batch["market_confirmed_uptrend"]) if latest_batch else None,
            "approved_symbols": latest_batch["approved_symbols"] if latest_batch else [],
            "setup_count": latest_batch["row_count"] if latest_batch else len(latest_setups),
        }

        return {
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "daily_pl": account.daily_pl,
                "daily_pl_pct": f"{account.daily_pl_pct:.2%}",
            },
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "entry": p.avg_entry_price,
                    "current": p.current_price,
                    "pl": p.unrealized_pl,
                    "pl_pct": f"{p.unrealized_plpc:.2%}",
                    "overlay_managed": self._is_overlay_symbol(p.symbol),
                }
                for p in positions
            ],
            "market": {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            },
            "performance": perf,
            "today": {
                "trade_summary": self.db.get_trade_summary(),
                "unrealized_pl": sum(p.unrealized_pl for p in positions),
            },
            "screening": screening,
            "overlay": {
                "enabled": self._overlay_enabled(),
                "symbol": self._overlay_symbol() if self._overlay_enabled() else None,
                "trigger": self.config.get("overlay_trigger"),
                "fraction": self.config.get("overlay_fraction"),
                "market_regime": overlay_context.get("market_regime") if overlay_context else None,
                "market_score": overlay_context.get("market_score") if overlay_context else None,
                "confirmed_uptrend": overlay_context.get("confirmed_uptrend") if overlay_context else None,
                "overlay_allowed": overlay_context.get("overlay_allowed") if overlay_context else None,
                "position_qty": overlay_position.qty if overlay_position else 0.0,
                "position_value": overlay_position.market_value if overlay_position else 0.0,
            },
            "notifications": {
                "enabled": self.notifier.enabled,
                "provider": "ntfy" if self.notifier.enabled else None,
                "topic": self.notifier.topic if self.notifier.enabled else None,
                "server": self.notifier.server if self.notifier.enabled else None,
            },
            "watchlist": watchlist,
            "paper_mode": self.config.get("paper_trading", True),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_position(positions: List[Position], symbol: str) -> Optional[Position]:
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    def _find_existing_open_order(self, symbol: str, side: Optional[str] = None):
        orders = self._get_open_orders(symbol, side)
        return orders[0] if orders else None

    def _get_latest_setup_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        if (
            self._latest_minervini_preflight is not None
            and self._latest_minervini_preflight.screen_df is not None
            and not self._latest_minervini_preflight.screen_df.empty
        ):
            matches = self._latest_minervini_preflight.screen_df[
                self._latest_minervini_preflight.screen_df["symbol"] == symbol
            ]
            if not matches.empty:
                return matches.iloc[-1].to_dict()

        for row in self.db.get_latest_setup_candidates():
            if str(row.get("symbol") or "").upper() != symbol.upper():
                continue
            payload = row.get("payload_json")
            if payload:
                try:
                    return json.loads(payload)
                except Exception:
                    pass
            return row
        return None

    def _get_position_earnings_context(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        today = date.today().isoformat()
        cached = self._latest_earnings_context.get(symbol)
        if cached is not None and cached.get("computed_on") == today:
            return cached

        context: Dict[str, Any] = {
            "symbol": symbol,
            "next_earnings_datetime": None,
            "earnings_days_away": None,
            "earnings_event_date": None,
            "source": None,
            "computed_on": today,
        }

        def _apply(next_earnings_datetime: Any, earnings_days_away: Any, source: str) -> bool:
            if next_earnings_datetime in (None, "", "NaT"):
                return False
            try:
                earnings_ts = pd.to_datetime(next_earnings_datetime)
            except Exception:
                return False
            if pd.isna(earnings_ts):
                return False
            if getattr(earnings_ts, "tzinfo", None) is not None:
                earnings_ts = earnings_ts.tz_localize(None)
            days_away = self._to_float(earnings_days_away)
            if days_away is None:
                days_away = float(
                    (earnings_ts.to_pydatetime() - datetime.utcnow()).total_seconds()
                    / 86400.0
                )
            context.update(
                {
                    "next_earnings_datetime": earnings_ts.isoformat(),
                    "earnings_days_away": days_away,
                    "earnings_event_date": earnings_ts.date().isoformat(),
                    "source": source,
                }
            )
            return True

        setup = self._get_latest_setup_for_symbol(symbol)
        if setup and _apply(
            setup.get("next_earnings_datetime"),
            setup.get("earnings_days_away"),
            "setup",
        ):
            self._latest_earnings_context[symbol] = context
            return context

        db_path = self.config.get("minervini_db_path", "research_data/market_data.duckdb")
        try:
            warehouse = MarketDataWarehouse(db_path, read_only=True)
            try:
                fundamentals = warehouse.get_latest_fundamentals([symbol])
            finally:
                warehouse.close()
            if not fundamentals.empty:
                row = fundamentals.iloc[0].to_dict()
                _apply(
                    row.get("next_earnings_datetime"),
                    None,
                    "warehouse",
                )
        except Exception as exc:
            logger.debug("Could not load earnings context for %s: %s", symbol, exc)

        self._latest_earnings_context[symbol] = context
        return context

    def _earnings_window_active(self, context: Optional[Dict[str, Any]]) -> bool:
        if not self.config.get("minervini_earnings_management_enabled", True):
            return False
        if not context:
            return False
        days_away = self._to_float(context.get("earnings_days_away"))
        if days_away is None:
            return False
        exit_days = max(int(self.config.get("minervini_earnings_exit_days", 5)), 0)
        return 0.0 <= days_away <= float(exit_days)

    def _load_position_management_state(
        self, symbol: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        inferred = self._infer_position_management_from_trades(symbol)
        inferred_cycle_entry = inferred.get("cycle_entry_date")
        cycle_entry_date = str(
            inferred_cycle_entry.isoformat()
            if inferred_cycle_entry is not None
            else state.get("entry_trade_date") or ""
        )
        row = self.db.get_position_management_state(symbol) or {}
        if row.get("cycle_entry_date") and row.get("cycle_entry_date") != cycle_entry_date:
            row = {}
        return {
            "cycle_entry_date": cycle_entry_date,
            "partial_profit_taken": bool(row.get("partial_profit_taken", False))
            or bool(inferred.get("partial_profit_taken", False)),
            "add_on_1_done": bool(row.get("add_on_1_done", False))
            or bool(inferred.get("add_on_1_done", False)),
            "add_on_2_done": bool(row.get("add_on_2_done", False))
            or bool(inferred.get("add_on_2_done", False)),
            "earnings_event_date": row.get("earnings_event_date"),
            "earnings_action": row.get("earnings_action"),
        }

    def _persist_position_management_state(
        self, symbol: str, state: Dict[str, Any]
    ) -> None:
        self.db.upsert_position_management_state(
            symbol=symbol,
            cycle_entry_date=state.get("cycle_entry_date"),
            partial_profit_taken=bool(state.get("partial_profit_taken")),
            add_on_1_done=bool(state.get("add_on_1_done")),
            add_on_2_done=bool(state.get("add_on_2_done")),
            earnings_event_date=state.get("earnings_event_date"),
            earnings_action=state.get("earnings_action"),
        )

    def _setup_supports_pyramiding(
        self,
        setup: Optional[Dict[str, Any]],
        current_price: Optional[float],
    ) -> bool:
        if not setup:
            return False

        candidate_status = str(setup.get("candidate_status") or "")
        if candidate_status not in {
            "leader_continuation_watch",
            "leader_continuation_actionable",
            "near_pivot",
            "breakout_ready",
        } and not bool(setup.get("breakout_signal")):
            return False

        buy_limit_price = self._to_float(setup.get("buy_limit_price"))
        if (
            current_price is not None
            and buy_limit_price is not None
            and current_price > buy_limit_price
        ):
            return False

        close_range_pct = self._to_float(setup.get("close_range_pct"))
        min_close_range_pct = float(
            self.config.get("leader_continuation_min_close_range_pct", 0.15)
        )
        if close_range_pct is not None and close_range_pct < min_close_range_pct:
            return False

        return (
            bool(setup.get("breakout_ready"))
            or bool(setup.get("breakout_signal"))
            or self._is_leader_continuation_setup(setup)
        )

    def _calculate_add_on_qty(
        self,
        *,
        account: Account,
        positions: List[Position],
        position: Position,
        price: float,
        stop_price: float,
        add_fraction: float,
    ) -> int:
        risk_per_share = max(price - stop_price, 0.0)
        if risk_per_share <= 0:
            return 0

        equity = float(account.equity or 0.0)
        if equity <= 0:
            return 0

        max_position_value = equity * float(self.config.get("max_position_pct", 0.12))
        current_value = float(position.market_value or 0.0)
        remaining_capacity = max(0.0, max_position_value - current_value)
        if remaining_capacity <= 0:
            return 0

        current_exposure_value = sum(float(p.market_value or 0.0) for p in positions)
        max_total_exposure_value = equity * float(self.config.get("max_total_exposure", 0.72))
        remaining_exposure_capacity = max(0.0, max_total_exposure_value - current_exposure_value)
        if remaining_exposure_capacity <= 0:
            return 0

        min_cash_reserve = equity * float(self.config.get("min_cash_reserve", 0.20))
        spendable_cash = max(0.0, float(account.cash or 0.0) - min_cash_reserve)
        if spendable_cash <= 0:
            return 0

        target_value = min(
            max_position_value * add_fraction,
            remaining_capacity,
            remaining_exposure_capacity,
            spendable_cash,
        )
        risk_budget = equity * float(self.config.get("risk_per_trade", 0.012)) * add_fraction

        qty_by_value = int(target_value / price)
        qty_by_risk = int(risk_budget / risk_per_share)
        return max(0, min(qty_by_value, qty_by_risk))

    def _get_open_orders(self, symbol: str, side: Optional[str] = None):
        getter = getattr(self.broker, "get_open_orders", None)
        if not callable(getter):
            return []
        try:
            orders = getter(symbol=symbol)
        except TypeError:
            orders = getter()
        except Exception as exc:
            logger.warning("Could not fetch open orders for %s: %s", symbol, exc)
            return []

        target_symbol = symbol.upper()
        target_side = side.lower() if side else None
        terminal_statuses = {"filled", "canceled", "cancelled", "expired", "rejected"}
        matches = []
        for order in orders or []:
            order_symbol = str(getattr(order, "symbol", "") or "").upper()
            order_side = self._enum_name(getattr(order, "side", ""))
            order_status = str(getattr(order, "status", "") or "").lower()
            if order_symbol != target_symbol:
                continue
            if target_side and order_side != target_side:
                continue
            if order_status in terminal_statuses:
                continue
            matches.append(order)
        return matches

    def _cancel_open_orders(self, symbol: str, side: Optional[str] = None) -> List[str]:
        canceled: List[str] = []
        for order in self._get_open_orders(symbol, side):
            order_id = str(getattr(order, "order_id", "") or "")
            if not order_id:
                continue
            try:
                self.broker.cancel_order(order_id)
                canceled.append(order_id)
            except Exception as exc:
                logger.warning("Could not cancel order %s for %s: %s", order_id, symbol, exc)
        return canceled

    def _target_exposure_for_regime(self, regime: Optional[str]) -> float:
        regime = (regime or "").lower()
        if regime == "confirmed_uptrend":
            return float(self.config.get("minervini_target_exposure_confirmed_uptrend", 0.72))
        if regime == "uptrend_under_pressure":
            return float(self.config.get("minervini_target_exposure_uptrend_under_pressure", 0.48))
        if regime == "market_correction":
            return float(self.config.get("minervini_target_exposure_market_correction", 0.0))
        return 0.0

    def _entries_allowed_for_regime(self, regime: Optional[str]) -> bool:
        regime = (regime or "").lower()
        if regime == "market_correction":
            return bool(self.config.get("minervini_allow_new_entries_in_correction", False)) and (
                self._target_exposure_for_regime(regime) > 0
            )
        return self._target_exposure_for_regime(regime) > 0

    def _is_leader_continuation_setup(self, setup: Optional[Dict]) -> bool:
        if not setup:
            return False
        if bool(setup.get("leader_continuation")):
            return True
        candidate_status = str(setup.get("candidate_status") or "")
        return candidate_status.startswith("leader_continuation")

    def _target_exposure_for_setup(
        self, setup: Optional[Dict], regime: Optional[str]
    ) -> float:
        regime = (regime or "").lower()
        if self._is_leader_continuation_setup(setup):
            if regime == "confirmed_uptrend":
                return float(
                    self.config.get(
                        "leader_continuation_target_exposure_confirmed_uptrend",
                        0.72,
                    )
                )
            if regime == "uptrend_under_pressure":
                return float(
                    self.config.get(
                        "leader_continuation_target_exposure_uptrend_under_pressure",
                        0.36,
                    )
                )
            if regime == "market_correction":
                return float(
                    self.config.get(
                        "leader_continuation_target_exposure_market_correction",
                        0.12,
                    )
                )
            return 0.0
        return self._target_exposure_for_regime(regime)

    def _entries_allowed_for_setup(
        self, setup: Optional[Dict], regime: Optional[str]
    ) -> bool:
        regime = (regime or "").lower()
        if self._is_leader_continuation_setup(setup):
            if regime == "market_correction":
                return bool(
                    self.config.get("leader_continuation_allow_in_correction", True)
                ) and self._target_exposure_for_setup(setup, regime) > 0
            return self._target_exposure_for_setup(setup, regime) > 0
        return self._entries_allowed_for_regime(regime)

    @staticmethod
    def _current_exposure(account: Account, positions: List[Position]) -> float:
        if account.equity <= 0:
            return 0.0
        return sum(p.market_value for p in positions) / float(account.equity)

    def _analysis_universe(self, preflight) -> list[str]:
        if preflight is not None and getattr(preflight, "screened_symbols", None):
            return list(preflight.screened_symbols)
        if self.watchlist:
            return list(self.watchlist)
        if self._latest_minervini_preflight is not None:
            return list(getattr(self._latest_minervini_preflight, "screened_symbols", []))
        return []

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _enum_name(value: Any) -> str:
        text = str(value or "").strip().lower()
        if "." in text:
            text = text.split(".")[-1]
        return text

    @classmethod
    def _order_type_name(cls, value: Any) -> str:
        return cls._enum_name(value)
