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
from typing import List, Dict, Optional

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
        self.watchlist = config.get("watchlist", [])
        self._latest_analysis_states: Dict[str, Dict] = {}
        self._latest_minervini_preflight = None
        self._active_universe: list[str] = list(self.watchlist)
        self._latest_overlay_context: Optional[Dict] = None

        # Broker
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
        self.tracker = PortfolioTracker(self.broker, self.db)

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

        lines = [
            f"Date: {preflight.trade_date}",
            f"Regime: {preflight.market_regime}",
            f"Approved: {len(preflight.approved_symbols)}",
            f"Watch: {len(watch)}",
        ]
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

        if not self.broker.is_market_open():
            logger.info("Market is closed. Skipping analysis.")
            return {"status": "market_closed"}

        preflight = None
        preflight_error = None
        try:
            preflight = self._run_minervini_preflight()
        except Exception as e:
            preflight_error = str(e)
            logger.error("Minervini preflight failed: %s", e, exc_info=True)
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
                result = self._analyze_and_trade(pos.symbol, account, stock_positions)
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
            order_result = self.broker.submit_bracket_order(
                order_request, stop_loss_price=sl_price, take_profit_price=tp_price
            )
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
        snapshot = self.tracker.take_daily_snapshot()
        self._notify_morning_scan(preflight, snapshot)
        return snapshot

    def run_daily_reflection(self) -> Dict:
        """After market close: reflect on today's trades and update memories."""
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

    def _get_overlay_context(self) -> Optional[Dict]:
        if not self._overlay_enabled():
            return None

        today = date.today().isoformat()
        if (
            self._latest_overlay_context is not None
            and self._latest_overlay_context.get("computed_on") == today
        ):
            return self._latest_overlay_context

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
                    needs_refresh = (date.today() - freshest).days > 5
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
                    "symbol": self._overlay_symbol(),
                    "trade_date": context_df.index[-1].date().isoformat(),
                    "market_score": int(score) if score is not None else None,
                    "market_regime": str(latest["market_regime"]),
                    "confirmed_uptrend": bool(latest["market_confirmed_uptrend"]),
                    "source": "market_context",
                }
        except Exception as exc:
            logger.warning("Overlay context refresh failed: %s", exc)

        if context is None:
            fallback_regime = (
                self._latest_minervini_preflight.market_regime
                if self._latest_minervini_preflight is not None
                else "unknown"
            )
            context = {
                "symbol": self._overlay_symbol(),
                "trade_date": today,
                "market_score": None,
                "market_regime": fallback_regime,
                "confirmed_uptrend": bool(
                    self._latest_minervini_preflight.confirmed_uptrend
                    if self._latest_minervini_preflight is not None
                    else False
                ),
                "source": "preflight_fallback",
            }

        context["overlay_allowed"] = self._overlay_context_allows_entry(context)
        context["computed_on"] = today
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
        getter = getattr(self.broker, "get_open_orders", None)
        if not callable(getter):
            return None
        try:
            orders = getter(symbol=symbol)
        except TypeError:
            orders = getter()
        except Exception as exc:
            logger.warning("Could not fetch open orders for %s: %s", symbol, exc)
            return None

        target_symbol = symbol.upper()
        target_side = side.lower() if side else None
        terminal_statuses = {"filled", "canceled", "cancelled", "expired", "rejected"}
        for order in orders or []:
            order_symbol = str(getattr(order, "symbol", "") or "").upper()
            order_side = str(getattr(order, "side", "") or "").lower()
            order_status = str(getattr(order, "status", "") or "").lower()
            if order_symbol != target_symbol:
                continue
            if target_side and order_side != target_side:
                continue
            if order_status in terminal_statuses:
                continue
            return order
        return None

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
