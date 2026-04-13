"""SQLite storage for trades, signals, and daily snapshots."""

import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class TradingDatabase:
    """Lightweight SQLite database for all trading records."""

    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"TradingDatabase opened at {db_path}")

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol        TEXT    NOT NULL,
                side          TEXT    NOT NULL,
                qty           REAL,
                notional      REAL,
                order_type    TEXT    NOT NULL DEFAULT 'market',
                status        TEXT    NOT NULL,
                filled_qty    REAL    DEFAULT 0,
                filled_price  REAL,
                order_id      TEXT,
                signal_id     INTEGER REFERENCES signals(id),
                reasoning     TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
                symbol        TEXT    NOT NULL,
                action        TEXT    NOT NULL,
                confidence    REAL    DEFAULT 0,
                reasoning     TEXT,
                stop_loss     REAL,
                take_profit   REAL,
                timeframe     TEXT,
                full_analysis TEXT,
                executed      INTEGER DEFAULT 0,
                rejected_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT    NOT NULL,
                equity        REAL    NOT NULL,
                cash          REAL    NOT NULL,
                buying_power  REAL,
                portfolio_value REAL,
                positions_json TEXT,
                daily_pl      REAL    DEFAULT 0,
                daily_pl_pct  REAL    DEFAULT 0,
                UNIQUE(date)
            );

            CREATE TABLE IF NOT EXISTS agent_memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_name   TEXT    NOT NULL,
                situation     TEXT    NOT NULL,
                recommendation TEXT   NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS setup_candidates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                screen_date   TEXT    NOT NULL,
                symbol        TEXT    NOT NULL,
                selected_for_analysis INTEGER DEFAULT 0,
                passed_template INTEGER DEFAULT 0,
                market_regime TEXT,
                market_confirmed_uptrend INTEGER DEFAULT 0,
                rs_percentile REAL,
                revenue_growth REAL,
                eps_growth REAL,
                revenue_acceleration REAL,
                eps_acceleration REAL,
                stage_number REAL,
                base_label TEXT,
                candidate_status TEXT,
                pivot_price REAL,
                buy_point REAL,
                buy_limit_price REAL,
                initial_stop_price REAL,
                initial_stop_pct REAL,
                distance_to_pivot_pct REAL,
                distance_to_buy_point_pct REAL,
                buy_zone_pct REAL,
                breakout_signal INTEGER DEFAULT 0,
                rule_watch_candidate INTEGER DEFAULT 0,
                rule_entry_candidate INTEGER DEFAULT 0,
                next_earnings_datetime TEXT,
                earnings_days_away REAL,
                payload_json TEXT,
                UNIQUE(screen_date, symbol)
            );

            CREATE TABLE IF NOT EXISTS screening_batches (
                screen_date   TEXT PRIMARY KEY,
                market_regime TEXT,
                market_confirmed_uptrend INTEGER DEFAULT 0,
                row_count     INTEGER DEFAULT 0,
                selected_count INTEGER DEFAULT 0,
                approved_symbols_json TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_snapshots_date ON daily_snapshots(date);
            CREATE INDEX IF NOT EXISTS idx_agent_memories_name ON agent_memories(memory_name);
            CREATE INDEX IF NOT EXISTS idx_setup_candidates_date ON setup_candidates(screen_date);
            CREATE INDEX IF NOT EXISTS idx_setup_candidates_symbol ON setup_candidates(symbol);
        """)
        self._ensure_columns(
            "setup_candidates",
            {
                "stage_number": "REAL",
                "candidate_status": "TEXT",
                "buy_point": "REAL",
                "buy_limit_price": "REAL",
                "initial_stop_price": "REAL",
                "initial_stop_pct": "REAL",
                "distance_to_buy_point_pct": "REAL",
                "buy_zone_pct": "REAL",
                "rule_watch_candidate": "INTEGER DEFAULT 0",
                "rule_entry_candidate": "INTEGER DEFAULT 0",
            },
        )
        self.conn.commit()

    def _ensure_columns(self, table: str, columns: Dict[str, str]):
        existing = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name in existing:
                continue
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

    # ── Signals ──────────────────────────────────────────────────────

    def log_signal(self, symbol: str, action: str, confidence: float = 0,
                   reasoning: str = "", stop_loss: float = None,
                   take_profit: float = None, timeframe: str = "",
                   full_analysis: str = "") -> int:
        cur = self.conn.execute(
            """INSERT INTO signals (symbol, action, confidence, reasoning,
               stop_loss, take_profit, timeframe, full_analysis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, action, confidence, reasoning,
             stop_loss, take_profit, timeframe, full_analysis),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_signal_executed(self, signal_id: int):
        self.conn.execute("UPDATE signals SET executed = 1 WHERE id = ?", (signal_id,))
        self.conn.commit()

    def mark_signal_rejected(self, signal_id: int, reason: str):
        self.conn.execute(
            "UPDATE signals SET rejected_reason = ? WHERE id = ?", (reason, signal_id)
        )
        self.conn.commit()

    # ── Trades ───────────────────────────────────────────────────────

    def log_trade(self, symbol: str, side: str, qty: float = None,
                  notional: float = None, order_type: str = "market",
                  status: str = "submitted", filled_qty: float = 0,
                  filled_price: float = None, order_id: str = None,
                  signal_id: int = None, reasoning: str = "") -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (symbol, side, qty, notional, order_type,
               status, filled_qty, filled_price, order_id, signal_id, reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, qty, notional, order_type, status,
             filled_qty, filled_price, order_id, signal_id, reasoning),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_trade_status(self, order_id: str, status: str,
                            filled_qty: float = None, filled_price: float = None):
        fields = ["status = ?"]
        values = [status]
        if filled_qty is not None:
            fields.append("filled_qty = ?")
            values.append(filled_qty)
        if filled_price is not None:
            fields.append("filled_price = ?")
            values.append(filled_price)
        values.append(order_id)
        self.conn.execute(
            f"UPDATE trades SET {', '.join(fields)} WHERE order_id = ?", values
        )
        self.conn.commit()

    # ── Daily Snapshots ──────────────────────────────────────────────

    def take_snapshot(self, equity: float, cash: float, buying_power: float,
                      portfolio_value: float, positions: List[Dict],
                      daily_pl: float = 0, daily_pl_pct: float = 0,
                      snapshot_date: str = None):
        d = snapshot_date or date.today().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO daily_snapshots
               (date, equity, cash, buying_power, portfolio_value,
                positions_json, daily_pl, daily_pl_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (d, equity, cash, buying_power, portfolio_value,
             json.dumps(positions), daily_pl, daily_pl_pct),
        )
        self.conn.commit()

    def get_snapshots(self, days: int = 30) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshot_on_date(self, snapshot_date: Optional[str] = None) -> Optional[Dict]:
        target_date = snapshot_date or date.today().isoformat()
        row = self.conn.execute(
            "SELECT * FROM daily_snapshots WHERE date = ? LIMIT 1", (target_date,)
        ).fetchone()
        return dict(row) if row else None

    def get_snapshots_between(self, start_date: str, end_date: str) -> List[Dict]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM daily_snapshots
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_starting_equity(self) -> Optional[float]:
        row = self.conn.execute(
            "SELECT equity FROM daily_snapshots ORDER BY date ASC LIMIT 1"
        ).fetchone()
        return row["equity"] if row else None

    # ── Agent Memories ───────────────────────────────────────────────

    def save_memories(self, memory_name: str, situations_and_recs: List[tuple]):
        for situation, recommendation in situations_and_recs:
            self.conn.execute(
                """INSERT INTO agent_memories (memory_name, situation, recommendation)
                   VALUES (?, ?, ?)""",
                (memory_name, situation, recommendation),
            )
        self.conn.commit()

    def load_memories(self, memory_name: str) -> List[tuple]:
        rows = self.conn.execute(
            "SELECT situation, recommendation FROM agent_memories WHERE memory_name = ?",
            (memory_name,),
        ).fetchall()
        return [(r["situation"], r["recommendation"]) for r in rows]

    # ── Minervini Setups ─────────────────────────────────────────────

    def save_setup_candidates(
        self,
        rows: List[Dict[str, Any]],
        screen_date: Optional[str] = None,
        selected_symbols: Optional[List[str]] = None,
    ):
        if not rows and screen_date is None:
            return

        selected = set(selected_symbols or [])
        batch_date = screen_date or rows[0].get("trade_date") or date.today().isoformat()
        self.conn.execute("DELETE FROM setup_candidates WHERE screen_date = ?", (batch_date,))
        if not rows:
            self.conn.commit()
            return

        for row in rows:
            symbol = row.get("symbol")
            next_earnings = row.get("next_earnings_datetime")
            payload = json.dumps(row, default=str)
            self.conn.execute(
                """
                INSERT INTO setup_candidates (
                    screen_date, symbol, selected_for_analysis, passed_template,
                    market_regime, market_confirmed_uptrend, rs_percentile,
                    revenue_growth, eps_growth, revenue_acceleration, eps_acceleration,
                    stage_number, base_label, candidate_status, pivot_price, buy_point,
                    buy_limit_price, initial_stop_price, initial_stop_pct,
                    distance_to_pivot_pct, distance_to_buy_point_pct, buy_zone_pct,
                    breakout_signal, rule_watch_candidate, rule_entry_candidate,
                    next_earnings_datetime, earnings_days_away, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_date,
                    symbol,
                    1 if symbol in selected else 0,
                    1 if row.get("passed_template") else 0,
                    row.get("market_regime"),
                    1 if row.get("market_confirmed_uptrend") else 0,
                    row.get("rs_percentile"),
                    row.get("revenue_growth"),
                    row.get("eps_growth"),
                    row.get("revenue_acceleration"),
                    row.get("eps_acceleration"),
                    row.get("stage_number"),
                    row.get("base_label"),
                    row.get("candidate_status"),
                    row.get("pivot_price"),
                    row.get("buy_point"),
                    row.get("buy_limit_price"),
                    row.get("initial_stop_price"),
                    row.get("initial_stop_pct"),
                    row.get("distance_to_pivot_pct"),
                    row.get("distance_to_buy_point_pct"),
                    row.get("buy_zone_pct"),
                    1 if row.get("breakout_signal") else 0,
                    1 if row.get("rule_watch_candidate") else 0,
                    1 if row.get("rule_entry_candidate") else 0,
                    next_earnings,
                    row.get("earnings_days_away"),
                    payload,
                ),
            )
        self.conn.commit()

    def save_screening_batch(
        self,
        screen_date: str,
        market_regime: Optional[str],
        confirmed_uptrend: bool,
        approved_symbols: Optional[List[str]] = None,
        row_count: int = 0,
    ):
        approved = approved_symbols or []
        self.conn.execute(
            """
            INSERT OR REPLACE INTO screening_batches (
                screen_date, market_regime, market_confirmed_uptrend,
                row_count, selected_count, approved_symbols_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                screen_date,
                market_regime,
                1 if confirmed_uptrend else 0,
                row_count,
                len(approved),
                json.dumps(approved),
            ),
        )
        self.conn.commit()

    # ── Queries ──────────────────────────────────────────────────────

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trades_for_symbol(self, symbol: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC", (symbol,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_today_trades(self) -> List[Dict]:
        return self.get_trades_on_date()

    def get_trades_on_date(self, trade_date: Optional[str] = None) -> List[Dict]:
        today = trade_date or date.today().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) = ?", (today,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_summary(self, trade_date: Optional[str] = None) -> Dict[str, Any]:
        summary_date = trade_date or date.today().isoformat()
        trades = self.get_trades_on_date(summary_date)
        filled_trades = [
            t for t in trades if "fill" in (t.get("status") or "").lower()
        ]

        gross_filled_notional = sum(
            (t.get("filled_qty") or t.get("qty") or 0) * (t.get("filled_price") or 0)
            for t in filled_trades
        )

        return {
            "date": summary_date,
            "total_orders": len(trades),
            "filled_orders": len(filled_trades),
            "buy_orders": sum(1 for t in trades if t.get("side") == "buy"),
            "sell_orders": sum(1 for t in trades if t.get("side") == "sell"),
            "symbols": sorted({t.get("symbol") for t in trades if t.get("symbol")}),
            "gross_filled_notional": gross_filled_notional,
        }

    def get_trades_between(self, start_date: str, end_date: str) -> List[Dict]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM trades
            WHERE date(timestamp) >= ? AND date(timestamp) <= ?
            ORDER BY timestamp ASC
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_summary_between(self, start_date: str, end_date: str) -> Dict[str, Any]:
        trades = self.get_trades_between(start_date, end_date)
        filled_trades = [
            t for t in trades if "fill" in (t.get("status") or "").lower()
        ]
        gross_filled_notional = sum(
            (t.get("filled_qty") or t.get("qty") or 0) * (t.get("filled_price") or 0)
            for t in filled_trades
        )
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_orders": len(trades),
            "filled_orders": len(filled_trades),
            "buy_orders": sum(1 for t in trades if t.get("side") == "buy"),
            "sell_orders": sum(1 for t in trades if t.get("side") == "sell"),
            "symbols": sorted({t.get("symbol") for t in trades if t.get("symbol")}),
            "gross_filled_notional": gross_filled_notional,
        }

    def get_today_pl(self) -> float:
        today = date.today().isoformat()
        row = self.conn.execute(
            "SELECT daily_pl FROM daily_snapshots WHERE date = ?", (today,)
        ).fetchone()
        return row["daily_pl"] if row else 0.0

    def get_setup_candidates_on_date(self, screen_date: Optional[str] = None) -> List[Dict]:
        target_date = screen_date or date.today().isoformat()
        rows = self.conn.execute(
            """
            SELECT *
            FROM setup_candidates
            WHERE screen_date = ?
            ORDER BY
                rule_entry_candidate DESC,
                selected_for_analysis DESC,
                passed_template DESC,
                rs_percentile DESC
            """,
            (target_date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_setup_candidates(self) -> List[Dict]:
        row = self.conn.execute(
            "SELECT MAX(screen_date) AS screen_date FROM setup_candidates"
        ).fetchone()
        if row is None or row["screen_date"] is None:
            return []
        return self.get_setup_candidates_on_date(row["screen_date"])

    def get_latest_screening_batch(self) -> Optional[Dict]:
        row = self.conn.execute(
            """
            SELECT *
            FROM screening_batches
            ORDER BY screen_date DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["approved_symbols"] = json.loads(result.get("approved_symbols_json") or "[]")
        return result

    def get_screening_batch_on_date(self, screen_date: str) -> Optional[Dict]:
        row = self.conn.execute(
            """
            SELECT *
            FROM screening_batches
            WHERE screen_date = ?
            LIMIT 1
            """,
            (screen_date,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["approved_symbols"] = json.loads(result.get("approved_symbols_json") or "[]")
        return result

    def get_win_rate(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            """SELECT symbol, side, filled_price FROM trades
               WHERE status LIKE '%fill%' ORDER BY timestamp"""
        ).fetchall()
        if not rows:
            return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
        return {"total_trades": len(rows), "note": "detailed P&L requires position pairing"}

    def close(self):
        self.conn.close()
