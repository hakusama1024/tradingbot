"""Track portfolio performance across time."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from tradingagents.broker.base_broker import BaseBroker
from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Captures daily snapshots and computes performance metrics."""

    def __init__(self, broker: BaseBroker, db: TradingDatabase, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.db = db
        self.config = config or {}

    def take_daily_snapshot(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        pos_list = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
            }
            for p in positions
        ]

        self.db.take_snapshot(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            portfolio_value=account.portfolio_value,
            positions=pos_list,
            daily_pl=account.daily_pl,
            daily_pl_pct=account.daily_pl_pct,
        )

        logger.info(
            f"Snapshot: equity=${account.equity:,.2f}  "
            f"daily_pl=${account.daily_pl:,.2f} ({account.daily_pl_pct:.2%})  "
            f"positions={len(positions)}"
        )
        return {
            "equity": account.equity,
            "cash": account.cash,
            "daily_pl": account.daily_pl,
            "positions": len(positions),
        }

    def get_total_position_value(self) -> float:
        positions = self.broker.get_positions()
        return sum(p.market_value for p in positions)

    def build_daily_report(self, report_date: Optional[str] = None) -> Dict:
        report_day = report_date or date.today().isoformat()
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        trades = self.db.get_trades_on_date(report_day)
        trade_summary = self.db.get_trade_summary(report_day)
        setups = self.db.get_setup_candidates_on_date(report_day)
        screening_batch = self.db.get_latest_screening_batch()

        position_rows = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "side": p.side,
                "entry": p.avg_entry_price,
                "current": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_pl_pct": p.unrealized_plpc,
            }
            for p in positions
        ]
        total_unrealized_pl = sum(p.unrealized_pl for p in positions)

        return {
            "date": report_day,
            "account": {
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
                "portfolio_value": account.portfolio_value,
                "daily_pl": account.daily_pl,
                "daily_pl_pct": account.daily_pl_pct,
            },
            "trade_summary": trade_summary,
            "trades": trades,
            "setups": setups,
            "screening_batch": screening_batch,
            "positions": position_rows,
            "position_summary": {
                "open_positions": len(position_rows),
                "total_unrealized_pl": total_unrealized_pl,
            },
            "performance": self.get_performance_summary(),
            "attribution": self.build_return_attribution(report_date=report_day),
        }

    def _download_benchmark_closes(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        closes: Dict[str, pd.Series] = {}
        for symbol in symbols:
            if not symbol:
                continue
            try:
                frame = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if frame is None or frame.empty:
                    continue
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                close = frame.get("Close")
                if close is None or close.empty:
                    continue
                series = close.copy()
                series.index = pd.to_datetime(series.index).date
                closes[symbol] = series
            except Exception as exc:
                logger.warning("Benchmark download failed for %s: %s", symbol, exc)
        return pd.DataFrame(closes).sort_index()

    def _build_snapshot_exposure_frame(self, days: int = 90) -> pd.DataFrame:
        snapshots = self.db.get_snapshots(days=days)
        if not snapshots:
            return pd.DataFrame()

        overlay_symbol = str(self.config.get("overlay_symbol", "SMH")).upper()
        rows = []
        for snapshot in reversed(snapshots):
            positions_raw = snapshot.get("positions_json") or "[]"
            try:
                positions = json.loads(positions_raw)
            except Exception:
                positions = []

            equity = float(snapshot.get("equity") or 0.0)
            cash = float(snapshot.get("cash") or 0.0)
            overlay_value = 0.0
            stock_value = 0.0
            for position in positions:
                market_value = float((position or {}).get("market_value") or 0.0)
                symbol = str((position or {}).get("symbol") or "").upper()
                if symbol == overlay_symbol:
                    overlay_value += market_value
                else:
                    stock_value += market_value

            rows.append(
                {
                    "date": snapshot.get("date"),
                    "equity": equity,
                    "cash": cash,
                    "overlay_value": overlay_value,
                    "stock_value": stock_value,
                    "overlay_weight": (overlay_value / equity) if equity > 0 else 0.0,
                    "stock_weight": (stock_value / equity) if equity > 0 else 0.0,
                    "cash_weight": (cash / equity) if equity > 0 else 0.0,
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        frame = frame[frame["equity"] > 0].reset_index(drop=True)
        return frame

    def build_return_attribution(self, report_date: Optional[str] = None) -> Dict[str, Any]:
        benchmark_symbol = str(
            self.config.get("attribution_benchmark_symbol", "QQQ")
        ).upper()
        comparison_symbols = list(self.config.get("attribution_comparison_symbols", ["SPY", "QQQ", "SMH"]))
        overlay_symbol = str(self.config.get("overlay_symbol", "SMH")).upper()
        period = str(self.config.get("attribution_period", "1M"))

        result: Dict[str, Any] = {
            "available": False,
            "benchmark_window": {},
            "snapshot_attribution": {},
        }

        try:
            history = self.broker.get_portfolio_history(period=period, timeframe="1D")
            history_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(history.get("timestamp", []), unit="s", utc=True)
                    .tz_convert("America/New_York")
                    .date,
                    "equity": history.get("equity", []),
                }
            ).dropna()
            if not history_df.empty:
                history_df = (
                    history_df.drop_duplicates(subset="date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                history_df = history_df[history_df["equity"] > 0].reset_index(drop=True)
            if not history_df.empty:
                start_date = str(history_df["date"].iloc[0])
                end_date = str(history_df["date"].iloc[-1])
                benchmark_frame = self._download_benchmark_closes(
                    list(dict.fromkeys(comparison_symbols)),
                    start_date,
                    (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                comparison = {
                    "period": period,
                    "start_date": start_date,
                    "end_date": end_date,
                    "strategy_return": float(history_df["equity"].iloc[-1] / history_df["equity"].iloc[0] - 1.0),
                    "benchmarks": {},
                }
                for symbol in comparison_symbols:
                    if symbol not in benchmark_frame.columns:
                        continue
                    close = benchmark_frame[symbol].dropna()
                    if close.empty:
                        continue
                    ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
                    comparison["benchmarks"][symbol] = ret
                comparison["alpha_vs_spy"] = comparison["strategy_return"] - comparison["benchmarks"].get("SPY", 0.0)
                comparison["alpha_vs_qqq"] = comparison["strategy_return"] - comparison["benchmarks"].get("QQQ", 0.0)
                comparison["alpha_vs_smh"] = comparison["strategy_return"] - comparison["benchmarks"].get("SMH", 0.0)
                result["benchmark_window"] = comparison
                result["available"] = True
        except Exception as exc:
            logger.warning("Failed to build benchmark comparison: %s", exc)
            result["benchmark_window"] = {"error": str(exc)}

        try:
            exposure_df = self._build_snapshot_exposure_frame(days=90)
            if len(exposure_df) >= 2:
                start_date = str(exposure_df["date"].iloc[0])
                end_date = str(exposure_df["date"].iloc[-1])
                benchmark_frame = self._download_benchmark_closes(
                    list(dict.fromkeys([benchmark_symbol, overlay_symbol])),
                    start_date,
                    (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if benchmark_symbol in benchmark_frame.columns and overlay_symbol in benchmark_frame.columns:
                    frame = exposure_df.copy()
                    frame["strategy_return"] = frame["equity"].pct_change().fillna(0.0)
                    frame["prev_equity"] = frame["equity"].shift(1)
                    frame["daily_pnl"] = frame["equity"].diff().fillna(0.0)

                    bench = benchmark_frame.reindex(frame["date"])
                    frame["benchmark_return"] = (
                        bench[benchmark_symbol].pct_change().fillna(0.0).to_numpy()
                    )
                    frame["overlay_return"] = (
                        bench[overlay_symbol].pct_change().fillna(0.0).to_numpy()
                    )
                    frame["lagged_stock_weight"] = frame["stock_weight"].shift(1).fillna(0.0)
                    frame["lagged_overlay_weight"] = frame["overlay_weight"].shift(1).fillna(0.0)
                    frame["overlay_pnl"] = (
                        frame["prev_equity"]
                        * frame["lagged_overlay_weight"]
                        * frame["overlay_return"]
                    ).fillna(0.0)
                    frame["beta_pnl"] = (
                        frame["prev_equity"]
                        * frame["lagged_stock_weight"]
                        * frame["benchmark_return"]
                    ).fillna(0.0)
                    frame["selection_pnl"] = (
                        frame["daily_pnl"] - frame["overlay_pnl"] - frame["beta_pnl"]
                    ).fillna(0.0)

                    start_equity = float(frame["equity"].iloc[0])
                    end_equity = float(frame["equity"].iloc[-1])
                    overlay_pnl = float(frame["overlay_pnl"].sum())
                    beta_pnl = float(frame["beta_pnl"].sum())
                    selection_pnl = float(frame["selection_pnl"].sum())
                    result["snapshot_attribution"] = {
                        "available": True,
                        "start_date": start_date,
                        "end_date": end_date,
                        "benchmark_symbol": benchmark_symbol,
                        "strategy_return": float(end_equity / start_equity - 1.0),
                        "benchmark_return": float(
                            bench[benchmark_symbol].dropna().iloc[-1]
                            / bench[benchmark_symbol].dropna().iloc[0]
                            - 1.0
                        ),
                        "overlay_symbol": overlay_symbol,
                        "overlay_pnl": overlay_pnl,
                        "overlay_return_equiv": overlay_pnl / start_equity if start_equity else 0.0,
                        "beta_pnl": beta_pnl,
                        "beta_return_equiv": beta_pnl / start_equity if start_equity else 0.0,
                        "selection_pnl": selection_pnl,
                        "selection_return_equiv": selection_pnl / start_equity if start_equity else 0.0,
                        "avg_overlay_weight": float(frame["overlay_weight"].mean()),
                        "avg_stock_weight": float(frame["stock_weight"].mean()),
                        "avg_cash_weight": float(frame["cash_weight"].mean()),
                        "note": "Approximate daily attribution using lagged snapshot weights.",
                    }
                    result["available"] = True
                else:
                    result["snapshot_attribution"] = {
                        "available": False,
                        "reason": "Benchmark history unavailable for attribution window.",
                    }
            else:
                result["snapshot_attribution"] = {
                    "available": False,
                    "reason": "Not enough snapshots yet.",
                }
        except Exception as exc:
            logger.warning("Failed to build snapshot attribution: %s", exc)
            result["snapshot_attribution"] = {"available": False, "error": str(exc)}

        return result

    def save_daily_report(self, report: Dict, output_dir: str) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{report['date']}.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        return report_path

    def get_performance_summary(self) -> Dict:
        snapshots = self.db.get_snapshots(days=365)
        if not snapshots:
            return {"message": "No snapshots yet"}

        latest = snapshots[0]
        starting = self.db.get_starting_equity()
        equities = [s["equity"] for s in reversed(snapshots)]

        total_return = 0.0
        if starting and starting > 0:
            total_return = (latest["equity"] - starting) / starting

        max_equity = 0.0
        max_drawdown = 0.0
        for eq in equities:
            if eq > max_equity:
                max_equity = eq
            dd = (max_equity - eq) / max_equity if max_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        win_days = sum(1 for r in daily_returns if r > 0)
        total_days = len(daily_returns)

        avg_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        std_return = (
            (sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
            if daily_returns else 0
        )
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0

        return {
            "current_equity": latest["equity"],
            "starting_equity": starting,
            "total_return": total_return,
            "total_return_pct": f"{total_return:.2%}",
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": f"{max_drawdown:.2%}",
            "sharpe_ratio": round(sharpe, 2),
            "win_day_rate": f"{win_days}/{total_days}" if total_days > 0 else "N/A",
            "total_snapshots": len(snapshots),
        }
