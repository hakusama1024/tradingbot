"""Portfolio-level CAN SLIM proxy backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .backtester import BacktestConfig
from .canslim import CANSLIMScreener
from .portfolio_backtester import PortfolioBacktestResult, PortfolioMinerviniBacktester


class PortfolioCANSLIMBacktester(PortfolioMinerviniBacktester):
    """Shared-capital backtester that swaps Minervini entry rules for CAN SLIM rules."""

    def __init__(
        self,
        screener: CANSLIMScreener | None = None,
        config: BacktestConfig | None = None,
        quarterly_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        super().__init__(screener=screener, config=config)
        self.screener = screener or CANSLIMScreener()
        self.quarterly_by_symbol = quarterly_by_symbol or {}

    def backtest_portfolio(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> PortfolioBacktestResult:
        prepared_frames = {}
        for symbol, df in data_by_symbol.items():
            quarterly_df = self.quarterly_by_symbol.get(symbol)
            prepared = self.screener.prepare_features(df, quarterly_df=quarterly_df)
            if prepared.empty:
                continue
            prepared_frames[symbol] = prepared
        if not prepared_frames:
            empty = pd.DataFrame()
            return PortfolioBacktestResult(
                summary={},
                trades=empty,
                equity_curve=empty,
                daily_state=empty,
                symbol_summary=empty,
            )

        regime_frame = self._build_regime_frame(
            benchmark_df,
            market_context_df=market_context_df,
        )
        all_dates = sorted(
            {
                ts
                for frame in prepared_frames.values()
                for ts in frame.index
            }
        )
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else all_dates[0]

        cash = float(self.config.initial_cash)
        positions: dict[str, dict] = {}
        trades: list[dict] = []
        equity_curve: list[dict] = []
        daily_state: list[dict] = []
        running_peak = float(self.config.initial_cash)
        max_drawdown = 0.0

        for trade_date in all_dates:
            regime_label, regime_ok, target_exposure = self._regime_state(
                regime_frame, trade_date
            )
            cash_ref = {"cash": cash}

            equity_before = cash_ref["cash"] + self._portfolio_market_value(
                positions, prepared_frames, trade_date
            )
            running_peak = max(running_peak, equity_before)
            max_drawdown = max(max_drawdown, (running_peak - equity_before) / running_peak)

            self._process_existing_positions(
                trade_date=trade_date,
                positions=positions,
                prepared_frames=prepared_frames,
                regime_frame=regime_frame,
                regime_ok=regime_ok,
                regime_label=regime_label,
                trades=trades,
                cash_ref=cash_ref,
            )
            cash = float(cash_ref["cash"])

            if trade_date >= trade_start_ts:
                self._process_new_entries(
                    trade_date=trade_date,
                    positions=positions,
                    prepared_frames=prepared_frames,
                    regime_ok=regime_ok,
                    regime_label=regime_label,
                    target_exposure=target_exposure,
                    cash_ref=cash_ref,
                )
                cash = float(cash_ref["cash"])

            market_value = self._portfolio_market_value(positions, prepared_frames, trade_date)
            equity = cash + market_value
            exposure = market_value / equity if equity > 0 else 0.0
            equity_curve.append(
                {
                    "trade_date": trade_date.date().isoformat(),
                    "equity": round(equity, 2),
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "exposure": round(exposure, 4),
                }
            )
            daily_state.append(
                {
                    "trade_date": trade_date.date().isoformat(),
                    "regime": regime_label,
                    "regime_confirmed_uptrend": bool(regime_ok),
                    "target_exposure": target_exposure,
                    "actual_exposure": round(exposure, 4),
                    "cash": round(cash, 2),
                    "equity": round(equity, 2),
                    "positions": len(positions),
                    "symbols": ",".join(sorted(positions)),
                }
            )

        if positions:
            final_date = all_dates[-1]
            for symbol in list(positions):
                row = prepared_frames[symbol].loc[final_date]
                price = float(row["close"])
                cash += self._liquidate_position(
                    symbol=symbol,
                    position=positions.pop(symbol),
                    exit_price=price,
                    exit_date=final_date,
                    trades=trades,
                    exit_reason="final_bar",
                )

        equity_curve_df = pd.DataFrame(equity_curve)
        daily_state_df = pd.DataFrame(daily_state)
        trades_df = pd.DataFrame(trades)
        symbol_summary_df = self._build_symbol_summary(trades_df)
        total_return = (cash / self.config.initial_cash) - 1.0 if self.config.initial_cash else 0.0
        benchmark_return = self._portfolio_benchmark_return(benchmark_df, trade_start_ts)
        summary = {
            "start_value": round(float(self.config.initial_cash), 2),
            "end_value": round(float(cash), 2),
            "total_return": round(float(total_return), 4),
            "benchmark_return": round(float(benchmark_return), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "symbols_tested": len(prepared_frames),
            "symbols_with_trades": int(trades_df["symbol"].nunique()) if not trades_df.empty else 0,
            "total_trades": int(len(trades_df)),
            "trade_win_rate": round(float((trades_df["pnl"] > 0).mean()), 4) if not trades_df.empty else 0.0,
            "avg_trade_return": round(float(trades_df["return_pct"].mean()), 4) if not trades_df.empty else 0.0,
            "median_trade_return": round(float(trades_df["return_pct"].median()), 4) if not trades_df.empty else 0.0,
            "avg_active_symbol_return": round(float(symbol_summary_df["total_return"].mean()), 4) if not symbol_summary_df.empty else 0.0,
            "median_active_symbol_return": round(float(symbol_summary_df["total_return"].median()), 4) if not symbol_summary_df.empty else 0.0,
            "positive_active_symbol_ratio": round(float((symbol_summary_df["total_return"] > 0).mean()), 4) if not symbol_summary_df.empty else 0.0,
            "realized_pnl": round(float(trades_df["pnl"].sum()), 2) if not trades_df.empty else 0.0,
            "trade_start_date": trade_start_ts.date().isoformat(),
            "end_date": all_dates[-1].date().isoformat(),
        }
        return PortfolioBacktestResult(
            summary=summary,
            trades=trades_df,
            equity_curve=equity_curve_df,
            daily_state=daily_state_df,
            symbol_summary=symbol_summary_df,
        )

    def _row_passes_entry(self, row: pd.Series, price: float, regime_ok: bool) -> bool:
        cfg = self.screener.config
        if cfg.require_market_uptrend and not regime_ok:
            return False
        if pd.isna(row.get("canslim_buy_point")) or price < float(row.get("canslim_buy_point")):
            return False
        if (
            pd.notna(row.get("canslim_buy_limit_price"))
            and price > float(row.get("canslim_buy_limit_price"))
        ):
            return False

        price_structure_ok = bool(
            row.get("close", 0) > row.get("sma_50", float("inf")) > row.get("sma_200", float("inf"))
        )
        liquidity_ok = bool(
            pd.notna(row.get("avg_volume_50"))
            and float(row.get("avg_volume_50")) >= cfg.min_avg_volume
            and pd.notna(row.get("avg_dollar_volume_50"))
            and float(row.get("avg_dollar_volume_50")) >= cfg.min_avg_dollar_volume
        )
        close_range_ok = bool(
            pd.notna(row.get("close_range_pct"))
            and float(row.get("close_range_pct")) >= cfg.min_close_range_pct
        )
        momentum_ok = bool(
            pd.notna(row.get("roc_60"))
            and float(row.get("roc_60")) > 0
            and pd.notna(row.get("roc_120"))
            and float(row.get("roc_120")) > 0
        )
        new_leader_ok = bool(row.get("near_52w_high")) and bool(row.get("new_high_signal"))
        volume_ok = bool(row.get("canslim_volume_signal")) or bool(row.get("breakout_signal"))
        fundamental_ok = bool(row.get("canslim_fundamental_ok"))

        return all(
            [
                price_structure_ok,
                liquidity_ok,
                close_range_ok,
                momentum_ok,
                new_leader_ok,
                volume_ok,
                fundamental_ok,
            ]
        )

    def _row_supports_pyramiding(self, row: pd.Series, price: float) -> bool:
        if pd.notna(row.get("ema_21")) and price < float(row["ema_21"]):
            return False
        if pd.notna(row.get("close_range_pct")) and float(row["close_range_pct"]) < self.screener.config.min_close_range_pct:
            return False
        return bool(row.get("new_high_signal")) or bool(row.get("breakout_signal"))

    def _process_new_entries(
        self,
        trade_date,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        regime_ok: bool,
        regime_label: str,
        target_exposure: float,
        cash_ref: dict[str, float],
    ) -> None:
        if regime_label == "market_correction" and not self.config.allow_new_entries_in_correction:
            return

        candidates = []
        for symbol, frame in prepared_frames.items():
            if symbol in positions or trade_date not in frame.index:
                continue
            row = frame.loc[trade_date]
            price = float(row["close"])
            if not self._row_passes_entry(row, price, regime_ok):
                continue
            candidates.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "row": row,
                    "rank": (
                        float(row.get("canslim_score") or 0.0),
                        float(row.get("current_eps_yoy_growth") or -999.0),
                        float(row.get("annual_eps_growth") or -999.0),
                        float(row.get("roc_120") or -999.0),
                        float(row.get("roc_60") or -999.0),
                        float(row.get("breakout_volume_ratio") or 0.0),
                        float(row.get("close_range_pct") or 0.0),
                    ),
                }
            )

        candidates.sort(key=lambda item: item["rank"], reverse=True)
        for candidate in candidates:
            if len(positions) >= self.config.max_positions:
                break

            portfolio_value = cash_ref["cash"] + self._portfolio_market_value(
                positions, prepared_frames, trade_date
            )
            market_value = self._portfolio_market_value(positions, prepared_frames, trade_date)
            current_exposure = market_value / portfolio_value if portfolio_value > 0 else 0.0
            if current_exposure >= target_exposure:
                break

            price = candidate["price"]
            row = candidate["row"]
            stop_candidate = row.get("initial_stop_price")
            stop_reference = (
                float(stop_candidate)
                if pd.notna(stop_candidate) and float(stop_candidate) < price
                else price * (1.0 - self.config.stop_loss_pct)
            )
            risk_per_share = price - stop_reference
            if risk_per_share <= 0:
                continue

            per_position_cap = portfolio_value * self.config.max_position_pct
            remaining_to_target = max(0.0, (target_exposure * portfolio_value) - market_value)
            budget = min(per_position_cap, remaining_to_target, cash_ref["cash"])
            if budget <= 0:
                continue
            risk_budget = portfolio_value * self.config.risk_per_trade * self.config.initial_entry_fraction
            qty = min(
                int((budget * self.config.initial_entry_fraction) / price),
                int(risk_budget / risk_per_share),
            )
            if qty <= 0:
                continue

            cash_ref["cash"] -= qty * price
            positions[candidate["symbol"]] = {
                "lots": [
                    {
                        "entry_price": price,
                        "shares": qty,
                        "entry_date": trade_date,
                        "leg_type": "initial",
                    }
                ],
                "entry_date": trade_date,
                "highest_close": price,
                "stop_price": stop_reference,
                "partial_taken": False,
                "add_on_1_done": False,
                "add_on_2_done": False,
            }
