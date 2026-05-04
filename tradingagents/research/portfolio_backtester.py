"""Portfolio-level Minervini backtesting with shared capital and exposure scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .backtester import BacktestConfig, MinerviniBacktester
from .minervini import MinerviniScreener


@dataclass
class PortfolioBacktestResult:
    summary: Dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    daily_state: pd.DataFrame
    symbol_summary: pd.DataFrame


class PortfolioMinerviniBacktester(MinerviniBacktester):
    """Shared-capital SEPA-style backtester."""

    def __init__(
        self,
        screener: MinerviniScreener | None = None,
        config: BacktestConfig | None = None,
    ):
        super().__init__(screener=screener, config=config)

    def backtest_portfolio(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> PortfolioBacktestResult:
        prepared_frames = {}
        for symbol, df in data_by_symbol.items():
            prepared = self.screener.prepare_features(df)
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

            equity_before = cash_ref["cash"] + self._portfolio_market_value(positions, prepared_frames, trade_date)
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
            market_score = None
            if (
                not regime_frame.empty
                and trade_date in regime_frame.index
                and "market_score" in regime_frame.columns
            ):
                market_score = regime_frame.loc[trade_date, "market_score"]
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
                    "market_score": (
                        int(market_score) if pd.notna(market_score) else None
                    ),
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

    def _process_existing_positions(
        self,
        trade_date,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        regime_frame: pd.DataFrame,
        regime_ok: bool,
        regime_label: str,
        trades: list[dict],
        cash_ref: dict[str, float],
    ) -> None:
        for symbol in list(positions):
            frame = prepared_frames.get(symbol)
            if frame is None or trade_date not in frame.index:
                continue
            row = frame.loc[trade_date]
            price = float(row["close"])
            position = positions[symbol]

            shares = self._total_shares(position["lots"])
            if shares <= 0:
                positions.pop(symbol, None)
                continue

            average_cost = self._average_cost(position["lots"])
            position["highest_close"] = max(position["highest_close"], price)
            hold_days = (trade_date - position["entry_date"]).days

            if price >= average_cost * (1.0 + self.config.breakeven_trigger_pct):
                position["stop_price"] = max(position["stop_price"], average_cost)

            if price >= average_cost * (1.0 + self.config.trailing_lock_trigger_pct_1):
                position["stop_price"] = max(
                    position["stop_price"],
                    average_cost * (1.0 + self.config.trailing_lock_floor_pct_1),
                )
            if price >= average_cost * (1.0 + self.config.trailing_lock_trigger_pct_2):
                position["stop_price"] = max(
                    position["stop_price"],
                    average_cost * (1.0 + self.config.trailing_lock_floor_pct_2),
                )

            if position["partial_taken"] and pd.notna(row.get("ema_21")):
                position["stop_price"] = max(position["stop_price"], float(row["ema_21"]))
            else:
                position["stop_price"] = max(
                    position["stop_price"],
                    position["highest_close"] * (1.0 - self.config.trail_stop_pct),
                )

            if (
                not position["partial_taken"]
                and
                not position["add_on_1_done"]
                and regime_label != "market_correction"
                and not self._market_extended_for_add_on(regime_frame, trade_date)
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_1)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash_ref["cash"],
                    equity=cash_ref["cash"] + self._portfolio_market_value(positions, prepared_frames, trade_date),
                    price=price,
                    stop_price=position["stop_price"],
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_1,
                )
                if added_qty > 0:
                    cash_ref["cash"] -= added_qty * price
                    position["lots"].append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_1",
                        }
                    )
                    position["add_on_1_done"] = True
                    shares = self._total_shares(position["lots"])

            if (
                not position["partial_taken"]
                and
                not position["add_on_2_done"]
                and regime_label != "market_correction"
                and not self._market_extended_for_add_on(regime_frame, trade_date)
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_2)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash_ref["cash"],
                    equity=cash_ref["cash"] + self._portfolio_market_value(positions, prepared_frames, trade_date),
                    price=price,
                    stop_price=position["stop_price"],
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_2,
                )
                if added_qty > 0:
                    cash_ref["cash"] -= added_qty * price
                    position["lots"].append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_2",
                        }
                    )
                    position["add_on_2_done"] = True

            shares = self._total_shares(position["lots"])
            average_cost = self._average_cost(position["lots"])
            if (
                not position["partial_taken"]
                and shares > 0
                and price >= average_cost * (1.0 + self.config.partial_profit_trigger_pct)
            ):
                partial_qty = max(1, int(shares * self.config.partial_profit_fraction))
                partial_qty = min(partial_qty, shares)
                cash_ref["cash"] += partial_qty * price
                trades.extend(
                    self._close_lots(
                        lots=position["lots"],
                        sell_qty=partial_qty,
                        exit_price=price,
                        exit_date=trade_date,
                        symbol=symbol,
                        exit_reason="partial_profit",
                    )
                )
                position["partial_taken"] = True
                if position["lots"]:
                    position["stop_price"] = max(
                        position["stop_price"],
                        self._average_cost(position["lots"]),
                    )

            shares = self._total_shares(position["lots"])
            if shares <= 0:
                positions.pop(symbol, None)
                continue

            exit_reason = None
            if price <= position["stop_price"]:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and regime_label == "market_correction":
                exit_reason = "market_regime"
            elif (
                self.config.use_ema21_exit
                and pd.notna(row.get("ema_21"))
                and hold_days >= 5
                and price < float(row["ema_21"])
            ):
                exit_reason = "lost_21ema"
            elif (
                not self.config.use_ema21_exit
                and pd.notna(row.get("sma_50"))
                and price < float(row["sma_50"])
            ):
                exit_reason = "lost_50dma"
            elif hold_days >= self.config.max_hold_days:
                exit_reason = "time"

            if exit_reason is None:
                continue

            cash_ref["cash"] += self._liquidate_position(
                symbol=symbol,
                position=positions.pop(symbol),
                exit_price=price,
                exit_date=trade_date,
                trades=trades,
                exit_reason=exit_reason,
            )

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
                        int(bool(row.get("breakout_signal"))),
                        float(row.get("template_score") or 0.0),
                        float(row.get("roc_60") or 0.0),
                        float(row.get("roc_120") or 0.0),
                        float(row.get("adx_14") or 0.0),
                        float(row.get("rsi_14") or 0.0),
                        -float(row.get("atr_pct_14") or 0.0),
                        float(row.get("close_range_pct") or 0.0),
                        float(row.get("breakout_volume_ratio") or 0.0),
                        -float(row.get("stage_number") or 99.0),
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
            budget = min(
                per_position_cap,
                remaining_to_target,
                cash_ref["cash"],
            )
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

    def _portfolio_market_value(
        self,
        positions: dict[str, dict],
        prepared_frames: dict[str, pd.DataFrame],
        trade_date,
    ) -> float:
        total = 0.0
        for symbol, position in positions.items():
            frame = prepared_frames.get(symbol)
            if frame is None or trade_date not in frame.index:
                continue
            total += self._total_shares(position["lots"]) * float(frame.loc[trade_date, "close"])
        return total

    def _regime_state(self, regime_frame: pd.DataFrame, trade_date) -> tuple[str, bool, float]:
        if not regime_frame.empty and trade_date in regime_frame.index:
            regime_label = str(regime_frame.loc[trade_date, "market_regime"])
            regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])
        else:
            regime_label = "uptrend_under_pressure"
            regime_ok = False

        exposure_map = {
            "confirmed_uptrend": self.config.target_exposure_confirmed_uptrend,
            "uptrend_under_pressure": self.config.target_exposure_uptrend_under_pressure,
            "market_correction": self.config.target_exposure_market_correction,
        }
        return regime_label, regime_ok, exposure_map.get(
            regime_label,
            self.config.target_exposure_uptrend_under_pressure,
        )

    def _portfolio_benchmark_return(
        self,
        benchmark_df: Optional[pd.DataFrame],
        trade_start_ts: pd.Timestamp,
    ) -> float:
        if benchmark_df is None or benchmark_df.empty:
            return 0.0
        frame = benchmark_df.loc[benchmark_df.index >= trade_start_ts]
        if frame.empty:
            return 0.0
        start_price = float(frame["close"].iloc[0])
        end_price = float(frame["close"].iloc[-1])
        if start_price <= 0:
            return 0.0
        return (end_price / start_price) - 1.0

    def _liquidate_position(
        self,
        symbol: str,
        position: dict,
        exit_price: float,
        exit_date,
        trades: list[dict],
        exit_reason: str,
    ) -> float:
        remaining = self._total_shares(position["lots"])
        trades.extend(
            self._close_lots(
                lots=position["lots"],
                sell_qty=remaining,
                exit_price=exit_price,
                exit_date=exit_date,
                symbol=symbol,
                exit_reason=exit_reason,
            )
        )
        return remaining * exit_price

    def _build_symbol_summary(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "total_trades",
                    "win_rate",
                    "total_pnl",
                    "total_return",
                    "avg_trade_return",
                    "avg_hold_days",
                ]
            )
        grouped = trades_df.groupby("symbol").agg(
            total_trades=("symbol", "size"),
            win_rate=("pnl", lambda s: float((s > 0).mean())),
            total_pnl=("pnl", "sum"),
            avg_trade_return=("return_pct", "mean"),
            avg_hold_days=("hold_days", "mean"),
        )
        grouped["total_return"] = grouped["total_pnl"] / float(self.config.initial_cash)
        return grouped.reset_index().sort_values("total_return", ascending=False)
