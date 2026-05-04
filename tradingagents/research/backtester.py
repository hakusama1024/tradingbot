"""Rule-based Minervini-style breakout backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .minervini import MinerviniScreener


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    max_position_pct: float = 0.10
    risk_per_trade: float = 0.01
    stop_loss_pct: float = 0.08
    trail_stop_pct: float = 0.10
    max_hold_days: int = 60
    min_template_score: int = 8
    require_volume_surge: bool = True
    require_base_pattern: bool = True
    require_market_regime: bool = True
    exit_on_market_correction: bool = False
    progressive_entries: bool = False
    initial_entry_fraction: float = 1.0
    add_on_trigger_pct_1: float = 0.025
    add_on_trigger_pct_2: float = 0.05
    add_on_fraction_1: float = 0.30
    add_on_fraction_2: float = 0.20
    breakeven_trigger_pct: float = 0.05
    trailing_lock_trigger_pct_1: float = 0.12
    trailing_lock_floor_pct_1: float = 0.03
    trailing_lock_trigger_pct_2: float = 0.20
    trailing_lock_floor_pct_2: float = 0.08
    partial_profit_trigger_pct: float = 0.12
    partial_profit_fraction: float = 0.33
    use_ema21_exit: bool = False
    use_close_range_filter: bool = False
    min_close_range_pct: float = 0.60
    scale_exposure_in_weak_market: bool = False
    weak_market_position_scale: float = 0.50
    target_exposure_confirmed_uptrend: float = 1.00
    target_exposure_uptrend_under_pressure: float = 0.60
    target_exposure_market_correction: float = 0.00
    allow_new_entries_in_correction: bool = False
    max_positions: int = 6
    min_rsi_14: float = 0.0
    min_adx_14: float = 0.0
    max_atr_pct_14: float = 1.0
    min_roc_20: float = -1.0
    min_roc_60: float = -1.0
    min_roc_120: float = -1.0
    market_extension_filter_enabled: bool = False
    market_extension_max_qqq_above_ema21_pct: float = 0.05
    market_extension_max_qqq_roc_5: float = 0.05


class MinerviniBacktester:
    """Backtest an approximate SEPA breakout workflow on daily data."""

    def __init__(
        self,
        screener: MinerviniScreener | None = None,
        config: BacktestConfig | None = None,
    ):
        self.screener = screener or MinerviniScreener()
        self.config = config or BacktestConfig()

    def _build_regime_frame(
        self,
        benchmark_df: Optional[pd.DataFrame],
        market_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if (
            market_context_df is not None
            and not market_context_df.empty
            and {"market_confirmed_uptrend", "market_regime"}.issubset(market_context_df.columns)
        ):
            columns = ["market_confirmed_uptrend", "market_regime"]
            if "market_score" in market_context_df.columns:
                columns.append("market_score")
            if "qqq_extension_pct" in market_context_df.columns:
                columns.append("qqq_extension_pct")
            if "qqq_roc_5" in market_context_df.columns:
                columns.append("qqq_roc_5")
            return market_context_df[columns].copy().sort_index()

        if benchmark_df is None or benchmark_df.empty:
            return pd.DataFrame()

        prepared = self.screener.prepare_features(benchmark_df)
        if prepared.empty:
            return prepared

        prepared["market_confirmed_uptrend"] = (
            (prepared["close"] > prepared["sma_50"])
            & (prepared["close"] > prepared["sma_200"])
            & (prepared["sma_200"] > prepared["sma_200_20d_ago"])
        )
        prepared["market_regime"] = "market_correction"
        prepared.loc[
            prepared["close"] > prepared["sma_200"],
            "market_regime",
        ] = "uptrend_under_pressure"
        prepared.loc[
            prepared["market_confirmed_uptrend"],
            "market_regime",
        ] = "confirmed_uptrend"
        return prepared[["market_confirmed_uptrend", "market_regime"]]

    def backtest_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        if self.config.progressive_entries:
            return self._backtest_symbol_progressive(
                symbol,
                df,
                benchmark_df=benchmark_df,
                market_context_df=market_context_df,
                trade_start_date=trade_start_date,
            )

        features = self.screener.prepare_features(df)
        if features.empty:
            return {"symbol": symbol, "error": "No data available"}
        regime_frame = self._build_regime_frame(benchmark_df, market_context_df=market_context_df)
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else None

        cash = self.config.initial_cash
        shares = 0
        entry_price = 0.0
        entry_date = None
        highest_close = 0.0
        stop_price = 0.0
        trades = []
        equity_curve = []

        running_peak = self.config.initial_cash
        max_drawdown = 0.0

        for trade_date, row in features.iterrows():
            price = float(row["close"])
            if price <= 0:
                continue
            regime_ok = True
            if not regime_frame.empty and trade_date in regime_frame.index:
                regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])

            position_value = shares * price
            equity = cash + position_value
            running_peak = max(running_peak, equity)
            max_drawdown = max(max_drawdown, (running_peak - equity) / running_peak)
            equity_curve.append({"trade_date": trade_date, "equity": equity})

            if trade_start_ts is not None and trade_date < trade_start_ts:
                continue

            if shares == 0:
                template_score = self._template_score(row)
                volume_ok = (
                    (not self.config.require_volume_surge)
                    or bool(row.get("breakout_signal"))
                )
                has_base = (
                    (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
                    or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
                )
                base_ok = (
                    (not self.config.require_base_pattern)
                    or has_base
                )
                setup_ready = (
                    True
                    if pd.isna(row.get("breakout_ready"))
                    else bool(row.get("breakout_ready"))
                )
                stage_ok = (
                    pd.notna(row.get("stage_number"))
                    and float(row.get("stage_number")) <= self.screener.config.max_stage_number
                )
                buy_point = row.get("buy_point")
                buy_limit_price = row.get("buy_limit_price")
                stop_candidate = row.get("initial_stop_price")
                if (
                    template_score < self.config.min_template_score
                    or not volume_ok
                    or not base_ok
                    or not setup_ready
                    or not stage_ok
                    or (self.config.require_market_regime and not regime_ok)
                ):
                    continue

                if pd.isna(buy_point) or price < float(buy_point):
                    continue
                if pd.notna(buy_limit_price) and price > float(buy_limit_price):
                    continue

                if pd.notna(stop_candidate) and float(stop_candidate) < price:
                    risk_per_share = price - float(stop_candidate)
                else:
                    risk_per_share = price * self.config.stop_loss_pct
                if risk_per_share <= 0:
                    continue

                capital_cap = cash * self.config.max_position_pct
                qty_by_capital = int(capital_cap / price)
                qty_by_risk = int((cash * self.config.risk_per_trade) / risk_per_share)
                shares = min(qty_by_capital, qty_by_risk)
                if shares <= 0:
                    shares = 0
                    continue

                entry_price = price
                entry_date = trade_date
                cash -= shares * entry_price
                highest_close = entry_price
                stop_price = (
                    float(stop_candidate)
                    if pd.notna(stop_candidate) and float(stop_candidate) < entry_price
                    else entry_price * (1.0 - self.config.stop_loss_pct)
                )
                continue

            highest_close = max(highest_close, price)
            stop_price = max(stop_price, highest_close * (1.0 - self.config.trail_stop_pct))
            hold_days = (trade_date - entry_date).days if entry_date is not None else 0

            exit_reason = None
            if price <= stop_price:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and not regime_ok:
                exit_reason = "market_regime"
            elif pd.notna(row.get("sma_50")) and price < float(row["sma_50"]):
                exit_reason = "lost_50dma"
            elif hold_days >= self.config.max_hold_days:
                exit_reason = "time"

            if exit_reason is None:
                continue

            cash += shares * price
            pnl = (price - entry_price) * shares
            return_pct = (price / entry_price) - 1.0 if entry_price > 0 else 0.0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else None,
                    "exit_date": trade_date.date().isoformat(),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return_pct": round(return_pct, 4),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                }
            )
            shares = 0
            entry_price = 0.0
            entry_date = None
            highest_close = 0.0
            stop_price = 0.0

        if shares > 0 and not features.empty:
            last_date = features.index[-1]
            last_price = float(features["close"].iloc[-1])
            cash += shares * last_price
            pnl = (last_price - entry_price) * shares
            return_pct = (last_price / entry_price) - 1.0 if entry_price > 0 else 0.0
            hold_days = (last_date - entry_date).days if entry_date is not None else 0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat() if entry_date is not None else None,
                    "exit_date": last_date.date().isoformat(),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(last_price, 2),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return_pct": round(return_pct, 4),
                    "hold_days": hold_days,
                    "exit_reason": "final_bar",
                }
            )

        final_value = cash
        benchmark_return = 0.0
        if not features.empty and float(features["close"].iloc[0]) > 0:
            benchmark_return = (float(features["close"].iloc[-1]) / float(features["close"].iloc[0])) - 1.0

        return {
            "symbol": symbol,
            "start_date": features.index[0].date().isoformat(),
            "end_date": features.index[-1].date().isoformat(),
            "trade_start_date": trade_start_ts.date().isoformat() if trade_start_ts is not None else features.index[0].date().isoformat(),
            "start_value": self.config.initial_cash,
            "end_value": round(final_value, 2),
            "total_return": round((final_value / self.config.initial_cash) - 1.0, 4),
            "benchmark_return": round(benchmark_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_trades": len(trades),
            "win_rate": round(
                sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                4,
            )
            if trades
            else 0.0,
            "profit_factor": self._profit_factor(trades),
            "avg_hold_days": round(
                sum(t["hold_days"] for t in trades) / len(trades),
                2,
            )
            if trades
            else 0.0,
            "regime_filter": self.config.require_market_regime,
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve),
        }

    def _backtest_symbol_progressive(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        market_context_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        features = self.screener.prepare_features(df)
        if features.empty:
            return {"symbol": symbol, "error": "No data available"}
        regime_frame = self._build_regime_frame(benchmark_df, market_context_df=market_context_df)
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date else None

        cash = self.config.initial_cash
        lots: list[dict] = []
        trades = []
        equity_curve = []
        highest_close = 0.0
        stop_price = 0.0
        partial_taken = False
        add_on_1_done = False
        add_on_2_done = False
        running_peak = self.config.initial_cash
        max_drawdown = 0.0

        for trade_date, row in features.iterrows():
            price = float(row["close"])
            if price <= 0:
                continue
            regime_ok = True
            if not regime_frame.empty and trade_date in regime_frame.index:
                regime_ok = bool(regime_frame.loc[trade_date, "market_confirmed_uptrend"])

            shares = self._total_shares(lots)
            equity = cash + (shares * price)
            running_peak = max(running_peak, equity)
            max_drawdown = max(max_drawdown, (running_peak - equity) / running_peak)
            equity_curve.append({"trade_date": trade_date, "equity": equity})

            if trade_start_ts is not None and trade_date < trade_start_ts:
                continue

            if shares <= 0:
                if not self._row_passes_entry(row, price, regime_ok):
                    continue

                stop_candidate = row.get("initial_stop_price")
                stop_reference = (
                    float(stop_candidate)
                    if pd.notna(stop_candidate) and float(stop_candidate) < price
                    else price * (1.0 - self.config.stop_loss_pct)
                )
                risk_per_share = price - stop_reference
                if risk_per_share <= 0:
                    continue

                regime_scale = self._regime_position_scale(regime_ok)
                position_cap_value = equity * self.config.max_position_pct * regime_scale
                initial_target_value = position_cap_value * self.config.initial_entry_fraction
                risk_budget = equity * self.config.risk_per_trade * self.config.initial_entry_fraction
                qty = min(
                    int(min(initial_target_value, cash) / price),
                    int(risk_budget / risk_per_share),
                )
                if qty <= 0:
                    continue

                cash -= qty * price
                lots.append(
                    {
                        "entry_price": price,
                        "shares": qty,
                        "entry_date": trade_date,
                        "leg_type": "initial",
                    }
                )
                highest_close = price
                stop_price = stop_reference
                partial_taken = False
                add_on_1_done = False
                add_on_2_done = False
                continue

            average_cost = self._average_cost(lots)
            shares = self._total_shares(lots)
            highest_close = max(highest_close, price)
            hold_days = (trade_date - lots[0]["entry_date"]).days if lots else 0

            if price >= average_cost * (1.0 + self.config.breakeven_trigger_pct):
                stop_price = max(stop_price, average_cost)

            if partial_taken and pd.notna(row.get("ema_21")):
                stop_price = max(stop_price, float(row["ema_21"]))
            else:
                stop_price = max(stop_price, highest_close * (1.0 - self.config.trail_stop_pct))

            if (
                not add_on_1_done
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_1)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash,
                    equity=equity,
                    price=price,
                    stop_price=stop_price,
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_1,
                )
                if added_qty > 0:
                    cash -= added_qty * price
                    lots.append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_1",
                        }
                    )
                    add_on_1_done = True
                    average_cost = self._average_cost(lots)
                    shares = self._total_shares(lots)

            if (
                not add_on_2_done
                and price >= average_cost * (1.0 + self.config.add_on_trigger_pct_2)
                and self._row_supports_pyramiding(row, price)
            ):
                added_qty = self._add_on_qty(
                    cash=cash,
                    equity=equity,
                    price=price,
                    stop_price=stop_price,
                    current_shares=shares,
                    regime_ok=regime_ok,
                    add_fraction=self.config.add_on_fraction_2,
                )
                if added_qty > 0:
                    cash -= added_qty * price
                    lots.append(
                        {
                            "entry_price": price,
                            "shares": added_qty,
                            "entry_date": trade_date,
                            "leg_type": "add_on_2",
                        }
                    )
                    add_on_2_done = True
                    average_cost = self._average_cost(lots)
                    shares = self._total_shares(lots)

            shares = self._total_shares(lots)
            average_cost = self._average_cost(lots)
            if (
                not partial_taken
                and shares > 0
                and price >= average_cost * (1.0 + self.config.partial_profit_trigger_pct)
            ):
                partial_qty = max(1, int(shares * self.config.partial_profit_fraction))
                partial_qty = min(partial_qty, shares)
                cash += partial_qty * price
                trades.extend(
                    self._close_lots(
                        lots=lots,
                        sell_qty=partial_qty,
                        exit_price=price,
                        exit_date=trade_date,
                        symbol=symbol,
                        exit_reason="partial_profit",
                    )
                )
                partial_taken = True
                if lots:
                    stop_price = max(stop_price, self._average_cost(lots))
                shares = self._total_shares(lots)

            exit_reason = None
            if shares <= 0:
                continue
            if price <= stop_price:
                exit_reason = "stop"
            elif self.config.exit_on_market_correction and not regime_ok:
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

            remaining = self._total_shares(lots)
            cash += remaining * price
            trades.extend(
                self._close_lots(
                    lots=lots,
                    sell_qty=remaining,
                    exit_price=price,
                    exit_date=trade_date,
                    symbol=symbol,
                    exit_reason=exit_reason,
                )
            )
            highest_close = 0.0
            stop_price = 0.0
            partial_taken = False
            add_on_1_done = False
            add_on_2_done = False

        if lots and not features.empty:
            last_date = features.index[-1]
            last_price = float(features["close"].iloc[-1])
            remaining = self._total_shares(lots)
            cash += remaining * last_price
            trades.extend(
                self._close_lots(
                    lots=lots,
                    sell_qty=remaining,
                    exit_price=last_price,
                    exit_date=last_date,
                    symbol=symbol,
                    exit_reason="final_bar",
                )
            )

        final_value = cash
        benchmark_return = 0.0
        if not features.empty and float(features["close"].iloc[0]) > 0:
            benchmark_return = (float(features["close"].iloc[-1]) / float(features["close"].iloc[0])) - 1.0

        return {
            "symbol": symbol,
            "start_date": features.index[0].date().isoformat(),
            "end_date": features.index[-1].date().isoformat(),
            "trade_start_date": trade_start_ts.date().isoformat() if trade_start_ts is not None else features.index[0].date().isoformat(),
            "start_value": self.config.initial_cash,
            "end_value": round(final_value, 2),
            "total_return": round((final_value / self.config.initial_cash) - 1.0, 4),
            "benchmark_return": round(benchmark_return, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_trades": len(trades),
            "win_rate": round(
                sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                4,
            )
            if trades
            else 0.0,
            "profit_factor": self._profit_factor(trades),
            "avg_hold_days": round(
                sum(t["hold_days"] for t in trades) / len(trades),
                2,
            )
            if trades
            else 0.0,
            "regime_filter": self.config.require_market_regime,
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve),
        }

    def backtest_universe(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None,
        trade_start_date: Optional[str] = None,
    ) -> Dict:
        results = []
        all_trades = []
        for symbol, df in data_by_symbol.items():
            result = self.backtest_symbol(
                symbol,
                df,
                benchmark_df=benchmark_df,
                trade_start_date=trade_start_date,
            )
            results.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"trades", "equity_curve"}
                }
            )
            all_trades.extend(result.get("trades", []))

        summary = pd.DataFrame(results)
        trades = pd.DataFrame(all_trades)
        portfolio_summary = {
            "symbols_tested": len(summary),
            "avg_return": round(float(summary["total_return"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_benchmark_return": round(float(summary["benchmark_return"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_win_rate": round(float(summary["win_rate"].mean()), 4)
            if not summary.empty
            else 0.0,
            "avg_max_drawdown": round(float(summary["max_drawdown"].mean()), 4)
            if not summary.empty
            else 0.0,
            "total_trades": int(summary["total_trades"].sum()) if not summary.empty else 0,
        }
        return {
            "summary": summary.sort_values("total_return", ascending=False)
            if not summary.empty
            else summary,
            "trades": trades,
            "portfolio_summary": portfolio_summary,
        }

    @staticmethod
    def _profit_factor(trades: list[dict]) -> float:
        gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return round(gross_win / gross_loss, 4)

    def _row_passes_entry(self, row: pd.Series, price: float, regime_ok: bool) -> bool:
        template_score = self._template_score(row)
        volume_ok = (not self.config.require_volume_surge) or bool(row.get("breakout_signal"))
        has_base = (
            (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
            or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
        )
        base_ok = (not self.config.require_base_pattern) or has_base
        setup_ready = True if pd.isna(row.get("breakout_ready")) else bool(row.get("breakout_ready"))
        stage_ok = (
            pd.notna(row.get("stage_number"))
            and float(row.get("stage_number")) <= self.screener.config.max_stage_number
        )
        buy_point = row.get("buy_point")
        buy_limit_price = row.get("buy_limit_price")
        close_range_ok = (
            (not self.config.use_close_range_filter)
            or (
                pd.notna(row.get("close_range_pct"))
                and float(row["close_range_pct"]) >= self.config.min_close_range_pct
            )
        )
        rsi_ok = (
            self.config.min_rsi_14 <= 0
            or (
                pd.notna(row.get("rsi_14"))
                and float(row["rsi_14"]) >= self.config.min_rsi_14
            )
        )
        adx_ok = (
            self.config.min_adx_14 <= 0
            or (
                pd.notna(row.get("adx_14"))
                and float(row["adx_14"]) >= self.config.min_adx_14
            )
        )
        atr_ok = (
            self.config.max_atr_pct_14 >= 1.0
            or (
                pd.notna(row.get("atr_pct_14"))
                and float(row["atr_pct_14"]) <= self.config.max_atr_pct_14
            )
        )
        roc_20_ok = (
            self.config.min_roc_20 <= -1.0
            or (
                pd.notna(row.get("roc_20"))
                and float(row["roc_20"]) >= self.config.min_roc_20
            )
        )
        roc_60_ok = (
            self.config.min_roc_60 <= -1.0
            or (
                pd.notna(row.get("roc_60"))
                and float(row["roc_60"]) >= self.config.min_roc_60
            )
        )
        roc_120_ok = (
            self.config.min_roc_120 <= -1.0
            or (
                pd.notna(row.get("roc_120"))
                and float(row["roc_120"]) >= self.config.min_roc_120
            )
        )
        if (
            template_score < self.config.min_template_score
            or not volume_ok
            or not base_ok
            or not setup_ready
            or not stage_ok
            or not close_range_ok
            or not rsi_ok
            or not adx_ok
            or not atr_ok
            or not roc_20_ok
            or not roc_60_ok
            or not roc_120_ok
            or (self.config.require_market_regime and not regime_ok)
        ):
            return False
        if pd.isna(buy_point) or price < float(buy_point):
            return False
        if pd.notna(buy_limit_price) and price > float(buy_limit_price):
            return False
        return True

    def _row_supports_pyramiding(self, row: pd.Series, price: float) -> bool:
        if pd.notna(row.get("ema_21")) and price < float(row["ema_21"]):
            return False
        if self.config.use_close_range_filter and pd.notna(row.get("close_range_pct")):
            if float(row["close_range_pct"]) < self.config.min_close_range_pct:
                return False
        return bool(row.get("breakout_ready")) or bool(row.get("breakout_signal"))

    def _market_extended_for_add_on(
        self,
        regime_frame: pd.DataFrame,
        trade_date,
    ) -> bool:
        if not self.config.market_extension_filter_enabled:
            return False
        if regime_frame.empty or trade_date not in regime_frame.index:
            return False

        row = regime_frame.loc[trade_date]
        extension_pct = row.get("qqq_extension_pct")
        roc_5 = row.get("qqq_roc_5")
        extension_hot = (
            pd.notna(extension_pct)
            and float(extension_pct) >= self.config.market_extension_max_qqq_above_ema21_pct
        )
        momentum_hot = (
            pd.notna(roc_5)
            and float(roc_5) >= self.config.market_extension_max_qqq_roc_5
        )
        if pd.notna(extension_pct) and pd.notna(roc_5):
            return extension_hot and momentum_hot
        if pd.notna(extension_pct):
            return extension_hot
        if pd.notna(roc_5):
            return momentum_hot
        return False

    def _regime_position_scale(self, regime_ok: bool) -> float:
        if self.config.scale_exposure_in_weak_market and not regime_ok:
            return self.config.weak_market_position_scale
        return 1.0

    def _add_on_qty(
        self,
        cash: float,
        equity: float,
        price: float,
        stop_price: float,
        current_shares: int,
        regime_ok: bool,
        add_fraction: float,
    ) -> int:
        risk_per_share = max(price - stop_price, 0.0)
        if risk_per_share <= 0:
            return 0
        regime_scale = self._regime_position_scale(regime_ok)
        max_position_value = equity * self.config.max_position_pct * regime_scale
        current_value = current_shares * price
        remaining_capacity = max(0.0, max_position_value - current_value)
        target_value = min(max_position_value * add_fraction, remaining_capacity, cash)
        risk_budget = equity * self.config.risk_per_trade * add_fraction
        return min(
            int(target_value / price),
            int(risk_budget / risk_per_share),
        )

    @staticmethod
    def _total_shares(lots: list[dict]) -> int:
        return int(sum(lot["shares"] for lot in lots))

    @staticmethod
    def _average_cost(lots: list[dict]) -> float:
        total_shares = sum(lot["shares"] for lot in lots)
        if total_shares <= 0:
            return 0.0
        total_cost = sum(lot["shares"] * lot["entry_price"] for lot in lots)
        return total_cost / total_shares

    @staticmethod
    def _close_lots(
        lots: list[dict],
        sell_qty: int,
        exit_price: float,
        exit_date,
        symbol: str,
        exit_reason: str,
    ) -> list[dict]:
        realized = []
        remaining = sell_qty
        while remaining > 0 and lots:
            lot = lots[0]
            lot_qty = min(remaining, int(lot["shares"]))
            pnl = (exit_price - lot["entry_price"]) * lot_qty
            return_pct = (exit_price / lot["entry_price"]) - 1.0 if lot["entry_price"] > 0 else 0.0
            hold_days = (exit_date - lot["entry_date"]).days
            realized.append(
                {
                    "symbol": symbol,
                    "entry_date": lot["entry_date"].date().isoformat(),
                    "exit_date": exit_date.date().isoformat(),
                    "entry_price": round(float(lot["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": lot_qty,
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float(return_pct), 4),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                    "leg_type": lot.get("leg_type", "initial"),
                }
            )
            lot["shares"] -= lot_qty
            remaining -= lot_qty
            if lot["shares"] <= 0:
                lots.pop(0)
        return realized

    @staticmethod
    def _template_score(row: pd.Series) -> int:
        score = 0
        score += int(row.get("close", 0) > row.get("sma_150", float("inf")))
        score += int(row.get("close", 0) > row.get("sma_200", float("inf")))
        score += int(row.get("sma_150", 0) > row.get("sma_200", float("inf")))
        score += int(row.get("sma_200", 0) > row.get("sma_200_20d_ago", float("inf")))
        score += int(
            row.get("sma_50", 0)
            > row.get("sma_150", float("inf"))
            > row.get("sma_200", float("inf"))
        )
        score += int(row.get("close", 0) > row.get("sma_50", float("inf")))
        score += int(
            row.get("close", 0)
            >= row.get("52w_low", float("inf")) * 1.30
        )
        score += int(
            row.get("close", 0)
            >= row.get("52w_high", float("inf")) * 0.75
        )
        score += int(row.get("avg_volume_50", 0) >= 200_000)
        score += int(
            (pd.notna(row.get("base_candidate")) and bool(row.get("base_candidate")))
            or (pd.notna(row.get("vcp_candidate")) and bool(row.get("vcp_candidate")))
        )
        score += int(
            pd.notna(row.get("stage_number")) and float(row.get("stage_number")) <= 2
        )
        score += int(
            pd.notna(row.get("breakout_ready")) and bool(row.get("breakout_ready"))
        )
        return score
