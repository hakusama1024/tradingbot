"""Pre-screening layer: use local technical indicators to decide if a stock
needs full LLM analysis, or can be handled by rules alone.

This avoids calling the LLM API for stocks that have no actionable signal,
cutting ~70-80% of API usage on quiet market days.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd
import duckdb
from stockstats import wrap

from tradingagents.research import (
    BroadMarketConfig,
    BroadMarketScreener,
    CANSLIMConfig,
    CANSLIMScreener,
    MarketDataWarehouse,
    MinerviniConfig,
    MinerviniScreener,
)
from tradingagents.storage.database import TradingDatabase

logger = logging.getLogger(__name__)


class ScreenResult(Enum):
    """What the pre-screener recommends."""
    SKIP = "skip"              # No signal — don't waste an LLM call
    RULE_BUY = "rule_buy"      # Strong technical BUY — can execute without LLM
    RULE_SELL = "rule_sell"    # Strong technical SELL — can execute without LLM
    NEEDS_LLM = "needs_llm"   # Ambiguous — worth running full LLM analysis


@dataclass
class PreScreenSignal:
    """Output of the pre-screener."""
    result: ScreenResult
    score: int                # -6 to +6, negative = bearish, positive = bullish
    confidence: float         # 0.0 to 1.0
    reasons: list             # Human-readable reasons
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass
class MinerviniPreflight:
    """Daily screen results used to gate new swing entries."""

    trade_date: str
    market_regime: str
    confirmed_uptrend: bool
    approved_symbols: list
    blocked_symbols: list
    screened_symbols: list
    screen_df: pd.DataFrame
    coarse_candidates_path: Optional[str] = None
    screen_path: Optional[str] = None
    regime_path: Optional[str] = None
    fundamentals_path: Optional[str] = None


class TechnicalPreScreener:
    """Scores stocks using local indicators — zero API calls.

    Scoring system (-6 to +6):
      +1  EMA9 > EMA21 (short-term uptrend)
      +1  Price > EMA50 (medium-term uptrend)
      +1  RSI 35-65 (healthy, not overbought)
      +1  MACD > Signal (bullish momentum)
      +1  Price > lower Bollinger (not in freefall)
      +1  Volume > 20-day average (confirmation)

    Negative scoring mirrors the above for bearish signals.

    Thresholds:
      score >= 5  → RULE_BUY (strong enough to skip LLM)
      score <= -4 → RULE_SELL
      |score| <= 1 → SKIP (no signal, don't bother LLM)
      else        → NEEDS_LLM (ambiguous, worth analyzing)
    """

    def __init__(
        self,
        buy_threshold: int = 5,
        sell_threshold: int = -4,
        skip_threshold: int = 1,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.skip_threshold = skip_threshold

    def screen(self, symbol: str, lookback_days: int = 365) -> PreScreenSignal:
        """Screen a single stock. Returns PreScreenSignal with recommendation."""
        try:
            df = self._get_data(symbol, lookback_days)
            if df is None or len(df) < 60:
                return PreScreenSignal(
                    result=ScreenResult.SKIP, score=0, confidence=0.0,
                    reasons=["Insufficient data"],
                )
            return self._score(df, symbol)
        except Exception as e:
            logger.error(f"PreScreener error for {symbol}: {e}")
            return PreScreenSignal(
                result=ScreenResult.NEEDS_LLM, score=0, confidence=0.5,
                reasons=[f"Screening error: {e}"],
            )

    def _get_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Download and prepare data using yfinance."""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=lookback_days)
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _score(self, df: pd.DataFrame, symbol: str) -> PreScreenSignal:
        """Compute a buy/sell score from technical indicators."""
        ss = wrap(df.copy())

        # Current values
        close = float(df["Close"].iloc[-1])
        reasons = []
        score = 0

        # 1. EMA crossover
        try:
            ema9 = float(ss["close_9_ema"].iloc[-1])
            ema21 = float(ss["close_21_ema"].iloc[-1])
            if ema9 > ema21:
                score += 1
                reasons.append(f"EMA9({ema9:.1f}) > EMA21({ema21:.1f})")
            else:
                score -= 1
                reasons.append(f"EMA9({ema9:.1f}) < EMA21({ema21:.1f})")
        except Exception:
            pass

        # 2. Price vs EMA50 (trend)
        try:
            ema50 = float(ss["close_50_ema"].iloc[-1])
            if close > ema50:
                score += 1
                reasons.append(f"Price({close:.1f}) > EMA50({ema50:.1f})")
            else:
                score -= 1
                reasons.append(f"Price({close:.1f}) < EMA50({ema50:.1f})")
        except Exception:
            pass

        # 3. RSI
        try:
            rsi = float(ss["rsi_14"].iloc[-1])
            if 35 < rsi < 65:
                score += 1
                reasons.append(f"RSI({rsi:.1f}) in healthy range")
            elif rsi >= 75:
                score -= 1
                reasons.append(f"RSI({rsi:.1f}) overbought")
            elif rsi <= 25:
                score += 1  # Oversold = potential bounce
                reasons.append(f"RSI({rsi:.1f}) oversold (bounce potential)")
            else:
                reasons.append(f"RSI({rsi:.1f}) neutral")
        except Exception:
            pass

        # 4. MACD
        try:
            macd = float(ss["macd"].iloc[-1])
            macds = float(ss["macds"].iloc[-1])
            if macd > macds:
                score += 1
                reasons.append("MACD bullish")
            else:
                score -= 1
                reasons.append("MACD bearish")
        except Exception:
            pass

        # 5. Bollinger Bands position
        try:
            boll_ub = float(ss["boll_ub"].iloc[-1])
            boll_lb = float(ss["boll_lb"].iloc[-1])
            boll_mid = float(ss["boll"].iloc[-1])
            if close > boll_mid:
                score += 1
                reasons.append("Above Bollinger mid")
            elif close < boll_lb:
                score -= 1
                reasons.append("Below Bollinger lower band")
        except Exception:
            pass

        # 6. Volume confirmation
        try:
            vol = float(df["Volume"].iloc[-1])
            vol_avg = float(df["Volume"].iloc[-21:].mean())
            if vol > vol_avg * 1.1:
                if score > 0:
                    score += 1
                    reasons.append("Volume confirms (above avg)")
                elif score < 0:
                    score -= 1
                    reasons.append("Volume confirms bearish (above avg)")
        except Exception:
            pass

        # ATR for stop/target calculation
        try:
            atr = float(ss["atr_14"].iloc[-1])
            sl_pct = round((atr * 1.5) / close, 4)
            tp_pct = round((atr * 2.5) / close, 4)
        except Exception:
            sl_pct = 0.05
            tp_pct = 0.15

        # Map score to result
        confidence = min(abs(score) / 6.0, 1.0)

        if score >= self.buy_threshold:
            result = ScreenResult.RULE_BUY
        elif score <= self.sell_threshold:
            result = ScreenResult.RULE_SELL
        elif abs(score) <= self.skip_threshold:
            result = ScreenResult.SKIP
        else:
            result = ScreenResult.NEEDS_LLM

        logger.info(f"{symbol}: score={score}, result={result.value}, reasons={reasons}")

        return PreScreenSignal(
            result=result,
            score=score,
            confidence=confidence,
            reasons=reasons,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
        )


class MinerviniPreScreener:
    """Run the Minervini research filter before allowing new swing entries."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, symbols: list[str]) -> MinerviniPreflight:
        end_date = date.today().isoformat()
        lookback_days = int(self.config.get("minervini_lookback_days", 730))
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        benchmark = self.config.get("minervini_benchmark", "SPY")
        results_dir = Path(self.config.get("results_dir", "./results")) / "minervini"
        results_dir.mkdir(parents=True, exist_ok=True)
        coarse_candidates_path = None

        if self._use_broad_market_scanner(symbols):
            broad_df = self._build_broad_market_candidates(benchmark=benchmark)
            if not broad_df.empty:
                coarse_path = results_dir / f"automation_coarse_candidates_{end_date}.csv"
                broad_df.to_csv(coarse_path, index=False)
                coarse_candidates_path = str(coarse_path)
                symbols = broad_df["symbol"].tolist()
            else:
                symbols = []

        if not symbols:
            return MinerviniPreflight(
                trade_date=end_date,
                market_regime="unknown",
                confirmed_uptrend=False,
                approved_symbols=[],
                blocked_symbols=[],
                screened_symbols=[],
                screen_df=pd.DataFrame(),
                coarse_candidates_path=coarse_candidates_path,
            )

        db_path = self.config.get("minervini_db_path", "research_data/market_data.duckdb")
        try:
            warehouse = MarketDataWarehouse(db_path)
        except duckdb.IOException as exc:
            if "Could not set lock" not in str(exc):
                raise
            logger.warning(
                "DuckDB locked for writes, falling back to cached screening batch using %s",
                db_path,
            )
            return self._load_cached_preflight(symbols)
        try:
            required_symbols = sorted(set(symbols + [benchmark]))
            warehouse.fetch_and_store_daily_bars(required_symbols, start_date, end_date)
            if self._should_refresh_fundamentals(warehouse, symbols, end_date):
                warehouse.fetch_and_store_fundamentals(symbols, snapshot_date=end_date)
                warehouse.fetch_and_store_quarterly_fundamentals(symbols)
                warehouse.fetch_and_store_earnings_events(symbols)

            data_by_symbol = {
                symbol: warehouse.get_daily_bars(symbol, start_date, end_date)
                for symbol in symbols
            }
            data_by_symbol = {
                symbol: frame for symbol, frame in data_by_symbol.items() if not frame.empty
            }
            benchmark_df = warehouse.get_daily_bars(benchmark, start_date, end_date)
            fundamentals_df = warehouse.get_latest_fundamentals(symbols)
            quarterly_df = warehouse.get_quarterly_fundamentals(symbols)

            selection_strategy = str(
                self.config.get("selection_strategy", "minervini")
            ).strip().lower()
            canslim_df = pd.DataFrame()
            if selection_strategy == "canslim_minervini":
                selected_symbols, canslim_df = self._apply_canslim_selection(
                    data_by_symbol=data_by_symbol,
                    quarterly_df=quarterly_df,
                    fundamentals_df=fundamentals_df,
                )
                if not selected_symbols:
                    if not canslim_df.empty:
                        canslim_df = canslim_df.copy()
                        canslim_df["selected_for_analysis"] = False
                    return MinerviniPreflight(
                        trade_date=end_date,
                        market_regime="screened_out",
                        confirmed_uptrend=False,
                        approved_symbols=[],
                        blocked_symbols=list(data_by_symbol.keys()),
                        screened_symbols=list(data_by_symbol.keys()),
                        screen_df=canslim_df,
                        coarse_candidates_path=coarse_candidates_path,
                    )
                data_by_symbol = {
                    symbol: frame
                    for symbol, frame in data_by_symbol.items()
                    if symbol in selected_symbols
                }
                symbols = list(data_by_symbol.keys())

            screener = MinerviniScreener(
                MinerviniConfig(
                    min_rs_percentile=self.config.get("minervini_min_rs_percentile", 70.0),
                    min_revenue_growth=self.config.get("minervini_min_revenue_growth", 0.15),
                    min_eps_growth=self.config.get("minervini_min_eps_growth", 0.15),
                    min_return_on_equity=self.config.get("minervini_min_roe", 0.15),
                    require_acceleration=self.config.get("minervini_require_acceleration", False),
                    min_days_to_earnings=self.config.get("minervini_min_days_to_earnings", 5),
                    require_fundamentals=self.config.get("minervini_require_fundamentals", True),
                    require_market_uptrend=self.config.get("minervini_require_market_uptrend", True),
                    max_stage_number=self.config.get("minervini_max_stage_number", 3),
                    pivot_buffer_pct=self.config.get("minervini_pivot_buffer_pct", 0.0),
                    max_buy_zone_pct=self.config.get("minervini_max_buy_zone_pct", 0.07),
                    leader_continuation_enabled=self.config.get("leader_continuation_enabled", True),
                    leader_continuation_min_rs_percentile=self.config.get(
                        "leader_continuation_min_rs_percentile",
                        75.0,
                    ),
                    leader_continuation_min_close_range_pct=self.config.get(
                        "leader_continuation_min_close_range_pct",
                        0.15,
                    ),
                    leader_continuation_min_adx_14=self.config.get(
                        "leader_continuation_min_adx_14",
                        12.0,
                    ),
                    leader_continuation_min_roc_60=self.config.get(
                        "leader_continuation_min_roc_60",
                        0.0,
                    ),
                    leader_continuation_min_roc_120=self.config.get(
                        "leader_continuation_min_roc_120",
                        0.0,
                    ),
                    leader_continuation_max_extension_pct=self.config.get(
                        "leader_continuation_max_extension_pct",
                        0.07,
                    ),
                    leader_continuation_max_pullback_pct=self.config.get(
                        "leader_continuation_max_pullback_pct",
                        0.08,
                    ),
                )
            )
            regime = screener.analyze_market_regime(benchmark_df)
            screen_df = screener.screen_universe(
                data_by_symbol,
                benchmark_df=benchmark_df,
                fundamentals_df=fundamentals_df,
            )
            if selection_strategy == "canslim_minervini" and not screen_df.empty and not canslim_df.empty:
                canslim_merge = canslim_df[
                    [
                        "symbol",
                        "canslim_score",
                        "canslim_selected",
                        "canslim_current_ok",
                        "canslim_annual_ok",
                        "canslim_fundamental_ok",
                        "new_high_signal",
                        "canslim_volume_signal",
                        "canslim_leader_signal",
                    ]
                ].drop_duplicates(subset="symbol")
                screen_df = screen_df.merge(canslim_merge, on="symbol", how="left")
                screen_df["selection_strategy"] = "canslim_minervini"
            elif not screen_df.empty:
                screen_df["selection_strategy"] = selection_strategy

            if not screen_df.empty:
                approved_mask = (
                    screen_df["rule_watch_candidate"]
                    if "rule_watch_candidate" in screen_df.columns
                    else screen_df["passed_template"]
                )
                approved_symbols = (
                    screen_df[approved_mask]["symbol"].tolist()
                    if self._entries_enabled_for_regime(regime["regime"])
                    else []
                )
                screen_df = screen_df.copy()
                screen_df["selected_for_analysis"] = screen_df["symbol"].isin(approved_symbols)
            else:
                approved_symbols = []

            blocked_symbols = [symbol for symbol in symbols if symbol not in approved_symbols]
            screen_path = results_dir / f"automation_screen_{end_date}.csv"
            regime_path = results_dir / f"automation_market_regime_{end_date}.csv"
            fundamentals_path = results_dir / f"automation_fundamentals_{end_date}.csv"

            pd.DataFrame([regime]).to_csv(regime_path, index=False)
            if not screen_df.empty:
                screen_df.to_csv(screen_path, index=False)
            if not fundamentals_df.empty:
                fundamentals_df.to_csv(fundamentals_path, index=False)

            return MinerviniPreflight(
                trade_date=end_date,
                market_regime=regime["regime"],
                confirmed_uptrend=bool(regime["confirmed_uptrend"]),
                approved_symbols=approved_symbols,
                blocked_symbols=blocked_symbols,
                screened_symbols=list(symbols),
                screen_df=screen_df,
                coarse_candidates_path=coarse_candidates_path,
                screen_path=str(screen_path) if screen_path.exists() else None,
                regime_path=str(regime_path),
                fundamentals_path=str(fundamentals_path) if fundamentals_path.exists() else None,
            )
        finally:
            warehouse.close()

    def _apply_canslim_selection(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        quarterly_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
    ) -> tuple[list[str], pd.DataFrame]:
        quarterly_by_symbol = (
            {
                symbol: frame.copy()
                for symbol, frame in quarterly_df.groupby("symbol")
            }
            if quarterly_df is not None and not quarterly_df.empty
            else {}
        )
        fundamentals_lookup = (
            fundamentals_df.set_index("symbol").to_dict("index")
            if fundamentals_df is not None and not fundamentals_df.empty
            else {}
        )

        screener = CANSLIMScreener(
            CANSLIMConfig(
                near_52w_high_pct=float(self.config.get("canslim_near_52w_high_pct", 0.15)),
                volume_surge_multiple=float(
                    self.config.get("canslim_volume_surge_multiple", 1.2)
                ),
                min_close_range_pct=float(
                    self.config.get("canslim_min_close_range_pct", 0.50)
                ),
                min_current_eps_growth=float(
                    self.config.get("canslim_min_current_eps_growth", 0.25)
                ),
                min_current_revenue_growth=float(
                    self.config.get("canslim_min_current_revenue_growth", 0.20)
                ),
                min_annual_eps_growth=float(
                    self.config.get("canslim_min_annual_eps_growth", 0.20)
                ),
                min_annual_revenue_growth=float(
                    self.config.get("canslim_min_annual_revenue_growth", 0.15)
                ),
                require_fundamentals=bool(
                    self.config.get("canslim_require_fundamentals", True)
                ),
            )
        )

        rows: list[dict] = []
        selected_symbols: list[str] = []
        min_score = float(self.config.get("canslim_min_score", 4.0))
        require_new_high = bool(self.config.get("canslim_require_new_high_signal", True))
        require_volume = bool(self.config.get("canslim_require_volume_signal", False))
        require_leader = bool(self.config.get("canslim_require_leader_signal", True))

        for symbol, df in data_by_symbol.items():
            prepared = screener.prepare_features(
                df,
                quarterly_df=quarterly_by_symbol.get(symbol),
            )
            if prepared.empty:
                continue
            row = prepared.iloc[-1].copy()
            latest = row.to_dict()
            latest["symbol"] = symbol
            latest["sector"] = fundamentals_lookup.get(symbol, {}).get("sector")
            latest["industry"] = fundamentals_lookup.get(symbol, {}).get("industry")

            selected = bool(latest.get("canslim_score", 0) >= min_score)
            selected = selected and bool(latest.get("near_52w_high", False))
            if require_new_high:
                selected = selected and bool(latest.get("new_high_signal", False))
            if require_volume:
                selected = selected and bool(latest.get("canslim_volume_signal", False))
            if require_leader:
                selected = selected and bool(latest.get("canslim_leader_signal", False))
            if bool(self.config.get("canslim_require_fundamentals", True)):
                annual_available = (
                    pd.notna(latest.get("annual_eps_growth"))
                    or pd.notna(latest.get("annual_revenue_growth"))
                )
                annual_gate = bool(latest.get("canslim_annual_ok", False)) or not annual_available
                selected = (
                    selected
                    and bool(latest.get("canslim_current_ok", False))
                    and bool(latest.get("canslim_positive_eps_ok", False))
                    and annual_gate
                )

            latest["canslim_selected"] = selected
            latest["selected_for_analysis"] = selected
            latest["selection_strategy"] = "canslim_minervini"
            rows.append(latest)
            if selected:
                selected_symbols.append(symbol)

        screen_df = pd.DataFrame(rows)
        if not screen_df.empty:
            sort_cols = [col for col in ["canslim_score", "rs_percentile"] if col in screen_df.columns]
            ascending = [False] * len(sort_cols)
            if sort_cols:
                screen_df = screen_df.sort_values(
                    by=sort_cols,
                    ascending=ascending,
                    na_position="last",
                ).reset_index(drop=True)

        logger.info(
            "CAN SLIM prefilter kept %s/%s symbols",
            len(selected_symbols),
            len(data_by_symbol),
        )
        return selected_symbols, screen_df

    def _load_cached_preflight(self, symbols: list[str]) -> MinerviniPreflight:
        db = TradingDatabase(self.config.get("db_path", "trading.db"))
        try:
            latest_batch = db.get_latest_screening_batch()
            if latest_batch is None:
                raise RuntimeError("DuckDB is locked and no cached screening batch exists")

            setup_rows = db.get_setup_candidates_on_date(latest_batch["screen_date"])
            screen_rows = []
            for row in setup_rows:
                payload = row.get("payload_json")
                if payload:
                    screen_rows.append(json.loads(payload))
                else:
                    screen_rows.append(row)

            approved_symbols = latest_batch["approved_symbols"]
            screened_symbols = sorted(
                {row.get("symbol") for row in screen_rows if row.get("symbol")}
            ) if screen_rows else list(symbols)
            blocked_symbols = [
                symbol for symbol in screened_symbols if symbol not in approved_symbols
            ]
            screen_df = pd.DataFrame(screen_rows) if screen_rows else pd.DataFrame()

            return MinerviniPreflight(
                trade_date=latest_batch["screen_date"],
                market_regime=latest_batch["market_regime"],
                confirmed_uptrend=bool(latest_batch["market_confirmed_uptrend"]),
                approved_symbols=approved_symbols,
                blocked_symbols=blocked_symbols,
                screened_symbols=screened_symbols,
                screen_df=screen_df,
            )
        finally:
            db.close()

    def _use_broad_market_scanner(self, symbols: list[str]) -> bool:
        return bool(self.config.get("broad_market_enabled", False)) or not symbols

    def _build_broad_market_candidates(self, benchmark: str) -> pd.DataFrame:
        screener = BroadMarketScreener(
            api_key=self.config.get("alpaca_api_key", ""),
            secret_key=self.config.get("alpaca_secret_key", ""),
            paper=self.config.get("paper_trading", True),
            config=BroadMarketConfig(
                min_price=float(self.config.get("broad_market_min_price", 10.0)),
                max_price=self.config.get("broad_market_max_price"),
                min_prev_volume=float(self.config.get("broad_market_min_prev_volume", 200_000)),
                min_prev_dollar_volume=float(
                    self.config.get("broad_market_min_prev_dollar_volume", 25_000_000)
                ),
                min_avg_dollar_volume=float(
                    self.config.get("broad_market_min_avg_dollar_volume", 20_000_000)
                ),
                max_seed_symbols=int(self.config.get("broad_market_max_seed_symbols", 600)),
                max_candidates=int(self.config.get("broad_market_max_candidates", 160)),
                snapshot_batch_size=int(self.config.get("broad_market_snapshot_batch_size", 200)),
                history_batch_size=int(self.config.get("broad_market_history_batch_size", 100)),
                history_period=str(self.config.get("broad_market_history_period", "1y")),
                exclude_funds=bool(self.config.get("broad_market_exclude_funds", True)),
                max_below_52w_high=float(
                    self.config.get("broad_market_max_below_52w_high", 0.30)
                ),
                min_above_52w_low=float(
                    self.config.get("broad_market_min_above_52w_low", 0.25)
                ),
            ),
        )
        return screener.build_candidates(benchmark=benchmark)

    @staticmethod
    def _should_refresh_fundamentals(
        warehouse: MarketDataWarehouse, symbols: list[str], target_date: str
    ) -> bool:
        latest = warehouse.get_latest_fundamentals(symbols)
        if latest.empty:
            return True
        latest_dates = set(str(value) for value in latest["snapshot_date"].dropna().tolist())
        quarterly = warehouse.get_quarterly_fundamentals(symbols)
        earnings_count = warehouse.conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM earnings_events WHERE symbol IN ({})".format(
                ", ".join(["?"] * len(symbols))
            ),
            symbols,
        ).fetchone()[0]
        return (
            target_date not in latest_dates
            or quarterly.empty
            or len(set(quarterly["symbol"].tolist())) < len(symbols)
            or earnings_count < len(symbols)
        )

    def _entries_enabled_for_regime(self, regime: str) -> bool:
        if regime == "confirmed_uptrend":
            return (
                self.config.get("minervini_target_exposure_confirmed_uptrend", 0.72) > 0
                or self.config.get(
                    "leader_continuation_target_exposure_confirmed_uptrend",
                    0.72,
                )
                > 0
            )
        if regime == "uptrend_under_pressure":
            return (
                self.config.get("minervini_target_exposure_uptrend_under_pressure", 0.48) > 0
                or self.config.get(
                    "leader_continuation_target_exposure_uptrend_under_pressure",
                    0.36,
                )
                > 0
            )
        if regime == "market_correction":
            return (
                bool(self.config.get("minervini_allow_new_entries_in_correction", False))
                and self.config.get("minervini_target_exposure_market_correction", 0.0) > 0
            ) or (
                bool(self.config.get("leader_continuation_allow_in_correction", True))
                and self.config.get(
                    "leader_continuation_target_exposure_market_correction",
                    0.12,
                )
                > 0
            )
        return False
