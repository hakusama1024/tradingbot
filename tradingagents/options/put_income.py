"""Cash-secured put income strategy research and scanner.

The backtester uses synthetic option prices from daily underlying bars and a
realized-volatility proxy. That makes it useful for rule validation, but not a
substitute for historical option quote data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _put_price_delta(
    spot: float,
    strike: float,
    dte: int,
    rate: float,
    iv: float,
) -> tuple[float, float]:
    t = max(dte, 1) / 365.0
    iv = max(iv, 0.01)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    price = strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    delta = _norm_cdf(d1) - 1.0
    return max(price, 0.01), delta


def _annualized_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window).std() * math.sqrt(252.0)


@dataclass
class PutIncomeParams:
    initial_equity: float = 100_000.0
    symbols: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "COST",
        "JPM",
        "V",
        "MA",
        "ORCL",
    )
    start: str = "2016-01-01"
    end: Optional[str] = None
    entry_dte: int = 45
    manage_dte: int = 21
    target_delta: float = 0.18
    min_delta: float = 0.12
    max_delta: float = 0.25
    min_iv: float = 0.22
    iv_multiplier: float = 1.20
    min_premium_yield: float = 0.012
    close_profit_pct: float = 0.50
    max_contracts_per_symbol: int = 2
    max_symbol_notional_pct: float = 0.20
    max_total_notional_pct: float = 0.65
    max_open_positions: int = 5
    trend_sma_days: int = 200
    min_open_interest: int = 100
    min_option_volume: int = 0
    entry_weekday: int = 0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.07
    rate: float = 0.04
    cash_yield: float = 0.00


class PutIncomeBacktester:
    """Synthetic daily-bar backtester for mechanical short-put rules."""

    def __init__(self, params: PutIncomeParams):
        self.params = params

    def run(self) -> Dict:
        data = self._download_data(self.params.symbols)
        if not data:
            raise RuntimeError("No price data downloaded")

        dates = sorted(set().union(*(df.index for df in data.values())))
        cash = self.params.initial_equity
        positions: List[Dict] = []
        trades: List[Dict] = []
        equity_curve = []

        for current_date in dates:
            cash *= 1.0 + self.params.cash_yield / 252.0
            marks = self._mark_positions(current_date, data, positions)
            equity = cash + sum(mark["mtm"] for mark in marks)
            closed, keep = self._manage_positions(current_date, data, positions, marks)
            for item in closed:
                cash += item["cash_delta"]
                trades.append(item["trade"])
            positions = keep

            if current_date.weekday() == self.params.entry_weekday:
                equity = cash + sum(self._position_mtm(current_date, data, p) for p in positions)
                for candidate in self._rank_candidates(current_date, data, equity, positions):
                    if len(positions) >= self.params.max_open_positions:
                        break
                    current_notional = sum(p["strike"] * 100 * p["contracts"] for p in positions)
                    if current_notional + candidate["notional"] > equity * self.params.max_total_notional_pct:
                        continue
                    cash += candidate["cash_delta"]
                    positions.append(candidate["position"])
                    trades.append(candidate["trade"])

            equity = cash + sum(self._position_mtm(current_date, data, p) for p in positions)
            equity_curve.append({"date": current_date, "equity": equity, "cash": cash})

        curve = pd.DataFrame(equity_curve).set_index("date")
        stats = self._stats(curve["equity"])
        return {
            "params": self.params.__dict__,
            "stats": stats,
            "trades": trades,
            "equity_curve": curve,
            "open_positions": positions,
        }

    def benchmark_stats(self, symbol: str = "SPY") -> Dict:
        df = yf.download(symbol, start=self.params.start, end=self.params.end, auto_adjust=True, progress=False)
        if df.empty:
            return {}
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        equity = self.params.initial_equity * df["Close"] / float(df["Close"].iloc[0])
        return self._stats(equity)

    def _download_data(self, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in symbols:
            df = yf.download(symbol, start=self.params.start, end=self.params.end, auto_adjust=True, progress=False)
            if df.empty or len(df) < self.params.trend_sma_days + 60:
                continue
            if hasattr(df.columns, "levels"):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df["sma"] = df["Close"].rolling(self.params.trend_sma_days).mean()
            df["rv"] = _annualized_realized_vol(df["Close"]).clip(lower=0.08, upper=1.50)
            data[symbol] = df
        return data

    def _rank_candidates(
        self,
        current_date,
        data: Dict[str, pd.DataFrame],
        equity: float,
        positions: List[Dict],
    ) -> List[Dict]:
        open_symbols = {p["symbol"] for p in positions}
        candidates = []
        for symbol, df in data.items():
            if symbol in open_symbols or current_date not in df.index:
                continue
            row = df.loc[current_date]
            if not row["Close"] > row["sma"]:
                continue
            iv = float(row["rv"]) * self.params.iv_multiplier
            if iv < self.params.min_iv:
                continue
            candidate = self._build_candidate(symbol, current_date, float(row["Close"]), iv, equity)
            if candidate:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (item["premium_yield"], item["iv"]), reverse=True)
        return candidates

    def _build_candidate(
        self,
        symbol: str,
        current_date,
        spot: float,
        iv: float,
        equity: float,
    ) -> Optional[Dict]:
        best = None
        for pct in [x / 100.0 for x in range(70, 99)]:
            strike = round(spot * pct, 2)
            price, delta = _put_price_delta(spot, strike, self.params.entry_dte, self.params.rate, iv)
            abs_delta = abs(delta)
            if not (self.params.min_delta <= abs_delta <= self.params.max_delta):
                continue
            score = abs(abs_delta - self.params.target_delta)
            if best is None or score < best["score"]:
                best = {"strike": strike, "premium": price, "delta": delta, "score": score}
        if best is None:
            return None

        premium = best["premium"] * (1.0 - self.params.slippage_pct)
        premium_yield = premium / best["strike"]
        if premium_yield < self.params.min_premium_yield:
            return None

        max_notional = equity * self.params.max_symbol_notional_pct
        contracts = min(
            self.params.max_contracts_per_symbol,
            int(max_notional // (best["strike"] * 100.0)),
        )
        if contracts < 1:
            return None

        notional = best["strike"] * 100.0 * contracts
        cash_delta = premium * 100.0 * contracts - self.params.commission_per_contract * contracts
        position = {
            "symbol": symbol,
            "entry_date": current_date,
            "expiry_date": self._expiry_for(current_date, self.params.entry_dte),
            "entry_dte": self.params.entry_dte,
            "strike": best["strike"],
            "contracts": contracts,
            "entry_premium": premium,
            "iv": iv,
            "delta": best["delta"],
        }
        trade = {
            "date": current_date,
            "symbol": symbol,
            "action": "SELL_PUT",
            "contracts": contracts,
            "strike": best["strike"],
            "premium": round(premium, 2),
            "delta": round(best["delta"], 3),
            "iv": round(iv, 3),
            "notional": round(notional, 2),
        }
        return {
            "position": position,
            "trade": trade,
            "cash_delta": cash_delta,
            "notional": notional,
            "premium_yield": premium_yield,
            "iv": iv,
        }

    def _manage_positions(self, current_date, data, positions: List[Dict], marks: List[Dict]):
        closed = []
        keep = []
        mark_by_id = {id(mark["position"]): mark for mark in marks}
        for position in positions:
            mark = mark_by_id.get(id(position))
            if mark is None:
                keep.append(position)
                continue
            dte = mark["dte"]
            current_price = mark["option_price"]
            profit_pct = 1.0 - current_price / max(position["entry_premium"], 0.01)
            should_close = profit_pct >= self.params.close_profit_pct or dte <= self.params.manage_dte
            if should_close:
                close_debit = current_price * (1.0 + self.params.slippage_pct)
                cash_delta = -close_debit * 100.0 * position["contracts"] - self.params.commission_per_contract * position["contracts"]
                closed.append(
                    {
                        "cash_delta": cash_delta,
                        "trade": {
                            "date": current_date,
                            "symbol": position["symbol"],
                            "action": "BUY_TO_CLOSE",
                            "contracts": position["contracts"],
                            "strike": position["strike"],
                            "debit": round(close_debit, 2),
                            "dte": dte,
                            "profit_pct": round(profit_pct, 3),
                        },
                    }
                )
            else:
                keep.append(position)
        return closed, keep

    def _mark_positions(self, current_date, data, positions: List[Dict]) -> List[Dict]:
        marks = []
        for position in positions:
            if position["symbol"] not in data or current_date not in data[position["symbol"]].index:
                continue
            df = data[position["symbol"]]
            row = df.loc[current_date]
            dte = max((position["expiry_date"] - current_date.date()).days, 0)
            iv = max(float(row["rv"]) * self.params.iv_multiplier, 0.08)
            price, _ = _put_price_delta(float(row["Close"]), position["strike"], dte, self.params.rate, iv)
            mtm = -price * 100.0 * position["contracts"]
            marks.append({"position": position, "option_price": price, "dte": dte, "mtm": mtm})
        return marks

    def _position_mtm(self, current_date, data, position: Dict) -> float:
        marks = self._mark_positions(current_date, data, [position])
        return marks[0]["mtm"] if marks else 0.0

    @staticmethod
    def _expiry_for(current_date, dte: int) -> date:
        return (pd.Timestamp(current_date) + pd.Timedelta(days=dte)).date()

    @staticmethod
    def _stats(equity: pd.Series) -> Dict:
        equity = equity.dropna()
        returns = equity.pct_change().fillna(0.0)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min())
        vol = float(returns.std() * math.sqrt(252.0))
        sharpe = float((returns.mean() * 252.0) / vol) if vol > 0 else 0.0
        calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0
        yearly = equity.resample("YE").last().pct_change().dropna()
        return {
            "start": str(equity.index[0].date()),
            "end": str(equity.index[-1].date()),
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "volatility": vol,
            "sharpe": sharpe,
            "calmar": calmar,
            "yearly_win_rate": float((yearly > 0).mean()) if len(yearly) else 0.0,
            "final_equity": float(equity.iloc[-1]),
        }


class PutIncomeScanner:
    """Current option-chain scanner for paper candidates."""

    def __init__(self, params: PutIncomeParams):
        self.params = params

    def scan(self) -> List[Dict]:
        rows = []
        for symbol in self.params.symbols:
            rows.extend(self._scan_symbol(symbol))
        rows.sort(key=lambda item: (item["premium_yield"], item["open_interest"]), reverse=True)
        return rows

    def _scan_symbol(self, symbol: str) -> List[Dict]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < self.params.trend_sma_days:
            return []
        spot = float(hist["Close"].iloc[-1])
        sma = float(hist["Close"].rolling(self.params.trend_sma_days).mean().iloc[-1])
        if spot <= sma:
            return []

        expirations = self._target_expirations(ticker.options)
        if not expirations:
            return []

        results = []
        for exp, dte in expirations:
            try:
                chain = ticker.option_chain(exp)
            except Exception:
                continue
            puts = chain.puts.copy()
            if puts.empty:
                continue
            for _, row in puts.iterrows():
                bid = float(row.get("bid") or 0.0)
                ask = float(row.get("ask") or 0.0)
                if bid <= 0 or ask <= 0:
                    continue
                mid = (bid + ask) / 2.0
                strike = float(row["strike"])
                iv = float(row.get("impliedVolatility") or 0.0)
                if iv < self.params.min_iv:
                    continue
                _, delta = _put_price_delta(spot, strike, dte, self.params.rate, iv)
                abs_delta = abs(delta)
                if not (self.params.min_delta <= abs_delta <= self.params.max_delta):
                    continue
                spread_pct = (ask - bid) / max(mid, 0.01)
                if spread_pct > 0.35:
                    continue
                premium_yield = mid / strike
                if premium_yield < self.params.min_premium_yield:
                    continue
                open_interest = int(row.get("openInterest") or 0)
                volume = int(row.get("volume") or 0)
                if open_interest < self.params.min_open_interest:
                    continue
                if volume < self.params.min_option_volume:
                    continue
                results.append(
                    {
                        "symbol": symbol,
                        "expiry": exp,
                        "dte": dte,
                        "spot": round(spot, 2),
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": round(mid, 2),
                        "delta_est": round(delta, 3),
                        "iv": round(iv, 3),
                        "premium_yield": round(premium_yield, 4),
                        "open_interest": open_interest,
                        "volume": volume,
                        "contract": row.get("contractSymbol"),
                    }
                )
        return results

    def _target_expirations(self, expirations: Iterable[str]) -> List[tuple[str, int]]:
        today = datetime.now(timezone.utc).date()
        out = []
        for exp in expirations:
            exp_date = datetime.fromisoformat(exp).date()
            dte = (exp_date - today).days
            if 30 <= dte <= 60:
                out.append((exp, dte))
        return out
