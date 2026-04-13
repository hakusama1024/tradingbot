"""Hard risk controls — the last gate before any order is submitted.

These rules cannot be overridden by the AI agents. They protect capital
regardless of what the analysis says.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from tradingagents.broker.models import OrderRequest, Account, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    passed: bool
    reason: str = ""


class RiskEngine:
    """Pre-trade risk checks that enforce hard limits."""

    def __init__(self, config: dict):
        # Position limits
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_total_exposure = config.get("max_total_exposure", 0.80)
        self.max_open_positions = config.get("max_open_positions", 10)
        self.min_cash_reserve = config.get("min_cash_reserve", 0.20)

        # Loss limits
        self.max_daily_loss = config.get("max_daily_loss", 0.03)
        self.max_drawdown = config.get("max_drawdown", 0.10)
        self.starting_equity = config.get("starting_equity", None)

        # Trading frequency limits
        self.max_trades_per_window = config.get("max_trades_per_window", 10)
        self.trade_window_hours = config.get("trade_window_hours", 12)

        # Stop-loss / take-profit defaults
        self.default_stop_loss_pct = config.get("default_stop_loss_pct", 0.05)
        self.default_take_profit_pct = config.get("default_take_profit_pct", 0.15)

        # Track trade timestamps for frequency limiting
        self._trade_timestamps: deque = deque()

    def check_order(
        self,
        order: OrderRequest,
        account: Account,
        positions: List[Position],
        current_price: float = 0,
    ) -> RiskCheckResult:
        """Run all risk checks. Returns (passed, reason)."""

        if order.side == "sell":
            return RiskCheckResult(passed=True, reason="Sell orders always allowed")

        checks = [
            self._check_trading_frequency(),
            self._check_single_position_limit(order, account, current_price),
            self._check_total_exposure(order, account, positions, current_price),
            self._check_cash_reserve(order, account, current_price),
            self._check_daily_loss(account),
            self._check_max_drawdown(account),
            self._check_max_positions(positions),
        ]

        for result in checks:
            if not result.passed:
                logger.warning(f"Risk check FAILED for {order.symbol}: {result.reason}")
                return result

        return RiskCheckResult(passed=True)

    def record_trade(self):
        """Call after a trade is executed to track frequency."""
        self._trade_timestamps.append(datetime.now())

    def get_stop_loss_price(self, entry_price: float, signal_sl_pct: float = None) -> float:
        """Calculate stop-loss price. Uses signal's SL or default."""
        pct = signal_sl_pct if signal_sl_pct and signal_sl_pct > 0 else self.default_stop_loss_pct
        return round(entry_price * (1 - pct), 2)

    def get_take_profit_price(self, entry_price: float, signal_tp_pct: float = None) -> float:
        """Calculate take-profit price. Uses signal's TP or default."""
        pct = signal_tp_pct if signal_tp_pct and signal_tp_pct > 0 else self.default_take_profit_pct
        return round(entry_price * (1 + pct), 2)

    # ── Individual Checks ────────────────────────────────────────────

    def _check_trading_frequency(self) -> RiskCheckResult:
        cutoff = datetime.now() - timedelta(hours=self.trade_window_hours)
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()

        if len(self._trade_timestamps) >= self.max_trades_per_window:
            return RiskCheckResult(
                passed=False,
                reason=f"Trading frequency limit reached: {len(self._trade_timestamps)} trades "
                       f"in the last {self.trade_window_hours}h (max {self.max_trades_per_window})"
            )
        return RiskCheckResult(passed=True)

    def _check_single_position_limit(
        self, order: OrderRequest, account: Account, current_price: float
    ) -> RiskCheckResult:
        if current_price <= 0:
            return RiskCheckResult(passed=True)
        order_value = (order.qty or 0) * current_price
        max_allowed = account.equity * self.max_position_pct
        if order_value > max_allowed:
            return RiskCheckResult(
                passed=False,
                reason=f"Order value ${order_value:,.0f} exceeds single-position limit "
                       f"${max_allowed:,.0f} ({self.max_position_pct:.0%} of equity)"
            )
        return RiskCheckResult(passed=True)

    def _check_total_exposure(
        self, order: OrderRequest, account: Account,
        positions: List[Position], current_price: float
    ) -> RiskCheckResult:
        total_current = sum(p.market_value for p in positions)
        new_value = (order.qty or 0) * current_price if current_price > 0 else 0
        total_after = total_current + new_value
        max_allowed = account.equity * self.max_total_exposure
        if total_after > max_allowed:
            return RiskCheckResult(
                passed=False,
                reason=f"Total exposure ${total_after:,.0f} would exceed limit "
                       f"${max_allowed:,.0f} ({self.max_total_exposure:.0%} of equity)"
            )
        return RiskCheckResult(passed=True)

    def _check_cash_reserve(
        self, order: OrderRequest, account: Account, current_price: float
    ) -> RiskCheckResult:
        order_cost = (order.qty or 0) * current_price if current_price > 0 else 0
        cash_after = account.cash - order_cost
        min_cash = account.equity * self.min_cash_reserve
        if cash_after < min_cash:
            return RiskCheckResult(
                passed=False,
                reason=f"Cash after trade ${cash_after:,.0f} would be below "
                       f"reserve ${min_cash:,.0f} ({self.min_cash_reserve:.0%} of equity)"
            )
        return RiskCheckResult(passed=True)

    def _check_daily_loss(self, account: Account) -> RiskCheckResult:
        if account.last_equity <= 0:
            return RiskCheckResult(passed=True)
        daily_loss_pct = (account.last_equity - account.equity) / account.last_equity
        if daily_loss_pct >= self.max_daily_loss:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss {daily_loss_pct:.2%} has reached limit "
                       f"{self.max_daily_loss:.2%}. Trading halted for today."
            )
        return RiskCheckResult(passed=True)

    def _check_max_drawdown(self, account: Account) -> RiskCheckResult:
        if self.starting_equity is None or self.starting_equity <= 0:
            return RiskCheckResult(passed=True)
        drawdown = (self.starting_equity - account.equity) / self.starting_equity
        if drawdown >= self.max_drawdown:
            return RiskCheckResult(
                passed=False,
                reason=f"Drawdown {drawdown:.2%} from starting equity "
                       f"${self.starting_equity:,.0f} has reached limit "
                       f"{self.max_drawdown:.2%}. Trading halted."
            )
        return RiskCheckResult(passed=True)

    def _check_max_positions(self, positions: List[Position]) -> RiskCheckResult:
        if self.max_open_positions is None or self.max_open_positions <= 0:
            return RiskCheckResult(passed=True)
        if len(positions) >= self.max_open_positions:
            return RiskCheckResult(
                passed=False,
                reason=f"Already at max positions ({self.max_open_positions})"
            )
        return RiskCheckResult(passed=True)
