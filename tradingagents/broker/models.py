"""Data models for broker interactions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal


@dataclass
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    qty: Optional[float] = None
    notional: Optional[float] = None
    order_type: Literal["market", "limit", "stop", "stop_limit", "trailing_stop"] = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: Literal["day", "gtc", "ioc"] = "day"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    qty: Optional[float]
    notional: Optional[float]
    order_type: str
    status: str
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class Account:
    account_id: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str
    day_trade_count: int = 0
    last_equity: float = 0.0

    @property
    def daily_pl(self) -> float:
        return self.equity - self.last_equity if self.last_equity > 0 else 0.0

    @property
    def daily_pl_pct(self) -> float:
        return self.daily_pl / self.last_equity if self.last_equity > 0 else 0.0


@dataclass
class MarketClock:
    is_open: bool
    next_open: datetime
    next_close: datetime
    timestamp: datetime
