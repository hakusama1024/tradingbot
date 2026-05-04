"""Abstract base class for broker integrations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import OrderRequest, OrderResult, Position, Account, MarketClock


class BaseBroker(ABC):
    """Abstract broker interface. Implement this to add a new broker."""

    @abstractmethod
    def get_account(self) -> Account:
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        ...

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        ...

    @abstractmethod
    def close_all_positions(self) -> List[OrderResult]:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult:
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        ...

    @abstractmethod
    def get_clock(self) -> MarketClock:
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_latest_prices(self, symbols: List[str]) -> dict:
        ...

    @abstractmethod
    def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        extended_hours: bool = False,
        intraday_reporting: str = "market_hours",
        pnl_reset: str = "no_reset",
    ) -> Dict[str, Any]:
        ...
