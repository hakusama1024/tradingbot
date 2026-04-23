"""Alpaca broker implementation for paper and live trading."""

import logging
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    ClosePositionRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from .base_broker import BaseBroker
from .models import OrderRequest, OrderResult, Position, Account, MarketClock

logger = logging.getLogger(__name__)

TIF_MAP = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
}


class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker — works for both paper and live accounts."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        mode = "PAPER" if paper else "LIVE"
        logger.info(f"AlpacaBroker initialized in {mode} mode")

    # ── Account ──────────────────────────────────────────────────────

    def get_account(self) -> Account:
        acct = self.trading_client.get_account()
        return Account(
            account_id=str(acct.id),
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value or acct.equity),
            status=str(acct.status),
            day_trade_count=int(acct.daytrade_count or 0),
            last_equity=float(acct.last_equity or 0),
        )

    # ── Positions ────────────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        raw = self.trading_client.get_all_positions()
        return [self._to_position(p) for p in raw]

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            raw = self.trading_client.get_open_position(symbol)
            return self._to_position(raw)
        except Exception:
            return None

    @staticmethod
    def _to_position(p) -> Position:
        return Position(
            symbol=p.symbol,
            qty=abs(float(p.qty)),
            side=str(p.side),
            avg_entry_price=float(p.avg_entry_price),
            current_price=float(p.current_price),
            market_value=abs(float(p.market_value)),
            unrealized_pl=float(p.unrealized_pl),
            unrealized_plpc=float(p.unrealized_plpc),
        )

    # ── Orders ───────────────────────────────────────────────────────

    def submit_order(self, order: OrderRequest) -> OrderResult:
        tif = TIF_MAP.get(order.time_in_force, TimeInForce.DAY)
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

        if order.order_type == "market":
            req = MarketOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                notional=order.notional,
                time_in_force=tif,
            )
        elif order.order_type == "limit":
            req = LimitOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                notional=order.notional,
                time_in_force=tif,
                limit_price=order.limit_price,
            )
        elif order.order_type == "stop":
            req = StopOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                stop_price=order.stop_price,
            )
        elif order.order_type == "stop_limit":
            req = StopLimitOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                stop_price=order.stop_price,
                limit_price=order.limit_price,
            )
        elif order.order_type == "trailing_stop":
            req = TrailingStopOrderRequest(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                time_in_force=tif,
                trail_percent=order.trail_percent,
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        result = self.trading_client.submit_order(req)
        logger.info(f"Order submitted: {order.side} {order.qty or order.notional} {order.symbol} -> {result.status}")
        return self._to_order_result(result)

    def submit_bracket_order(
        self,
        order: OrderRequest,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit a protected entry order.

        If take_profit_price is provided, use a bracket order:
        - A stop order at stop_loss_price
        - A limit order at take_profit_price

        If take_profit_price is omitted, use an OTO order with stop-loss only.
        """
        if order.side != "buy":
            return self.submit_order(order)

        order_class = OrderClass.BRACKET if take_profit_price is not None else OrderClass.OTO
        req_kwargs = {
            "symbol": order.symbol,
            "side": OrderSide.BUY,
            "qty": order.qty,
            "time_in_force": TimeInForce.GTC,
            "order_class": order_class,
            "stop_loss": {"stop_price": round(stop_loss_price, 2)},
        }
        if take_profit_price is not None:
            req_kwargs["take_profit"] = {"limit_price": round(take_profit_price, 2)}
        req = MarketOrderRequest(**req_kwargs)

        result = self.trading_client.submit_order(req)
        if take_profit_price is not None:
            logger.info(
                f"Bracket order: BUY {order.qty} {order.symbol} "
                f"SL=${stop_loss_price:.2f} TP=${take_profit_price:.2f} -> {result.status}"
            )
        else:
            logger.info(
                f"OTO order: BUY {order.qty} {order.symbol} "
                f"SL=${stop_loss_price:.2f} -> {result.status}"
            )
        return self._to_order_result(result)

    def cancel_order(self, order_id: str) -> None:
        self.trading_client.cancel_order_by_id(order_id)
        logger.info(f"Order cancelled: {order_id}")

    def get_order(self, order_id: str) -> OrderResult:
        raw = self.trading_client.get_order_by_id(order_id)
        return self._to_order_result(raw)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol] if symbol else None)
        raw = self.trading_client.get_orders(filter=req)
        return [self._to_order_result(order) for order in raw]

    @staticmethod
    def _to_order_result(o) -> OrderResult:
        return OrderResult(
            order_id=str(o.id),
            symbol=o.symbol,
            side=str(o.side),
            qty=float(o.qty) if o.qty else None,
            notional=float(o.notional) if o.notional else None,
            order_type=str(o.type),
            status=str(o.status),
            filled_qty=float(o.filled_qty or 0),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            limit_price=float(o.limit_price) if getattr(o, "limit_price", None) else None,
            stop_price=float(o.stop_price) if getattr(o, "stop_price", None) else None,
            submitted_at=o.submitted_at,
            filled_at=o.filled_at,
        )

    # ── Close Positions ──────────────────────────────────────────────

    def close_position(self, symbol: str) -> OrderResult:
        result = self.trading_client.close_position(symbol)
        logger.info(f"Position closed: {symbol}")
        return self._to_order_result(result)

    def close_all_positions(self) -> List[OrderResult]:
        results = self.trading_client.close_all_positions(cancel_orders=True)
        logger.info(f"All positions closed ({len(results)} orders)")
        return [self._to_order_result(r) for r in results if hasattr(r, "id")]

    # ── Market Data ──────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        return self.trading_client.get_clock().is_open

    def get_clock(self) -> MarketClock:
        c = self.trading_client.get_clock()
        return MarketClock(
            is_open=c.is_open,
            next_open=c.next_open,
            next_close=c.next_close,
            timestamp=c.timestamp,
        )

    def get_latest_price(self, symbol: str) -> float:
        quotes = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        q = quotes[symbol]
        mid = (float(q.ask_price) + float(q.bid_price)) / 2
        return mid if mid > 0 else float(q.ask_price or q.bid_price)

    def get_latest_prices(self, symbols: List[str]) -> dict:
        quotes = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbols)
        )
        prices = {}
        for sym, q in quotes.items():
            mid = (float(q.ask_price) + float(q.bid_price)) / 2
            prices[sym] = mid if mid > 0 else float(q.ask_price or q.bid_price)
        return prices
