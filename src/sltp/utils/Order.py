# -------- Order.py --------
# A request to open a position, as submitted by a strategy.
# The manager fills it at the next open, copies the risk levels onto the Trade, then drops it.
# allocation is sized by `sizing`: EQUITY is a fraction of equity, FIXED is cash, AVAILABLE is
# a fraction capped by uninvested cash. stop_loss and take_profit are absolute prices.

# -------- Imports --------
from dataclasses import dataclass
from enum import Enum

from sltp.utils.PositionType import PositionType


# -------- Order Sizing --------
class OrderSizing(Enum):
    EQUITY = "equity"
    FIXED = "fixed"
    AVAILABLE = "available"


# -------- Order --------
@dataclass(frozen=True)
class Order:
    ticker: str
    position_type: PositionType
    allocation: float
    order_date: str
    stop_loss: float
    take_profit: float
    meta_data: dict | None = None
    sizing: OrderSizing = OrderSizing.EQUITY

    # Resolve the order's size into a fraction of current portfolio equity.
    def calculate_weight(self, current_equity: float, available_cash: float) -> float:
        if self.sizing is OrderSizing.EQUITY:
            return self.allocation

        if self.sizing is OrderSizing.FIXED:
            if current_equity <= 0:
                return 0.0
            return self.allocation / current_equity

        if self.sizing is OrderSizing.AVAILABLE:
            if current_equity <= 0 or available_cash <= 0:
                return 0.0
            return min(self.allocation, available_cash / current_equity)

        raise ValueError(f"Unknown order sizing: {self.sizing}")
