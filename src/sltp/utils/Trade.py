# -------- Trade.py --------
# A filled position and its lifecycle record.
# One Trade is one lot, so two entries in a ticker stay separate and never average prices.
# allocation is a fraction of equity_at_entry; both are kept so P&L stays recoverable.

# -------- Imports --------
from dataclasses import dataclass
from sltp.utils.Event import Event
from sltp.utils.StockData import StockData
from sltp.utils.PositionType import PositionType


# -------- Trade --------
@dataclass
class Trade:
    id: int
    ticker: str
    history: StockData
    position_type: PositionType
    entry_date: str
    entry_price: float
    allocation: float
    equity_at_entry: float
    stop_loss: float
    take_profit: float
    fees: float
    dividend_income: float = 0.0
    borrow_cost: float = 0.0
    exit: Event | None = None

    # Open means "has no exit event"; derived on purpose so it can never disagree with the event that closed the trade.
    @property
    def is_open(self) -> bool:
        return self.exit is None
