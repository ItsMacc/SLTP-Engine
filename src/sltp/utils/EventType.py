# -------- EventType.py --------
# Why a trade closed. Exit causes only; anything else gets its own type.

# -------- Imports --------
from enum import Enum


# -------- Event Type --------
class EventType(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL = "signal"
    END_OF_BACKTEST = "end_of_backtest"
