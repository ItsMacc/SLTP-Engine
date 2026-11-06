# -------- PositionType.py --------
# Which way a position is held: LONG or SHORT.
# Buy/sell is an action, not a direction; a buy can open a long or cover a short.

# -------- Imports --------
from enum import Enum


# -------- Position Type --------
class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
