# -------- Event.py --------
# A dated fact that closed a trade. Immutable, since it records what already happened.

# -------- Imports --------
from dataclasses import dataclass
from sltp.utils.EventType import EventType


# -------- Event --------
@dataclass(frozen=True)
class Event:
    date: str
    type: EventType
    price: float
