# -------- RebalanceFrequency.py --------
# How often a strategy rebuilds its universe. The manager owns the actual dates.
# NEVER exists so a strategy can opt out without a null.

# -------- Imports --------
from enum import Enum


# -------- Rebalance Frequency --------
class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NEVER = "never"
