# -------- __init__.py --------
# Re-exports the backtester, so callers import from the package.

# -------- Imports --------
from sltp.backtesting.backtester import Backtester


# -------- Exports --------
__all__ = ["Backtester"]
