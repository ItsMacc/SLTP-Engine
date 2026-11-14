# -------- __init__.py --------
# The package's public surface:
#
#   from sltp import Strategy, Backtester, Order, PositionType
#
# sltp.data is deliberately not exported: a strategy asks the engine for bars, it does not
# read Parquet itself. Set SLTP_DATA_DIR before importing when the bars live elsewhere.
#
# Nothing heavy is imported here. The names below resolve on first use, so importing
# the package (which `import sltp.cli` always does) never waits on numpy or
# pyarrow. Only --version and --help stay fast; real work still loads everything.

# -------- Imports --------
import importlib

from typing import TYPE_CHECKING


# -------- Version --------
try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    __version__ = _dist_version("sltp")
except PackageNotFoundError:
    __version__ = "0.0.0"


# -------- Lazy Exports --------
# Public name to the submodule that defines it.
_LAZY_EXPORTS: dict[str, str] = {
    "Backtester": "sltp.backtesting",
    "PortfolioManager": "sltp.portfolio",
    "Strategy": "sltp.strategy",
    "CALENDAR": "sltp.utils",
    "Event": "sltp.utils",
    "EventType": "sltp.utils",
    "Order": "sltp.utils",
    "OrderSizing": "sltp.utils",
    "PositionType": "sltp.utils",
    "RebalanceFrequency": "sltp.utils",
    "StockData": "sltp.utils",
    "Trade": "sltp.utils",
}


# Static mirror of the lazy table below, so IDEs and type checkers resolve
# `from sltp import Strategy` without importing anything. Never costs a numpy
# import: TYPE_CHECKING is always False at runtime.
if TYPE_CHECKING:
    from sltp.backtesting import Backtester
    from sltp.portfolio import PortfolioManager
    from sltp.strategy import Strategy
    from sltp.utils import (
        CALENDAR,
        Event,
        EventType,
        Order,
        OrderSizing,
        PositionType,
        RebalanceFrequency,
        StockData,
        Trade,
    )


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    value = getattr(module, name)
    globals()[name] = value
    return value


# -------- Exports --------
__all__ = [
    "Backtester",
    "CALENDAR",
    "Event",
    "EventType",
    "Order",
    "OrderSizing",
    "PortfolioManager",
    "PositionType",
    "RebalanceFrequency",
    "StockData",
    "Strategy",
    "Trade",
    "__version__",
]
