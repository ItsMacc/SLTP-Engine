# -------- StockData.py --------
# One ticker's daily OHLCV history as parallel numpy arrays.
# The engine slices them up to the current date, so the last element is "today" and there
# is no lookahead. Dates are int32 day numbers, but .dates still returns "YYYY-MM-DD".

# -------- Imports --------
import numpy as np

from dataclasses import dataclass

# Engine imports
from sltp.utils.Dates import CALENDAR, DateLabels, codes_from_labels


# -------- Helpers --------
# Accept date codes, "YYYY-MM-DD" labels or datetimes and return int32 codes.
def _coerce_codes(value) -> np.ndarray:
    if value is None:
        raise TypeError("StockData requires date_codes (or legacy dates).")
    if isinstance(value, DateLabels):
        return value._codes
    array = np.asarray(value)
    if array.dtype == np.int32:
        return array
    if array.dtype.kind in "USOM":
        return codes_from_labels(array)
    if array.dtype.kind in "iu":
        return array.astype(np.int32)
    raise TypeError(f"Cannot build dates from {array.dtype} values.")


# Make an array read-only, unless it already is. Slicing a read-only history hands
# back read-only views, so the flag is already False most of the time and setting it
# again costs more than building the object around it.
def _frozen(value):
    if isinstance(value, np.ndarray) and value.flags.writeable:
        value.flags.writeable = False
    return value


# -------- Stock Data --------
# Read-only container for one ticker's bars.
@dataclass(frozen=True, init=False)
class StockData:
    ticker: str
    date_codes: np.ndarray
    open_price: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close_price: np.ndarray
    volume: np.ndarray

    # Store the bars and make every array read-only.
    # `dates` is the legacy keyword: old readers pass "YYYY-MM-DD" arrays and they
    # are converted to codes here.
    def __init__(
        self,
        ticker: str,
        date_codes=None,
        open_price=None,
        high=None,
        low=None,
        close_price=None,
        volume=None,
        *,
        dates=None,
    ):
        put = object.__setattr__
        put(self, "ticker", ticker)
        put(self, "date_codes", _frozen(_coerce_codes(date_codes if date_codes is not None else dates)))
        put(self, "open_price", _frozen(open_price))
        put(self, "high", _frozen(high))
        put(self, "low", _frozen(low))
        put(self, "close_price", _frozen(close_price))
        put(self, "volume", _frozen(volume))

    # The engine's own slices: the arrays are already int32 date codes from another
    # StockData, so this skips the coercion above. The result is read-only, like any
    # other history, because every array is frozen before it is stored.
    @classmethod
    def view(
        cls,
        ticker: str,
        date_codes: np.ndarray,
        open_price,
        high,
        low,
        close_price,
        volume,
    ) -> "StockData":
        instance: StockData = object.__new__(cls)
        put = object.__setattr__
        put(instance, "ticker", ticker)
        put(instance, "date_codes", _frozen(date_codes))
        put(instance, "open_price", _frozen(open_price))
        put(instance, "high", _frozen(high))
        put(instance, "low", _frozen(low))
        put(instance, "close_price", _frozen(close_price))
        put(instance, "volume", _frozen(volume))
        return instance

    # -------- Date views --------
    # The bar dates as a sequence of "YYYY-MM-DD" strings.
    @property
    def dates(self) -> DateLabels:
        return DateLabels(self.date_codes)

    # The bar dates as a "<U10" array, for JSON and tabular output.
    @property
    def labels(self) -> np.ndarray:
        return CALENDAR.label_array(self.date_codes)
