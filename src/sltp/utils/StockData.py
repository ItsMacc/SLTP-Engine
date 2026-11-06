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
        fields: dict[str, object] = {
            "ticker": ticker,
            "date_codes": _coerce_codes(date_codes if date_codes is not None else dates),
            "open_price": open_price,
            "high": high,
            "low": low,
            "close_price": close_price,
            "volume": volume,
        }
        for name, value in fields.items():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            object.__setattr__(self, name, value)

    # -------- Date views --------
    # The bar dates as a sequence of "YYYY-MM-DD" strings.
    @property
    def dates(self) -> DateLabels:
        return DateLabels(self.date_codes)

    # The bar dates as a "<U10" array, for JSON and tabular output.
    @property
    def labels(self) -> np.ndarray:
        return CALENDAR.label_array(self.date_codes)
