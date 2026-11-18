# -------- StockData.py --------
# One ticker's daily OHLCV history, plus the window of it a strategy is allowed to see.
# The arrays are frozen on construction and never change. Only the window moves, and it lives
# in a two-item list because rebinding a frozen field raises while mutating a list does not.
# A `data` handed to a callback is valid for that callback alone: keeping a reference across
# days reads whatever window the engine has moved to since, which is a silent lookahead bug.

# -------- Imports --------
import numpy as np

from dataclasses import dataclass

from sltp.utils.Dates import CALENDAR, DateLabels, codes_from_labels


# -------- Stock Data --------
# Read-only container for one ticker's bars and the window over them.
# eq=False on purpose: the fields are numpy arrays, so the generated __eq__ compares
# them elementwise and raises on bool(), and the generated __hash__ hashes an array.
# These objects are compared by identity anyway -- there is only one per ticker.
@dataclass(frozen=True, init=False, eq=False)
class StockData:
    ticker: str
    _codes: np.ndarray
    _opens: np.ndarray
    _highs: np.ndarray
    _lows: np.ndarray
    _closes: np.ndarray
    _volumes: np.ndarray

    # Store the full bars and make every array read-only.
    def __init__(self, ticker: str, date_codes=None, opens=None, highs=None,
                 lows=None, closes=None, volumes=None):
        put = object.__setattr__
        put(self, "ticker", ticker)
        put(self, "_codes", _frozen(_coerce_codes(date_codes)))
        put(self, "_opens", _frozen(opens))
        put(self, "_highs", _frozen(highs))
        put(self, "_lows", _frozen(lows))
        put(self, "_closes", _frozen(closes))
        put(self, "_volumes", _frozen(volumes))

        # The whole history is visible until the engine narrows it.
        put(self, "_window", [0, int(np.size(self._closes))])
        put(self, "_views", {})

    # -------- Window --------
    # Show bars [start, end), the last of which is today.
    def set_window(self, start: int, end: int) -> None:
        length: int = int(np.size(self._closes))
        if start < 0 or end < start or end > length:
            raise ValueError(
                f"bad window [{start}, {end}) for {self.ticker} with {length} bars"
            )

        self._window[0] = int(start)
        self._window[1] = int(end)
        self._views.clear()

    # Where the window sits, for a rebalance snapshot. Two ints, never an object.
    @property
    def window(self) -> tuple[int, int]:
        return self._window[0], self._window[1]

    def __len__(self) -> int:
        return self._window[1] - self._window[0]

    # -------- Bar Views --------
    @property
    def date_codes(self) -> np.ndarray:
        return self._view("codes")

    @property
    def opens(self) -> np.ndarray:
        return self._view("opens")

    @property
    def highs(self) -> np.ndarray:
        return self._view("highs")

    @property
    def lows(self) -> np.ndarray:
        return self._view("lows")

    @property
    def closes(self) -> np.ndarray:
        return self._view("closes")

    @property
    def volumes(self) -> np.ndarray:
        return self._view("volumes")

    # -------- Engine View --------
    # The complete arrays, ignoring the window.
    @property
    def full(self) -> tuple[np.ndarray, ...]:
        return self._codes, self._opens, self._highs, self._lows, self._closes, self._volumes

    # -------- Date Views --------
    # The bar dates as "YYYY-MM-DD" labels, so dates[-1] is today.
    @property
    def dates(self) -> DateLabels:
        labels = self._views.get("dates")
        if labels is None:
            labels = DateLabels(self.date_codes)
            self._views["dates"] = labels
        return labels

    # The bar dates as a "<U10" array, for JSON and tabular output.
    @property
    def labels(self) -> np.ndarray:
        return CALENDAR.label_array(self.date_codes)


    # One cached slice per name for the current window.
    def _view(self, name: str) -> np.ndarray:
        view = self._views.get(name)
        if view is None:
            start, end = self._window
            view = getattr(self, "_" + name)[start:end]
            self._views[name] = view
        return view


# -------- Helpers --------
# Accept date codes, "YYYY-MM-DD" labels or datetimes and return int32 codes.
def _coerce_codes(value) -> np.ndarray:
    if value is None:
        raise TypeError("StockData requires date_codes.")
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


# Make an array read-only, unless it already is.
def _frozen(value):
    if isinstance(value, np.ndarray) and value.flags.writeable:
        value.flags.writeable = False
    return value
