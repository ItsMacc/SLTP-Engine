# -------- test_streaming_indicators.py --------
# Verify the streaming indicator states against the batch functions bar by bar.
#
# Each case advances a state exactly like the engine does (one prefix slice per day)
# and demands bit-identical values to the batch array at every bar, through the
# real Strategy._ta front door. Run it with:
#
#   .venv/bin/python test/test_streaming_indicators.py

# -------- Imports --------
import sys

import numpy as np

sys.path.insert(0, "src")

from sltp import indicators as ours
from sltp.strategy.base_strategy import Strategy


# -------- Fixtures --------
# Random-walk bars with a fixed seed, so failures reproduce exactly.
def bars(count: int = 900, seed: int = 20240105) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, count)))
    open_price = close * (1.0 + rng.normal(0.0, 0.002, count))
    high = np.maximum.reduce([close * (1.0 + np.abs(rng.normal(0.0, 0.004, count))), close, open_price])
    low = np.minimum.reduce([close * (1.0 - np.abs(rng.normal(0.0, 0.004, count))), close, open_price])
    volume = rng.integers(100_000, 5_000_000, count).astype(np.float64)
    return open_price, high, low, close, volume


# A strategy with a fabricated series registry, driven exactly like the backtester
# drives it: full float32 history arrays, one growing prefix slice per day. Slices
# must come from the stored float32 fulls so the pointer registry resolves them.
class Harness(Strategy):
    def __init__(self, ticker: str, full: tuple[np.ndarray, ...]) -> None:
        super().__init__()
        self._full = tuple(np.asarray(array, dtype=np.float32) for array in full)
        addresses = {array.ctypes.data: array for array in self._full}
        self._indicator_series = addresses
        self._indicator_ticker = ticker

    def day(self, length: int) -> tuple[np.ndarray, ...]:
        return tuple(array[:length] for array in self._full)


# The batch reference for one configuration, computed on the widened fulls exactly
# like _ta's batch path does.
def expected_outputs(helper: str, params: tuple, full: tuple[np.ndarray, ...]) -> tuple:
    if helper == "_macd":
        fast, slow, signal = params
        expected = ours.MACD(np.asarray(full[0], dtype=np.float64), fast, slow, signal)
    elif helper in ("_atr", "_adx"):
        raw = getattr(ours, helper.strip("_").upper())
        widened = [np.asarray(array, dtype=np.float64) for array in full]
        expected = raw(*widened, *params)
    else:
        raw = getattr(ours, helper.strip("_").upper())
        expected = raw(np.asarray(full[0], dtype=np.float64), *params)
    return expected if isinstance(expected, tuple) else (expected,)


# Advance one indicator day by day and compare each day against the batch array.
def check_daily(
    label: str, helper: str, params: tuple, full: tuple[np.ndarray, ...], failures: list
) -> None:
    strategy = Harness("AAA", full)
    method = getattr(strategy, helper)
    expected = expected_outputs(helper, params, strategy._full)

    worst = 0.0
    days = strategy._full[0].size
    for length in range(1, days + 1):
        got = method(*strategy.day(length), *params)
        got_outputs = got if isinstance(got, tuple) else (got,)
        if len(got_outputs) != len(expected):
            failures.append(f"{label}: output count {len(got_outputs)} vs {len(expected)}")
            return
        for position, (got_array, want_array) in enumerate(zip(got_outputs, expected)):
            values = np.asarray(got_array, dtype=np.float64).ravel()
            if values.size != 1:
                failures.append(f"{label} output {position}: expected 1 value, got {values.size}")
                return
            want = float(want_array[length - 1])
            if values[0] != values[0] and want != want:
                continue
            if values[0] != want:
                failures.append(f"{label} output {position} day {length}: {values[0]!r} vs {want!r}")
                return
            if values[0] == values[0]:
                worst = max(worst, abs(values[0] - want))
    print(f"ok: {label} ({days} days, max abs diff {worst:.3g})")


# -------- Cases --------
def main() -> int:
    failures: list[str] = []
    _, high, low, close, _ = bars()

    for label, helper, params, full in [
        ("EMA 20", "_ema", (20,), (close,)),
        ("EMA 5", "_ema", (5,), (close,)),
        ("EMA 50", "_ema", (50,), (close,)),
        ("RSI 14", "_rsi", (14,), (close,)),
        ("RSI 7", "_rsi", (7,), (close,)),
        ("RSI 21", "_rsi", (21,), (close,)),
        ("ATR 14", "_atr", (14,), (high, low, close)),
        ("ATR 7", "_atr", (7,), (high, low, close)),
        ("ADX 14", "_adx", (14,), (high, low, close)),
        ("ADX 10", "_adx", (10,), (high, low, close)),
        ("ADX 20", "_adx", (20,), (high, low, close)),
        ("MACD 12,26,9", "_macd", (12, 26, 9), (close,)),
        ("MACD 8,17,9", "_macd", (8, 17, 9), (close,)),
    ]:
        check_daily(label, helper, params, full, failures)

    # Two tickers interleaved day by day, sharing nothing.
    other = bars(seed=99)[3]
    first = Harness("AAA", (close,))
    second = Harness("BBB", (other,))
    want_first = expected_outputs("_rsi", (14,), first._full)[0]
    want_second = expected_outputs("_rsi", (14,), second._full)[0]
    for length in range(1, first._full[0].size + 1):
        a = float(np.asarray(first._rsi(*first.day(length), 14)).ravel()[0])
        b = float(np.asarray(second._rsi(*second.day(length), 14)).ravel()[0])
        same_a = a == float(want_first[length - 1]) or (a != a and want_first[length - 1] != want_first[length - 1])
        same_b = b == float(want_second[length - 1]) or (b != b and want_second[length - 1] != want_second[length - 1])
        if not (same_a and same_b):
            failures.append(f"interleaved tickers diverge at day {length}")
            break
    else:
        print("ok: interleaved tickers stay independent")

    # First request halfway through the run still matches.
    late = Harness("AAA", (close,))
    want_late = expected_outputs("_ema", (20,), late._full)[0]
    for length in range(400, late._full[0].size + 1):
        got = float(np.asarray(late._ema(*late.day(length), 20)).ravel()[0])
        want = float(want_late[length - 1])
        if got != want and not (got != got and want != want):
            failures.append(f"late first request diverges at day {length}")
            break
    else:
        print("ok: mid-run first request matches")

    # Asking for an earlier bar rewinds and still matches.
    rewound = Harness("AAA", (close,))
    for length in range(1, 501):
        rewound._rsi(*rewound.day(length), 14)
    early = float(np.asarray(rewound._rsi(*rewound.day(50), 14)).ravel()[0])
    if early != float(expected_outputs("_rsi", (14,), rewound._full)[0][49]):
        failures.append("rewind does not match batch")
    else:
        print("ok: rewind matches")

    # NaN bars sprinkled through the series.
    messy_close = close.copy()
    messy_high = high.copy()
    messy_low = low.copy()
    messy_close[100:105] = np.nan
    messy_low[300] = np.nan
    messy_high[500:510] = np.nan
    for label, helper, params, full in [
        ("NaN EMA 20", "_ema", (20,), (messy_close,)),
        ("NaN RSI 14", "_rsi", (14,), (messy_close,)),
        ("NaN ATR 14", "_atr", (14,), (messy_high, messy_low, messy_close)),
        ("NaN ADX 14", "_adx", (14,), (messy_high, messy_low, messy_close)),
        ("NaN MACD", "_macd", (12, 26, 9), (messy_close,)),
    ]:
        check_daily(label, helper, params, full, failures)

    # Flat prices: RSI reads 0, never NaN.
    check_daily("flat RSI 14", "_rsi", (14,), (np.full(300, 50.0),), failures)

    # Series shorter than the warm-up never produce a value.
    check_daily("short EMA 20", "_ema", (20,), (close[:10],), failures)
    check_daily("short ADX 14", "_adx", (14,), (high[:10], low[:10], close[:10]), failures)

    # The batch flag still returns full windows identical to the module functions.
    plain = Harness("AAA", (close,))
    plain.STREAMING_INDICATORS = False
    windowed = plain._rsi(*plain.day(100), 14)
    if not isinstance(windowed, np.ndarray) or windowed.size != 100:
        failures.append("batch fallback changed shape")
    elif not np.array_equal(np.asarray(windowed), expected_outputs("_rsi", (14,), plain._full)[0][:100], equal_nan=True):
        failures.append("batch fallback changed values")
    else:
        print("ok: batch fallback untouched")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("\nall streaming parity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
