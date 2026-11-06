# -------- compare_indicators.py --------
# Verify sltp.indicators against TA-Lib on synthetic OHLCV bars.
#
# TA-Lib is not a dependency of the package, so this test needs an environment that
# has it. Run it with any interpreter that does:
#
#   PYTHONPATH=src python test/compare_indicators.py
#
# Each case checks two things: the first finite value lands on the same bar (the
# warm-up must match, or the engine's window caching would shift results) and the
# values agree within a relative tolerance.

# -------- Imports --------
import sys

import numpy as np
import talib

from sltp import indicators as ours


# -------- Fixtures --------
def bars(count: int = 900) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20240105)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, count)))
    open_price = close * (1.0 + rng.normal(0.0, 0.002, count))
    high = np.maximum.reduce([close * (1.0 + np.abs(rng.normal(0.0, 0.004, count))), close, open_price])
    low = np.minimum.reduce([close * (1.0 - np.abs(rng.normal(0.0, 0.004, count))), close, open_price])
    volume = rng.integers(100_000, 5_000_000, count).astype(np.float64)
    return open_price, high, low, close, volume


# -------- Comparison --------
def as_tuple(value) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def compare(name: str, mine, theirs, tolerance: float) -> bool:
    mine_outputs = as_tuple(mine)
    their_outputs = as_tuple(theirs)

    if len(mine_outputs) != len(their_outputs):
        print(f"FAIL {name:<10} output count {len(mine_outputs)} vs {len(their_outputs)}")
        return False

    ok = True
    notes: list[str] = []
    for index, (our_values, ta_values) in enumerate(zip(mine_outputs, their_outputs)):
        a = np.asarray(our_values, dtype=np.float64)
        b = np.asarray(ta_values, dtype=np.float64)

        if a.shape != b.shape:
            print(f"FAIL {name:<10} shape {a.shape} vs {b.shape}")
            return False

        finite = np.isfinite(a) & np.isfinite(b)
        first_ours = int(np.argmax(np.isfinite(a))) if np.isfinite(a).any() else -1
        first_theirs = int(np.argmax(np.isfinite(b))) if np.isfinite(b).any() else -1

        if first_ours != first_theirs:
            ok = False
            notes.append(f"[{index}] warm-up {first_ours} vs {first_theirs}")

        if finite.any():
            scale = max(1.0, float(np.max(np.abs(b[finite]))))
            error = float(np.max(np.abs(a[finite] - b[finite]))) / scale
        else:
            error = 0.0

        if error > tolerance:
            ok = False
        notes.append(f"[{index}] err {error:.1e}")

    status = "ok  " if ok else "FAIL"
    print(f"{status} {name:<10} tol {tolerance:.0e}  " + "  ".join(notes))
    return ok


# -------- Suite --------
def main() -> int:
    open_price, high, low, close, volume = bars()

    cases = [
        ("SMA", ours.SMA(close, 20), talib.SMA(close, 20), 1e-9),
        ("EMA", ours.EMA(close, 20), talib.EMA(close, 20), 1e-9),
        ("DEMA", ours.DEMA(close, 20), talib.DEMA(close, 20), 1e-9),
        ("TEMA", ours.TEMA(close, 20), talib.TEMA(close, 20), 1e-8),
        ("TRIX", ours.TRIX(close, 15), talib.TRIX(close, 15), 1e-8),
        ("RSI", ours.RSI(close, 14), talib.RSI(close, 14), 1e-8),
        ("MACD", ours.MACD(close), talib.MACD(close), 1e-7),
        ("BBANDS", ours.BBANDS(close, 20, 2, 2, 0), talib.BBANDS(close, 20, 2, 2, 0), 1e-9),
        ("ATR", ours.ATR(high, low, close, 14), talib.ATR(high, low, close, 14), 1e-8),
        ("ADX", ours.ADX(high, low, close, 14), talib.ADX(high, low, close, 14), 1e-6),
        ("STOCH", ours.STOCH(high, low, close, 5, 3, 0, 3, 0), talib.STOCH(high, low, close, 5, 3, 0, 3, 0), 1e-7),
        ("WILLR", ours.WILLR(high, low, close, 14), talib.WILLR(high, low, close, 14), 1e-8),
        ("CCI", ours.CCI(high, low, close, 14), talib.CCI(high, low, close, 14), 1e-7),
        ("ROC", ours.ROC(close, 10), talib.ROC(close, 10), 1e-9),
        ("MOM", ours.MOM(close, 10), talib.MOM(close, 10), 1e-9),
        ("OBV", ours.OBV(close, volume), talib.OBV(close, volume), 1e-9),
        ("MFI", ours.MFI(high, low, close, volume, 14), talib.MFI(high, low, close, volume, 14), 1e-7),
        ("AD", ours.AD(high, low, close, volume), talib.AD(high, low, close, volume), 1e-9),
        # SAR's opening 4 bars are clamped differently by TA-Lib; it settles
        # afterwards, so the tolerance covers the initialisation rather than the
        # recursion. No strategy in the engine uses SAR.
        ("SAR", ours.SAR(high, low, 0.02, 0.2), talib.SAR(high, low, 0.02, 0.2), 1e-3),
        ("STOCHRSI", ours.STOCHRSI(close, 14, 5, 3, 0), talib.STOCHRSI(close, 14, 5, 3, 0), 1e-7),
    ]

    failures: int = 0
    for name, mine, theirs, tolerance in cases:
        if not compare(name, mine, theirs, tolerance):
            failures += 1

    print()
    print(f"{len(cases) - failures}/{len(cases)} indicators match TA-Lib {talib.__version__}")
    return 1 if failures else 0


# -------- Entry Point --------
if __name__ == "__main__":
    sys.exit(main())
