import time
import numpy as np


# ==========================================
# 1. PROBE & CURSOR ENGINE CLASSES
# ==========================================

class _SeriesProbe:
    """Wraps a NumPy array to record comparisons & math inside on_signal()."""
    __slots__ = ("arr", "log", "branch_hints")

    def __init__(self, arr: np.ndarray, log: list, branch_hints: dict):
        self.arr = arr
        self.log = log
        self.branch_hints = branch_hints

    def _u(self, other):
        return other.arr if isinstance(other, _SeriesProbe) else other

    def _rec(self, op_name: str, mask: np.ndarray) -> np.ndarray:
        self.log.append((op_name, mask))
        return mask

    # Lookback helper: e.g. bar.close.ago(2)
    def ago(self, n: int):
        shifted = np.empty_like(self.arr, dtype=np.float32)
        shifted[:n] = np.nan
        shifted[n:] = self.arr[:-n]
        return _SeriesProbe(shifted, self.log, self.branch_hints)

    # Comparisons
    def __gt__(self, o): return self._rec(">", self.arr > self._u(o))
    def __lt__(self, o): return self._rec("<", self.arr < self._u(o))
    def __ge__(self, o): return self._rec(">=", self.arr >= self._u(o))
    def __le__(self, o): return self._rec("<=", self.arr <= self._u(o))
    def __eq__(self, o): return self._rec("==", self.arr == self._u(o))
    def __ne__(self, o): return self._rec("!=", self.arr != self._u(o))

    # Arithmetic
    def __add__(self, o): return _SeriesProbe(self.arr + self._u(o), self.log, self.branch_hints)
    def __radd__(self, o): return _SeriesProbe(self._u(o) + self.arr, self.log, self.branch_hints)
    def __sub__(self, o): return _SeriesProbe(self.arr - self._u(o), self.log, self.branch_hints)
    def __rsub__(self, o): return _SeriesProbe(self._u(o) - self.arr, self.log, self.branch_hints)
    def __mul__(self, o): return _SeriesProbe(self.arr * self._u(o), self.log, self.branch_hints)
    def __rmul__(self, o): return _SeriesProbe(self._u(o) * self.arr, self.log, self.branch_hints)
    def __truediv__(self, o): return _SeriesProbe(self.arr / self._u(o), self.log, self.branch_hints)


class _ScalarSeries:
    """Fast scalar wrapper for real bar-by-bar execution."""
    __slots__ = ("arr", "cursor")

    def __init__(self, arr: np.ndarray, cursor):
        self.arr = arr
        self.cursor = cursor

    @property
    def val(self) -> float:
        return float(self.arr[self.cursor.i])

    def ago(self, n: int) -> float:
        idx = self.cursor.i - n
        return float(self.arr[idx]) if idx >= 0 else float("nan")

    def __gt__(self, o): return self.val > float(o)
    def __lt__(self, o): return self.val < float(o)
    def __ge__(self, o): return self.val >= float(o)
    def __le__(self, o): return self.val <= float(o)
    def __eq__(self, o): return self.val == float(o)
    def __ne__(self, o): return self.val != float(o)
    def __add__(self, o): return self.val + float(o)
    def __radd__(self, o): return float(o) + self.val
    def __sub__(self, o): return self.val - float(o)
    def __rsub__(self, o): return float(o) - self.val
    def __mul__(self, o): return self.val * float(o)
    def __rmul__(self, o): return float(o) * self.val
    def __truediv__(self, o): return self.val / float(o)
    def __float__(self): return self.val


class BarCursor:
    """Zero-allocation cursor reused across all bars."""
    __slots__ = ("open", "high", "low", "close", "volume", "i")

    def __init__(self, cols: dict):
        self.open = _ScalarSeries(cols["open"], self)
        self.high = _ScalarSeries(cols["high"], self)
        self.low = _ScalarSeries(cols["low"], self)
        self.close = _ScalarSeries(cols["close"], self)
        self.volume = _ScalarSeries(cols["volume"], self)
        self.i = 0


def discover_candidates(strategy, cols: dict):
    """
    Runs on_signal() once on full NumPy arrays to capture all condition masks.
    Returns (candidate_indices, captured_logs, fell_back_to_brute_force).
    """
    n = len(cols["close"])
    log = []
    probe = type("ProbeBar", (), {
        k: _SeriesProbe(cols[k], log, {}) for k in ("open", "high", "low", "close", "volume")
    })()

    try:
        res = strategy.on_signal(probe)
        # If the user returned a pure boolean mask directly without `if` statements!
        if isinstance(res, np.ndarray) and res.dtype == bool:
            return np.flatnonzero(res), log, False
    except ValueError:
        # Expected! Python stopped at the first `if <numpy_array>:` after recording `log`.
        pass
    except Exception:
        # User wrote un-vectorizable Python (e.g., math.sin, external APIs, etc.) -> safe fallback!
        return np.arange(n, dtype=np.int32), log, True

    if not log:
        return np.arange(n, dtype=np.int32), log, True

    # Keep Day 0 + any bar where ANY logged condition is True OR flips True <-> False
    active = np.zeros(n, dtype=bool)
    active[0] = True
    for _, mask in log:
        clean = np.nan_to_num(mask, nan=False)
        active |= clean
        active[1:] |= (clean[1:] != clean[:-1])

    return np.flatnonzero(active), log, False


# ==========================================
# 2. GOOFY USER STRATEGY (EDIT THIS!)
# ==========================================

class GoofyUserStrategy:
    def on_signal(self, bar):
        # --- Write whatever goofy conditions you want here! ---
        c2 = bar.close.ago(2)
        c5 = bar.close.ago(5)

        # Weird multi-variable math + nested conditions
        weird_momentum = (bar.high - c2) / (c5 + 0.0001)
        if weird_momentum > 0.08:
            if bar.close > bar.open * 1.015:
                return {"action": "BUY", "stop_loss": float(bar.close) * 0.95}

        return None


# ==========================================
# 3. VERIFICATION & BENCHMARK HARNESS
# ==========================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 7_056  # 28 years of daily bars for 1 stock

    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, N))).astype(np.float32)
    cols = {
        "open": (close * rng.uniform(0.99, 1.01, N)).astype(np.float32),
        "high": (close * rng.uniform(1.00, 1.04, N)).astype(np.float32),
        "low": (close * rng.uniform(0.96, 1.00, N)).astype(np.float32),
        "close": close,
        "volume": rng.integers(100_000, 2_000_000, N).astype(np.float32),
    }

    strat = GoofyUserStrategy()

    # --- A. Brute-Force Ground Truth (Check every single bar 0..7055) ---
    t0 = time.perf_counter()
    cursor = BarCursor(cols)
    truth_orders = []
    for i in range(N):
        cursor.i = i
        order = strat.on_signal(cursor)
        if order is not None:
            truth_orders.append((i, order))
    t_brute = (time.perf_counter() - t0) * 1000

    # --- B. Fast Probe Engine (Discover conditions -> run only candidates) ---
    t0 = time.perf_counter()
    candidates, logs, fell_back = discover_candidates(strat, cols)
    fast_orders = []
    for i in candidates:
        cursor.i = int(i)
        order = strat.on_signal(cursor)
        if order is not None:
            fast_orders.append((int(i), order))
    t_fast = (time.perf_counter() - t0) * 1000

    # --- Report ---
    print(f"Total bars:          {N}")
    print(f"Conditions captured: {len(logs)} -> ops: {[op for op, _ in logs]}")
    print(f"Fell back to full?   {fell_back}")
    print(f"Bars checked:        {len(candidates)} / {N} (skipped {N - len(candidates)} quiet bars!)")
    print(f"Brute-force orders:  {len(truth_orders)} in {t_brute:.2f} ms")
    print(f"Probe-fast orders:   {len(fast_orders)} in {t_fast:.2f} ms")
    assert truth_orders == fast_orders, "MISMATCH!"
    print("EXACT MATCH: 100% identical orders!")
