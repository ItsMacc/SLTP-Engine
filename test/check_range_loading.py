# -------- check_range_loading.py --------
# Prove that loading only the bundles a backtest touches is lossless. Three claims are checked: 1. 2. 3.

# -------- Imports --------
import os
import sys
import time

import numpy as np

from sltp.market import market_data as service

from sltp import Backtester, Strategy
from sltp.market.market_data import files_for as flat_files_for
from sltp.utils import Order, PositionType


# -------- Baseline --------
# The buy-and-hold result recorded before range loading: if warm-up bars go missing, this number moves.
BASELINE_END_VALUE: float = 118.0106477949049
BASELINE_TOLERANCE: float = 1e-9


# -------- Strategies --------
class BuyAndHold(Strategy):
    TICKER: str = "SPY"

    def __init__(self) -> None:
        super().__init__()
        self.name = "Buy and Hold"
        self.MAX_POSITIONS = 1
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = 1

    def initialize_universe(self, current_date: str, histories=None) -> bool:
        self.tickers.clear()
        if self.TICKER not in self.available_universe:
            return False
        self.add_ticker(self.TICKER)
        return True

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers or float(data.closes[-1]) <= 0.0:
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=1.0,
            stop_loss=0.0,
            take_profit=0.0,
        )


class TrendFilter(Strategy):
    """Hold SPY only while it trades above its 200-day average.

    The 200-day window is the point: it needs bars from before the run's first
    session, so it fails loudly if range loading starves the warm-up.
    """

    TICKER: str = "SPY"
    WINDOW: int = 200

    def __init__(self) -> None:
        super().__init__()
        self.name = "Trend Filter (200-day)"
        self.MAX_POSITIONS = 1
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = self.WINDOW

    def initialize_universe(self, current_date: str, histories=None) -> bool:
        self.tickers.clear()
        if self.TICKER not in self.available_universe:
            return False
        self.add_ticker(self.TICKER)
        return True

    def _above(self, data) -> bool:
        closes = data.closes
        if len(closes) < self.WINDOW:
            return False
        return float(closes[-1]) > float(np.mean(closes[-self.WINDOW:]))

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers or not self._above(data):
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=1.0,
            stop_loss=0.0,
            take_profit=0.0,
        )

    def should_exit(self, ticker: str, trade, history=None) -> bool:
        data = history if history is not None else trade.history
        if data is None:
            return False
        return not self._above(data)


# -------- Runner --------
def run(strategy: type, label: str, lookback: int | None = None) -> dict:
    service.refresh()
    instance = strategy()
    if lookback is not None:
        instance.LOOKBACK_BARS = lookback
    backtester = Backtester(
        strategy=instance,
        start_date="2020-01-02",
        end_date="2020-12-31",
    )
    # The engine's own window, so the count matches the files it really reads.
    reading = len(flat_files_for(backtester.load_start, backtester.end_date))
    started = time.time()
    result = backtester.run()
    elapsed = time.time() - started
    result["files_read"] = reading
    print(f"  {label:<22} {elapsed:5.1f}s  files={reading}  "
          f"sessions={len(result['equity']):>4}  trades={len(result['trades'])}")
    return result


# -------- Entry Point --------
def main() -> int:
    everything = flat_files_for(None, None)
    window = flat_files_for("2015-01-02", "2020-12-31")

    print(f"bundles on disk   : {len(everything)}")
    print(f"a 2020 run reads  : {len(window)}")
    for path in window:
        print(f"   {os.path.basename(path)}")
    for path in everything:
        if path not in window:
            print(f"   skipped {os.path.basename(path)}")
    print()

    limited = run(TrendFilter, "range-aware")
    buy_hold = run(BuyAndHold, "range-aware B&H")

    # The same run reaching back to the first bundle: if pruning dropped warm-up bars, this differs.
    full = run(TrendFilter, "long warm-up", lookback=7000)

    print()
    problems: list[str] = []

    if len(window) >= len(everything):
        problems.append(f"pruning selected every file ({len(window)} of {len(everything)})")

    if full["files_read"] <= limited["files_read"]:
        problems.append("the long warm-up run read no extra bundles, so the comparison proves nothing")

    if not limited["trades"]:
        problems.append("the 200-day strategy never traded: warm-up bars are missing")

    if limited["equity"] != full["equity"]:
        problems.append("equity curves differ between the pruned and the long warm-up run")
    if limited["metrics"] != full["metrics"]:
        problems.append("metrics differ between the pruned and the long warm-up run")
    if len(limited["trades"]) != len(full["trades"]):
        problems.append("trade counts differ between the pruned and the long warm-up run")

    if buy_hold["equity"]:
        end_value = float(buy_hold["equity"][-1]["portfolio_value"])
        if abs(end_value - BASELINE_END_VALUE) > BASELINE_TOLERANCE:
            problems.append(f"buy and hold moved off the baseline: {end_value!r}")
        else:
            print(f"buy and hold ends at {end_value!r}, matching the baseline")

    if problems:
        print()
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    print()
    print("ok: a pruned run matches one that reaches back to the first bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
