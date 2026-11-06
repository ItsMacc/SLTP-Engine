# -------- smoke_backtest.py --------
# Run the whole engine once, against the real bar files.
#
# This is the check that the package is actually wired up: the path resolver found
# the data, the Parquet reader loaded it, the indicator module stood in for TA-Lib,
# the portfolio manager filled an order and the metrics came back.
#
#   python test/smoke_backtest.py

# -------- Imports --------
import os
import resource
import sys

from sltp import Backtester, Strategy
from sltp.data.market_data_service import bar_files, get_latest_trading_date
from sltp.utils import Order, PositionType


# -------- Strategy --------
# The smallest strategy that proves the plumbing: one fund, bought and held.
class BuyAndHold(Strategy):
    TICKER: str = "SPY"
    ALLOCATION: float = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.name = "Smoke Buy and Hold"
        self.MAX_POSITIONS = 1
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = 1

    def initialize_universe(self, current_date: str, histories: dict) -> bool:
        self.tickers.clear()
        if self.TICKER not in histories:
            return False
        self.add_ticker(self.TICKER)
        return True

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers:
            return None
        if float(data.close_price[-1]) <= 0.0:
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=self.ALLOCATION,
            order_date=str(data.dates[-1]),
            stop_loss=0.0,
            take_profit=0.0,
        )


# -------- Peak Memory --------
# macOS reports bytes and Linux kilobytes, so normalise to MiB.
def peak_memory_mib() -> float:
    peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


# -------- Entry Point --------
def main() -> int:
    files: list[str] = bar_files()
    print(f"bar files: {len(files)}")
    for path in files:
        print(f"  {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB)")
    print(f"latest session in the bar files: {get_latest_trading_date()}")

    result = Backtester(
        strategy=BuyAndHold(),
        start_date="2020-01-02",
        end_date="2020-12-31",
    ).run()

    metrics = result["metrics"]
    equity = result["equity"]
    trades = result["trades"]

    print(f"metric keys: {', '.join(sorted(metrics))}")
    print(f"equity points: {len(equity)}")
    print(f"trades: {len(trades)}")

    if equity:
        first = equity[0]
        last = equity[-1]
        print(f"first point: {first}")
        print(f"last point : {last}")

    print(f"peak resident memory: {peak_memory_mib():.0f} MiB")

    problems: list[str] = []
    if not equity:
        problems.append("no equity curve")
    if len(equity) < 200:
        problems.append(f"only {len(equity)} sessions, expected a full year")
    if not trades:
        problems.append("no trades were filled")
    if equity and abs(float(equity[0]["portfolio_value"]) - 100.0) > 1e-6:
        problems.append(f"index did not start at 100: {equity[0]['portfolio_value']}")
    if equity:
        final = float(equity[-1]["portfolio_value"])
        if not final > 0.0:
            problems.append(f"final index is not positive: {final}")

    if problems:
        print()
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    print()
    print("ok: engine ran end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
