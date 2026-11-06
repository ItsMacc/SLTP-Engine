# -------- two_strategies.py --------
# Two tiny strategies that exercise the CLI: one that trades on a trend filter and
# one that holds a fixed mix. They are fast, need no tuning and use only the public
# imports a user is meant to rely on.
#
#   sltp backtest test/fixtures/two_strategies.py:Momentum
#   sltp compare test/fixtures/two_strategies.py:Momentum test/fixtures/two_strategies.py:Value

# -------- Imports --------
import numpy as np

from sltp import Order, PositionType, Strategy


# -------- Strategies --------
class Momentum(Strategy):
    TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    LOOKBACK: int = 60
    TREND: int = 20

    def __init__(self) -> None:
        super().__init__()
        self.name = "Momentum 20"
        self.MAX_POSITIONS = 2
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = self.LOOKBACK

    def initialize_universe(self, current_date: str, histories: dict) -> bool:
        self.tickers.clear()
        for ticker in self.TICKERS:
            if ticker in histories:
                self.add_ticker(ticker)
        return bool(self.tickers)

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers:
            return None
        closes = data.close_price
        if len(closes) < self.LOOKBACK:
            return None
        if float(closes[-1]) <= float(np.mean(closes[-self.TREND:])):
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=0.5,
            order_date=str(data.dates[-1]),
            stop_loss=0.0,
            take_profit=0.0,
        )


class Value(Strategy):
    TICKERS: tuple[str, ...] = ("SPY", "TLT")
    WEIGHTS: tuple[float, ...] = (0.6, 0.4)

    def __init__(self) -> None:
        super().__init__()
        self.name = "Static 60/40"
        self.MAX_POSITIONS = 2
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = 1

    def initialize_universe(self, current_date: str, histories: dict) -> bool:
        self.tickers.clear()
        for ticker in self.TICKERS:
            if ticker in histories:
                self.add_ticker(ticker)
        return bool(self.tickers)

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers or ticker not in self.TICKERS:
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=self.WEIGHTS[self.TICKERS.index(ticker)],
            order_date=str(data.dates[-1]),
            stop_loss=0.0,
            take_profit=0.0,
        )
