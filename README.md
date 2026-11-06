# SLTP

SLTP is a backtesting engine for daily stock bars, built on plain NumPy. You describe a strategy in a small Python class, and SLTP walks it through years of history one session at a time, fills the orders, tracks the portfolio, and reports back with an equity curve, a trade list, and a full set of performance statistics.

A few things define how it works. Strategies only ever see data up to the current bar, so lookahead bias is structurally impossible rather than something you have to police yourself. Splits and dividends are applied to positions automatically. Orders support allocation sizing, stop-losses, and take-profits. The portfolio manager enforces your position limits and records why any order was rejected. And because the hot path is vectorized NumPy over a local columnar archive, a multi-decade run over thousands of tickers completes in seconds on an ordinary laptop.

## Installation

Requires Python 3.11 or newer.

```bash
pip install sltp
```

## Writing a strategy

Create a file called `momentum.py`:

```python
import numpy as np

from sltp import Order, PositionType, Strategy


class Momentum(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Momentum 20"
        self.MAX_POSITIONS = 2
        self.MAX_LOTS_PER_TICKER = 1
        self.MIN_BARS_REQUIRED = 60

    def initialize_universe(self, current_date: str, histories: dict) -> bool:
        self.tickers.clear()
        for ticker in ("SPY", "QQQ", "IWM", "DIA"):
            if ticker in histories:
                self.add_ticker(ticker)
        return bool(self.tickers)

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers:
            return None
        closes = data.close_price
        if len(closes) < 60:
            return None
        if float(closes[-1]) <= float(np.mean(closes[-20:])):
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=0.5,
            order_date=str(data.dates[-1]),
            stop_loss=0.0,
            take_profit=0.0,
        )
```

Two methods carry the strategy. `initialize_universe` runs as the backtest steps forward and decides which tickers are tradable on each date. Returning `False` tells the engine the universe is not ready yet, and it keeps walking forward until you accept. `on_signal` is called once per ticker per bar with that ticker's point-in-time history (`data.close_price`, `data.dates`, and the rest of the OHLCV fields). Return an `Order` to open a position, or `None` to pass.

### Portfolio controls

Set these as attributes, usually in `__init__`:

| Attribute | Default | Meaning |
|---|---|---|
| `MAX_POSITIONS` | `10` | maximum distinct open positions (`None` for unlimited) |
| `MAX_ALLOCATION_PER_TICKER` | `None` | cap on portfolio fraction per ticker |
| `MAX_LOTS_PER_TICKER` | `1` | maximum concurrent lots in one ticker |
| `REBALANCE_FREQUENCY` | `NEVER` | how often `on_rebalance` fires |
| `MIN_BARS_REQUIRED` | `0` | bars of history needed before trading starts |
| `LIQUIDATE_ON_END` | `False` | close everything on the final session |

The strategy also exposes read-only state during a run: `open_tickers`, `open_positions`, `open_position_count`, and current `equity`. Prebuilt universes are available as `available_universe`, `available_stocks`, `available_etfs`, and `available_gold`.

## Running a backtest

From Python:

```python
from sltp import Backtester
from momentum import Momentum

result = Backtester(
    strategy=Momentum(),
    start_date="2019-01-01",
    end_date="2023-12-31",
).run()

print(result["metrics"])
print(result["trades"][-1])
```

`Backtester` takes the strategy, a start date, and an optional end date (defaults to the latest session on disk). Pass `include_dividends=False` to skip cash distributions. The result holds `metrics`, the `equity` curve records, and the full `trades` list.

From the terminal:

```bash
sltp backtest momentum.py --start 2019-01-01 --chart
```

### What gets reported

Every run returns the same metrics dictionary:

| Metric | Meaning |
|---|---|
| `starting_value`, `ending_value` | index level at each end (starts at 100) |
| `cagr` | compound annual growth rate, percent |
| `sharpe`, `sortino` | risk-adjusted return, total and downside-only |
| `psr` | probabilistic Sharpe ratio against a benchmark of 1.0 |
| `expectancy` | expected return per trade |
| `win_rate`, `loss_rate` | share of closed winning and losing trades |
| `average_win`, `average_loss` | mean return of winners and losers, percent |
| `max_drawdown` | worst peak-to-trough fall, percent |
| `drawdown_recovery` | longest stretch below a peak, in sessions |
| `turnover`, `cost_drag` | annualized traded notional and trading costs as a share of equity |
| `total_trading_cost` | cumulative costs over the run |
| `dividend_income`, `unrealized_profit` | cash distributions received and open profit at the end |
| `rejected_orders` | count of refused orders by reason |

## CLI help

```
sltp backtest STRATEGY   run one strategy and report what it did
sltp compare A B [...]   run two or more strategies over one period and compare them
```

`STRATEGY` is a `.py` file. When a file defines several strategies, select one with `file.py:ClassName`.

| Flag | Meaning |
|---|---|
| `-s, --start DATE` | first session, `YYYY-MM-DD` (default: 3 years before the end) |
| `-e, --end DATE` | last session, `YYYY-MM-DD` (default: the latest session on disk) |
| `--no-dividends` | ignore cash distributions |
| `--json` | print JSON instead of the summary |
| `-g, --chart` | draw an SVG equity chart and open it |
| `-o, --out FILE` | write the chart to `FILE` (implies `--chart`) |
| `--version` | print the version |

```bash
sltp backtest momentum.py
sltp backtest momentum.py --start 2019-01-01 --chart
sltp backtest strategies.py:Momentum --start 2019-01-01 --end 2023-12-31
sltp compare value.py momentum.py --out compare.svg
sltp compare a.py b.py c.py --json > results.json
```

## Notes

First run may take some time, but all subsequent runs will be significantly faster.

Market data lives outside the package. If your bars are not in the default location, set `SLTP_DATA_DIR` before importing.

A strategy file is ordinary Python. Its folder goes on `sys.path`, so it can import the files sitting next to it.
