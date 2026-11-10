# SLTP

SLTP is a backtesting engine for daily stock bars, built on plain NumPy. You describe a strategy in a small Python class, and SLTP walks it through years of history one session at a time, fills the orders, tracks the portfolio, and reports back with an equity curve, a trade list, and a full set of performance statistics.

A few things define how it works. Strategies only ever see data up to the current bar, so lookahead bias is structurally impossible rather than something you have to police yourself. Splits and dividends are applied to positions automatically. Orders support allocation sizing, stop-losses, and take-profits. The portfolio manager enforces your position limits and records why any order was rejected. And because the hot path is vectorized NumPy over a local columnar archive, a multi-decade run over thousands of tickers completes in seconds on an ordinary laptop.

## Installation

Requires Python 3.11 or newer.

```bash
pip install sltp
```

## Writing a strategy

Everything lives in one class. Copy this skeleton into a file (say `my_strategy.py`), fill in the pieces, and you have a working strategy.

```python
from sltp import Order, PositionType, Strategy


class MyStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.name = "My strategy"

    def initialize_universe(self, current_date: str, histories=None) -> bool:
        return bool(self.tickers)

    def on_rebalance(self, current_date: str, histories=None) -> None:
        return None

    def run_daily(self, current_date: str) -> None:
        return None

    def on_signal(self, ticker: str, data) -> Order | None:
        return None

    def should_exit(self, ticker: str, trade, history=None) -> bool:
        return False
```

Only `on_signal` is required. The rest have sensible defaults, so delete whatever you don't need. The sections below go through each piece in the order you'll actually write it.

### Step 1: `__init__`, name it and set your limits

The constructor is where you give the strategy an identity and decide how much rope the portfolio manager gives it. Always call `super().__init__()` first.

```python
def __init__(self) -> None:
    super().__init__()
    self.name = "Momentum 20"
    self.description = "Buys index ETFs above their 20-day average"
    self.MAX_POSITIONS = 2
    self.MAX_ALLOCATION_PER_TICKER = 0.5
    self.MAX_LOTS_PER_TICKER = 1
    self.REBALANCE_FREQUENCY = RebalanceFrequency.MONTHLY
    self.MIN_BARS_REQUIRED = 60
    self.LIQUIDATE_ON_END = True
```

What each knob does:

| Setting | Default | Meaning |
|---|---|---|
| `name` | `"Base Strategy"` | shows up in reports, charts, and comparisons |
| `description` | `"Base Strategy Interface"` | a line about what the strategy does |
| `MAX_POSITIONS` | `10` | maximum distinct open positions (`None` for unlimited) |
| `MAX_ALLOCATION_PER_TICKER` | `None` | cap on portfolio fraction in one ticker (`None` for no cap) |
| `MAX_LOTS_PER_TICKER` | `1` | maximum concurrent lots in one ticker |
| `REBALANCE_FREQUENCY` | `NEVER` | how often `on_rebalance` fires: `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY`, or `NEVER` (import from `sltp`) |
| `MIN_BARS_REQUIRED` | `0` | bars of history needed before any trading starts |
| `LIQUIDATE_ON_END` | `False` | close everything on the final session |

When an order breaks one of these limits, it isn't silently dropped. The rejection is counted by reason in the result's `rejections`, so you can see exactly what stopped you.

Two extra read-only values are handy inside your logic: `open_tickers` (which tickers currently hold positions), `open_positions` (the open trades themselves), `open_position_count`, and current `equity` (the portfolio index level, starting at 100).

### Step 2: `initialize_universe`, choose what you can trade

As the backtest steps forward, the engine hands you every ticker's history up to the current date and asks: what is tradable today? Add tickers with `add_ticker` (or `add_tickers` for a list, `remove_ticker` to drop one). Return `True` once the universe is ready. Returning `False` tells the engine it isn't ready yet, and it keeps walking forward until you accept.

```python
def initialize_universe(self, current_date: str, histories: dict) -> bool:
    self.tickers.clear()
    for ticker in ("SPY", "QQQ", "IWM", "DIA"):
        if ticker in histories:
            self.add_ticker(ticker)
    return bool(self.tickers)
```

You don't have to hardcode names. Four properties list everything with local data behind it:

- `available_universe`, every ticker on disk
- `available_stocks`, individual companies only
- `available_etfs`, funds only
- `available_gold`, gold proxies only

That makes custom filters short. Say you want liquid large caps and nothing under five dollars:

```python
def initialize_universe(self, current_date: str, histories: dict) -> bool:
    self.tickers.clear()
    for ticker in self.available_stocks:
        data = histories[ticker]
        if float(data.close_price[-1]) > 5.0 and float(data.volume[-1]) > 1_000_000:
            self.add_ticker(ticker)
    return bool(self.tickers)
```

Because `data` only reaches the current bar, a filter like this can never peek at the future. Pair it with `REBALANCE_FREQUENCY` and `on_rebalance` to rebuild the list on a schedule instead of every day.

One performance note: declaring `histories` without a default tells the engine to load every ticker's bars before calling you. If your universe is hardcoded (or your rebalance just rotates a fixed list), write `histories=None` instead and the full-universe load is skipped entirely. The same rule applies to `on_rebalance`.

### Step 3: `on_signal`, the actual decision

Once per ticker per bar, the engine calls `on_signal` with that ticker's point-in-time history. `data` carries parallel NumPy arrays where the last element is today: `open_price`, `high`, `low`, `close_price`, `volume`, plus `dates` (`"YYYY-MM-DD"` strings). Return an `Order` to open a position, or `None` to pass.

```python
def on_signal(self, ticker: str, data) -> Order | None:
    if ticker in self.open_tickers:
        return None
    closes = data.close_price
    if len(closes) < 60:
        return None
    if float(closes[-1]) <= float(sum(closes[-20:]) / 20):
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

A few helpers round out the class. `on_rebalance` rebuilds the universe whenever `REBALANCE_FREQUENCY` comes due. `run_daily` runs once per session before any ticker is polled, good for warmup math or daily bookkeeping. `should_exit` holds custom exit logic per open trade, and three helpers cover the common cases: `_time_stop_days(trade, days)`, `_trailing_stop(trade, pct)`, and `_atr_stop(trade, multiplier)`.

You also get a full indicator shelf that runs on the point-in-time bars: `_sma`, `_ema`, `_dema`, `_tema`, `_rsi`, `_macd`, `_bollinger_bands`, `_atr`, `_adx`, `_stochastic`, `_stochastic_rsi`, `_williams_r`, `_cci`, `_roc`, `_momentum`, `_obv`, `_mfi`, `_ad`, `_trix`, and `_parabolic_sar`. So the 20-day check above can just be `self._sma(closes, 20)[-1]`.

### Step 4: orders, how you ask for a position

Every order names a direction, a size, and optional risk levels. Fills happen at the next open.

```python
Order(
    ticker="SPY",
    position_type=PositionType.LONG,  # or PositionType.SHORT
    allocation=0.5,
    order_date="2023-06-01",
    stop_loss=380.0,     # absolute price, 0.0 means none
    take_profit=460.0,   # absolute price, 0.0 means none
)
```

Direction is `LONG` or `SHORT`. Shorts borrow the shares, so they pay borrow costs when those are enabled and receive the same dividend treatment in reverse.

Size is set by `sizing`, which reads `allocation` three ways:

- `OrderSizing.EQUITY` (the default): a fraction of portfolio equity. `allocation=0.5` invests half the book.
- `OrderSizing.FIXED`: a cash amount. `allocation=10000` invests ten thousand dollars regardless of book size.
- `OrderSizing.AVAILABLE`: a fraction, but capped by uninvested cash. `allocation=1.0` means "put all remaining cash to work" without ever overspending.

```python
from sltp import OrderSizing

# Half the book, whatever the book is worth.
Order(ticker="QQQ", position_type=PositionType.LONG, allocation=0.5,
      order_date="2023-06-01", stop_loss=0.0, take_profit=0.0)

# Exactly $10,000 of SPY.
Order(ticker="SPY", position_type=PositionType.LONG, allocation=10_000,
      order_date="2023-06-01", stop_loss=0.0, take_profit=0.0,
      sizing=OrderSizing.FIXED)

# Everything left in cash, into TLT.
Order(ticker="TLT", position_type=PositionType.LONG, allocation=1.0,
      order_date="2023-06-01", stop_loss=0.0, take_profit=0.0,
      sizing=OrderSizing.AVAILABLE)

# Half the book short QQQ.
Order(ticker="QQQ", position_type=PositionType.SHORT, allocation=0.5,
      order_date="2023-06-01", stop_loss=0.0, take_profit=0.0)
```

`stop_loss` and `take_profit` are absolute prices, checked against the daily bars. `0.0` on either means no level there. Anything extra you want to remember (a signal score, a regime tag) rides along in `meta_data`, a plain dict the engine ignores.

## Running a backtest

From Python:

```python
from sltp import Backtester
from my_strategy import MyStrategy

result = Backtester(
    strategy=MyStrategy(),
    start_date="2019-01-01",
    end_date="2023-12-31",
).run()

print(result["metrics"])
print(result["trades"][-1])
```

`end_date` defaults to the latest session on disk. Pass `include_dividends=False` to skip cash distributions. Three optional knobs model friction: `spread_pct` (one-way spread paid on every fill), `max_participation` (largest share of a bar's volume one fill may take, the rest waits for the next open), and `borrow_rate` plus `hard_to_borrow` (annual cost of holding shorts, and tickers that can't be shorted at all). All default to zero friction, so leave them out and the run behaves exactly as before.

The result is a dict with `metrics`, the `equity` curve records, the full `trades` list, `rejections` (refused orders by reason), and a `stamp` recording the strategy, dates, and costs behind the run.

Comparing strategies in Python is just two runs side by side:

```python
from sltp import Backtester
from momentum import Momentum
from value import Value

runs = {
    "momentum": Backtester(strategy=Momentum(), start_date="2019-01-01", end_date="2023-12-31").run(),
    "value": Backtester(strategy=Value(), start_date="2019-01-01", end_date="2023-12-31").run(),
}
for name, result in runs.items():
    m = result["metrics"]
    print(name, round(m["cagr"], 2), round(m["sharpe"], 2), round(m["max_drawdown"], 2))
```

From the terminal, `backtest` runs one strategy and `compare` runs several over the same period:

```bash
sltp backtest my_strategy.py --start 2019-01-01 --chart
sltp compare momentum.py value.py --out compare.svg
```

A lone chart automatically adds a buy-and-hold SPY line for reference.

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
| `borrow_cost` | cumulative short borrow costs over the run |
| `dividend_income`, `unrealized_profit` | cash distributions received and open profit at the end |
| `rejected_orders` | count of refused orders by reason |

## CLI commands

Two commands, `backtest` and `compare`. `STRATEGY` is a `.py` file, and when a file defines several strategies you pick one with `file.py:ClassName`.

```
sltp backtest STRATEGY   run one strategy and report what it did
sltp compare A B [...]   run two or more strategies over one period and compare them
```

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

## Links

Website: https://your-website.com

Docs: https://your-website.com/docs
