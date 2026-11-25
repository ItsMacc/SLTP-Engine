# SLTP

A backtesting engine for daily stock bars. You write a strategy in one small Python class, SLTP walks it through twenty-eight years of history one session at a time, and you get an equity curve, a trade list and a full metric sheet back.

The whole archive is 59 million bars across 19,362 symbols, from 1998 to last week, sitting in 1.4 GB of flat binary files. A thirteen-year run that ranks 5,735 names every rebalance finishes in about two seconds on an M2 with the pages warm. That is the full ranking path, not one ticker held for show.

At a glance:

| | |
|---|---|
| bars on disk | 59,098,794 |
| symbols | 19,362 across 14 markets |
| history | 1998-01-01 to 2026-11-20 |
| archive size | 1.42 GB, 16 files |
| peak RSS, one-year run | 216 MiB |
| dependencies | numpy, pyarrow, dateutil |

Jump to: [Why it is fast](#why-it-is-fast) · [Quickstart](#quickstart) · [Writing a strategy](#writing-a-strategy) · [Running a backtest](#running-a-backtest) · [Market data](#market-data) · [CLI](#cli-commands)

## Why it is fast

Four decisions do most of the work, and none of them are micro-optimisation.

**Indicators stream instead of recomputing.** A naive engine rebuilds a 200-day EMA from scratch on every bar, which is 200 multiplications to learn one new number. SLTP keeps one recurrence state per ticker per indicator and carries it forward, so `_ema`, `_rsi`, `_atr`, `_adx` and `_macd` cost O(1) per call instead of O(history). That single change made indicator-heavy runs roughly 25 to 35 times faster, with bit-identical values. The other fifteen indicators stay on the batch path, and `STREAMING_INDICATORS = False` turns the whole thing off if you ever want to compare.

**One object per ticker, not one per ticker per day.** `StockData` holds a ticker's full loaded bars plus a window over them. The engine moves the window instead of allocating a new object, and each window's views are cached, so a strategy that reads only `closes` pays for one slice per day instead of six. A trade still gets its own frozen cut of the bars through its entry day, because its history must never shift under it.

**The archive is a layout, not a pile of files to parse.** Each bundle carries a 24-byte preamble and a JSON header, then seven contiguous column blocks. Dates and symbol codes are `uint16`, rows are symbol-major and date-ascending, so one ticker's year is a byte range rather than a filter. A sidecar index per market is sorted by symbol and holds row counts, which turns "where is AAPL in 2011" into a binary search plus arithmetic. No pandas in the run path, no per-row Python, no copies.

**The universe is lazy.** Declaring `histories` with no default is a promise that you need every ticker's bars, so the engine loads them. Writing `histories=None` is a promise that you do not, so it skips the full-market read entirely and hands you a membership set instead. The same rule applies to `on_rebalance`, and it is the difference between a one-second run and a thirty-second one on a fixed list.

## What it will not let you do

Lookahead bias is not policed, it is impossible. A strategy receives arrays that end at today's bar, the window moves forward under it, and the bar reader is not reachable from strategy code. You cannot accidentally call `get_stock_history` for next Tuesday because there is no such call to make.

Reruns are byte-identical. Pin the dates and the same data version, and the numbers reproduce exactly, including the costs. Every run stamps the bar source it read (`own:<dir>` or `cache@<version>`), so a metric that moved is always explainable rather than mysterious.

Rejected orders are not silent. The portfolio manager counts every refusal by reason, so `result["rejections"]` tells you that 412 orders died on "not enough equity to allocate", 31 on "maximum lots reached", and 6 on "hard to borrow", instead of that quiet feeling that something was ignored.

## Installation

Requires Python 3.11 or newer.

```bash
pip install sltp
```

## Quickstart

Save this as `momentum.py`. It buys the four index ETFs whenever price sits above its twenty-day average, five positions maximum, rebalanced monthly.

```python
from sltp import Order, PositionType, RebalanceFrequency, Strategy


class Momentum(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Momentum 20"
        self.MAX_POSITIONS = 5
        self.MAX_ALLOCATION_PER_TICKER = 0.2
        self.REBALANCE_FREQUENCY = RebalanceFrequency.MONTHLY
        self.MIN_BARS_REQUIRED = 60

    def initialize_universe(self, current_date: str, histories=None) -> bool:
        self.tickers.clear()
        for ticker in ("SPY", "QQQ", "IWM", "DIA", "TLT"):
            if ticker in self.available_universe:
                self.add_ticker(ticker)
        return bool(self.tickers)

    def on_signal(self, ticker: str, data) -> Order | None:
        if ticker in self.open_tickers:
            return None
        closes = data.closes
        if len(closes) < 60 or float(closes[-1]) <= self._sma(closes, 20)[-1]:
            return None
        return Order(
            ticker=ticker,
            position_type=PositionType.LONG,
            allocation=0.2,
            stop_loss=0.0,
            take_profit=0.0,
        )
```

```bash
sltp backtest momentum.py --start 2014-01-01 --chart
```

That prints the hooks, the metrics, the trade list, and writes an SVG equity curve with a buy-and-hold SPY line for reference. The rest of this document goes through the pieces in the order you will actually touch them.

## Writing a strategy

Everything lives in one class. Copy the skeleton below, delete what you do not need, and you have a working strategy. Only `on_signal` is required.

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

### Step 1: `__init__`, name it and set your limits

The constructor gives the strategy an identity and decides how much rope the portfolio manager hands it. Always call `super().__init__()` first.

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
| `REBALANCE_FREQUENCY` | `NEVER` | when `on_rebalance` fires: `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY`, or `NEVER` |
| `MIN_BARS_REQUIRED` | `0` | bars of history needed before any trading starts |
| `LIQUIDATE_ON_END` | `False` | close everything on the final session |
| `LOOKBACK_BARS` | `None` | sessions loaded before the start date, for indicators and ranking (`None` keeps the five-year default) |

When an order breaks one of these limits it is counted by reason in `result["rejections"]`, never dropped quietly.

A few read-only values are handy inside your logic: `open_tickers`, `open_positions`, `open_position_count`, and `equity` (the portfolio index level, starting at 100).

### Step 2: `initialize_universe`, choose what you can trade

As the backtest steps forward, the engine asks one question: what is tradable today? Add tickers with `add_ticker` (or `add_tickers` for a list, `remove_ticker` to drop one), and return `True` once the universe is ready. Returning `False` says not yet, and the engine keeps walking forward until you accept.

```python
def initialize_universe(self, current_date: str, histories=None) -> bool:
    self.tickers.clear()
    for ticker in ("SPY", "QQQ", "IWM", "DIA"):
        if ticker in self.available_universe:
            self.add_ticker(ticker)
    return bool(self.tickers)
```

You do not have to hardcode names. Four properties list everything with local data behind it:

- `available_universe`, every ticker on disk
- `available_stocks`, individual companies only
- `available_etfs`, funds only
- `available_gold`, gold proxies only

When the universe itself is the question, declare `histories` with no default and the engine loads every ticker's bars before calling you. Say you want liquid large caps and nothing under five dollars:

```python
def initialize_universe(self, current_date: str, histories: dict) -> bool:
    self.tickers.clear()
    for ticker in self.available_stocks:
        data = histories[ticker]
        if float(data.closes[-1]) > 5.0 and float(data.volumes[-1]) > 1_000_000:
            self.add_ticker(ticker)
    return bool(self.tickers)
```

Because `data` only reaches the current bar, a filter like this can never peek at the future. Pair it with `REBALANCE_FREQUENCY` and `on_rebalance` to rebuild the list on a schedule instead of every session.

The one performance note that matters: declaring `histories` without a default orders the full-universe load. If your universe is hardcoded, or your rebalance just rotates a fixed list, write `histories=None` and the engine skips that read.

### Step 3: `on_signal`, the actual decision

Once per ticker per bar, the engine calls `on_signal` with that ticker's point-in-time history. `data` carries parallel NumPy arrays whose last element is today: `opens`, `highs`, `lows`, `closes`, `volumes`, plus `dates` as `"YYYY-MM-DD"` strings. `len(data)` is the number of bars visible so far. Return an `Order` to open a position, or `None` to pass.

```python
def on_signal(self, ticker: str, data) -> Order | None:
    if ticker in self.open_tickers:
        return None
    closes = data.closes
    if len(closes) < 60:
        return None
    if float(closes[-1]) <= float(sum(closes[-20:]) / 20):
        return None
    return Order(
        ticker=ticker,
        position_type=PositionType.LONG,
        allocation=0.5,
        stop_loss=0.0,
        take_profit=0.0,
    )
```

A few helpers round out the class. `on_rebalance` rebuilds the universe whenever `REBALANCE_FREQUENCY` comes due. `run_daily` runs once per session before any ticker is polled, good for warmup math or bookkeeping. `should_exit` holds custom exit logic per open trade, and three helpers cover the common cases: `_time_stop_days(trade, days)`, `_trailing_stop(trade, pct)`, `_atr_stop(trade, multiplier)`.

You also get an indicator shelf that runs on the point-in-time bars: `_sma`, `_ema`, `_dema`, `_tema`, `_rsi`, `_macd`, `_bollinger_bands`, `_atr`, `_adx`, `_stochastic`, `_stochastic_rsi`, `_williams_r`, `_cci`, `_roc`, `_momentum`, `_obv`, `_mfi`, `_ad`, `_trix`, and `_parabolic_sar`. So the twenty-day check above is just `self._sma(closes, 20)[-1]`.

### Step 4: orders, how you ask for a position

Every order names a direction, a size, and optional risk levels. Fills happen at the next open.

```python
Order(
    ticker="SPY",
    position_type=PositionType.LONG,  # or PositionType.SHORT
    allocation=0.5,
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
      stop_loss=0.0, take_profit=0.0)

# Exactly $10,000 of SPY.
Order(ticker="SPY", position_type=PositionType.LONG, allocation=10_000,
      stop_loss=0.0, take_profit=0.0,
      sizing=OrderSizing.FIXED)

# Everything left in cash, into TLT.
Order(ticker="TLT", position_type=PositionType.LONG, allocation=1.0,
      stop_loss=0.0, take_profit=0.0,
      sizing=OrderSizing.AVAILABLE)

# Half the book short QQQ.
Order(ticker="QQQ", position_type=PositionType.SHORT, allocation=0.5,
      stop_loss=0.0, take_profit=0.0)
```

`stop_loss` and `take_profit` are absolute prices, checked against the daily bars. `0.0` on either means no level there. Anything extra you want to remember, a signal score or a regime tag, rides along in `meta_data`, a plain dict the engine ignores.

### How history is stored

One `StockData` object exists per ticker for the whole run, not one per ticker per day. It holds that ticker's full loaded bars plus a window over them, and the engine moves the window instead of building a new object:

- `data.closes` (and `opens`, `highs`, `lows`, `volumes`, `dates`) return the window only, so the last element is always today and there is no lookahead. `len(data)` is the number of bars visible so far. Each window's views are cached, so a strategy that reads only `closes` pays for one slice per day rather than six.
- `data` is only valid for the callback it was passed to. Holding onto it and reading it on a later day reads that day's window, not the one you saw, which is a silent lookahead bug.
- `set_window(start, end)` and `data.window` are engine-facing. A strategy never calls them: the engine cuts each ticker to today before `on_signal` and before `should_exit`.
- `data.full` returns `(codes, opens, highs, lows, closes, volumes)`, the complete arrays ignoring the window. The engine uses it to scan dates and to cut a trade's snapshot; strategies should read the windowed arrays instead.

Inside a run there is one store: `_store` maps ticker to its single `StockData`, and the portfolio manager reads that same dict, so bars are loaded once and referenced everywhere. The universe cache holds `(start, end)` spans per rebalance date instead of objects, so an as-of snapshot costs two integers. A trade still gets its own cut of the bars through its entry day, because its history must never move under it the way the shared window does.

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

`end_date` defaults to the latest session on disk. Pass `include_dividends=False` to skip cash distributions. Four optional knobs model friction: `spread_pct` (one-way spread paid on every fill), `max_participation` (largest share of a bar's volume one fill may take, the rest waits for the next open), `min_fill_notional` (smallest fill worth booking, in dollars, default `0.01`, `0.0` disables it), and `borrow_rate` plus `hard_to_borrow` (annual cost of holding shorts, and tickers that cannot be shorted at all). All default to zero friction, so leave them out and the run behaves exactly as before.

The result is a dict with `metrics`, the `equity` curve records, the full `trades` list, `rejections` (refused orders by reason), and a `stamp` recording the strategy, dates, and costs behind the run.

Comparing strategies in Python is two runs side by side:

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

## Market data

Every run reads bars from the first source that has them:

1. Your own directory, if you set one (`sltp data use`, or `SLTP_DATA_DIR` for one-off runs).
2. The managed cache in `~/.sltp/cache`, refreshed from R2.

Fresh installs just work: the first run downloads what its dates need, later runs check a version stamp and top up only when R2 moved. Each run records its bar source in `result["stamp"]["data_source"]` (`own:<dir>` or `cache@<version>`), so a number that moved is always explainable. Pin `end_date` and a rerun is byte-identical.

```bash
sltp data                 # where am I reading from, and how fresh is it
sltp data update          # pull the latest bars now
sltp data auto off        # stay offline; run on whatever is cached (default: on)
sltp data use ./my_bars   # read your own files instead of the cache
sltp data use --clear     # forget that and go back to the default
sltp data check ./my_bars # validate a folder before you trust it
sltp reset                # forget every setting, behave like a fresh install
```

Bringing your own bars: put one flat bundle per calendar year named `1998.bin`, … plus `actions.parquet` in a folder, then run `sltp data check` on it and `sltp data use` it. Build the bundles from yearly parquet (`Symbol, Date, Open, High, Low, Close, Volume` columns, rows sorted) with `python -m sltp.data.build_flat <parquet dir> <out dir>`. No network, no surprises: with `auto off` the engine never phones home and runs on local files, warning when they go stale instead of failing.

## CLI commands

Four commands. `STRATEGY` is a `.py` file, and when a file defines several strategies you pick one with `file.py:ClassName`.

```
sltp backtest STRATEGY   run one strategy and report what it did
sltp compare A B [...]   run two or more strategies over one period and compare them
sltp data [ACTION]       show, refresh, or redirect the bar source
sltp reset               restore all settings to their defaults
```

| Flag | Meaning |
|---|---|
| `-s, --start DATE` | first session, `YYYY-MM-DD` (default: 3 years before the end) |
| `-e, --end DATE` | last session, `YYYY-MM-DD` (default: the latest session on disk) |
| `--no-dividends` | ignore cash distributions |
| `--spread-bps BPS` | one-way spread paid on every fill, in basis points (default: 0) |
| `--max-participation FRAC` | largest share of a bar's volume one fill may take, 0-1 (default: no cap) |
| `--min-fill DOLLARS` | smallest fill worth booking; smaller waits whole (default: 0.01, 0 disables) |
| `--borrow-rate RATE` | annual cost of holding shorts, as a fraction (default: 0) |
| `--json` | print JSON instead of the summary |
| `-g, --chart` | draw an SVG equity chart and open it |
| `-o, --out FILE` | write everything (hook breakdown, report, chart) to `FILE`; the terminal stays quiet (implies `--chart`) |
| `--version` | print the version |

```bash
sltp backtest momentum.py
sltp backtest momentum.py --start 2019-01-01 --chart
sltp backtest strategies.py:Momentum --start 2019-01-01 --end 2023-12-31
sltp compare value.py momentum.py --out compare.svg
sltp compare a.py b.py c.py --json > results.json
```

## Benchmarks

Measured on an idle M2 with the archive in page cache. Both numbers move with disk warmth and with whatever else the machine is doing, so treat them as a shape rather than a contract.

| run | what it does | time |
|---|---|---|
| minimum volatility, 2013 to 2026 | ranks 5,735 names per rebalance, 330 trades | ~2 s |
| statistical mean reversion, 2013 to 2026 | ranks 11,897 names, 492 trades | ~4 s |
| one year, one ticker | end-to-end smoke test | 0.6 s, 216 MiB peak |

For scale, that works out to roughly 2.5 million bars a second through the full engine path: read, window, signal, fill, account. The streaming indicator work is what makes the ranked universes affordable, and the lazy universe load is what keeps the fixed-list runs cheap.

Two checks keep it honest. `python test/smoke_backtest.py` runs the whole engine against the real archive end to end. `test/test_streaming_indicators.py` compares every streaming indicator against the batch implementation it replaced, so "same values, less work" is a test rather than a claim.

## Notes

Bars live outside the package. If yours are not in the default location, point at them once with `sltp data use <dir>`, or set `SLTP_DATA_DIR` for a single run.

A strategy file is ordinary Python. Its folder goes on `sys.path`, so it can import the files sitting next to it.

The first run may take a while if it has to download bars. Everything after that reads local files.

## Links

Website: https://sltp-app.com

Docs: https://sltp-app.com/#/docs/introduction
