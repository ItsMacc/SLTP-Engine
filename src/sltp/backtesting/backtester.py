# -------- backtester.py --------
# Runs a strategy day by day and returns what happened.

# -------- Imports --------
import inspect
import math
import sys
import time

import numpy as np

from collections.abc import Mapping
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Engine imports
from sltp.portfolio import PortfolioManager
from sltp.strategy import Strategy
from sltp.utils.Dates import CALENDAR, to_code
from sltp.utils.StockData import StockData
from sltp.utils.PositionType import PositionType

# Database imports
from sltp.market.market_data import _histories, calendar, dividends, run_guard


# -------- Constants --------
# How far past the requested start the universe may look for a date it can trade.
MAX_START_PROBE_DAYS: int = 1826
# No LOOKBACK_BARS? Load five years of sessions: enough for the hungriest indicator or ranking window.
DEFAULT_LOOKBACK_DAYS: int = 1826
# As-of universe snapshots kept per date; probe dates never repeat, so a few suffice.
_ASOF_CACHE_LIMIT: int = 4
# A cash distribution above the whole pre-dividend share price is a data artefact.
MAX_DIVIDEND_YIELD: float = 1.0
# Hook profiling: every call is timed exactly, so the breakdown is a measurement, never an estimate.


# -------- Backtester --------
# One simulation: a strategy, a date range, and the bars it needs.
class Backtester:
    # Wire up the manager and the per-run state.
    def __init__(self, strategy: Strategy, start_date: str, end_date: str | None = None, include_dividends: bool = True,
                 spread_pct: float = 0.0, max_participation: float | None = None,
                 borrow_rate: float = 0.0, hard_to_borrow: list[str] | set[str] | tuple[str, ...] | None = None,
                 min_fill_notional: float = 0.01) -> None:
        if spread_pct < 0:
            raise ValueError(f"spread_pct must be non-negative, got {spread_pct}.")
        if max_participation is not None and max_participation <= 0:
            raise ValueError(f"max_participation must be positive, got {max_participation}.")
        if borrow_rate < 0:
            raise ValueError(f"borrow_rate must be non-negative, got {borrow_rate}.")
        if min_fill_notional < 0:
            raise ValueError(f"min_fill_notional must be non-negative, got {min_fill_notional}.")
        self.strategy: Strategy = strategy
        self.manager: PortfolioManager = PortfolioManager(strategy)
        self.manager.SPREAD_PCT = float(spread_pct)
        self.manager.MAX_PARTICIPATION = None if max_participation is None else float(max_participation)
        self.manager.BORROW_RATE = float(borrow_rate)
        self.manager.HARD_TO_BORROW = frozenset(hard_to_borrow) if hard_to_borrow else frozenset()
        self.manager.MIN_FILL_NOTIONAL = float(min_fill_notional)
        self.manager.set_history_loader(self._history_for_trade)
        self.start_date: str = start_date
        self.end_date: str | None = end_date
        self.include_dividends: bool = include_dividends
        self.liquidate_on_end: bool = self.strategy.LIQUIDATE_ON_END

        # Only the bundles overlapping this window are read, so a 2020 run skips the 1998 file.
        self.load_start: str = self._load_start_for(self.start_date)

        # Every bar the run uses: one record per ticker, created once and reused.
        self._tickers: dict[str, _Ticker] = {}
        self._dividend_events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.skipped: set[str] = set()
        self._unrealized_at_end: float = 0.0
        self._equity_records: list[dict] = []
        self._has_run: bool = False
        self._has_custom_exit: bool = getattr(strategy.should_exit, "__func__", None) is not Strategy.should_exit
        self._has_resting_exits: bool = False
        self._universe_tickers: set[str] = set()
        # Ranking windows per date: ticker -> (start, end). Spans, never objects.
        self._asof_cache: dict[str, dict[str, tuple[int, int]]] = {}
        # The ranking pool, read once: a run ranks the same names from start to finish.
        self._pool_cache: tuple[list[str], frozenset[str]] | None = None
        self._universe_skipped: set[str] = set()
        self._cut_cursors: dict[str, tuple[int, int, int]] = {}
        self._pending_rebalance: bool = False
        self._day_code = np.int32(0)
        self._hook_time: dict[str, float] = {}
        self._hook_calls: dict[str, int] = {}
        self._universe_warned: bool = False
        # The sorted scan-ticker list behind _bars_on.
        self._scan_tickers_cache: set[str] = set()
        self._scan_tickers_sorted: list[str] = []
        # Fills ask for today's volume here instead of the manager keeping a second copy of the bars.
        self.manager.set_volume_lookup(self._today_volume)

    # -------- Run --------
    # Simulate the strategy and return metrics, equity, trades, rejections and a stamp.
    def run(self, quiet: bool = False) -> dict:
        if self._has_run:
            raise RuntimeError("This Backtester has already run. Build a new one for another simulation.")
        self._has_run = True
        with run_guard():
            return self._run(quiet)


    # Where the run's Python time went, by strategy hook. Empty when nothing was timed.
    def hook_breakdown_text(self) -> str:
        if not self._hook_calls:
            return ""
        lines: list[str] = [f"hook time: {getattr(self.strategy, 'name', '?')}"]
        width: int = max(len(name) for name in self._hook_time)
        for name, seconds in sorted(self._hook_time.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  {name.ljust(width)}   {seconds:>7.2f}s x{self._hook_calls[name]:>7,}")
        return "\n".join(lines)


    # -------- Metrics --------
    # Turn the equity curve and closed trades into the reported statistics.
    def performance_metrics(self) -> dict:
        dates, equity_values, _cash_values = self.manager.equity_curve
        if len(equity_values) == 0:
            return {}

        starting_value: float = float(equity_values[0])
        ending_value: float = float(equity_values[-1])
        start_dt: datetime = datetime.strptime(str(dates[0]), "%Y-%m-%d")
        end_dt: datetime = datetime.strptime(str(dates[-1]), "%Y-%m-%d")
        elapsed_days: int = max(1, (end_dt - start_dt).days)
        years: float = max(1, elapsed_days) / 365.25
        # Turnover and cost drag use real elapsed time; CAGR treats a short run as a year.
        elapsed_years: float = elapsed_days / 365.25
        cagr: float = float(((ending_value / starting_value) ** (1.0 / years) - 1.0) * 100.0) if starting_value > 0 else 0.0

        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else np.array([0.0])
        mean_return: float = float(np.mean(daily_returns))
        std_return: float = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
        sharpe: float = round(mean_return / std_return * np.sqrt(252), 4) if std_return > 0 else 0.0

        # Probabilistic Sharpe: adjusts for skew, fat tails and sample size
        n: int = len(daily_returns)
        sr_benchmark_daily: float = 1.0 / np.sqrt(252)
        if n > 2 and std_return > 0:
            sr_daily: float = mean_return / std_return
            sk: float = _skew(daily_returns)
            kt: float = _kurtosis(daily_returns)
            daily_variance: float = (1.0 - sk * sr_daily + ((kt - 1.0) / 4.0) * sr_daily ** 2) / (n - 1)
            psr: float = round(_normal_cdf((sr_daily - sr_benchmark_daily) / np.sqrt(daily_variance)), 4) if daily_variance > 0 else 0.0
        else:
            psr = 0.0

        negative_returns = daily_returns[daily_returns < 0]
        downside_std: float = float(np.std(negative_returns, ddof=1)) if len(negative_returns) > 1 else 0.0
        sortino: float = round(mean_return / downside_std * np.sqrt(252), 4) if downside_std > 0 else 0.0

        peaks = np.maximum.accumulate(equity_values)
        drawdowns = np.divide(peaks - equity_values, peaks, out=np.zeros_like(peaks), where=peaks != 0)
        max_drawdown: float = round(float(np.max(drawdowns)) * 100.0, 2)
        average_equity: float = float(np.mean(equity_values))
        total_trading_cost: float = round(self.manager.total_trading_cost, 2)
        turnover: float = round(self.manager.traded_notional / average_equity / elapsed_years * 100.0, 2) if average_equity > 0 else 0.0
        cost_drag: float = round(total_trading_cost / average_equity / elapsed_years * 100.0, 2) if average_equity > 0 else 0.0
        below_peak = equity_values < peaks
        max_duration: int = 0
        current_duration: int = 0
        for is_below in below_peak:
            current_duration = current_duration + 1 if is_below else 0
            max_duration = max(max_duration, current_duration)

        # Only trades that have actually exited count towards the win and loss figures
        trade_returns: list[float] = []
        for trade in self.manager.trades:
            if trade.exit is not None:
                exit_price: float = trade.exit.price
                direction: float = -1.0 if trade.position_type is PositionType.SHORT else 1.0
                notional: float = trade.equity_at_entry * trade.allocation
                dividend_return: float = trade.dividend_income / notional if notional > 0 else 0.0
                trade_returns.append(direction * (exit_price - trade.entry_price) / trade.entry_price + dividend_return)

        wins: list[float] = [value for value in trade_returns if value > 0]
        losses: list[float] = [value for value in trade_returns if value < 0]
        total_trades: int = len(trade_returns)
        win_rate: float = len(wins) / total_trades * 100.0 if total_trades else 0.0
        average_win: float = sum(wins) / len(wins) * 100.0 if wins else 0.0
        average_loss: float = sum(losses) / len(losses) * 100.0 if losses else 0.0
        expectancy: float = (
            (len(wins) / total_trades * average_win / 100.0) - (len(losses) / total_trades * abs(average_loss) / 100.0)
            if total_trades else 0.0
        )

        return {
            "starting_value": starting_value,
            "ending_value": ending_value,
            "cagr": round(cagr, 2),
            "sharpe": sharpe,
            "psr": psr,
            "sortino": sortino,
            "expectancy": round(expectancy, 4),
            "win_rate": round(win_rate, 2),
            "loss_rate": round(100.0 - win_rate, 2) if total_trades else 0.0,
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "max_drawdown": max_drawdown,
            "turnover": turnover,
            "cost_drag": cost_drag,
            "total_trading_cost": total_trading_cost,
            "drawdown_recovery": max_duration,
            "unrealized_profit": round(self._unrealized_at_end, 2),
            "dividend_income": round(self.manager.dividend_income, 2),
            "borrow_cost": round(self.manager.total_borrow_cost, 2),
            "rejected_orders": dict(sorted(self.manager.rejected_orders.items())),
            "equity": self._equity_records,
        }

    # -------- Helpers --------
    # The simulation itself, under the read guard so no hook can reach past the run window through the data service.
    def _run(self, quiet: bool = False) -> dict:
        self.strategy._reset_indicator_state()

        actual_start: str = self._initialize_universe()
        self._ensure_universe()
        calendar: list = self._build_calendar(actual_start)
        if not calendar:
            raise ValueError("No trading days available between the start date and the loaded data.")

        self.manager.mark_rebalanced(CALENDAR.label(calendar[0]))
        for day_index, code in enumerate(calendar):
            self._day_code = code
            current_date: str = CALENDAR.label(code)

            # Strategy lifecycle callbacks come before order processing
            self.manager.refresh_view()
            self._timed_hook("run_daily", self.strategy.run_daily, current_date)

            if self.manager.rebalance_due(current_date):
                self._pending_rebalance = True

            # A rebalance can change the active universe; cached tickers are cheap to revisit
            self._ensure_universe()
            bars = self._bars_on()
            opens = {ticker: bar[0] for ticker, bar in bars.items()}

            # Dividends belong to lots held from the previous close, before today's fills
            self.manager.pay_dividends(int(self._day_code))

            # Fill pending exits and entries at today's open
            self.manager.fill_pending_exits(current_date, opens)
            self.manager.fill_pending_orders(current_date, opens)

            # Evaluate entries against history through today's completed bar
            self.manager.refresh_view()
            if self._pending_rebalance:
                self._call_universe_hook(self.strategy.on_rebalance, current_date)
                self.manager.mark_rebalanced(current_date)
                self._pending_rebalance = False
                self._ensure_universe()
                bars = self._bars_on()
            self._process_tickers(bars)

            # Custom exits use today's completed data and queue next-open orders
            if self._has_custom_exit and self.manager.open_trades:
                self._process_custom_exits(bars)

            # Close-based protective exits use today's final OHLC
            self._check_resting_exits(current_date, bars)

            # Every fill is done, so one close per ticker serves the final mark and the liquidation.
            closes = {ticker: bar[3] for ticker, bar in bars.items()}
            is_last_day: bool = day_index == len(calendar) - 1
            if is_last_day:
                self._unrealized_at_end = self._unrealized_pnl(closes)
                if self.liquidate_on_end:
                    self.manager.close_all(current_date, closes)

            # Exactly one mark and one log per date, after all final-day fills
            self.manager.update_daily_accounting(current_date, closes)
            self._equity_records.append({
                "date": current_date,
                "portfolio_value": float(self.manager.portfolio_value),
                "active_positions": self.manager.active_positions,
                "total_transaction_cost": float(self.manager.total_cost),
                "total_trading_cost": float(self.manager.total_trading_cost),
                "total_borrow_cost": float(self.manager.total_borrow_cost),
            })

        self.manager.refresh_view()
        if not quiet:
            self._print_hook_breakdown()
        metrics: dict = self.performance_metrics()
        equity: list[dict] = self._equity_records
        return {
            "metrics": metrics,
            "equity": equity,
            "trades": self.manager.trades,
            "rejections": dict(sorted(self.manager.rejected_orders.items())),
            "stamp": self._stamp(),
        }

    def _print_hook_breakdown(self) -> None:
        text: str = self.hook_breakdown_text()
        if text:
            print(text, file=sys.stderr)

    # What this run is made of: versions, range, costs and strategy knobs.
    def _stamp(self) -> dict:
        return {
            "sltp_version": _package_version(),
            "data_version": _data_version(),
            "data_source": _data_source(),
            "start": self.start_date,
            "end": self.end_date,
            "actual_start": self._equity_records[0]["date"] if self._equity_records else None,
            "costs": {
                "fee_rate": self.manager.FEE_RATE,
                "slippage_pct": self.manager.SLIPPAGE_PCT,
                "spread_pct": self.manager.SPREAD_PCT,
                "max_participation": self.manager.MAX_PARTICIPATION,
                "min_fill_notional": self.manager.MIN_FILL_NOTIONAL,
                "borrow_rate": self.manager.BORROW_RATE,
                "hard_to_borrow": sorted(self.manager.HARD_TO_BORROW),
                "include_dividends": self.include_dividends,
            },
            "strategy": _strategy_params(self.strategy),
        }

    # The earliest bar a call on current_date may look back on.
    def _load_start_for(self, current_date: str) -> str:
        lookback: int | None = self.strategy.LOOKBACK_BARS
        days: int = DEFAULT_LOOKBACK_DAYS if lookback is None else max(0, lookback * 2)
        return (datetime.strptime(current_date, "%Y-%m-%d") - relativedelta(days=days)).strftime("%Y-%m-%d")

    # Build the universe, walking the start date forward until the strategy accepts it.
    def _initialize_universe(self) -> str:
        start_dt: datetime = datetime.strptime(self.start_date, "%Y-%m-%d")
        if self._universe_at(start_dt, 0):
            return self._accepted_universe_date(start_dt, 0)

        # The earliest success and the tickers it built, so the winning probe never runs (and reloads) a second time.
        best: tuple[int, list[str]] | None = None
        low: int = 0
        high: int = 1
        while True:
            probe: int = min(high, MAX_START_PROBE_DAYS)
            if self._universe_at(start_dt, probe):
                best = (probe, list(self.strategy.tickers))
                high = probe
                break
            if probe >= MAX_START_PROBE_DAYS:
                raise ValueError(
                    f"Could not initialize universe from {self.start_date} within {MAX_START_PROBE_DAYS} days."
                )
            low, high = probe, probe * 2

        while high - low > 1:
            middle: int = (low + high) // 2
            if self._universe_at(start_dt, middle):
                best = (middle, list(self.strategy.tickers))
                high = middle
            else:
                low = middle

        if best is None:
            raise ValueError(
                f"Could not initialize universe from {self.start_date} within {MAX_START_PROBE_DAYS} days."
            )
        offset, tickers = best
        self.strategy.tickers.clear()
        self.strategy.tickers.extend(tickers)
        return self._accepted_universe_date(start_dt, offset)

    # Ask the universe hook for one date, counted in calendar days from the requested start.
    def _universe_at(self, start_dt: datetime, offset: int) -> bool:
        date_str: str = (start_dt + relativedelta(days=offset)).strftime("%Y-%m-%d")
        return self._call_universe_hook(self.strategy.initialize_universe, date_str)

    # The date the winning probe landed on, refusing a hook that claims success with no tickers.
    def _accepted_universe_date(self, start_dt: datetime, offset: int) -> str:
        if not self.strategy.tickers:
            raise ValueError("initialize_universe() returned True but declared no tickers.")
        return (start_dt + relativedelta(days=offset)).strftime("%Y-%m-%d")

    # Time one strategy hook call for the end-of-run breakdown.
    def _timed_hook(self, name: str, hook, *args):
        start: float = time.perf_counter()
        try:
            return hook(*args)
        finally:
            self._hook_time[name] = self._hook_time.get(name, 0.0) + (time.perf_counter() - start)
            self._hook_calls[name] = self._hook_calls.get(name, 0) + 1

    # Call a universe hook, passing as-of histories whenever it declares the parameter.
    def _call_universe_hook(self, hook, current_date: str):
        # The base hooks only report what is already there, so they never need a snapshot.
        if getattr(hook, "__func__", None) in (Strategy.initialize_universe, Strategy.on_rebalance):
            return self._timed_hook(hook.__name__, hook, current_date)
        parameters = list(inspect.signature(hook).parameters.values())
        accepts_histories: bool = len(parameters) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )
        if accepts_histories:
            return self._timed_hook(hook.__name__, hook, current_date, self._histories_as_of(current_date))
        return self._timed_hook(hook.__name__, hook, current_date)

    # Every ticker's bars up to the callback date, loaded only when the hook reaches for them.
    def _histories_as_of(self, current_date: str) -> Mapping[str, StockData]:
        return _AsOfHistories(self, current_date)

    # The whole universe as of one date, for a hook that ranks every candidate.
    def _snapshot(self, current_date: str) -> dict[str, StockData]:
        spans: dict[str, tuple[int, int]] | None = self._spans_as_of(current_date)
        if spans is None:
            return self._load_histories(current_date)

        # The store's objects are shared; the window is what makes a snapshot. Everyone gets the
        # same objects the next caller will see, so a hook must not hold on to them.
        histories: dict[str, StockData] = {}
        for ticker, (lo, hi) in spans.items():
            history = self._tickers[ticker].history
            history.set_window(lo, hi)
            histories[ticker] = history
        return histories

    # One candidate as of one date: the same window, without loading the other ten thousand.
    def _snapshot_one(self, current_date: str, ticker: str) -> StockData | None:
        if ticker not in self._ranking_pool()[1]:
            return None
        if self.end_date is not None and current_date > self.end_date:
            # A warm-up probe reaches past the run, so that rare date is read alone and not cached.
            return _histories([ticker], self._load_start_for(current_date), current_date).get(ticker)
        self._ensure_ranked([ticker])
        record: _Ticker | None = self._tickers.get(ticker)
        if record is None:
            return None
        span: tuple[int, int] | None = self._cut_span(
            ticker, record, to_code(self._load_start_for(current_date)), to_code(current_date)
        )
        if span is None:
            return None
        history: StockData = record.history
        history.set_window(*span)
        return history

    # The ranking windows for one date, cached; None means the date is past the run's end.
    def _spans_as_of(self, current_date: str) -> dict[str, tuple[int, int]] | None:
        # A warm-up probe can reach past the run's end: that rare date gets its own load and is not cached.
        if self.end_date is not None and current_date > self.end_date:
            return None
        spans: dict[str, tuple[int, int]] | None = self._asof_cache.get(current_date)
        if spans is None:
            spans = self._cut_spans(current_date)
            self._asof_cache[current_date] = spans
            while len(self._asof_cache) > _ASOF_CACHE_LIMIT:
                self._asof_cache.pop(next(iter(self._asof_cache)))
        return spans

    # The candidates the strategy ranks, in its own order: ticker order decides ties, so it is kept.
    def _ranking_pool(self) -> tuple[list[str], frozenset[str]]:
        if self._pool_cache is None:
            listed: list[str] = list(self.strategy.available_universe)
            self._pool_cache = (listed, frozenset(listed))
        return self._pool_cache

    # Read the universe for one date straight from the files.
    def _load_histories(self, current_date: str) -> dict[str, StockData]:
        code = to_code(current_date)
        histories: dict[str, StockData] = {}
        # Ranking only reads back from today, so the files after today cannot contribute.
        window_start: str = self._load_start_for(current_date)
        for ticker, history in _histories(self.strategy.available_universe, window_start, current_date).items():
            index: int = int(np.searchsorted(history.full[0], code, side="right")) - 1
            if index >= 0:
                history.set_window(0, index + 1)
                histories[ticker] = history
        return histories

    # Warn once when most of a large universe is unusable: names short of MIN_BARS_REQUIRED can never rank.
    def _warn_once_for_unusable_symbols(self) -> None:
        if self._universe_warned or len(self._tickers) < 1000:
            return
        self._universe_warned = True
        min_bars: int = int(getattr(self.strategy, "MIN_BARS_REQUIRED", 0) or 0)
        if min_bars <= 0:
            return
        short: int = sum(1 for record in self._tickers.values() if record.arrays[0].size < min_bars)
        if short:
            total: int = len(self._tickers)
            print(
                f"note: {total} symbols loaded for ranking; {short} have fewer than "
                f"{min_bars} bars and can never meet MIN_BARS_REQUIRED.",
                file=sys.stderr,
            )

    # Load the whole ranking universe once, over the run's window.
    def _ensure_ranked(self, symbols: list[str]) -> None:
        missing: list[str] = [
            symbol for symbol in dict.fromkeys(symbols)
            if symbol not in self._tickers and symbol not in self._universe_skipped
        ]
        if not missing:
            return
        loaded: dict[str, StockData] = _histories(missing, self.load_start, self.end_date)
        for symbol in missing:
            history: StockData | None = loaded.get(symbol)
            if history is None:
                self._universe_skipped.add(symbol)
            else:
                self._tickers[symbol] = _Ticker(history)
        self._warn_once_for_unusable_symbols()

    # One history for a ticker the cache never saw, cut from the same bulk window so a fill reads no files.
    def _history_for_trade(self, ticker: str, current_date: str) -> StockData | None:
        self._ensure_ranked([ticker])
        record: _Ticker | None = self._tickers.get(ticker)
        if record is None:
            return None
        index: int = int(np.searchsorted(record.arrays[0], to_code(current_date), side="right")) - 1
        if index < 0:
            return None
        return record.history

    # Cut every ticker's bars to this date's ranking window.
    def _cut_spans(self, current_date: str) -> dict[str, tuple[int, int]]:
        code = to_code(current_date)
        low_code = to_code(self._load_start_for(current_date))
        symbols, _pool = self._ranking_pool()
        self._ensure_ranked(symbols)

        spans: dict[str, tuple[int, int]] = {}
        for ticker in symbols:
            record: _Ticker | None = self._tickers.get(ticker)
            if record is None:
                continue
            span: tuple[int, int] | None = self._cut_span(ticker, record, low_code, code)
            # A ticker with no bar inside this window is absent, exactly as a single-date read would have left it out.
            if span is not None:
                spans[ticker] = span
        return spans

    # One ticker's window for a date, reusing the last cut when its bars have not moved.
    def _cut_span(self, ticker: str, record, low_code: int, code: int) -> tuple[int, int] | None:
        codes = record.arrays[0]
        cursor: tuple[int, int, int] | None = self._cut_cursors.get(ticker)
        # Bars only arrive at the end, so a name unchanged since the last rebalance keeps its cut, no search.
        if (
            cursor is not None
            and code >= cursor[2]
            and (cursor[1] >= codes.size or codes[cursor[1]] > code)
            and (cursor[0] >= cursor[1] or codes[cursor[0]] >= low_code)
        ):
            lo, hi = cursor[0], cursor[1]
        else:
            lo = int(np.searchsorted(codes, low_code, side="left"))
            # Searching the tail for the far edge finds the second bound without walking the whole history twice.
            hi = lo + int(np.searchsorted(codes[lo:], code, side="right"))
            self._cut_cursors[ticker] = (lo, hi, code)
        return None if lo >= hi else (lo, hi)

    # Make sure every ticker the strategy currently wants is loaded.
    def _ensure_universe(self) -> None:
        wanted: list[str] = list(dict.fromkeys([*self.strategy.tickers, *self.manager.open_tickers]))
        self._universe_tickers.update(self.strategy.tickers)
        self._ensure_tickers(wanted)

    # Load the given tickers and mark them ready for the day scan.
    def _ensure_tickers(self, tickers: list[str]) -> None:
        # A record can already exist from ranking; "ready" is what marks it wired for this run.
        unread: list[str] = [
            ticker for ticker in tickers
            if ticker not in self._tickers and ticker not in self.skipped and ticker not in self._universe_skipped
        ]
        if unread:
            for ticker, history in _histories(unread, self.load_start, self.end_date).items():
                self._tickers[ticker] = _Ticker(history)
        fresh: list[str] = []
        for ticker in tickers:
            record: _Ticker | None = self._tickers.get(ticker)
            if record is None:
                self.skipped.add(ticker)
            elif not record.ready:
                record.ready = True
                fresh.append(ticker)
        if fresh and self.include_dividends:
            self._load_dividends(fresh)

    # Load dividends for the tickers just added, dropping impossible rows.
    def _load_dividends(self, tickers: list[str]) -> None:
        for ticker in tickers:
            record: _Ticker | None = self._tickers.get(ticker)
            if record is None or ticker in self._dividend_events:
                continue
            history: StockData = record.history
            codes, amounts = dividends(ticker, self.load_start, self.end_date)
            codes, amounts = self._plausible_dividends(history, codes, amounts)
            if codes.size:
                self._dividend_events[ticker] = (codes, amounts)
        self.manager.set_dividend_events(self._dividend_events)

    # Keep dividends that could have been paid: a previous close must exist and fit the price.
    @staticmethod
    def _plausible_dividends(history: StockData, codes: np.ndarray, amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if codes.size == 0:
            return codes, amounts
        all_codes, _opens, _highs, _lows, all_closes, _volumes = history.full
        index: np.ndarray = np.searchsorted(all_codes, codes, side="left") - 1
        valid: np.ndarray = index >= 0
        previous: np.ndarray = np.zeros(codes.size, dtype=np.float64)
        previous[valid] = all_closes[index[valid]]
        keep: np.ndarray = valid & (previous > 0) & (amounts <= previous * MAX_DIVIDEND_YIELD)
        return codes[keep], amounts[keep]

    # The market's own trading days inside the run window.
    def _build_calendar(self, actual_start: str) -> list:
        if not self._tickers:
            raise ValueError("No price data loaded. initialize_universe() must add locally available tickers.")
        codes = calendar(actual_start, self.end_date)
        start_code = to_code(actual_start)
        end_code = to_code(self.end_date) if self.end_date is not None else codes[-1]
        return codes[(codes >= start_code) & (codes <= end_code)].tolist()

    # Only the active universe, anything still held, and orders about to fill.
    def _scan_tickers(self) -> set[str]:
        tickers: set[str] = set(self.strategy.tickers)
        tickers.update(self.manager.open_tickers)
        tickers.update(order.ticker for order in self.manager.pending_orders)
        return tickers

    # Collect today's bars. Cursors only advance, so a run touches each ticker's codes once.
    def _bars_on(self) -> dict[str, tuple[float, float, float, float]]:
        code = self._day_code
        bars: dict[str, tuple[float, float, float, float]] = {}
        tickers = self._scan_tickers()
        if tickers != self._scan_tickers_cache:
            self._scan_tickers_cache = tickers
            self._scan_tickers_sorted = sorted(tickers)
        for ticker in self._scan_tickers_sorted:
            record: _Ticker | None = self._tickers.get(ticker)
            if record is None:
                continue
            codes = record.arrays[0]
            index: int = record.cursor
            length: int = codes.size
            while index < length and codes[index] < code:
                index += 1
            record.cursor = index
            if index < length and codes[index] == code:
                record.index = index
                record.index_code = code
                bars[ticker] = (
                    float(record.arrays[1][index]),
                    float(record.arrays[2][index]),
                    float(record.arrays[3][index]),
                    float(record.arrays[4][index]),
                )
            else:
                record.index = None
                record.index_code = None
        return bars

    # Volume of today's bar for one ticker, or None when it did not trade today.
    def _today_volume(self, ticker: str) -> float | None:
        record: _Ticker | None = self._tickers.get(ticker)
        if record is None or record.index is None or record.index_code != self._day_code:
            return None
        return float(record.arrays[5][record.index])

    # The index of today's bar, or None when this ticker did not trade.
    def _index_on_day(self, record) -> int | None:
        if record is None or record.index_code != self._day_code:
            return None
        return record.index

    # Hand the strategy this ticker's loaded series so its indicators compute once.
    def _publish_series(self, ticker: str) -> None:
        record: _Ticker | None = self._tickers.get(ticker)
        if record is not None:
            self.strategy._indicator_series = record.series
            self.strategy._indicator_ticker = ticker

    # Today's view of one loaded ticker: the shared object, cut to today's bar.
    def _history_as_of(self, ticker: str, index: int) -> StockData:
        history = self._tickers[ticker].history
        history.set_window(0, index + 1)
        return history

    # Ask the strategy for entries on every name it may still buy.
    def _process_tickers(self, bars: dict[str, tuple[float, float, float, float]]) -> None:
        universe: set[str] = self._universe_tickers
        ordered: list[str] = list(self.strategy.tickers)
        for ticker in self.manager.open_tickers:
            if ticker not in universe:
                ordered.append(ticker)

        max_positions: int | None = self.strategy.MAX_POSITIONS
        min_bars: int = self.strategy.MIN_BARS_REQUIRED
        for ticker in ordered:
            if ticker not in bars:
                continue
            held: int = self.manager.lot_count(ticker)
            if ticker in universe:
                max_lots: int | None = self.strategy.MAX_LOTS_PER_TICKER
                can_add_lot: bool = max_lots is None or held < max_lots
                has_position_capacity: bool = max_positions is None or bool(held) or self.manager.active_positions < max_positions
                if can_add_lot and has_position_capacity:
                    index = self._index_on_day(self._tickers.get(ticker))
                    # len(dates) is index + 1, so a certainly-too-short history never builds a view.
                    if index is None or index + 1 < min_bars:
                        continue
                    self._publish_series(ticker)
                    history = self._history_as_of(ticker, index)
                    if len(history.dates) >= min_bars:
                        order = self._timed_hook("on_signal", self.strategy.on_signal, ticker, history)
                        if order is not None:
                            self.manager.submit_order(order)
                            if (order.stop_loss > 0 or order.take_profit > 0) and not self._has_resting_exits:
                                self._has_resting_exits = True

    # Let the strategy close its own positions and queue an exit for the next open.
    def _process_custom_exits(self, bars: dict[str, tuple[float, float, float, float]]) -> None:
        min_bars: int = self.strategy.MIN_BARS_REQUIRED
        for ticker in self.manager.open_tickers:
            if ticker not in bars:
                continue
            self._publish_series(ticker)
            # Same skip-fast-path as entries: the len() check below stays.
            index = self._index_on_day(self._tickers.get(ticker))
            if index is None or index + 1 < min_bars:
                continue
            history = self._history_as_of(ticker, index)
            if len(history.dates) < min_bars:
                continue
            for trade in self.manager.lots(ticker):
                trade.history = history
                exit_signal = self._timed_hook("should_exit", self.strategy.should_exit, ticker, trade, history)
                if exit_signal:
                    self.manager.queue_exit(trade.id)

    # Check stop-loss and take-profit levels against today's bar.
    def _check_resting_exits(self, current_date: str, bars: dict[str, tuple[float, float, float, float]]) -> None:
        if not self._has_resting_exits:
            return

        for ticker in self.manager.open_tickers:
            bar = bars.get(ticker)
            if bar is not None:
                self.manager.check_exits(ticker, current_date, bar[0], bar[1], bar[2])

    # Mark the open book to today's closes.
    def _unrealized_pnl(self, closes: dict[str, float]) -> float:
        unrealized: float = 0.0
        for trade in self.manager.open_trades.values():
            price: float = closes.get(trade.ticker, self.manager.last_prices.get(trade.ticker, trade.entry_price))
            direction: float = -1.0 if trade.position_type is PositionType.SHORT else 1.0
            notional: float = trade.equity_at_entry * trade.allocation
            unrealized += notional * direction * (price - trade.entry_price) / trade.entry_price
        return unrealized


# -------- Per-Ticker Record --------
# One ticker's bars and the run state that walks them: the arrays the day scan reads, the
# indicator series the strategy polls, a cursor and today's index.
class _Ticker:
    __slots__ = ("history", "arrays", "_series", "cursor", "index", "index_code", "ready")

    def __init__(self, history: StockData) -> None:
        self.history = history
        self.arrays = history.full
        self._series = None
        self.cursor = 0
        self.index = None
        self.index_code = None
        self.ready = False

    # The full array behind each column, keyed by address. Built on first poll: ranking names never need it.
    @property
    def series(self) -> dict[int, np.ndarray]:
        if self._series is None:
            self._series = _series_by_address(self.history)
        return self._series


# -------- As-Of Histories --------
# One date's view of the universe, handed to a universe hook. Iterating or measuring it loads
# every candidate, exactly as the eager snapshot did; asking for one ticker loads only that
# ticker. A strategy with a fixed ticker list therefore never pays for the rest of the market.
class _AsOfHistories(Mapping):
    __slots__ = ("_run", "_date", "_filled")

    def __init__(self, run: Backtester, current_date: str) -> None:
        self._run = run
        self._date = current_date
        self._filled: dict[str, StockData] | None = None

    # The full snapshot, built on first use and kept for the rest of this view.
    def _all(self) -> dict[str, StockData]:
        if self._filled is None:
            self._filled = self._run._snapshot(self._date)
        return self._filled

    def __getitem__(self, ticker: str) -> StockData:
        if self._filled is not None:
            return self._filled[ticker]
        history: StockData | None = self._run._snapshot_one(self._date, ticker)
        if history is None:
            raise KeyError(ticker)
        return history

    def __iter__(self):
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())

    # A single lookup never materializes the rest of the universe.
    def get(self, ticker: str, default=None):
        try:
            return self[ticker]
        except KeyError:
            return default


# One ticker's price columns, keyed by the memory address of their first bar.
def _series_by_address(history: StockData) -> dict[int, np.ndarray]:
    _codes, opens, highs, lows, closes, volumes = history.full
    return {
        array.ctypes.data: array
        for array in (opens, highs, lows, closes, volumes)
        if isinstance(array, np.ndarray)
    }

# -------- Statistics --------
# Moment skewness and Pearson kurtosis in numpy
def _skew(values: np.ndarray) -> float:
    spread: float = float(np.mean((values - np.mean(values)) ** 2))
    if spread <= 0:
        return 0.0
    return float(np.mean((values - np.mean(values)) ** 3) / spread ** 1.5)


def _kurtosis(values: np.ndarray) -> float:
    spread: float = float(np.mean((values - np.mean(values)) ** 2))
    if spread <= 0:
        return 0.0
    return float(np.mean((values - np.mean(values)) ** 4) / spread ** 2)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


# -------- Stamp --------
# What a run is made of, so results cache and compare fairly: package, data versions, range, costs, strategy knobs.
_MISSING = object()


# The installed package version, or "unknown" when it cannot be determined.
def _package_version() -> str:
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("sltp")
    except Exception:
        return "unknown"


# Where the run's bars came from: "own:<dir>" or "cache@<version>".
def _data_source() -> str:
    try:
        from sltp.market.layout import source

        info = source()
        if info.get("kind") == "cache":
            return f"cache@{info.get('version', 'unknown')}"
        return f"own:{info.get('dir', '?')}"
    except Exception:
        return "unknown"

    # The published data version, or "unknown" when the manifest was never read.
def _data_version() -> str:
    try:
        from sltp.market.remote import read_manifest
        return str(read_manifest().get("version", "unknown"))
    except Exception:
        return "unknown"


# A value that survives json.dumps, or _MISSING when it cannot be represented.
def _stamp_scalar(value):
    value = getattr(value, "value", value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    ):
        return list(value)
    return _MISSING


# The strategy's knobs: its constraints plus any public scalar the subclass set.
def _strategy_params(strategy) -> dict:
    base: set[str] = set(vars(Strategy()))
    custom: dict = {}
    for key, value in vars(strategy).items():
        if key in base or key.startswith("_"):
            continue
        scalar = _stamp_scalar(value)
        if scalar is not _MISSING:
            custom[key] = scalar
    return {
        "class": type(strategy).__name__,
        "name": getattr(strategy, "name", None),
        "constraints": {
            "max_positions": _stamp_scalar(getattr(strategy, "MAX_POSITIONS", None)),
            "max_allocation_per_ticker": _stamp_scalar(getattr(strategy, "MAX_ALLOCATION_PER_TICKER", None)),
            "max_lots_per_ticker": _stamp_scalar(getattr(strategy, "MAX_LOTS_PER_TICKER", None)),
            "rebalance_frequency": _stamp_scalar(getattr(strategy, "REBALANCE_FREQUENCY", None)),
            "min_bars_required": _stamp_scalar(getattr(strategy, "MIN_BARS_REQUIRED", None)),
            "lookback_bars": _stamp_scalar(getattr(strategy, "LOOKBACK_BARS", None)),
            "liquidate_on_end": _stamp_scalar(getattr(strategy, "LIQUIDATE_ON_END", None)),
        },
        "custom": custom,
    }
