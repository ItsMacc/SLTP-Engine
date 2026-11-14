# -------- backtester.py --------
# Runs a strategy day by day and returns what happened.
# Callbacks only ever see bars up to the current date, and orders fill at the next open.

# -------- Imports --------
import inspect
import math
import sys
import time

import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta

# Engine imports
from sltp.portfolio import PortfolioManager
from sltp.strategy import Strategy
from sltp.utils.Dates import CALENDAR, to_code
from sltp.utils.StockData import StockData
from sltp.utils.PositionType import PositionType

# Database imports
from sltp.data.market_data_service import get_dividends, get_trading_calendar, _get_histories, _run_guard


# -------- Constants --------
# How far past the requested start the universe may look for a date it can trade.
# A start-date search budget, not a warmup: it bounds the forward probe for a
# tradable date and has nothing to do with how much history a strategy reads.
MAX_START_PROBE_DAYS: int = 1826
# History loaded when a strategy declares no LOOKBACK_BARS of its own: five
# years of sessions, enough for the hungriest indicator or ranking window.
DEFAULT_LOOKBACK_DAYS: int = 1826
# As-of universe snapshots kept per date; probe dates never repeat, so a few suffice.
_ASOF_CACHE_LIMIT: int = 4
# A cash distribution above the whole pre-dividend share price is a data artefact.
MAX_DIVIDEND_YIELD: float = 1.0
# Hook profiling: the first 16,589 calls of a hot hook are timed exactly (two
# clock reads are cheap at that scale), the rest are timed on whole sampled
# days only, so untimed days cost the inner loops nothing at all. The sampled
# budget is the finite-population formula for +-1% (99% accuracy) at 99%
# confidence: n = n0 / (1 + n0 / remaining) with
#   n0 = z^2 * p(1-p) / E^2 = 2.576^2 * 0.25 / 0.01^2 = 16,589.
# (No power of two needed: days are counted with plain modulo, not a bitmask.)
# p = 1/2 is the worst case (maximum variance), so real timing data lands
# inside +-1% with room to spare. The absolute count rises with N but never
# exceeds 16,589; only the fraction falls (50% at N = 16.6k, 0.04% at 39M).
# The day is the sampling unit, so at least 50 days are always sampled:
# 16,589 calls can collapse into 3 days on a wide universe, and three days
# cannot represent 28 years of market regimes. Only these hooks are sampled;
# the rest fire rarely enough to time always.
PROFILE_EXACT_CALLS: int = 16_589
PROFILE_SAMPLE_N0: float = 16589.0
PROFILE_MIN_SAMPLE_DAYS: int = 50
PROFILE_SAMPLED_HOOKS: frozenset = frozenset({"on_signal", "should_exit"})


# -------- Stamp --------
# What a run is made of, so results can be cached honestly and compared fairly:
# package and data versions, the requested and actual range, the cost knobs and
# the strategy's own knobs. Everything in it survives json.dumps.
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
        from sltp.data.paths import source

        info = source()
        if info.get("kind") == "cache":
            return f"cache@{info.get('version', 'unknown')}"
        return f"own:{info.get('dir', '?')}"
    except Exception:
        return "unknown"

    # The published data version, or "unknown" when the manifest was never read.
    # Sync paths already parse it, so this costs nothing whenever data was fetched.
def _data_version() -> str:
    try:
        from sltp.data.remote import read_manifest
        return str(read_manifest().get("version", "unknown"))
    except Exception:
        return "unknown"


# A value that survives json.dumps, or _MISSING when it cannot be represented.
# Plain scalars pass through; enums unwrap to their value; lists stay lists.
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
# The universe itself stays out: it is rebuilt from code and data at run time.
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


# -------- Backtester --------
# One simulation: a strategy, a date range, and the bars it needs.
class Backtester:
    # Wire up the manager and the per-run state.
    def __init__(self, strategy: Strategy, start_date: str, end_date: str | None = None, include_dividends: bool = True,
                 spread_pct: float = 0.0, max_participation: float | None = None,
                 borrow_rate: float = 0.0, hard_to_borrow: list[str] | set[str] | tuple[str, ...] | None = None) -> None:
        if spread_pct < 0:
            raise ValueError(f"spread_pct must be non-negative, got {spread_pct}.")
        if max_participation is not None and max_participation <= 0:
            raise ValueError(f"max_participation must be positive, got {max_participation}.")
        if borrow_rate < 0:
            raise ValueError(f"borrow_rate must be non-negative, got {borrow_rate}.")
        self.strategy: Strategy = strategy
        self.manager: PortfolioManager = PortfolioManager(strategy)
        self.manager.SPREAD_PCT = float(spread_pct)
        self.manager.MAX_PARTICIPATION = None if max_participation is None else float(max_participation)
        self.manager.BORROW_RATE = float(borrow_rate)
        self.manager.HARD_TO_BORROW = frozenset(hard_to_borrow) if hard_to_borrow else frozenset()
        self.manager.set_history_loader(self._history_for_trade)
        self.start_date: str = start_date
        self.end_date: str | None = end_date
        self.include_dividends: bool = include_dividends
        self.liquidate_on_end: bool = self.strategy.LIQUIDATE_ON_END

        # Only the chunks overlapping this window are read, so a 2020 run skips the 1998 file.
        self.load_start: str = self._load_start_for(self.start_date)

        # Ticker histories are loaded once and indexed once for the whole run
        self._data: dict[str, dict] = {}
        self._dividend_events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.skipped: set[str] = set()
        self._unrealized_at_end: float = 0.0
        self._equity_records: list[dict] = []
        self._has_run: bool = False
        self._has_custom_exit: bool = getattr(strategy.should_exit, "__func__", None) is not Strategy.should_exit
        self._has_resting_exits: bool = False
        self._universe_tickers: set[str] = set()
        self._asof_cache: dict[str, dict[str, StockData]] = {}
        self._universe_histories: dict[str, StockData] = {}
        self._universe_skipped: set[str] = set()
        self._cut_cursors: dict[str, tuple[int, int, int]] = {}
        self._pending_rebalance: bool = False
        self._day_code = np.int32(0)
        self._hook_time: dict[str, float] = {}
        self._hook_calls: dict[str, int] = {}
        # Profiling state: exact-prefix snapshots taken when the prefix ends,
        # plus the day counts that scale sampled hooks (see _run).
        self._profile_prefix_time: dict[str, float] = {}
        self._profile_prefix_calls: dict[str, int] = {}
        self._profile_remaining_days: int = 0
        self._profile_sampled_days: int = 0
        self._universe_warned: bool = False

    # -------- Run --------
    # Simulate the strategy and return metrics, equity, trades, rejections and a stamp.
    # Quiet skips the stderr breakdown; fetch it later with hook_breakdown_text().
    def run(self, quiet: bool = False) -> dict:
        if self._has_run:
            raise RuntimeError("This Backtester has already run. Build a new one for another simulation.")
        self._has_run = True
        with _run_guard():
            return self._run(quiet)

    # The simulation itself, under the read guard so strategy hooks cannot reach
    # past the run window through the data service.
    def _run(self, quiet: bool = False) -> dict:
        self.strategy._reset_indicator_state()

        actual_start: str = self._initialize_universe()
        self._ensure_universe()
        calendar: list = self._build_calendar(actual_start)
        if not calendar:
            raise ValueError("No trading days available between the start date and the loaded data.")

        # Profiling plan, decided once. The leading prefix_days are timed exactly
        # (the first ~20,000 hot-hook calls, warmup included); later days are
        # timed one in sample_every, spread evenly so history growth is covered
        # uniformly. The decision is per day, so untimed days are pure engine.
        tickers_per_day = max(1, len(self._scan_tickers()))
        prefix_days = min(len(calendar), (PROFILE_EXACT_CALLS + tickers_per_day - 1) // tickers_per_day)
        remaining_calls = (len(calendar) - prefix_days) * tickers_per_day
        if remaining_calls:
            sampled_calls = PROFILE_SAMPLE_N0 / (1.0 + PROFILE_SAMPLE_N0 / remaining_calls)
            sample_every = max(1, round(remaining_calls / sampled_calls))
            sample_every = min(sample_every, max(1, (len(calendar) - prefix_days) // PROFILE_MIN_SAMPLE_DAYS))
        else:
            sample_every = 1
        self._profile_remaining_days = len(calendar) - prefix_days
        self._profile_sampled_days = 0

        self.manager.mark_rebalanced(CALENDAR.label(calendar[0]))
        for day_index, code in enumerate(calendar):
            self._day_code = code
            current_date: str = CALENDAR.label(code)

            # The day's timing decision, made once: everything below reads it.
            timed_day = day_index < prefix_days or (day_index - prefix_days) % sample_every == 0
            if day_index == prefix_days:
                # The exact prefix ends here. What was measured so far stays as
                # the warmup-inclusive baseline; sampled days scale on top.
                self._profile_prefix_time = dict(self._hook_time)
                self._profile_prefix_calls = dict(self._hook_calls)
            if timed_day and day_index >= prefix_days:
                self._profile_sampled_days += 1

            # Strategy lifecycle callbacks come before order processing
            self.manager.refresh_view()
            self._timed_hook("run_daily", self.strategy.run_daily, current_date)

            if self.manager.rebalance_due(current_date):
                self._pending_rebalance = True

            # A rebalance can change the active universe; cached tickers are cheap to revisit
            self._ensure_universe()
            bars = self._bars_on()
            opens = {ticker: bar[0] for ticker, bar in bars.items()}
            closes = {ticker: bar[3] for ticker, bar in bars.items()}

            # Dividends belong to lots held from the previous close, before today's fills
            self.manager.pay_dividends(int(self._day_code))

            # Fill pending exits and entries at today's open
            self.manager.fill_pending_exits(current_date, opens)
            filled_tickers: set[str] = {order.ticker for order in self.manager.pending_orders}
            self.manager.fill_pending_orders(current_date, opens)
            for ticker in filled_tickers:
                fill_index: int | None = self._index_on_day(self._data.get(ticker))
                if fill_index is not None:
                    for trade in self.manager.lots(ticker):
                        if trade.entry_date == current_date:
                            trade.history = self._history_as_of(ticker, fill_index)

            # Evaluate entries against history through today's completed bar
            self.manager.refresh_view()
            if self._pending_rebalance:
                self._call_universe_hook(self.strategy.on_rebalance, current_date)
                self.manager.mark_rebalanced(current_date)
                self._pending_rebalance = False
                self._ensure_universe()
                bars = self._bars_on()
                opens = {ticker: bar[0] for ticker, bar in bars.items()}
                closes = {ticker: bar[3] for ticker, bar in bars.items()}
            self._process_tickers(current_date, bars, timed_day)

            # Custom exits use today's completed data and queue next-open orders
            if self._has_custom_exit and self.manager.open_trades:
                self._process_custom_exits(current_date, bars, timed_day)

            # Close-based protective exits use today's final OHLC
            self._check_resting_exits(current_date, bars)

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

    # Where the run's Python time went, by strategy hook. To stderr so it never
    # pollutes --json output; a slow strategy names itself here. The time column
    # starts past the longest hook name, so every row lines up no matter which
    # hooks a strategy defines.
    # The breakdown as text, for callers (like `sltp -o`) that file it instead
    # of printing it. Empty when nothing was timed.
    def hook_breakdown_text(self) -> str:
        if not self._hook_calls:
            return ""
        lines: list[str] = [f"hook time: {getattr(self.strategy, 'name', '?')}"]
        width: int = max(len(name) for name in self._hook_time)
        rows: list = []
        for name in self._hook_time:
            seconds: float = self._hook_time[name]
            calls: int = self._hook_calls[name]
            sampled = name in PROFILE_SAMPLED_HOOKS and self._profile_sampled_days > 0
            if sampled:
                # Estimate the whole run from the exact prefix plus the sampled
                # days. A '~' marks rows that are estimates, not measurements.
                scale = self._profile_remaining_days / self._profile_sampled_days
                prefix_seconds = self._profile_prefix_time.get(name, 0.0)
                prefix_calls = self._profile_prefix_calls.get(name, 0)
                seconds = prefix_seconds + (seconds - prefix_seconds) * scale
                calls = prefix_calls + round((calls - prefix_calls) * scale)
            rows.append((name, seconds, calls, sampled))
        for name, seconds, calls, sampled in sorted(rows, key=lambda row: row[1], reverse=True):
            mark = "~" if sampled else " "
            lines.append(f"  {name.ljust(width)}   {seconds:>7.2f}s x{calls:>7,}{mark}")
        return "\n".join(lines)

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
                "borrow_rate": self.manager.BORROW_RATE,
                "hard_to_borrow": sorted(self.manager.HARD_TO_BORROW),
                "include_dividends": self.include_dividends,
            },
            "strategy": _strategy_params(self.strategy),
        }

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

    # -------- Universe and Data --------
    # The earliest bar a call on current_date may look back on. A strategy that
    # declares LOOKBACK_BARS gets that many sessions, doubled so weekends and
    # holidays never eat into it; anything else keeps the five-year default.
    def _load_start_for(self, current_date: str) -> str:
        lookback: int | None = self.strategy.LOOKBACK_BARS
        days: int = DEFAULT_LOOKBACK_DAYS if lookback is None else max(0, lookback * 2)
        return (datetime.strptime(current_date, "%Y-%m-%d") - relativedelta(days=days)).strftime("%Y-%m-%d")
    # Build the universe, walking the start date forward until the strategy accepts it.
    def _initialize_universe(self) -> str:
        start_dt: datetime = datetime.strptime(self.start_date, "%Y-%m-%d")
        if self._universe_at(start_dt, 0):
            return self._accepted_universe_date(start_dt, 0)

        # The earliest success and the tickers it built, so the winning probe
        # never has to run (and reload) a second time.
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

    # Time one strategy hook call for the end-of-run breakdown. Hot hooks reach
    # this on timed days only (see the profiling plan in _run).
    def _timed_hook(self, name: str, hook, *args):
        start: float = time.perf_counter()
        try:
            return hook(*args)
        finally:
            self._hook_time[name] = self._hook_time.get(name, 0.0) + (time.perf_counter() - start)
            self._hook_calls[name] = self._hook_calls.get(name, 0) + 1

    # Call a universe hook, passing as-of histories when the strategy asks for them.
    # A histories parameter defaulting to None promises the hook ignores it,
    # so the full-universe load is skipped entirely.
    def _call_universe_hook(self, hook, current_date: str):
        parameters = list(inspect.signature(hook).parameters.values())
        accepts_histories: bool = len(parameters) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )
        if accepts_histories:
            histories_param = parameters[1] if len(parameters) >= 2 else None
            if histories_param is not None and histories_param.default is None:
                return self._timed_hook(hook.__name__, hook, current_date)
            return self._timed_hook(hook.__name__, hook, current_date, self._histories_as_of(current_date))
        return self._timed_hook(hook.__name__, hook, current_date)

    # Every ticker's bars up to the callback date, for universe ranking.
    # One entry per date asked, so a repeated probe is a dict hit, not a reload.
    def _histories_as_of(self, current_date: str) -> dict[str, StockData]:
        cached: dict[str, StockData] | None = self._asof_cache.get(current_date)
        if cached is not None:
            return dict(cached)

        histories: dict[str, StockData] = {}
        # A warm-up probe can look past the run's own end date, and the bulk read
        # stops there; that rare date keeps the old one-shot load.
        if self.end_date is not None and current_date > self.end_date:
            histories = self._load_histories(current_date)
        else:
            histories = self._cut_histories(current_date)

        self._asof_cache[current_date] = histories
        while len(self._asof_cache) > _ASOF_CACHE_LIMIT:
            self._asof_cache.pop(next(iter(self._asof_cache)))
        return dict(histories)

    # Read the universe for one date straight from the files. Only the runaway
    # warm-up probe uses this; every normal ranking date is cut from the bulk read.
    def _load_histories(self, current_date: str) -> dict[str, StockData]:
        code = to_code(current_date)
        histories: dict[str, StockData] = {}
        # Ranking only reads back from today, so the files after today cannot contribute.
        window_start: str = self._load_start_for(current_date)
        for ticker, history in _get_histories(self.strategy.available_universe, window_start, current_date).items():
            index: int = int(np.searchsorted(history.date_codes, code, side="right")) - 1
            if index >= 0:
                histories[ticker] = self._slice_history(history, index)
        return histories

    # Warn once when a large universe is mostly unusable: names with fewer bars
    # than MIN_BARS_REQUIRED can never rank, so the user sees where the bulk
    # read went instead of wondering why a full-universe strategy is slow.
    def _warn_once_for_unusable_symbols(self) -> None:
        if self._universe_warned or len(self._universe_histories) < 1000:
            return
        self._universe_warned = True
        min_bars: int = int(getattr(self.strategy, "MIN_BARS_REQUIRED", 0) or 0)
        if min_bars <= 0:
            return
        short: int = sum(1 for history in self._universe_histories.values() if len(history.date_codes) < min_bars)
        if short:
            total: int = len(self._universe_histories)
            print(
                f"note: {total} symbols loaded for ranking; {short} have fewer than "
                f"{min_bars} bars and can never meet MIN_BARS_REQUIRED.",
                file=sys.stderr,
            )

    # Load the whole ranking universe once, over the run's window.
    # Tickers with no bars anywhere in it are remembered so they are not re-read.
    def _ensure_universe_histories(self, symbols: list[str]) -> None:
        missing: list[str] = [
            symbol for symbol in dict.fromkeys(symbols)
            if symbol not in self._universe_histories and symbol not in self._universe_skipped
        ]
        if not missing:
            return
        loaded: dict[str, StockData] = _get_histories(missing, self.load_start, self.end_date)
        for symbol in missing:
            history: StockData | None = loaded.get(symbol)
            if history is None:
                self._universe_skipped.add(symbol)
            else:
                self._universe_histories[symbol] = history
        self._warn_once_for_unusable_symbols()

    # One history for a trade opened in a ticker the handed-over cache never saw,
    # cut from the same bulk window so a fill never triggers a side read.
    def _history_for_trade(self, ticker: str, current_date: str) -> StockData | None:
        self._ensure_universe_histories([ticker])
        history: StockData | None = self._universe_histories.get(ticker)
        if history is None:
            return None
        index: int = int(np.searchsorted(history.date_codes, to_code(current_date), side="right")) - 1
        if index < 0:
            return None
        return self._slice_history(history, index)

    # Cut every ticker's bars to this date's ranking window. The cuts are views of
    # the arrays already in memory, so a rebalance costs slicing, not reading files.
    def _cut_histories(self, current_date: str) -> dict[str, StockData]:
        code = to_code(current_date)
        low_code = to_code(self._load_start_for(current_date))
        symbols: list[str] = self.strategy.available_universe
        self._ensure_universe_histories(symbols)

        cursors: dict[str, tuple[int, int, int]] = self._cut_cursors
        histories: dict[str, StockData] = {}
        for ticker in symbols:
            history: StockData | None = self._universe_histories.get(ticker)
            if history is None:
                continue
            codes = history.date_codes
            cursor: tuple[int, int, int] | None = cursors.get(ticker)
            # Bars only ever arrive at the end, so a name that did not trade since the
            # last rebalance keeps its cut and needs no binary search at all.
            if (
                cursor is not None
                and code >= cursor[2]
                and (cursor[1] >= codes.size or codes[cursor[1]] > code)
                and (cursor[0] >= cursor[1] or codes[cursor[0]] >= low_code)
            ):
                lo, hi = cursor[0], cursor[1]
            else:
                lo = int(np.searchsorted(codes, low_code, side="left"))
                # Searching the tail for the far edge finds the second bound without
                # a second walk over the whole history.
                hi = lo + int(np.searchsorted(codes[lo:], code, side="right"))
                cursors[ticker] = (lo, hi, code)
            # A ticker with no bar inside this window is absent, exactly as a
            # single-date read would have left it out.
            if lo >= hi:
                continue
            histories[ticker] = self._slice_range(history, lo, hi)
        return histories

    # Make sure every ticker the strategy currently wants is loaded.
    def _ensure_universe(self) -> None:
        wanted: list[str] = list(dict.fromkeys([*self.strategy.tickers, *self.manager.open_tickers]))
        self._universe_tickers.update(self.strategy.tickers)
        self._ensure_tickers(wanted)

    # Load the given tickers, remembering the ones with no local data.
    def _ensure_tickers(self, tickers: list[str]) -> None:
        missing: list[str] = [ticker for ticker in tickers if ticker not in self._data and ticker not in self.skipped]
        if not missing:
            return

        # Anything the ranking universe already loaded (same window) is reused, not re-read.
        loaded: dict[str, StockData] = {
            ticker: self._universe_histories[ticker] for ticker in missing if ticker in self._universe_histories
        }
        unread: list[str] = [
            ticker for ticker in missing if ticker not in loaded and ticker not in self._universe_skipped
        ]
        if unread:
            loaded.update(_get_histories(unread, self.load_start, self.end_date))
        for ticker in missing:
            history: StockData | None = loaded.get(ticker)
            if history is None:
                self.skipped.add(ticker)
                continue
            self._data[ticker] = {
                "history": history,
                "codes": history.date_codes,
                "open": history.open_price,
                "high": history.high,
                "low": history.low,
                "close": history.close_price,
                "volume": history.volume,
                "series": _series_by_address(history),
                "cursor": 0,
                "index": None,
                "index_code": None,
            }
        if self.include_dividends:
            self._load_dividends(missing)
        self.manager.set_history_cache({ticker: data["history"] for ticker, data in self._data.items()})

    # Load dividends for the tickers just added, dropping impossible rows.
    def _load_dividends(self, tickers: list[str]) -> None:
        for ticker in tickers:
            data = self._data.get(ticker)
            if data is None or ticker in self._dividend_events:
                continue
            codes, amounts = get_dividends(ticker, self.load_start, self.end_date)
            codes, amounts = self._plausible_dividends(data["history"], codes, amounts)
            if codes.size:
                self._dividend_events[ticker] = (codes, amounts)
        self.manager.set_dividend_events(self._dividend_events)

    # Keep dividends that could have been paid: a previous close must exist and fit the price.
    @staticmethod
    def _plausible_dividends(history: StockData, codes: np.ndarray, amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if codes.size == 0:
            return codes, amounts
        index: np.ndarray = np.searchsorted(history.date_codes, codes, side="left") - 1
        valid: np.ndarray = index >= 0
        previous: np.ndarray = np.zeros(codes.size, dtype=np.float64)
        previous[valid] = history.close_price[index[valid]]
        keep: np.ndarray = valid & (previous > 0) & (amounts <= previous * MAX_DIVIDEND_YIELD)
        return codes[keep], amounts[keep]

    # The market's own trading days inside the run window.
    def _build_calendar(self, actual_start: str) -> list:
        if not self._data:
            raise ValueError("No price data loaded. initialize_universe() must add locally available tickers.")
        codes = get_trading_calendar(actual_start, self.end_date)
        start_code = to_code(actual_start)
        end_code = to_code(self.end_date) if self.end_date is not None else codes[-1]
        return codes[(codes >= start_code) & (codes <= end_code)].tolist()

    # -------- Per-day Bars --------
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
        for ticker in sorted(self._scan_tickers()):
            data = self._data.get(ticker)
            if data is None:
                continue
            codes = data["codes"]
            index: int = data["cursor"]
            length: int = len(codes)
            while index < length and codes[index] < code:
                index += 1
            data["cursor"] = index
            if index < length and codes[index] == code:
                data["index"] = index
                data["index_code"] = code
                bars[ticker] = (
                    float(data["open"][index]),
                    float(data["high"][index]),
                    float(data["low"][index]),
                    float(data["close"][index]),
                )
            else:
                data["index"] = None
                data["index_code"] = None
        return bars

    # The index of today's bar, or None when this ticker did not trade.
    def _index_on_day(self, data) -> int | None:
        if data is None or data["index_code"] != self._day_code:
            return None
        return data["index"]

    # Hand the strategy this ticker's loaded series so its indicators compute once.
    def _publish_series(self, ticker: str) -> None:
        data = self._data.get(ticker)
        if data is not None:
            self.strategy._indicator_series = data["series"]
            self.strategy._indicator_ticker = ticker

    # Today's view of one loaded ticker.
    def _history_as_of(self, ticker: str, index: int) -> StockData:
        return self._slice_history(self._data[ticker]["history"], index)

    # Cut a history so its last bar is the given index, giving strategies a no-lookahead view.
    @staticmethod
    def _slice_history(history: StockData, index: int) -> StockData:
        end: int = index + 1
        return StockData.view(
            history.ticker,
            history.date_codes[:end],
            history.open_price[:end],
            history.high[:end],
            history.low[:end],
            history.close_price[:end],
            history.volume[:end],
        )

    # Cut a history to a half-open bar range. Slicing NumPy gives views, so this
    # shares memory with the bulk read instead of copying it.
    @staticmethod
    def _slice_range(history: StockData, lo: int, hi: int) -> StockData:
        return StockData.view(
            history.ticker,
            history.date_codes[lo:hi],
            history.open_price[lo:hi],
            history.high[lo:hi],
            history.low[lo:hi],
            history.close_price[lo:hi],
            history.volume[lo:hi],
        )

    # -------- Daily Loop Helpers --------
    # Ask the strategy for entries on every name it may still buy.
    def _process_tickers(self, current_date: str, bars: dict[str, tuple[float, float, float, float]], timed_day: bool) -> None:
        universe: set[str] = self._universe_tickers
        held_tickers: set[str] = self.manager.open_tickers
        ordered: list[str] = list(self.strategy.tickers)
        for ticker in held_tickers:
            if ticker not in universe:
                ordered.append(ticker)

        max_positions: int | None = self.strategy.MAX_POSITIONS
        for ticker in ordered:
            if ticker not in bars:
                continue
            held: list = self.manager.lots(ticker)
            if ticker in universe:
                max_lots: int | None = self.strategy.MAX_LOTS_PER_TICKER
                can_add_lot: bool = max_lots is None or len(held) < max_lots
                has_position_capacity: bool = max_positions is None or bool(held) or self.manager.active_positions < max_positions
                if can_add_lot and has_position_capacity:
                    self._publish_series(ticker)
                    history = self._history_for_date(ticker)
                    if len(history.dates) >= self.strategy.MIN_BARS_REQUIRED:
                        if timed_day:
                            order = self._timed_hook("on_signal", self.strategy.on_signal, ticker, history)
                        else:
                            order = self.strategy.on_signal(ticker, history)
                        if order is not None:
                            self.manager.submit_order(order)
                            if (order.stop_loss > 0 or order.take_profit > 0) and not self._has_resting_exits:
                                self._has_resting_exits = True

    # Let the strategy close its own positions and queue an exit for the next open.
    def _process_custom_exits(self, current_date: str, bars: dict[str, tuple[float, float, float, float]], timed_day: bool) -> None:
        for ticker in self.manager.open_tickers:
            if ticker not in bars:
                continue
            self._publish_series(ticker)
            history = self._history_for_date(ticker)
            if len(history.dates) < self.strategy.MIN_BARS_REQUIRED:
                continue
            for trade in self.manager.lots(ticker):
                trade.history = history
                if timed_day:
                    exit_signal = self._timed_hook("should_exit", self.strategy.should_exit, ticker, trade, history)
                else:
                    exit_signal = self.strategy.should_exit(ticker, trade, history)
                if exit_signal:
                    self.manager.queue_exit(trade.id)

    # The strategy's view of one held ticker, ending on today's bar.
    def _history_for_date(self, ticker: str) -> StockData:
        index: int | None = self._index_on_day(self._data[ticker])
        if index is None:
            raise KeyError(f"{ticker} has no bar on {CALENDAR.label(self._day_code)}")
        return self._history_as_of(ticker, index)

    # Check stop-loss and take-profit levels against today's bar.
    def _check_resting_exits(self, current_date: str, bars: dict[str, tuple[float, float, float, float]]) -> None:
        if not self._has_resting_exits:
            return

        for ticker in self.manager.open_tickers:
            bar = bars.get(ticker)
            if bar is None:
                continue
            ticker_lots = self.manager.lots(ticker)
            if not any(trade.stop_loss > 0 or trade.take_profit > 0 for trade in ticker_lots):
                continue
            current_open, current_high, current_low, _close = bar
            self.manager.check_exits(ticker, current_date, current_open, current_high, current_low)

    # Mark the open book to today's closes.
    def _unrealized_pnl(self, closes: dict[str, float]) -> float:
        unrealized: float = 0.0
        for trade in self.manager.open_trades.values():
            price: float = closes.get(trade.ticker, self.manager.last_prices.get(trade.ticker, trade.entry_price))
            direction: float = -1.0 if trade.position_type is PositionType.SHORT else 1.0
            notional: float = trade.equity_at_entry * trade.allocation
            unrealized += notional * direction * (price - trade.entry_price) / trade.entry_price
        return unrealized


# One ticker's price columns, keyed by the memory address of their first bar.
def _series_by_address(history: StockData) -> dict[int, np.ndarray]:
    return {
        array.ctypes.data: array
        for array in (history.open_price, history.high, history.low, history.close_price, history.volume)
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

