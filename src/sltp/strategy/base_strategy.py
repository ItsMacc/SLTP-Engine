# -------- base_strategy.py --------
# Base class for a strategy: hooks, universe access and indicators.
# Only on_signal is required; the other hooks do nothing unless you override them.
# A strategy makes the decisions and keeps no book, and sees the portfolio read-only.

# -------- Imports --------
import numpy as np
import sltp.indicators as ta

from collections import OrderedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Engine imports
from sltp.utils.Order import Order
from sltp.utils.Trade import Trade
from sltp.utils.Dates import to_code
from sltp.utils.StockData import StockData
from sltp.utils.PositionType import PositionType
from sltp.utils.RebalanceFrequency import RebalanceFrequency

# Database imports
from sltp.data.market_data_service import get_all_tickers, get_available_etfs, get_available_gold, get_available_stocks


# -------- Constants --------
# Indicator cache budget
INDICATOR_CACHE_MAX_BYTES: int = 30 * 1024 * 1024


# -------- Strategy --------
# Base class for every strategy: hooks, universe access and indicators.
class Strategy:
    # Set False on a strategy to keep every indicator on the batch path below.
    STREAMING_INDICATORS: bool = True

    # Set up the identity, the portfolio limits and the empty universe.
    def __init__(self):
        # -------- Identity --------
        self.name: str = "Base Strategy"
        self.description: str = "Base Strategy Interface"

        # -------- Portfolio Constraints --------
        self.MAX_POSITIONS: int | None = 10
        self.MAX_ALLOCATION_PER_TICKER: float | None = None
        self.MAX_LOTS_PER_TICKER: int = 1
        self.REBALANCE_FREQUENCY: RebalanceFrequency = RebalanceFrequency.NEVER
        self.MIN_BARS_REQUIRED: int = 0
        self.LIQUIDATE_ON_END: bool = False

        # -------- Tickers --------
        self.tickers: list[str] = []

        # -------- Indicators --------
        # The series being polled, published by the backtester, and what has been
        # computed from it so far. Both stay None until the first call needs them.
        self._indicator_series: dict[int, np.ndarray] | None = None
        self._indicator_cache: OrderedDict = OrderedDict()
        self._indicator_cache_bytes: int = 0
        self._indicator_ticker: str | None = None
        self._stream_states: dict = {}

    # -------- Universe --------
    # Everything with local price data.
    @property
    def available_universe(self) -> list[str]:
        return get_all_tickers()

    # Exchange-traded funds only.
    @property
    def available_etfs(self) -> list[str]:
        return get_available_etfs()

    # Individual companies only.
    @property
    def available_stocks(self) -> list[str]:
        return get_available_stocks()

    # Gold proxies only.
    @property
    def available_gold(self) -> list[str]:
        return get_available_gold()

    # The tickers currently holding open positions.
    @property
    def open_tickers(self) -> frozenset[str]:
        return getattr(self, "_open_ticker_set", frozenset())

    # The active open trades.
    @property
    def open_positions(self) -> tuple[Trade, ...]:
        return getattr(self, "_open_trades", tuple())

    # The number of distinct active positions.
    @property
    def open_position_count(self) -> int:
        return len(self.open_tickers)

    # The current index value of the book; 100 before the first mark.
    @property
    def equity(self) -> float:
        return getattr(self, "_equity", 100.0)

    # Add one symbol to the universe, ignoring a duplicate.
    def add_ticker(self, ticker: str) -> None:
        if ticker not in self.tickers:
            self.tickers.append(ticker)

    # Add several symbols at once, keeping their order.
    def add_tickers(self, tickers: list[str]) -> None:
        for ticker in tickers:
            self.add_ticker(ticker)

    # Drop one symbol from the universe.
    def remove_ticker(self, ticker: str) -> None:
        if ticker in self.tickers:
            self.tickers.remove(ticker)

    # -------- Lifecycle Hooks --------
    # Build the starting universe; the default accepts the declared tickers.
    # The None default is a promise the hook ignores histories, so the engine skips loading them.
    def initialize_universe(self, current_date: str, histories: dict[str, StockData] | None = None) -> bool:
        return bool(self.tickers)

    # Refresh self.tickers when REBALANCE_FREQUENCY comes due.
    # The None default is a promise the hook ignores histories, so the engine skips loading them.
    def on_rebalance(self, current_date: str, histories: dict[str, StockData] | None = None) -> None:
        return None

    # Evaluate one ticker's history and return an order, or None for no trade.
    def on_signal(self, ticker: str, data: StockData) -> Order | None:
        raise NotImplementedError("Strategy subclasses must implement on_signal()")

    # Run once per day, before any ticker is polled.
    def run_daily(self, current_date: str) -> None:
        return None

    # Custom exit logic using the helper methods below.
    def should_exit(self, ticker: str, trade: Trade, history: StockData | None = None) -> bool:
        return False

    # -------- Indicator Helpers --------
    # Simple moving average.
    def _sma(self, values, window):
        return self._ta(ta.SMA, (window,), (values,))

    # Exponential moving average.
    def _ema(self, values, window):
        return self._ta(ta.EMA, (window,), (values,))

    # Double exponential moving average.
    def _dema(self, values, window):
        return self._ta(ta.DEMA, (window,), (values,))

    # Relative Strength Index, from 0 to 100.
    def _rsi(self, values, window):
        return self._ta(ta.RSI, (window,), (values,))

    # Moving average convergence divergence.
    def _macd(self, values, fast_period=12, slow_period=26, signal_period=9):
        return self._ta(ta.MACD, (fast_period, slow_period, signal_period), (values,))

    # Bollinger Bands above and below a moving average.
    def _bollinger_bands(self, values, window, deviations=2):
        return self._ta(ta.BBANDS, (window, deviations, deviations), (values,))

    # Average true range.
    def _atr(self, high, low, close, window):
        return self._ta(ta.ATR, (window,), (high, low, close))

    # Average directional index: trend strength.
    def _adx(self, high, low, close, window):
        return self._ta(ta.ADX, (window,), (high, low, close))

    # Stochastic oscillator: the close relative to its recent range.
    # The two zeros keep TA-Lib's default SMA smoothing for %K and %D.
    def _stochastic(self, high, low, close, fast_k=5, slow_k=3, slow_d=3):
        return self._ta(ta.STOCH, (fast_k, slow_k, 0, slow_d, 0), (high, low, close))

    # Williams %R.
    def _williams_r(self, high, low, close, window):
        return self._ta(ta.WILLR, (window,), (high, low, close))

    # Commodity Channel Index.
    def _cci(self, high, low, close, window):
        return self._ta(ta.CCI, (window,), (high, low, close))

    # Rate of change: percentage difference from `window` bars ago.
    def _roc(self, values, window):
        return self._ta(ta.ROC, (window,), (values,))

    # Price change over `window` bars.
    def _momentum(self, values, window):
        return self._ta(ta.MOM, (window,), (values,))

    # On-balance volume.
    def _obv(self, close, volume):
        return self._ta(ta.OBV, (), (close, volume))

    # Money flow index from price and volume.
    def _mfi(self, high, low, close, volume, window):
        return self._ta(ta.MFI, (window,), (high, low, close, volume))

    # Accumulation/distribution line.
    def _ad(self, high, low, close, volume):
        return self._ta(ta.AD, (), (high, low, close, volume))

    # Triple exponential average.
    def _trix(self, values, window):
        return self._ta(ta.TRIX, (window,), (values,))

    # Triple exponential moving average.
    def _tema(self, values, window):
        return self._ta(ta.TEMA, (window,), (values,))

    # Parabolic SAR.
    def _parabolic_sar(self, high, low, acceleration=0.02, maximum=0.2):
        return self._ta(ta.SAR, (acceleration, maximum), (high, low))

    # Stochastic oscillator applied to RSI.
    def _stochastic_rsi(self, values, window, fast_k=5, fast_d=3):
        return self._ta(ta.STOCHRSI, (window, fast_k, fast_d), (values,))

    # -------- Exit Helpers --------
    # True once the trade has been held for the allowed number of days.
    def _time_stop_days(self, trade: Trade, days: int) -> bool:
        entry_dt = datetime.strptime(trade.entry_date, "%Y-%m-%d")
        exit_deadline = to_code((entry_dt + relativedelta(days=days)).strftime("%Y-%m-%d"))

        return int(trade.history.date_codes[-1]) >= int(exit_deadline)

    # True when price has pulled back from its best level since entry.
    def _trailing_stop(self, trade: Trade, pct: float) -> bool:
        entry_idx = self._get_entry_index(trade)
        data = trade.history

        if trade.position_type == PositionType.LONG:
            # Long: stop below the highest price since entry.
            highest_since_entry = np.max(data.high[entry_idx:])
            trailing_stop = highest_since_entry * (1.0 - pct)
            return bool(data.low[-1] <= trailing_stop)
        else:
            # Short: stop above the lowest price since entry.
            lowest_since_entry = np.min(data.low[entry_idx:])
            trailing_stop = lowest_since_entry * (1.0 + pct)
            return bool(data.high[-1] >= trailing_stop)

    # True when price has moved against the entry by a multiple of the average true range.
    def _atr_stop(self, trade: Trade, multiplier: float, window: int = 14) -> bool:
        entry_idx = self._get_entry_index(trade)

        # Slice backward from entry_idx, without going below 0.
        start_idx = max(0, entry_idx - window - 1)

        data = trade.history
        high = data.high[start_idx:entry_idx + 1]
        low = data.low[start_idx:entry_idx + 1]
        close = data.close_price[start_idx:entry_idx + 1]

        atr_values = self._atr(high, low, close, window)
        # Only the last value has the full window behind it; the rest are warmup.
        current_atr: float = float(atr_values[-1]) if len(atr_values) else float("nan")
        if not np.isfinite(current_atr) or current_atr <= 0:
            return False

        if trade.position_type == PositionType.LONG:
            # Long: the stop sits below the entry by ATR * multiplier.
            stop_price: float = trade.entry_price - (current_atr * multiplier)
            return bool(data.low[-1] <= stop_price)
        else:
            # Short: the stop sits above the entry by ATR * multiplier.
            stop_price = trade.entry_price + (current_atr * multiplier)
            return bool(data.high[-1] >= stop_price)

    # Position of the entry bar, falling back to the nearest bar if the date is absent.
    def _get_entry_index(self, trade: Trade) -> int:
        data = trade.history
        entry_code = to_code(trade.entry_date)
        match_indices = np.flatnonzero(data.date_codes == entry_code)
        if len(match_indices) == 0:
            return int(np.searchsorted(data.date_codes, entry_code))
        return int(match_indices[0])

    # -------- Indicator Plumbing --------
    # TA-Lib wants float64 and the bars are float32, so widen here.
    @staticmethod
    def _widen(*arrays) -> tuple:
        return tuple(np.asarray(array, dtype=np.float64) if isinstance(array, np.ndarray) else array for array in arrays)

    # The full loaded series behind each array, or None to fall back to a normal call.
    def _series_window(self, arrays: tuple) -> tuple | None:
        series = self._indicator_series
        if not series:
            return None

        window: list[np.ndarray] = []
        for array in arrays:
            if not isinstance(array, np.ndarray):
                return None
            full = _series_behind(array, series)
            if full is None:
                return None
            window.append(full)
        return tuple(window)

    # Drop per-run indicator state so a reused strategy never carries values across runs.
    def _reset_indicator_state(self) -> None:
        self._stream_states = {}
        self._indicator_cache = OrderedDict()
        self._indicator_cache_bytes = 0
        self._indicator_ticker = None

    # Run one indicator and return today's value. The streaming path carries one
    # scalar state per ticker and configuration forward to today's bar; anything
    # unsupported, untracked or disabled falls through to the batch cache below.
    def _ta(self, func, params: tuple, arrays: tuple):
        series = self._series_window(arrays)
        if series is None:
            return func(*self._widen(*arrays), *params)

        name = getattr(func, "__name__", "")
        ticker = self._indicator_ticker
        if self.STREAMING_INDICATORS and ticker is not None and ta.stream_supported(name):
            today = arrays[0].size - 1
            if today < 0:
                return np.asarray([float("nan")])
            key = (ticker, name, params, id(series[0]))
            state = self._stream_states.get(key)
            if state is None or today < state.pos:
                state = ta.new_stream(name, params)
                self._stream_states[key] = state
            current = ta.stream_to(state, name, series, today)
            if isinstance(current, tuple):
                return tuple(np.asarray([value], dtype=np.float64) for value in current)
            return np.asarray([current], dtype=np.float64)

        key: tuple = (func, params, tuple(id(full) for full in series))
        entry = self._indicator_cache.get(key)
        if entry is None:
            computed = func(*self._widen(*series), *params)
            raw: tuple = computed if isinstance(computed, tuple) else (computed,)
            # float32 halves the entry (~7 significant digits: a 6-digit price
            # like berk.A's 1234.56 round-trips to <0.0001) while raw bars and
            # fill accounting stay float32/float64 respectively, so signals and
            # fills never lose a cent.
            outputs: tuple = tuple(
                np.asarray(output, dtype=np.float32) if isinstance(output, np.ndarray) else output
                for output in raw
            )
            entry = (outputs, series)
            self._indicator_cache[key] = entry
            self._indicator_cache_bytes += sum(
                output.nbytes if isinstance(output, np.ndarray) else 0 for output in outputs
            )
            while self._indicator_cache_bytes > INDICATOR_CACHE_MAX_BYTES:
                _old_key, (old_outputs, _old_series) = self._indicator_cache.popitem(last=False)
                if _old_key == key:
                    # Single entry larger than the whole budget: keep it and stop.
                    self._indicator_cache[key] = entry
                    break
                self._indicator_cache_bytes -= sum(
                    output.nbytes if isinstance(output, np.ndarray) else 0 for output in old_outputs
                )

        outputs = entry[0]
        length: int = arrays[0].size
        if len(outputs) == 1:
            return outputs[0][:length]
        return tuple(output[:length] for output in outputs)


# The full series this array is the start of, or None when it is a slice or a new array.
def _series_behind(array: np.ndarray, series: dict[int, np.ndarray]) -> np.ndarray | None:
    full = series.get(array.ctypes.data)
    if full is None or array.size > full.size:
        return None
    if array.dtype != full.dtype or array.strides != full.strides:
        return None
    return full if array.size else None
