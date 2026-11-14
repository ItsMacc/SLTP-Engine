# -------- market_data_service.py --------
# Cached read layer over the bar files and market_actions.parquet.
# Universe lists keep delisted symbols too, so a backtest has no survivorship bias.

# -------- Imports --------
import bisect
import contextlib
import os
import re
import threading
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from datetime import datetime
from collections import OrderedDict
from typing import Literal

# Engine imports
from sltp.utils.Dates import CALENDAR, labels_from_codes, to_code
from sltp.utils.StockData import StockData

# Package imports
from sltp.data import remote as _remote
from sltp.data.catalog import ETF, GOLD, STOCK, CATEGORY_COLUMN
from sltp.data.paths import market_dir, symbol_data_dir, using_remote_cache


# -------- Paths & Constants --------
MARKET_DIR: str = str(market_dir())
SYMBOL_DATA_DIR: str = str(symbol_data_dir())
RECENT_FILE: str = os.path.join(MARKET_DIR, "market_data_recent.parquet")
ACTIONS_FILE: str = os.path.join(MARKET_DIR, "market_actions.parquet")
CATALOG_FILE: str = os.path.join(SYMBOL_DATA_DIR, "symbol_catalog.parquet")
# Pre-merge filenames, kept only as a local fallback so existing checkouts keep working.
LEGACY_SYMBOLS_FILE: str = os.path.join(SYMBOL_DATA_DIR, "symbols.parquet")
LEGACY_STATUS_FILE: str = os.path.join(SYMBOL_DATA_DIR, "symbol_status.parquet")

ACTION_SPLIT: int = 0
ACTION_DIVIDEND: int = 1
# A few rows record a split as a cash amount; cap it so it never becomes real income.
MAX_DIVIDEND_PER_SHARE: float = 1_000_000.0
DEFAULT_START: str = "1998-01-01"
DATE_FORMAT: str = "%Y-%m-%d"
PRICE_COLS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
BAR_COLS: list[str] = ["Symbol", "Date", *PRICE_COLS]
SYMBOL_CACHE_LIMIT: int = 500
# Largest request served through the shared cache. Bigger requests are read in one
# pass straight to the caller: batching them would only re-scan the files per batch
# and churn the cache that selective reads rely on, for identical final bytes.

Scope = Literal["all", "stocks", "missing"]

# Statuses that still trade: a current last bar, or one a few sessions behind.
TRADABLE_STATUSES: tuple[str, ...] = ("active", "lagging")

# Fewest bars a symbol needs before it is worth charting.
MIN_TRADABLE_BARS: int = 30

_CACHE: dict[str, tuple[tuple | None, object]] = {}  # key -> (file signature, value)
_SYMBOL_CACHE: OrderedDict[str, StockData] = OrderedDict()  # bounded cache for selective reads
_SYMBOL_CACHE_SIGNATURE: tuple | None = None
_LOAD_LOCK = threading.RLock()  # without it, concurrent cold requests each load the file


# -------- Bar Files --------
# A chunk is one calendar year: market_data_<year>.parquet. Nothing else is read.
CHUNK_PATTERN: re.Pattern = re.compile(r"^market_data_(\d{4})\.parquet$")


# Every chunk on disk, oldest first. Empty when the bars still live in one file.
def chunk_files() -> list[str]:
    if not os.path.isdir(MARKET_DIR):
        return []

    chunks: list[tuple[int, str]] = []
    for name in os.listdir(MARKET_DIR):
        match = CHUNK_PATTERN.match(name)
        if match is None:
            continue
        path: str = os.path.join(MARKET_DIR, name)
        if os.path.getsize(path) > 0:
            chunks.append((int(match.group(1)), path))
    chunks.sort()
    return [path for _, path in chunks]


# Pull the R2 chunks a range touches; a no-op whenever local bars exist, so the
# desktop path never pays for a check it does not need.
def _ensure_remote_bars(start: str | None, end: str | None) -> None:
    if not using_remote_cache():
        return
    if _remote.offline():
        try:
            missing: bool = not _list_bar_files()
        except RuntimeError:
            missing = True
        if missing:
            raise RuntimeError(
                "no bars cached and auto-update is off;"
                " run `sltp data update` once or `sltp data auto on` to fetch them"
            )
        return
    _remote.sync_bars(start, end)


# The catalog, fetched once on machines without local data.
def _ensure_remote_catalog() -> None:
    if not using_remote_cache():
        return
    if _remote.offline():
        return
    _remote.sync_catalog()


# The corporate-actions file, fetched once on machines without local data.
def _ensure_remote_actions() -> None:
    if not using_remote_cache():
        return
    if _remote.offline():
        return
    _remote.sync_actions()


# Every file holding bars, oldest first: one chunk per year plus the recent tail.
def bar_files() -> list[str]:
    _ensure_remote_bars(None, None)
    return _list_bar_files()


# The bar files already on disk, without triggering any download.
def _list_bar_files() -> list[str]:
    files: list[str] = chunk_files()
    if not files:
        raise RuntimeError(
            f"no bar files in {MARKET_DIR} (want market_data_YYYY.parquet);"
            " run `sltp data update` or point at your own files with `sltp data use`"
        )
    if os.path.exists(RECENT_FILE) and os.path.getsize(RECENT_FILE) > 0:
        files.append(RECENT_FILE)
    return files


# The year a YYYY-MM-DD bound falls in; callers validate the date before this.
def _year_of(value: str) -> int:
    return int(str(value)[:4])


# Only the files a date range can touch, so a 2020 run never opens the 1998-2003 file.
# Results are unchanged: callers still slice to the range they asked for.
def bar_files_for(start: str | None, end: str | None) -> list[str]:
    _ensure_remote_bars(start, end)
    files: list[str] = _list_bar_files()
    if start is None and end is None:
        return files

    first_year: int | None = _year_of(start) if start is not None else None
    last_year: int | None = _year_of(end) if end is not None else None

    chunks: list[tuple[int, int, str]] = []
    others: list[str] = []
    for path in files:
        match = CHUNK_PATTERN.match(os.path.basename(path))
        if match is None:
            others.append(path)
            continue
        year = int(match.group(1))
        chunks.append((year, year, path))

    # No year chunks (only the recent tail, say): nothing to prune by date.
    if not chunks:
        return files

    selected: list[str] = [
        path
        for low, high, path in chunks
        if (first_year is None or high >= first_year) and (last_year is None or low <= last_year)
    ]

    # A range outside the stored years reads everything, so the caller gets the usual error.
    if not selected:
        return files

    # The tail is newer than every chunk, so it only belongs in a range that reaches it.
    newest: int = max(high for _, high, _ in chunks)
    if last_year is None or last_year >= newest:
        selected.extend(others)
    return selected


# -------- Market Data --------
# The latest trading date stored in the bar files.
def get_latest_trading_date() -> str:
    return _cached_value("latest_date", _bars_signature(), _read_latest_date)


# Every trading day as sorted int32 codes; a range reads only the chunks it touches.
def get_trading_calendar(start: str | None = None, end: str | None = None) -> np.ndarray:
    if start is None and end is None:
        return _cached_value("trading_calendar", _bars_signature(), _read_trading_calendar)

    files: list[str] = bar_files_for(start, end)
    key: str = f"trading_calendar:{start}:{end}"

    # Cut to the window while building, not after: the caller slices identically,
    # so the cached value holds only the sessions the run uses.
    def build() -> np.ndarray:
        codes = _read_trading_calendar(files)
        if start is not None:
            codes = codes[codes >= to_code(start)]
        if end is not None:
            codes = codes[codes <= to_code(end)]
        return codes

    return _cached_value(key, _files_signature(files), build)


# -------- Run Guard --------
# A backtest in flight: direct reads from strategy hooks would bypass the run
# window (lookahead) and the bulk cache, so the public entries below refuse
# while one is active. Engine internals call the underscored readers, which
# stay open. Thread-local so concurrent runs guard independently.
_RUN_STATE = threading.local()


def _run_depth() -> int:
    return int(getattr(_RUN_STATE, "depth", 0))


def _require_outside_run(caller: str) -> None:
    if _run_depth() > 0:
        raise RuntimeError(
            f"{caller} cannot be called from inside a running backtest. "
            "Use the histories the engine passes to initialize_universe/on_rebalance "
            "or the data object in on_signal: direct reads bypass the run window "
            "(lookahead) and the bulk cache."
        )


@contextlib.contextmanager
def _run_guard():
    _RUN_STATE.depth = _run_depth() + 1
    try:
        yield
    finally:
        _RUN_STATE.depth = _run_depth() - 1


# One ticker's bars between start and end, inclusive; None on either side means the full range.
def get_stock_history(ticker: str, start: str | None = None, end: str | None = None) -> StockData:
    _require_outside_run("get_stock_history")
    start = start or DEFAULT_START
    _validate_range(start, end)
    symbol: str = ticker.strip().upper()
    history: StockData | None = _selected_histories([symbol], start, end).get(symbol)
    if history is None:
        raise ValueError(f"No local data found for {symbol} in master database.")

    sliced: StockData | None = _slice_history(history, start, end)
    if sliced is None:
        raise ValueError(f"No data found for {symbol}{_describe_range(start, end)}.")
    return sliced


# Batch form of get_stock_history; tickers with no bars in the range are left out.
def get_stock_histories(tickers: list[str], start: str | None = None, end: str | None = None) -> dict[str, StockData]:
    _require_outside_run("get_stock_histories")
    if not tickers:
        return {}

    start = start or DEFAULT_START
    _validate_range(start, end)
    return _get_histories(tickers, start, end)


# The engine's own entry: same read, no guard. Only the backtester calls this.
def _get_histories(tickers: list[str], start: str, end: str | None) -> dict[str, StockData]:
    symbols: list[str] = [ticker.strip().upper() for ticker in tickers]

    # One read per request, whatever the size. A request bigger than the shared cache
    # can never benefit from it, so it goes straight to the files in a single pass
    # instead of cycling batches through a cache it would only evict.
    unique: list[str] = list(dict.fromkeys(symbols))
    if len(unique) > SYMBOL_CACHE_LIMIT:
        # The bulk read already cut every history to this range, so the request order is
        # the only thing left to apply.
        histories: dict[str, StockData] = _bulk_histories(unique, start, end)
        return {symbol: histories[symbol] for symbol in symbols if symbol in histories}

    histories = _selected_histories(unique, start, end)

    result: dict[str, StockData] = {}
    for symbol in symbols:
        history: StockData | None = histories.get(symbol)
        sliced: StockData | None = _slice_history(history, start, end) if history is not None else None
        if sliced is not None:
            result[symbol] = sliced
    return result


# Dividends for one ticker as (day codes, amount per share), already split-adjusted by the source.
def get_dividends(ticker: str, start: str | None = None, end: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    entry: tuple[np.ndarray, np.ndarray] | None = _events_by_symbol(ACTION_DIVIDEND).get(ticker.strip().upper())
    if entry is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    codes, amounts = _events_in_range(entry[0], entry[1], start, end)
    keep: np.ndarray = np.isfinite(amounts) & (amounts > 0) & (amounts <= MAX_DIVIDEND_PER_SHARE)
    return codes[keep], amounts[keep]


# Split ratios for one ticker: 7.0 is a 7-for-1 split, 0.1 is a 1-for-10 reverse split.
def get_splits(ticker: str, start: str | None = None, end: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    entry: tuple[np.ndarray, np.ndarray] | None = _events_by_symbol(ACTION_SPLIT).get(ticker.strip().upper())
    if entry is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    return _events_in_range(entry[0], entry[1], start, end)


# Drop every cached value so the next read reloads from disk.
def refresh_market_data_cache() -> None:
    global _SYMBOL_CACHE_SIGNATURE
    with _LOAD_LOCK:
        _CACHE.clear()
        _SYMBOL_CACHE.clear()
        _SYMBOL_CACHE_SIGNATURE = None


# -------- Catalog --------
# Tickers for a scope: everything, the stock category, or the symbols that never returned a bar.
def get_tickers(scope: Scope = "all") -> list[str]:
    if scope == "all":
        return get_all_tickers()
    if scope == "stocks":
        return get_available_stocks()
    if scope == "missing":
        return get_missing_tickers()
    raise ValueError(f"Unknown scope {scope!r}. Expected one of: all, stocks, missing.")


# Every valid ticker in the catalog, sorted.
def get_all_tickers() -> list[str]:
    return sorted(_names_by_symbol())


# Ticker to company name, for callers that need the whole map.
def get_all_ticker_names() -> dict[str, str]:
    return dict(_names_by_symbol())


# Ticker and name pairs, for search boxes.
def get_ticker_directory() -> list[dict[str, str]]:
    return [{"ticker": ticker, "name": name} for ticker, name in _names_by_symbol().items()]


# Ticker and name pairs for symbols that still trade and have enough bars to chart.
def get_tradable_ticker_directory() -> list[dict[str, str]]:
    bars: dict[str, int] = _tradable_bars_by_symbol()
    return [
        {"ticker": ticker, "name": name}
        for ticker, name in _names_by_symbol().items()
        if bars.get(ticker, 0) >= MIN_TRADABLE_BARS
    ]


# One ticker's name, falling back to the symbol when the catalog has no entry.
def get_stock_name(ticker: str) -> str:
    symbol: str = ticker.strip().upper()
    return _names_by_symbol().get(symbol, symbol)


# Each catalog symbol's category: stock, etf or gold.
def get_symbol_categories() -> dict[str, str]:
    return dict(_categories_by_symbol())


# Company tickers that have bars, sorted.
def get_available_stocks() -> list[str]:
    return _tickers_in_categories((STOCK,))


# ETF tickers that have bars, sorted. Gold funds count as ETFs too.
def get_available_etfs() -> list[str]:
    return _tickers_in_categories((ETF, GOLD))


# Gold-fund tickers that have bars, sorted.
def get_available_gold() -> list[str]:
    return _tickers_in_categories((GOLD,))


# Symbols that never returned a single bar, sorted.
def get_missing_tickers() -> list[str]:
    return sorted(symbol for symbol, status in _status_by_symbol().items() if status == "missing")


# Each symbol's status: active, lagging, stopped or missing.
def get_status_by_symbol() -> dict[str, str]:
    return dict(_status_by_symbol())


# Symbols that stopped trading, meaning their last bar is five or more sessions back.
def get_stopped_tickers() -> list[str]:
    return sorted(symbol for symbol, status in _status_by_symbol().items() if status == "stopped")


# -------- Cache Helpers --------
# (path, size, mtime) of one file, which is what tells the caches to reload.
def _signature(path: str, required: bool = True) -> tuple | None:
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"Market data file not found: {path}")
        return None
    stat = os.stat(path)
    return os.path.abspath(path), stat.st_size, stat.st_mtime_ns


# The signature of a file set, so each date range gets its own cache entry.
def _files_signature(files: list[str]) -> tuple:
    return tuple(_signature(path) for path in files)


# The signature of every bar file, so a write to any part of the set invalidates the caches.
def _bars_signature() -> tuple:
    return _files_signature(bar_files())


# The cached value for a key, rebuilt when its signature changed.
def _cached_value(key: str, signature: tuple | None, build):
    entry = _CACHE.get(key)
    if entry is not None and entry[0] == signature:
        return entry[1]
    value = build()
    _CACHE[key] = (signature, value)
    return value


# The merged symbol catalog, falling back to the pre-merge symbols.parquet on
# checkouts that have not re-downloaded the data yet.
def _catalog_path() -> str:
    if os.path.exists(CATALOG_FILE):
        return CATALOG_FILE
    _ensure_remote_catalog()
    if os.path.exists(CATALOG_FILE):
        return CATALOG_FILE
    return LEGACY_SYMBOLS_FILE


# Whatever file holds the status columns: the merged catalog when it has them,
# else the legacy symbol_status.parquet. None when neither exists, so callers
# degrade to "no status info" instead of raising.
def _status_source() -> str | None:
    catalog: str = _catalog_path()
    if os.path.exists(catalog) and "Status" in pq.read_schema(catalog).names:
        return catalog
    if os.path.exists(LEGACY_STATUS_FILE):
        return LEGACY_STATUS_FILE
    return None


# The catalog's symbol to name map, cached per file change.
def _names_by_symbol() -> dict[str, str]:
    return _cached_value("symbols", _signature(_catalog_path()), _read_symbol_names)


# Every symbol that has bars, cached per file change.
def _tickers_with_bars() -> list[str]:
    return _cached_value("tickers_with_bars", _bars_signature(), _read_tickers_with_bars)


# The catalog's symbol to category map, cached per file change.
def _categories_by_symbol() -> dict[str, str]:
    return _cached_value("categories", _signature(_catalog_path()), _read_symbol_categories)


# The symbol to status map, cached per file change.
def _status_by_symbol() -> dict[str, str]:
    return _cached_value("status", _signature(_catalog_path(), required=False), _read_symbol_status)


# Bar counts for the symbols that still trade, cached per file change.
def _tradable_bars_by_symbol() -> dict[str, int]:
    return _cached_value("tradable_bars", _signature(_catalog_path(), required=False), _read_tradable_bars)


# Symbols of the given categories that have bars, memoized per file change.
# Strategies read this on every rebalance; without the cache each call re-reads
# the catalog and re-sorts ~11k symbols.
def _tickers_in_categories(kinds: tuple[str, ...]) -> list[str]:
    kinds = tuple(sorted(kinds))

    def build() -> list[str]:
        categories: dict[str, str] = _categories_by_symbol()
        available: set[str] = set(_tickers_with_bars())
        return sorted(symbol for symbol, category in categories.items() if category in kinds and symbol in available)

    return _cached_value(
        f"tickers_in_categories_{'_'.join(kinds)}",
        (_signature(_catalog_path()), _bars_signature()),
        build,
    )


# -------- History Helpers --------
# Histories for a request, reading only those symbols' row groups.
def _selected_histories(symbols: list[str], start: str | None = None, end: str | None = None) -> dict[str, StockData]:
    global _SYMBOL_CACHE_SIGNATURE
    wanted: list[str] = list(dict.fromkeys(symbols))
    files: list[str] = bar_files_for(start, end)
    with _LOAD_LOCK:
        signature: tuple = _files_signature(files)
        if signature != _SYMBOL_CACHE_SIGNATURE:
            _SYMBOL_CACHE.clear()
            _SYMBOL_CACHE_SIGNATURE = signature

        missing: list[str] = [symbol for symbol in wanted if symbol not in _SYMBOL_CACHE]
        if missing:
            for symbol, history in _read_requested_histories(missing, files).items():
                _SYMBOL_CACHE[symbol] = history
                _SYMBOL_CACHE.move_to_end(symbol)
            while len(_SYMBOL_CACHE) > SYMBOL_CACHE_LIMIT:
                _SYMBOL_CACHE.popitem(last=False)

        return {symbol: _SYMBOL_CACHE[symbol] for symbol in wanted if symbol in _SYMBOL_CACHE}


# The newest bulk read, kept so a compare or a parameter sweep does not re-read the
# same bars for every run. The key carries the bar files' signature, the range and the
# universe, so changed files or a different question miss rather than serve stale bars.
_BULK_CACHE: OrderedDict[tuple, dict[str, StockData]] = OrderedDict()
BULK_CACHE_LIMIT: int = 2


# A big request: one pass over the files, remembered for the next run that asks for it.
def _bulk_histories(symbols: list[str], start: str, end: str | None) -> dict[str, StockData]:
    files: list[str] = bar_files_for(start, end)
    key: tuple = (_files_signature(files), start, end, len(symbols), hash(frozenset(symbols)))
    with _LOAD_LOCK:
        cached: dict[str, StockData] | None = _BULK_CACHE.get(key)
        if cached is not None:
            _BULK_CACHE.move_to_end(key)
            return cached
        histories: dict[str, StockData] = _read_requested_histories(symbols, files, start, end)
        _BULK_CACHE[key] = histories
        while len(_BULK_CACHE) > BULK_CACHE_LIMIT:
            _BULK_CACHE.popitem(last=False)
        return histories


# One history cut to a date range; the slices stay views because the codes are sorted.
def _slice_history(history: StockData, start: str, end: str | None) -> StockData | None:
    codes = history.date_codes
    lo: int = int(np.searchsorted(codes, to_code(start), side="left"))
    hi: int = len(codes) if end is None else int(np.searchsorted(codes, to_code(end), side="right"))
    if lo >= hi:
        return None
    if lo == 0 and hi == len(codes):
        return history
    return StockData.view(
        history.ticker,
        codes[lo:hi],
        history.open_price[lo:hi],
        history.high[lo:hi],
        history.low[lo:hi],
        history.close_price[lo:hi],
        history.volume[lo:hi],
    )


# Reject a range whose start sits after its end.
def _validate_range(start: str, end: str | None) -> None:
    start_dt: datetime = _parse_date(start, "start")
    if end is None:
        return
    if start_dt > _parse_date(end, "end"):
        raise ValueError(f"Invalid date range: start ({start}) must not be after end ({end}).")


# Parse one YYYY-MM-DD string, naming which bound was wrong when it fails.
def _parse_date(value: str, label: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        raise ValueError(f"Invalid {label} date ({value}). Expected YYYY-MM-DD.") from None


# The range as a phrase, for error messages.
def _describe_range(start: str, end: str | None) -> str:
    return f" from {start}" + (f" to {end}" if end is not None else "")


# -------- Readers --------
# The catalog's Symbol and Name columns as a dict.
def _read_symbol_names() -> dict[str, str]:
    table = pq.read_table(_catalog_path(), columns=["Symbol", "Name"])
    return dict(zip(table.column("Symbol").to_pylist(), table.column("Name").to_pylist()))


# The catalog's Symbol and Category columns, insisting the category column exists.
def _read_symbol_categories() -> dict[str, str]:
    catalog: str = _catalog_path()
    if CATEGORY_COLUMN not in pq.read_schema(catalog).names:
        raise RuntimeError(
            f"{os.path.relpath(catalog)} has no {CATEGORY_COLUMN} column. "
            "Run: python -m sltp.data.symbol_categories"
        )
    table = pq.read_table(catalog, columns=["Symbol", CATEGORY_COLUMN])
    return dict(zip(table.column("Symbol").to_pylist(), table.column(CATEGORY_COLUMN).to_pylist()))


# Every symbol with bars; the dictionary encoding means one small read per row group.
def _read_tickers_with_bars() -> list[str]:
    table = pq.read_table(bar_files(), columns=["Symbol"], read_dictionary=["Symbol"])
    column = table.column("Symbol")
    if isinstance(column, pa.ChunkedArray):
        if pa.types.is_dictionary(column.type):
            return sorted({label for chunk in column.chunks for label in chunk.dictionary.to_pylist()})
        return column.unique().to_pylist()
    if pa.types.is_dictionary(column.type):
        return column.dictionary.to_pylist()
    return column.unique().to_pylist()


# The newest Date in one file, taken from row-group statistics when they are there.
def _newest_code_in(path: str) -> int | None:
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_row_groups == 0:
        return None
    date_index = next(
        (
            index
            for index in range(metadata.num_columns)
            if metadata.row_group(0).column(index).path_in_schema == "Date"
        ),
        None,
    )
    if date_index is None:
        return None

    newest = None
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(date_index).statistics
        if statistics is None or statistics.max is None:
            return None
        value = int(statistics.max)
        if newest is None or value > newest:
            newest = value
    return newest


# The newest session across every bar file, falling back to reading the Date column.
def _read_latest_date() -> str:
    newest: int | None = None
    for path in bar_files():
        value: int | None = _newest_code_in(path)
        if value is None:
            newest = None
            break
        if newest is None or value > newest:
            newest = value
    if newest is None:
        newest = int(np.max(_codes_from_table(_read_table(["Date"]))))
    CALENDAR.register(labels_from_codes(np.array([newest], dtype=np.int32)))
    return CALENDAR.label(newest)


# Read the requested columns; filters still skip row groups. Defaults to every bar file.
def _read_table(columns: list[str], filters: list | None = None, files: list[str] | None = None) -> pa.Table:
    return pq.read_table(files if files is not None else bar_files(), columns=columns, filters=filters)


# The Date column as int32 day numbers.
def _codes_from_table(table: pa.Table) -> np.ndarray:
    column = table.column("Date")
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column.to_numpy(zero_copy_only=False)


# Every trading day in the given files, as sorted int32 codes.
def _read_trading_calendar(files: list[str] | None = None) -> np.ndarray:
    codes: np.ndarray = np.unique(_codes_from_table(_read_table(["Date"], files=files)))
    CALENDAR.register(labels_from_codes(codes))
    return codes


# Footer metadata per file, parsed once per file version. Only metadata is kept,
# never an open handle, so a data refresh can still replace the files underneath.
_FOOTERS: dict[str, tuple[tuple, object, list[tuple[str | None, str | None]] | None]] = {}


# The file's metadata plus each row group's (min, max) Symbol; None when any group lacks statistics.
def _footer(path: str):
    signature = _signature(path)
    entry = _FOOTERS.get(path)
    if entry is not None and entry[0] == signature:
        return entry
    metadata = pq.ParquetFile(path).metadata
    bounds: list[tuple[str | None, str | None]] | None = []
    if metadata.num_row_groups:
        column = next(
            i for i in range(metadata.num_columns)
            if metadata.row_group(0).column(i).path_in_schema == "Symbol"
        )
        for group in range(metadata.num_row_groups):
            stats = metadata.row_group(group).column(column).statistics
            if stats is None or not stats.has_min_max:
                bounds = None
                break
            bounds.append((stats.min, stats.max))
    entry = (signature, metadata, bounds)
    _FOOTERS[path] = entry
    return entry


# Row groups of one file that can hold any wanted symbol. `wanted` is sorted.
def _needed_groups(bounds, wanted: list[str]) -> list[int]:
    if bounds is None:
        return []
    chosen: list[int] = []
    for group, (low, high) in enumerate(bounds):
        at = bisect.bisect_left(wanted, low)
        if at < len(wanted) and wanted[at] <= high:
            chosen.append(group)
    return chosen


# Only the requested symbols, reading just the row groups their Symbol ranges can touch.
# With a start/end the bars are cut while the block is split, not in a second pass.
# A big request streams each file whole (per-group overhead thousands of times would
# dominate); a small request reads only matching groups, so cached histories stop
# pinning whole decoded files and top-ups stop decoding them.
def _read_requested_histories(
    symbols: list[str],
    files: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, StockData]:
    wanted: list[str] = sorted(set(symbols))
    if not wanted:
        return {}
    # A small request keeps copies so cached histories do not pin whole row groups.
    copy: bool = len(wanted) <= SYMBOL_CACHE_LIMIT
    file_list: list[str] = files if files is not None else bar_files()
    if not copy:
        filters = [("Symbol", "in", wanted)]
        if start is not None:
            filters.append(("Date", ">=", int(to_code(start))))
        if end is not None:
            filters.append(("Date", "<=", int(to_code(end))))

        def whole_file_batches():
            for path in file_list:
                table = pq.read_table([path], columns=BAR_COLS, filters=filters)
                yield from table.to_batches()
                del table

        return _build_histories(whole_file_batches(), 1, start, end, None, False)

    pieces: dict[str, list[tuple[np.ndarray, ...]]] = {}
    cut: bool = start is not None or end is not None
    low_code: int = int(to_code(start)) if start is not None else 0
    high_code: int = int(to_code(end)) if end is not None else int(np.iinfo(np.int32).max)
    wanted_set: frozenset = frozenset(wanted)
    for path in file_list:
        _, metadata, bounds = _footer(path)
        if bounds is None:  # no statistics: fall back to the dataset scan for this file
            filters = [("Symbol", "in", wanted)]
            if start is not None:
                filters.append(("Date", ">=", int(to_code(start))))
            if end is not None:
                filters.append(("Date", "<=", int(to_code(end))))
            table = pq.read_table([path], columns=BAR_COLS, filters=filters)
            batches = table.to_batches()
            del table
        else:
            groups: list[int] = _needed_groups(bounds, wanted)
            if not groups:
                continue
            reader = pq.ParquetFile(path, metadata=metadata, read_dictionary=["Symbol"])
            try:
                table = reader.read_row_groups(groups, columns=BAR_COLS)
            finally:
                reader.close()
            batches = table.to_batches()
            del table
        for batch in batches:
            _accumulate_pieces(pieces, batch, cut, low_code, high_code, wanted_set, copy)
    return _finalize_histories(pieces)


# Every event of one kind grouped by symbol, so a dividend lookup is a dict hit, not a parquet read.
def _events_by_symbol(kind: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _ensure_remote_actions()
    return _cached_value(f"actions_{kind}", _signature(ACTIONS_FILE, required=False), lambda: _read_events_by_symbol(kind))


# One kind's events per symbol; the file is sorted, so each run of equal symbols is one ticker.
def _read_events_by_symbol(kind: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(ACTIONS_FILE):
        return {}

    table: pa.Table = pq.read_table(
        ACTIONS_FILE, columns=["Symbol", "Date", "Amount"], filters=[("Kind", "=", kind)]
    )
    if table.num_rows == 0:
        return {}

    codes = table.column("Date").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int32)
    amounts = table.column("Amount").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32)
    symbols: np.ndarray = np.array(table.column("Symbol").combine_chunks().to_pylist(), dtype=object)

    edges: np.ndarray = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), edges)).tolist()
    ends = np.concatenate((edges, np.array([len(symbols)], dtype=np.int64))).tolist()
    return {str(symbols[start]): (codes[start:end], amounts[start:end]) for start, end in zip(starts, ends)}


# One event series cut to the requested range; start=None means from DEFAULT_START.
def _events_in_range(codes: np.ndarray, amounts: np.ndarray, start: str | None, end: str | None) -> tuple[np.ndarray, np.ndarray]:
    if codes.size == 0:
        return codes, amounts

    start_code: np.int32 = to_code(start or DEFAULT_START)
    end_code: np.int32 = to_code(end) if end is not None else np.int32(np.iinfo(np.int32).max)
    keep = (codes >= start_code) & (codes <= end_code)
    return codes[keep], amounts[keep]


# The half-open bar range covering [low_code, high_code] in sorted date codes.
def _window_edges(codes: np.ndarray, low_code: int, high_code: int) -> tuple[int, int]:
    lo: int = int(np.searchsorted(codes, low_code, side="left"))
    hi: int = int(np.searchsorted(codes, high_code, side="right"))
    return lo, hi


# A column as numpy.
def _column_values(column) -> np.ndarray:
    try:
        return column.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        return column.to_numpy(zero_copy_only=False)


# One ticker's bars put back together when a file left them in more than one piece.
# Sorted by date, and a repeated date keeps the row from the later piece, as before.
def _merge_runs(ticker: str, runs: list[tuple[np.ndarray, ...]]) -> StockData:
    codes: np.ndarray = np.concatenate([run[0] for run in runs])
    columns: list[np.ndarray] = [np.concatenate([run[index] for run in runs]) for index in range(1, 6)]
    if codes.size < 2 or bool(np.all(codes[1:] > codes[:-1])):
        return StockData(
            ticker=ticker,
            date_codes=codes,
            open_price=columns[0],
            high=columns[1],
            low=columns[2],
            close_price=columns[3],
            volume=columns[4],
        )
    order: np.ndarray = np.argsort(codes, kind="stable")
    codes = codes[order]

    keep = np.ones(codes.size, dtype=bool)
    if codes.size > 1:
        keep[:-1] = codes[1:] != codes[:-1]

    ordered = [values[order][keep] for values in columns]
    return StockData(
        ticker=ticker,
        date_codes=codes[keep],
        open_price=ordered[0],
        high=ordered[1],
        low=ordered[2],
        close_price=ordered[3],
        volume=ordered[4],
    )


# One batch split into per-ticker pieces. `wanted` skips wide groups' extra
# names; `copy` keeps small cached slices from pinning whole decoded groups.
def _accumulate_pieces(
    pieces: dict[str, list[tuple[np.ndarray, ...]]],
    batch,
    cut: bool,
    low_code: int,
    high_code: int,
    wanted: frozenset | None = None,
    copy: bool = False,
) -> None:
    encoded = pc.dictionary_encode(batch.column("Symbol"))
    labels: list[str] = encoded.dictionary.to_pylist()
    indices: np.ndarray = encoded.indices.to_numpy(zero_copy_only=True)
    if indices.size == 0:
        return  # a selected group that holds no wanted row
    columns: tuple[np.ndarray, ...] = tuple(
        _column_values(batch.column(name)) for name in ("Date", "Open", "High", "Low", "Close", "Volume")
    )
    date: np.ndarray = columns[0]
    CALENDAR.register(labels_from_codes(np.unique(date)))

    # Within one row group the file's own order holds, so equal symbols sit together.
    edges: np.ndarray = np.flatnonzero(indices[1:] != indices[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), edges)).tolist()
    ends = np.concatenate((edges, np.array([indices.size], dtype=np.int64))).tolist()
    for head, tail in zip(starts, ends):
        ticker: str = labels[indices[head]]
        if wanted is not None and ticker not in wanted:
            continue
        if cut:
            lo, hi = _window_edges(date[head:tail], low_code, high_code)
            if lo >= hi:
                continue
            head, tail = head + lo, head + hi
        if copy:
            pieces.setdefault(ticker, []).append(tuple(column[head:tail].copy() for column in columns))
        else:
            pieces.setdefault(ticker, []).append(tuple(column[head:tail] for column in columns))


# The pile into one StockData per symbol, freeing each ticker's pieces the moment
# its merged arrays exist so the peak stays one shelf, not two.
def _finalize_histories(pieces: dict[str, list[tuple[np.ndarray, ...]]]) -> dict[str, StockData]:
    histories: dict[str, StockData] = {}
    for ticker in list(pieces.keys()):
        runs = pieces.pop(ticker)
        # The common case: one piece, in date order, stays a view of the read buffer.
        if len(runs) == 1:
            codes: np.ndarray = runs[0][0]
            if codes.size < 2 or bool(np.all(codes[1:] > codes[:-1])):
                histories[ticker] = StockData.view(ticker, *runs[0])
                continue
        histories[ticker] = _merge_runs(ticker, runs)
    return histories


# Bars into one StockData per symbol, as views of the columns they were read from.
# Given start/end each symbol is cut here, so callers get one pass over the bars.
def _build_histories(
    batches,
    rows: int,
    start: str | None = None,
    end: str | None = None,
    wanted: frozenset | None = None,
    copy: bool = False,
) -> dict[str, StockData]:
    if rows == 0:
        return {}

    cut: bool = start is not None or end is not None
    low_code: int = int(to_code(start)) if start is not None else 0
    high_code: int = int(to_code(end)) if end is not None else int(np.iinfo(np.int32).max)

    pieces: dict[str, list[tuple[np.ndarray, ...]]] = {}
    for batch in batches:
        _accumulate_pieces(pieces, batch, cut, low_code, high_code, wanted, copy)
    return _finalize_histories(pieces)


# The status file as symbol -> status.
def _read_symbol_status() -> dict[str, str]:
    path: str | None = _status_source()
    if path is None:
        return {}
    table = pq.read_table(path, columns=["Symbol", "Status"])
    return dict(zip(table.column("Symbol").to_pylist(), table.column("Status").to_pylist()))


# The status file as symbol -> bar count, for the symbols that still trade.
def _read_tradable_bars() -> dict[str, int]:
    path: str | None = _status_source()
    if path is None:
        return {}
    if "Bars" not in pq.read_schema(path).names:
        return {}
    table = pq.read_table(path, columns=["Symbol", "Status", "Bars"])
    return {
        symbol: int(count)
        for symbol, status, count in zip(
            table.column("Symbol").to_pylist(),
            table.column("Status").to_pylist(),
            table.column("Bars").to_pylist(),
        )
        if status in TRADABLE_STATUSES
    }
