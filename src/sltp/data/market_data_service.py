# -------- market_data_service.py --------
# Cached read layer over the bar files and market_actions.parquet.
# Universe lists keep delisted symbols too, so a backtest has no survivorship bias.

# -------- Imports --------
import contextlib
import gzip
import json
import os
import struct
import re
import threading
import numpy as np
import pyarrow as pa
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
# One flat bundle per year: MARKET_DAILY/market_data_<year>.bin. Nothing else is read.
FLAT_YEAR_PATTERN: re.Pattern = re.compile(r"^market_data_(\d{4})\.bin$")
FLAT_MAGIC: bytes = b"SLTPFLAT"
# 1: raw column payload. 2: gzip-compressed column payload, header still raw.
FLAT_FORMAT: int = 2
FLAT_PREAMBLE: struct.Struct = struct.Struct("<8sIIQ")
FLAT_RECENT_NAME: str = "market_data_recent.bin"
FLAT_COLS: tuple = (
    ("sym", np.dtype("<u2"), 2),
    ("date", np.dtype("<u2"), 2),
    ("open", np.dtype("<f4"), 4),
    ("high", np.dtype("<f4"), 4),
    ("low", np.dtype("<f4"), 4),
    ("close", np.dtype("<f4"), 4),
    ("vol", np.dtype("<f4"), 4),
)
FLAT_DTYPES: dict[str, np.dtype] = {name: dtype for name, dtype, _ in FLAT_COLS}


# The header table of contents at the head of a container file. The one parser
# for the format: the engine, the converter's manifest entries and `sltp data
# check` all read through here. Anything foreign, truncated or ragged raises.
def read_flat_header(bin_path: str) -> dict:
    with open(bin_path, "rb") as handle:
        preamble: bytes = handle.read(FLAT_PREAMBLE.size)
        if len(preamble) != FLAT_PREAMBLE.size:
            raise ValueError(f"{bin_path}: truncated preamble")
        magic, fmt, header_len, rows = FLAT_PREAMBLE.unpack(preamble)
        if magic != FLAT_MAGIC or fmt not in (1, FLAT_FORMAT):
            raise ValueError(f"{bin_path}: not a flat container this engine wrote")
        raw: bytes = handle.read(header_len)
        if len(raw) != header_len:
            raise ValueError(f"{bin_path}: truncated header")
        index = json.loads(raw.decode("utf-8"))
    counts = [int(c) for c in index.get("counts", [])]
    runs_ok = sum(counts) == rows and (not counts or int(index.get("starts", [0])[0]) == 0)
    if not runs_ok:
        raise ValueError(f"{bin_path}: header runs do not cover all rows")
    index["offsets"] = {name: int(at) for name, at in index.get("offsets", {}).items()}
    index["rows"] = rows
    index["header_len"] = header_len
    index["format"] = fmt
    return index


# The column payload of one container: raw bytes for format 1, gunzipped for
# format 2. Offsets in the index are payload-relative in both cases.
def _flat_payload(bin_path: str, index: dict) -> bytes:
    start: int = FLAT_PREAMBLE.size + int(index["header_len"])
    with open(bin_path, "rb") as handle:
        handle.seek(start)
        blob: bytes = handle.read()
    if index.get("format", 1) == 2:
        blob = gzip.decompress(blob)
    return blob


# One full column as numpy, however the payload is stored. Views, never copies:
# gunzip materializes the payload once, then every column is a view of it.
def _flat_column(bin_path: str, index: dict, name: str) -> np.ndarray:
    rows: int = int(index.get("rows", 0))
    offset: int = int(index["offsets"][name])
    dtype = FLAT_DTYPES[name]
    if index.get("format", 1) == 2:
        return np.frombuffer(_flat_payload(bin_path, index), dtype=dtype, count=rows, offset=offset)
    base: int = FLAT_PREAMBLE.size + int(index["header_len"])
    return np.fromfile(bin_path, dtype=dtype, count=rows, offset=base + offset)


# Every flat bundle on disk, oldest first: (year, bin path, index). A bundle with a
# foreign magic or format is ignored, never read. Parsed once per directory
# signature; later calls reuse the cached list.
_FLAT_BUNDLES_CACHE: list | None = None
_FLAT_BUNDLES_SIG: tuple | None = None


def _flat_bundles() -> list:
    if not os.path.isdir(MARKET_DIR):
        return []
    global _FLAT_BUNDLES_CACHE, _FLAT_BUNDLES_SIG
    try:
        sig = tuple(
            (name, os.path.getsize(os.path.join(MARKET_DIR, name)))
            for name in os.listdir(MARKET_DIR)
            if name.startswith("market_data_") and name.endswith(".bin")
        )
    except OSError:
        return []
    with _LOAD_LOCK:
        if sig == _FLAT_BUNDLES_SIG and _FLAT_BUNDLES_CACHE is not None:
            return _FLAT_BUNDLES_CACHE
    found: list = []
    for name in os.listdir(MARKET_DIR):
        match = FLAT_YEAR_PATTERN.match(name)
        if match is None and name != FLAT_RECENT_NAME:
            continue
        bin_path = os.path.join(MARKET_DIR, name)
        if not os.path.isfile(bin_path):
            continue
        try:
            index = read_flat_header(bin_path)
        except (OSError, ValueError):
            continue
        year = int(match.group(1)) if match is not None else None
        found.append((year, bin_path, index))
    found.sort(key=lambda item: item[0] if item[0] is not None else 9999)
    with _LOAD_LOCK:
        _FLAT_BUNDLES_CACHE = found
        _FLAT_BUNDLES_SIG = sig
    return found


# Only the flat bundles a date range can touch. Reads the container headers
# only: no column bytes move until the read itself.
def flat_files_for(start: str | None, end: str | None) -> list[str]:
    _ensure_remote_bars(start, end)
    first_year: int | None = _year_of(start) if start is not None else None
    last_year: int | None = _year_of(end) if end is not None else None
    bundles = _flat_bundles()
    years = [(year, path) for year, path, _ in bundles if year is not None]
    recent = [path for year, path, _ in bundles if year is None]
    if not years:
        return recent
    selected = [
        path
        for year, path in years
        if (first_year is None or year >= first_year)
        and (last_year is None or year <= last_year)
    ]
    if not selected:
        return [path for _, path in years] + recent
    newest: int = max(year for year, _ in years)
    if last_year is None or last_year >= newest:
        selected.extend(recent)
    return selected


# The flat bundles' signature for cache keys: (path, rows) per year.
def _flat_signature() -> tuple:
    return tuple((path, index["rows"]) for _, path, index in _flat_bundles())


# Pull the R2 bundles a range touches; a no-op whenever local bars exist, so the
# desktop path never pays for a check it does not need.
def _ensure_remote_bars(start: str | None, end: str | None) -> None:
    if not using_remote_cache():
        return
    if _remote.offline():
        if not _flat_bundles():
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


# The year a YYYY-MM-DD bound falls in; callers validate the date before this.
def _year_of(value: str) -> int:
    return int(str(value)[:4])


# -------- Market Data --------
# The latest trading date stored in the bar files.
def get_latest_trading_date() -> str:
    return _cached_value("latest_date", _bars_signature(), _read_latest_date)


# Every trading day as sorted day codes; a range reads only the bundles it touches.
def get_trading_calendar(start: str | None = None, end: str | None = None) -> np.ndarray:
    if start is None and end is None:
        return _cached_value(
            "trading_calendar", _bars_signature(), lambda: _data_calendar()
        )

    flats: list[str] = flat_files_for(start, end)
    key: str = f"trading_calendar:{start}:{end}"

    # Cut to the window while building, not after: the caller slices identically,
    # so the cached value holds only the sessions the run uses.
    def build() -> np.ndarray:
        codes = _data_calendar(flats)
        if start is not None:
            codes = codes[codes >= to_code(start)]
        if end is not None:
            codes = codes[codes <= to_code(end)]
        return codes

    return _cached_value(key, _flat_dir_signature(flats), build)


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


# The signature of every bar file, so a write to any part of the set invalidates the caches.
def _bars_signature() -> tuple:
    return _flat_signature()


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
# Histories for a request, slicing only those symbols' runs.
def _selected_histories(symbols: list[str], start: str | None = None, end: str | None = None) -> dict[str, StockData]:
    global _SYMBOL_CACHE_SIGNATURE
    wanted: list[str] = list(dict.fromkeys(symbols))
    if not flat_files_for(start, end):
        raise RuntimeError(
            f"no bar files in {MARKET_DIR} (want market_data_YYYY.bin);"
            " run `sltp data update` or point at your own files with `sltp data use`"
        )
    with _LOAD_LOCK:
        signature: tuple = _flat_signature()
        if signature != _SYMBOL_CACHE_SIGNATURE:
            _SYMBOL_CACHE.clear()
            _SYMBOL_CACHE_SIGNATURE = signature

        missing: list[str] = [symbol for symbol in wanted if symbol not in _SYMBOL_CACHE]
        if missing:
            for symbol, history in _read_requested_histories(missing).items():
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
    key: tuple = (
        _flat_signature(),
        start,
        end,
        len(symbols),
        hash(frozenset(symbols)),
    )
    with _LOAD_LOCK:
        cached: dict[str, StockData] | None = _BULK_CACHE.get(key)
        if cached is not None:
            _BULK_CACHE.move_to_end(key)
            return cached
        histories: dict[str, StockData] = _read_requested_histories(symbols, start, end)
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


# Every symbol with bars, from the bundles' index symbol tables.
def _read_tickers_with_bars() -> list[str]:
    names: set[str] = set()
    for _, _, index in _flat_bundles():
        names.update(str(symbol) for symbol in index["symbols"])
    return sorted(names)


# The newest session across every bundle, from their index session ranges.
def _read_latest_date() -> str:
    newest: int | None = None
    for _, _, index in _flat_bundles():
        code = int(to_code(str(index["last_session"])))
        if newest is None or code > newest:
            newest = code
    if newest is None:
        newest = int(np.max(_data_calendar()))
    CALENDAR.register(labels_from_codes(np.array([newest], dtype=np.int32)))
    return CALENDAR.label(newest)


# The signature of one flat-bundle list, for cache keys.
def _flat_dir_signature(paths: list[str]) -> tuple:
    return tuple((path, os.path.getsize(path)) for path in paths if os.path.isfile(path))


# Union of trading sessions across the flat bundles in range.
def _data_calendar(flat_dirs: list[str] | None = None) -> np.ndarray:
    parts: list[np.ndarray] = []
    if flat_dirs is None:
        flat_dirs = flat_files_for(None, None)
    if flat_dirs:
        by_path: dict[str, dict] = {path: index for _, path, index in _flat_bundles()}
        blocks: list[np.ndarray] = []
        for path in flat_dirs:
            index = by_path.get(path)
            if index is None:
                continue
            try:
                blocks.append(np.unique(_flat_column(path, index, "date")))
            except (OSError, ValueError, KeyError):
                continue
        if blocks:
            parts.append(np.unique(np.concatenate(blocks)))
    codes = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int32)
    CALENDAR.register(labels_from_codes(codes))
    return codes


# One flat year cut to the requested symbols and date range. Column bytes load
# straight into numpy with no parsing, and each run is a view of them.
# Unreadable or ragged bundles are skipped.
def _flat_runs(
    path: str,
    index: dict,
    wanted: list[str],
    cut: bool,
    low_code: int,
    high_code: int,
    pieces: dict[str, list[tuple[np.ndarray, ...]]],
) -> None:
    rows: int = int(index.get("rows", 0))
    if rows <= 0:
        return
    cols: dict[str, np.ndarray] = {}
    try:
        for name, _, _ in FLAT_COLS:
            cols[name] = _flat_column(path, index, name)
    except (OSError, ValueError, KeyError):
        return
    if any(len(values) != rows for values in cols.values()):
        return
    CALENDAR.register(labels_from_codes(np.unique(cols["date"])))
    symbols: list = index["symbols"]
    starts: list = index["starts"]
    counts: list = index["counts"]
    position: dict[str, int] = {str(symbol): i for i, symbol in enumerate(symbols)}
    date: np.ndarray = cols["date"]
    bars: tuple[np.ndarray, ...] = (
        cols["date"], cols["open"], cols["high"], cols["low"], cols["close"], cols["vol"]
    )
    for ticker in wanted:
        at = position.get(ticker)
        if at is None:
            continue
        head: int = int(starts[at])
        tail: int = head + int(counts[at])
        if cut:
            lo, hi = _window_edges(date[head:tail], low_code, high_code)
            if lo >= hi:
                continue
            head, tail = head + lo, head + hi
        pieces.setdefault(ticker, []).append(tuple(column[head:tail] for column in bars))


# Flat bundles for the requested symbols, merged across years with _finalize so
# a symbol split over years is stitched oldest-first through one finalize.
def _read_flat_histories(
    symbols: list[str],
    paths: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, StockData]:
    wanted: list[str] = sorted(set(symbols))
    if not wanted or not paths:
        return {}
    cut: bool = start is not None or end is not None
    low_code: int = int(to_code(start)) if start is not None else 0
    high_code: int = int(to_code(end)) if end is not None else int(np.iinfo(np.int32).max)
    by_path: dict[str, dict] = {}
    for path in paths:
        try:
            by_path[path] = read_flat_header(path)
        except (OSError, ValueError):
            continue
    pieces: dict[str, list[tuple[np.ndarray, ...]]] = {}
    for path in sorted(by_path):
        _flat_runs(path, by_path[path], wanted, cut, low_code, high_code, pieces)
    return _finalize_histories(pieces)


# Only the requested symbols, sliced straight out of the flat bundles.
# With a start/end the bars are cut while the block is split, not in a second pass.
def _read_requested_histories(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, StockData]:
    wanted: list[str] = sorted(set(symbols))
    if not wanted:
        return {}
    return _read_flat_histories(wanted, flat_files_for(start, end), start, end)


# Every event of one kind grouped by symbol, so a dividend lookup is a dict hit, not a file scan.
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
