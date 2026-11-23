# -------- market_data.py --------
# Everything the engine reads: one ticker's bars, a batch of them, the sessions they traded,
# the actions that adjust them, and the universe to choose from. A read takes `markets`,
# which is None for the US, a bundle name ("US", "INTERNATIONAL") or a catalogue market
# ("Japan", "India"). Bars are assembled into six contiguous arrays per ticker, which is
# the one copy the engine needs; service.py holds the same data as views and copies nothing.

# -------- Imports --------
import contextlib
import mmap
import threading

import numpy as np
import pyarrow.parquet as pq

from sltp.market.catalog import ETF, GOLD, STOCK
from sltp.market import index, layout, service
from sltp.utils.Dates import labels_from_codes, register, to_code
from sltp.utils.StockData import StockData


# -------- Choices --------
# The archive starts here, so a request without a start means "everything there is".
START: str = "1998-01-01"
# No markets argument means the US, which is all the engine traded before international.
DEFAULT: tuple = ("US",)
FOREIGN: str = "INTERNATIONAL"
# Bundle column -> StockData keyword, in the order the bundle stores them.
FIELDS: tuple = (("open", "opens"), ("high", "highs"), ("low", "lows"),
                 ("close", "closes"), ("vol", "volumes"))
# The action file beside each market's bundles, and its two kinds.
ACTIONS: str = "actions.parquet"
SPLIT: int = 0
DIVIDEND: int = 1
EPOCH: np.datetime64 = np.datetime64("1970-01-01")
# Day codes are u2, so one boolean per code is the cheapest way to union session sets.
_CODES: int = 1 << 16
# A backtest in flight, so a read from inside a strategy hook can be refused.
_RUN: threading.local = threading.local()
# Once-only work: the calendar registration and everything the other helpers cache.
_STATE: dict = {}


# -------- Histories --------
# One ticker's bars, oldest first, cut to the requested range.
def get_stock_history(ticker: str, start: str | None = None, end: str | None = None,
                      markets: list[str] | None = None) -> StockData:
    _require_outside_run("get_stock_history")
    symbol: str = ticker.strip().upper()
    panel = _panel()
    if not _selected(symbol, markets):
        raise ValueError(f"{symbol} is not in {_names(markets)}: "
                         f"the archive holds it under {panel.market_of(symbol) or 'nothing'}")
    history: StockData | None = _build(panel, symbol, start, end)
    if history is None:
        raise ValueError(f"no bars for {symbol} in {_span(start, end)}")
    return history


# The same read for several tickers; names outside the selection or the range are left out.
def get_stock_histories(tickers: list[str], start: str | None = None, end: str | None = None,
                        markets: list[str] | None = None) -> dict[str, StockData]:
    _require_outside_run("get_stock_histories")
    return _histories(tickers, start, end, markets)


# The engine's own entry: the same read with no guard, for callers that already hold it.
def _histories(tickers: list[str], start: str | None = None, end: str | None = None,
               markets: list[str] | None = None) -> dict[str, StockData]:
    if not tickers:
        return {}
    panel = _panel()
    histories: dict = {}
    for symbol in dict.fromkeys(ticker.strip().upper() for ticker in tickers):
        if not _selected(symbol, markets):
            continue
        history: StockData | None = _build(panel, symbol, start, end)
        if history is not None:
            histories[symbol] = history
    return histories


# -------- Universe --------
# Every ticker the selection holds bars for, sorted. Each requested name adds its own
# tickers, so ["US", "Japan"] is the US plus Japan rather than their intersection.
def universe(markets: list[str] | None = None) -> list[str]:
    return sorted(_allowed(markets))


# -------- Windows --------
# The chunk labels a selection covers, oldest first.
def windows(markets: list[str] | None = None) -> list[int]:
    labels: set = set()
    for name in layout.market_dirs():
        index.require_intact(name)
    for market in sorted(_bundles(markets)):
        labels.update(int(year) for year, _path in layout.bundles(market))
    return sorted(labels)


# Every ticker's bars for one chunk, as views into that chunk's single mapping. Only the day
# codes are copied; the mapping itself is released as soon as the returned dict is dropped.
def window(label: int, start: str | None = None, end: str | None = None,
           markets: list[str] | None = None) -> dict[str, StockData]:
    _register_calendar()
    low: int = int(to_code(start or START))
    high: int | None = None if end is None else int(to_code(end))
    allowed: set = _allowed(markets)
    histories: dict = {}
    for market in sorted(_bundles(markets)):
        bundles = layout.bundles(market)
        position = next((at for at, (year, _path) in enumerate(bundles)
                         if int(year) == int(label)), None)
        if position is None:
            continue
        book = index.book(market)
        handle = open(bundles[position][1], "rb")
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        handle.close()
        payload, rows = int(book["payloads"][position]), int(book["rows"][position])
        columns = {name: np.frombuffer(mapping, dtype=kind, count=rows,
                                       offset=payload + int(book["columns"][position, slot]))
                   for slot, (name, kind) in enumerate(index.COLUMNS)}
        counts = book["counts"][:, position]
        prefix = book["prefix"][:, position]
        for rank, symbol in enumerate(index.symbols(market)):
            count = int(counts[rank])
            if count and symbol in allowed:
                history = _stock(symbol, columns, int(prefix[rank]) - count, count, low, high)
                if history is not None:
                    histories[symbol] = history
        del columns
    return histories


# -------- Sessions --------
# Every session the selection traded in the range, as day codes, oldest first.
def calendar(start: str | None = None, end: str | None = None,
             markets: list[str] | None = None) -> np.ndarray:
    low: int = int(to_code(start or START))
    high: int | None = None if end is None else int(to_code(end))
    if high is not None and low > high:
        raise ValueError(f"invalid range: start ({start}) is after end ({end})")
    codes = _sessions(markets)
    keep = codes >= low
    if high is not None:
        keep &= codes <= high
    return codes[keep]


# The newest session the selection traded, as YYYY-MM-DD.
def latest_date(markets: list[str] | None = None) -> str:
    codes = _sessions(markets)
    return _label(int(codes[-1])) if codes.size else ""


# -------- Availability --------
# Tickers whose catalogue row falls in one or more categories and which hold bars.
def available(category, markets: list[str] | None = None) -> list[str]:
    wanted: tuple = (category,) if isinstance(category, str) else tuple(category)
    held: set = _allowed(markets)
    return sorted(row["Symbol"] for row in service.catalogue()
                  if row["Category"] in wanted and row["Symbol"] in held)


# Company tickers with bars, sorted.
def available_stocks(markets: list[str] | None = None) -> list[str]:
    return available(STOCK, markets)


# Fund tickers with bars, sorted; gold proxies count as funds too.
def available_etfs(markets: list[str] | None = None) -> list[str]:
    return available((ETF, GOLD), markets)


# Gold-proxy tickers with bars, sorted.
def available_gold(markets: list[str] | None = None) -> list[str]:
    return available(GOLD, markets)


# -------- Cache --------
# Drop every cached lookup, so a rewritten archive is picked up by the next read.
def refresh() -> None:
    _STATE.clear()
    service.reset()


# -------- Files --------
# The bundle files a date range can touch, oldest first.
def files_for(start: str | None = None, end: str | None = None,
              markets: list[str] | None = None) -> list[str]:
    low: int = int(to_code(start or START))
    high: int | None = None if end is None else int(to_code(end))
    found: list = []
    for market in sorted(_bundles(markets)):
        book = index.book(market)
        for at, (_label, path) in enumerate(layout.bundles(market)):
            if int(book["lasts"][at]) < low:
                continue
            if high is not None and int(book["firsts"][at]) > high:
                continue
            found.append(str(path))
    return found


# -------- Guard --------
# A backtest in flight: the public reads refuse to run inside one, because a read from a
# strategy hook bypasses the run window (lookahead) and the loaded universe.
@contextlib.contextmanager
def run_guard():
    _RUN.depth = _depth() + 1
    try:
        yield
    finally:
        _RUN.depth = _depth() - 1


# -------- Actions --------
# A ticker's cash dividends as (day codes, cash per share in the bar's own price unit).
def dividends(ticker: str, start: str | None = None, end: str | None = None) -> tuple:
    return _actions(ticker, DIVIDEND, start, end)


# A ticker's splits as (day codes, ratio where 7.0 is seven-for-one).
def splits(ticker: str, start: str | None = None, end: str | None = None) -> tuple:
    return _actions(ticker, SPLIT, start, end)


# -------- Helpers --------
# Assemble one ticker's bars: pick the bundles that overlap the range, concatenate each
# column once, then cut it. None when the ticker has no bar inside the range.
def _build(panel, symbol: str, start: str | None, end: str | None) -> StockData | None:
    _register_calendar()
    blocks: list = panel.bars(symbol)
    if not blocks:
        return None
    low: int = int(to_code(start or START))
    high: int | None = None if end is None else int(to_code(end))
    if high is not None and low > high:
        raise ValueError(f"invalid range: start ({start}) is after end ({end})")

    chosen: list = [block for block in blocks if _overlaps(block, low, high)]
    if not chosen:
        return None

    # One concatenation per column, in C: a per-year assignment loop costs ~10 us a call.
    codes: np.ndarray = np.concatenate([block["date"] for block in chosen]).astype(np.int32)
    values: dict = {name: np.concatenate([block[column] for block in chosen])
                    for column, name in FIELDS}

    total: int = int(codes.size)
    first: int = int(np.searchsorted(codes, low, side="left"))
    last: int = total if high is None else int(np.searchsorted(codes, high, side="right"))
    if first >= last:
        return None
    if first or last != total:
        codes = codes[first:last]
        values = {name: column[first:last] for name, column in values.items()}
    return StockData(ticker=symbol, date_codes=codes, opens=values["opens"], highs=values["highs"],
                     lows=values["lows"], closes=values["closes"], volumes=values["volumes"])


# One ticker's slice of a mapped chunk, cut to the range without copying.
def _stock(symbol: str, columns: dict, first: int, count: int, low: int,
           high: int | None) -> StockData | None:
    dates: np.ndarray = columns["date"][first:first + count]
    begin: int = int(np.searchsorted(dates, low, side="left"))
    end: int = count if high is None else int(np.searchsorted(dates, high, side="right"))
    if begin >= end:
        return None
    span = slice(first + begin, first + end)
    return StockData(ticker=symbol, date_codes=columns["date"][span], opens=columns["open"][span],
                     highs=columns["high"][span], lows=columns["low"][span],
                     closes=columns["close"][span], volumes=columns["vol"][span])


# Whether a bundle's span reaches into the requested range.
def _overlaps(block: dict, low: int, high: int | None) -> bool:
    return int(block["date"][-1]) >= low and (high is None or int(block["date"][0]) <= high)


# Every session the selection traded, as sorted day codes. A bundle contributes its own
# calendar; a catalogue market contributes only the sessions its own tickers traded.
def _sessions(markets: list[str] | None) -> np.ndarray:
    _register_calendar()
    panel = _panel()
    known: set = {name.upper() for name in _known()}
    marks = np.zeros(_CODES, dtype=bool)
    for name in _names(markets):
        if name.upper() in known:
            for year, _path in layout.bundles(name.upper()):
                column = panel.block(name.upper(), year).get("date")
                if column is not None:
                    marks[column] = True
        else:
            for ticker in _label_set(_canonical(name)):
                for block in panel.bars(ticker):
                    marks[block["date"]] = True
    return np.flatnonzero(marks).astype(np.int32)


# The archive as one shared panel: building it sorts 19,342 names, so it is done once.
def _panel():
    if "panel" not in _STATE:
        _STATE["panel"] = service.all_tickers()
    return _STATE["panel"]


# The bundle markets a selection reads, cached by the selection as written.
def _bundles(markets: list[str] | None) -> set:
    key: tuple = tuple(_names(markets))
    cache: dict = _STATE.setdefault("bundles_of", {})
    if key not in cache:
        known: set = {name.upper() for name in _known()}
        picked: set = set()
        for name in key:
            if name.upper() in known:
                picked.add(name.upper())
            else:
                _canonical(name)  # validates a catalogue market and raises on a typo
                picked.add(FOREIGN)
        cache[key] = picked
    return cache[key]


# Every ticker a selection covers, as a set, cached by the selection as written.
def _allowed(markets: list[str] | None) -> set:
    key: tuple = tuple(_names(markets))
    cache: dict = _STATE.setdefault("allowed", {})
    if key not in cache:
        known: set = {name.upper() for name in _known()}
        chosen: set = set()
        for name in key:
            if name.upper() in known:
                chosen |= _bundle_tickers().get(name.upper(), set())
            else:
                chosen |= _label_set(_canonical(name))
        cache[key] = chosen
    return cache[key]


# Every ticker grouped by the bundle market it lives in, built once.
def _bundle_tickers() -> dict:
    if "by_bundle" not in _STATE:
        panel = _panel()
        groups: dict = {}
        for ticker in panel.tickers:
            groups.setdefault(panel.market_of(ticker), set()).add(ticker)
        _STATE["by_bundle"] = groups
    return _STATE["by_bundle"]


# Whether one ticker falls inside a selection, without building the whole list.
def _selected(symbol: str, markets: list[str] | None) -> bool:
    market = _panel().market_of(symbol)
    if market is None:
        return False
    known: set = {name.upper() for name in _known()}
    for name in _names(markets):
        if name.upper() == market:
            return True
        if name.upper() not in known and symbol in _label_set(_canonical(name)):
            return True
    return False


# A selection as the caller wrote it, defaulting to the US.
def _names(markets: list[str] | None) -> list[str]:
    return [str(name).strip() for name in (markets or DEFAULT)]


# The bundle markets the archive holds, listed once.
def _known() -> tuple:
    if "bundles" not in _STATE:
        _STATE["bundles"] = tuple(layout.markets())
    return _STATE["bundles"]


# The catalogue's spelling of a market, matched without case, so "japan" and "JAPAN" work.
def _canonical(name: str) -> str:
    cache: dict = _STATE.setdefault("canonical", {})
    if name.upper() not in cache:
        found: str | None = None
        for label in service.catalogue_markets():
            if label.upper() == name.upper():
                found = label
                break
        if found is None:
            raise ValueError(f"unknown market {name!r}: bundles {sorted(_known())}, "
                             f"catalogue {sorted(service.catalogue_markets())}")
        cache[name.upper()] = found
    return cache[name.upper()]


# The tickers of one catalogue market, kept because a label check runs per ticker.
def _label_set(label: str) -> set:
    known: dict = _STATE.setdefault("labels", {})
    if label not in known:
        known[label] = set(service.international(label).tickers)
    return known[label]


# A day count back to YYYY-MM-DD.
def _label(days: int) -> str:
    return str(EPOCH + np.timedelta64(int(days), "D"))


# One ticker's actions of a given kind, as (day codes, amounts) inside the range.
def _actions(ticker: str, kind: int, start: str | None, end: str | None) -> tuple:
    empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
    symbol: str = ticker.strip().upper()
    market = _panel().market_of(symbol)
    table = _actions_table(market) if market is not None else None
    if table is None:
        return empty
    # Rows are grouped by symbol once, so a lookup is one dict hit and a slice, not a scan.
    span: tuple | None = table["rows"].get(symbol)
    if span is None:
        return empty
    window = slice(*span)
    low: int = int(to_code(start or START))
    high: int | None = None if end is None else int(to_code(end))
    codes: np.ndarray = table["date"][window]
    keep = (table["kind"][window] == kind) & (codes >= low)
    if high is not None:
        keep &= codes <= high
    return codes[keep].astype(np.int32, copy=False), table["amount"][window][keep].astype(np.float32, copy=False)


# One market's action file as plain arrays, read once per file version.
def _actions_table(market: str) -> dict | None:
    path = layout.frequency_dir() / market / ACTIONS
    if not path.is_file():
        return None
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns)
    entry = _STATE.get("actions")
    if entry is not None and entry[0] == signature:
        return entry[1]
    table = pq.read_table(path, columns=["Symbol", "Date", "Kind", "Amount"])
    symbols = table.column("Symbol").to_numpy(zero_copy_only=False)
    order = np.argsort(symbols, kind="stable")
    ordered = symbols[order]
    edges = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[edges[1:], len(order)]
    frame = {
        "rows": {str(name): (int(first), int(last))
                 for name, first, last in zip(ordered[edges], edges, ends)},
        "date": table.column("Date").to_numpy()[order],
        "kind": table.column("Kind").to_numpy()[order],
        "amount": table.column("Amount").to_numpy()[order],
    }
    _STATE["actions"] = (signature, frame)
    return frame


# The calendar decodes codes on demand, so every session any bundle holds is registered once.
# The span comes from the index, which records each bundle's first and last session.
def _register_calendar() -> None:
    if _STATE.get("calendar"):
        return
    _panel()
    _STATE["calendar"] = True
    lows: list = []
    highs: list = []
    for market in _known():
        book = index.book(market)
        if book["firsts"].size:
            lows.append(int(book["firsts"].min()))
            highs.append(int(book["lasts"].max()))
    if lows:
        register(labels_from_codes(np.arange(min(lows), max(highs) + 1, dtype=np.int32)))


# How deep inside a running backtest this thread is.
def _depth() -> int:
    return int(getattr(_RUN, "depth", 0))


# Refuse a public read that would bypass the run window.
def _require_outside_run(caller: str) -> None:
    if _depth() > 0:
        raise RuntimeError(
            f"{caller} cannot be called from inside a running backtest. Use the histories the "
            f"engine passes to initialize_universe/on_rebalance or the data object in on_signal."
        )


# The range as it should read in an error message.
def _span(start: str | None, end: str | None) -> str:
    return f"[{start or START}, {end or 'latest'}]"
