# -------- service.py --------
# Reads for the rest of the app: one ticker's bars as views into the mapped bundles, and
# the whole archive as one set of column views. Nothing is copied, so asking for every
# ticker costs the index lookups and nothing else; the OS pages in what is actually read.
# The sidecar index is gitignored, so it is built here whenever it is missing or stale.
# Every read goes through here; the old sltp/data reader is gone.

# -------- Imports --------
import mmap

import numpy as np
import pyarrow.parquet as pq

from sltp.data.paths import CATALOG_NAME, symbol_data_dir
from sltp.market import index, layout


# -------- Choices --------
# The catalogue columns the selections need; the rest of the table is long text.
FIELDS: tuple = ("Symbol", "Universe", "Market", "Category")
EQUITY: str = "stock"
FOREIGN: str = "INTERNATIONAL"
_CACHE: dict = {}


# -------- Panel --------
# A ticker set over the whole archive. The views stay valid as long as this object lives.
class Panel:
    def __init__(self, tickers: list, store: dict):
        self.tickers = tickers
        self._store = store

    def __len__(self) -> int:
        return len(self.tickers)

    def __contains__(self, ticker: object) -> bool:
        return ticker in self._store["where"]

    # Every bar the selection can serve, straight from the index and no file reads.
    @property
    def bar_count(self) -> int:
        totals = self._store["totals"]
        return sum(totals.get(ticker, 0) for ticker in self.tickers)

    # The market a ticker's bars live in, or None when the archive does not hold it.
    def market_of(self, ticker: str) -> str | None:
        return self._store["where"].get(ticker)

    # One ticker's bars, oldest year first, as a dict of column views per calendar year.
    # The rank comes from the store, so no call searches the name block.
    def bars(self, ticker: str) -> list:
        market = self._store["where"].get(ticker)
        rank = self._store["ranks"].get(ticker)
        if market is None or rank is None:
            return []
        book = self._store["books"][market]
        counts = book["counts"][rank]
        prefix = book["prefix"][rank]
        years = book["years"]
        out: list = []
        for position in np.flatnonzero(counts):
            year = int(years[position])
            block = self._store["blocks"].get((market, year))
            if block is None:
                continue
            count = int(counts[position])
            first = int(prefix[position]) - count
            year_rows = {"year": year}
            year_rows.update({name: values[first:first + count] for name, values in block.items()})
            out.append(year_rows)
        return out

    # One column of one ticker, as one view per year.
    def column(self, ticker: str, name: str) -> list:
        return [year_rows[name] for year_rows in self.bars(ticker)]

    # One market-year's whole column set, for sweeps that read all of it anyway.
    def block(self, market: str, year: int) -> dict:
        return self._store["blocks"].get((market, year), {})


# -------- Selections --------
# Every ticker the archive holds bars for, optionally only one catalogue market.
def all_tickers(market: str | None = None) -> Panel:
    store = _store()
    if market is None:
        return Panel(sorted(store["where"]), store)
    return _selected(store, market=market)


# Every stock in the archive, optionally only one catalogue market.
def all_equities(market: str | None = None) -> Panel:
    return _selected(_store(), category=EQUITY, market=market)


# Every international stock, optionally only one international market: international("Japan").
def international(market: str | None = None) -> Panel:
    return _selected(_store(), universe=FOREIGN, category=EQUITY, market=market)


# Drop the cached panel so the next read rebuilds it from the files as they are now.
def reset() -> None:
    _CACHE.clear()
    index.forget()


# The catalogue rows as plain dicts, so a caller can read categories and names.
def catalogue() -> list:
    return _catalog()


# Every market label the catalogue uses, so a caller can spell a selection without guessing.
def catalogue_markets() -> set:
    return _markets()


# A panel of the tickers whose catalogue rows match every given field.
def _selected(store: dict, universe: str | None = None, category: str | None = None,
              market: str | None = None) -> Panel:
    if market is not None and market not in _markets():
        raise ValueError(f"unknown market {market!r}; the catalogue has {sorted(_markets())}")
    wanted = _matching(universe, category, market)
    return Panel([name for name in sorted(store["where"]) if name in wanted], store)


# -------- Store --------
# The whole archive as one view set, built once per process
def _store() -> dict:
    if "store" in _CACHE:
        return _CACHE["store"]
    where: dict = {}
    ranks: dict = {}
    books: dict = {}
    blocks: dict = {}
    rows: dict = {}
    totals: dict = {}
    mappings: dict = {}
    # Sweep every market directory first: a market whose bundles are all damaged is not
    # discovered by markets(), and dropping it here is exactly the silent loss to avoid.
    for name in layout.market_dirs():
        index.require_intact(name)
    found = layout.markets()
    if not found:
        raise RuntimeError(
            f"no bar files in {layout.frequency_dir()}. Run `sltp data update`, or point at a "
            f"folder of .bin files with `sltp data use <dir>`."
        )
    for market in found:
        _ensure_index(market)
        book = index.book(market)
        # A bundle covers a span, so its slot is its position in the list, never its label.
        for position, (_label, path) in enumerate(layout.bundles(market)):
            year = int(book["years"][position])
            count = int(book["rows"][position])
            payload = int(book["payloads"][position])
            handle = open(path, "rb")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            handle.close()
            blocks[(market, year)] = {
                name: np.frombuffer(mapping, dtype=kind, count=count,
                                    offset=payload + int(book["columns"][position, slot]))
                for slot, (name, kind) in enumerate(index.COLUMNS)
            }
            mappings[(market, year)] = mapping
            rows[(market, year)] = count
        names = index.symbols(market)
        where.update(dict.fromkeys(names, market))
        ranks.update({name: at for at, name in enumerate(names)})
        totals.update(dict(zip(names, book["counts"].sum(axis=1).tolist())))
        books[market] = book
    store = {"where": where, "ranks": ranks, "books": books, "blocks": blocks,
             "rows": rows, "totals": totals, "mappings": mappings}
    _CACHE["store"] = store
    return store


# Rebuild one market's sidecar when it is absent, unreadable, or predates the bundles,
# which is the normal state after a data sync: .idx never travels with the data.
def _ensure_index(market: str) -> None:
    path = layout.index_path(market)
    index.require_intact(market)
    try:
        if path.is_file() and not index.verify(market):
            return
    except (RuntimeError, ValueError):
        pass
    index.write_index(market)


# The symbols whose catalogue row matches every given field, as a set.
def _matching(universe: str | None = None, category: str | None = None,
              market: str | None = None) -> set:
    wanted: set = set()
    for row in _catalog():
        if universe is not None and row["Universe"] != universe:
            continue
        if category is not None and row["Category"] != category:
            continue
        if market is not None and row["Market"] != market:
            continue
        wanted.add(str(row["Symbol"]))
    return wanted


# The catalogue's market labels, so a misspelled filter fails loudly instead of empty.
def _markets() -> set:
    if "markets" not in _CACHE:
        _CACHE["markets"] = {str(row["Market"]) for row in _catalog()}
    return _CACHE["markets"]


# The catalogue columns above, read once and kept as plain dicts.
def _catalog() -> list:
    if "catalog" in _CACHE:
        return _CACHE["catalog"]
    path = symbol_data_dir() / CATALOG_NAME
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing, so no symbol has a name or a market. Run `sltp data update`, "
            f"or point at a data folder with `sltp data use <dir>`."
        )
    table = pq.read_table(path, columns=list(FIELDS))
    columns = [table.column(name).to_pylist() for name in FIELDS]
    _CACHE["catalog"] = [dict(zip(FIELDS, values)) for values in zip(*columns)]
    return _CACHE["catalog"]
