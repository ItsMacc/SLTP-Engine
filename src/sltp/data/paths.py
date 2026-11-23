# -------- paths.py --------
# Where the data lives. The data is not shipped with the package, so lookup order:
#   1. SLTP_DATA_DIR or `sltp data use`, naming the data root or a directory inside it.
#   2. The managed R2 cache in ~/.sltp/cache, refreshed according to auto-update.
#
#   MARKET_DATA/
#     symbol_catalog.parquet        every symbol of every market, one table
#     DAILY/US/*.bin                one bundle per chunk, any size and any name
#     DAILY/INTERNATIONAL/*.bin     all international markets share one bundle per chunk
#     1-MINUTE/  1-HOUR/            the intraday tiers, same shape
#
# A bundle is read for its header, never its name, so any chunking works.
# The resolver returns a path even when the files are absent; the read layer raises.

# -------- Imports --------
from pathlib import Path

from sltp.data import config as _config
from sltp.data.remote import CACHE_MARKET_DIR


# -------- Layout --------
# The bar market the engine reads by default, and the tier it reads it at.
BAR_MARKET: str = "US"
FREQUENCY: str = "DAILY"
CATALOG_NAME: str = "symbol_catalog.parquet"
ACTIONS_NAME: str = "actions.parquet"


# True when the name is a bar file. Only the suffix is read; the header decides the rest.
def is_bin(name: str) -> bool:
    return str(name).lower().endswith(".bin")


# -------- Resolution --------
# The shapes a chosen path may take, most specific first: a market directory, the tier
# above it, the data root, or a directory of bundles straight from the user. Anything else
# resolves to the managed cache, so a repo checkout never shadows it by accident.
def _locate(configured: Path) -> Path | None:
    candidates = (
        configured / FREQUENCY / BAR_MARKET,
        configured / "MARKET_DATA" / FREQUENCY / BAR_MARKET,
        configured / BAR_MARKET,
        configured / FREQUENCY,
        configured / "MARKET_DATA" / FREQUENCY,
        configured / "MARKET_DATA",
        configured,
    )
    for root in candidates:
        if _has_data(root):
            return root
    return None


# True when a chosen path holds bars in any shape the resolver accepts.
def holds_bars(directory) -> bool:
    return _locate(Path(str(directory)).expanduser()) is not None


# True when a path holds bars: at least one non-trivial bundle.
def _has_data(root: Path) -> bool:
    if not root.is_dir():
        return False
    for entry in root.iterdir():
        if not (entry.is_file() and is_bin(entry.name)):
            continue
        try:
            if entry.stat().st_size > 24:
                return True
        except OSError:
            continue
    return False


# The directory holding the bar files for BAR_MARKET.
def market_dir() -> Path:
    configured = _config.data_override()
    if configured is not None:
        found = _locate(Path(configured).expanduser())
        if found is not None:
            return found
    # No local bars, so the R2 cache becomes the market directory. Reads fill it
    # lazily; a machine with local files never reaches this line.
    CACHE_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_MARKET_DIR


# The directory holding the symbol catalogue, which sits above the bar folders.
def symbol_data_dir() -> Path:
    market: Path = market_dir()
    for parent in (market, market.parent, market.parent.parent):
        if (parent / CATALOG_NAME).is_file():
            return parent
    return market.parent.parent


# True when reads resolve to the R2 cache, meaning the network may be used.
def using_remote_cache() -> bool:
    return market_dir() == CACHE_MARKET_DIR


# Where a run's bars come from, for status output and the run stamp.
# "own" is an explicit or repo directory; "cache" is the managed R2 copy.
def source() -> dict:
    chosen: Path = market_dir()
    if chosen == CACHE_MARKET_DIR:
        from sltp.data.remote import cached_version

        return {"kind": "cache", "dir": str(chosen), "version": cached_version() or "unknown"}
    return {"kind": "own", "dir": str(chosen), "version": "local"}


# True when the bar files are present, so callers can fail early with a good message.
def data_available() -> bool:
    try:
        return _has_data(market_dir())
    except Exception:
        return False
