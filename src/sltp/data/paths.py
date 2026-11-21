# -------- paths.py --------
# Where the data lives. The data is not shipped with the package, so lookup order:
#   1. SLTP_DATA_DIR or `sltp data use`, naming the data root or a directory inside it.
#   2. The managed R2 cache in ~/.sltp/cache, refreshed according to auto-update.
#
#   MARKET_DATA/
#     symbol_catalog.parquet        every symbol of every market, one table
#     DAILY/US/1998.bin             one bundle per calendar year, plus recent.bin
#     DAILY/INTERNATIONAL/1998.bin  all international markets share one bundle per year
#     1-MINUTE/  1-HOUR/            reserved for the intraday tiers
#
# The resolver returns a path even when the files are absent; the read layer raises.

# -------- Imports --------
import re

from pathlib import Path

from sltp.data import config as _config
from sltp.data.remote import CACHE_MARKET_DIR


# -------- Layout --------
# The bar market the engine reads. International bundles publish under their own
# directory and switch through this name once they exist.
BAR_MARKET: str = "US"
# A bundle is one calendar year, or the recent tail: 1998.bin, 2026.bin, recent.bin.
BUNDLE_PATTERN: re.Pattern = re.compile(r"^(\d{4}|recent)\.bin$")
CATALOG_NAME: str = "symbol_catalog.parquet"
ACTIONS_NAME: str = "actions.parquet"


# True when the name is a bar bundle, so discovery never picks up stray files.
def is_bundle(name: str) -> bool:
    return BUNDLE_PATTERN.match(name) is not None


# -------- Resolution --------
# Explicit choices only: the data root, its MARKET_DATA/DAILY/<market> directory,
# or that directory itself. Anything else resolves to the managed cache, so the
# default is always fresh R2 bars and a repo checkout never shadows them by accident.
def _candidates() -> list[Path]:
    roots: list[Path] = []

    configured = _config.data_override()
    if configured is not None:
        roots.append(configured / "MARKET_DATA" / "DAILY" / BAR_MARKET)
        roots.append(configured / "DAILY" / BAR_MARKET)
        roots.append(configured / BAR_MARKET)
        roots.append(configured)

    return roots


# True when a path holds bars: at least one non-trivial bundle.
def _has_data(root: Path) -> bool:
    if not root.is_dir():
        return False
    for entry in root.iterdir():
        if not (entry.is_file() and is_bundle(entry.name)):
            continue
        try:
            if entry.stat().st_size > 24:
                return True
        except OSError:
            continue
    return False


def has_bars(directory) -> bool:
    return _has_data(Path(str(directory)).expanduser())


# The directory holding the bar files for BAR_MARKET.
def market_dir() -> Path:
    roots: list[Path] = _candidates()
    for root in roots:
        if _has_data(root):
            return root
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
