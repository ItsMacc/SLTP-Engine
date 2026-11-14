# -------- paths.py --------
# Where the bar files live. The data is not shipped with the package, so lookup order:
#   1. SLTP_DATA_DIR or `sltp data use`, naming the market directory or its parent.
#   2. The managed R2 cache in ~/.sltp/cache, refreshed according to auto-update.
# Bars live in one file per year (market_data_YYYY.parquet) plus the recent tail.
# The resolver returns a path even when the files are absent; the read layer raises.

# -------- Imports --------
from pathlib import Path

from sltp.data import config as _config
from sltp.data.remote import CACHE_MARKET_DIR


# -------- Resolution --------
# Explicit choices only: `sltp data use` (or SLTP_DATA_DIR for one-off runs).
# Anything else resolves to the managed cache, so the default is always fresh
# R2 bars and a repo checkout never shadows them by accident.
def _candidates() -> list[Path]:
    roots: list[Path] = []

    configured = _config.data_override()
    if configured is not None:
        roots.append(configured / "market")
        roots.append(configured)

    return roots


# True when a path holds bars: at least one year chunk.
def _has_data(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(
        path.name.startswith("market_data_") and path.suffix == ".parquet"
        for path in root.glob("*.parquet")
    )


def has_bars(directory) -> bool:
    return _has_data(Path(str(directory)).expanduser())


# The directory holding the bar files.
def market_dir() -> Path:
    roots: list[Path] = _candidates()
    for root in roots:
        if _has_data(root):
            return root
    # No local bars, so the R2 cache becomes the market directory. Reads fill it
    # lazily; a machine with local files never reaches this line.
    CACHE_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_MARKET_DIR


# True when reads resolve to the R2 cache, meaning the network may be used.
def using_remote_cache() -> bool:
    return market_dir() == CACHE_MARKET_DIR


# Where a run's bars come from, for status output and the run stamp.
def source() -> dict:
    chosen: Path = market_dir()
    if chosen == CACHE_MARKET_DIR:
        from sltp.data.remote import cached_version

        return {"kind": "cache", "dir": str(chosen), "version": cached_version() or "unknown"}
    return {"kind": "own", "dir": str(chosen), "version": "local"}


# Where a run's bars come from, for status output and the run stamp.
# "own" is an explicit or repo directory; "cache" is the managed R2 copy.
def source() -> dict:
    chosen: Path = market_dir()
    if chosen == CACHE_MARKET_DIR:
        from sltp.data.remote import cached_version

        return {"kind": "cache", "dir": str(chosen), "version": cached_version() or "unknown"}
    return {"kind": "own", "dir": str(chosen), "version": "local"}


# The directory holding the symbol catalog.
def symbol_data_dir() -> Path:
    market: Path = market_dir()
    inside: Path = market / "symbol_data"
    if inside.is_dir():
        return inside
    return market.parent / "symbol_data"


# True when the bar files are present, so callers can fail early with a good message.
def data_available() -> bool:
    try:
        return _has_data(market_dir())
    except Exception:
        return False
