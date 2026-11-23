# -------- layout.py --------
# Where the market data lives, and how the bundles inside it are found. The data is not
# shipped with the package, so lookup order:
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
import struct

from pathlib import Path

from sltp.market import config as _config
from sltp.market.remote import CACHE_MARKET_DIR


# -------- Constants --------
# The bar market the engine reads by default, the tier it reads it at, and the two files
# that sit beside the bundles rather than inside them.
BAR_MARKET: str = "US"
FREQUENCY: str = "DAILY"
CATALOG_NAME: str = "symbol_catalog.parquet"
ACTIONS_NAME: str = "actions.parquet"
# FREQUENCY is the tier the engine reads by default; any directory name is a valid one.
INDEX_SUFFIX: str = ".idx"


# -------- Resolution --------
# True when the name is a bar file. Only the suffix is read; the header decides the rest.
def is_bin(name: str) -> bool:
    return str(name).lower().endswith(".bin")


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
        from sltp.market.remote import cached_version

        return {"kind": "cache", "dir": str(chosen), "version": cached_version() or "unknown"}
    return {"kind": "own", "dir": str(chosen), "version": "local"}


# True when the bar files are present, so callers can fail early with a good message.
def data_available() -> bool:
    try:
        return _has_data(market_dir())
    except Exception:
        return False


# -------- Discovery --------
# The directory holding one frequency's markets. A folder the user pointed at directly
# has no MARKET_DATA/DAILY shape, so there the resolved market's parent is the tier.
def frequency_dir(freq: str = FREQUENCY, root: Path | None = None) -> Path:
    if root is not None:
        return Path(root) / freq
    canonical = Path(symbol_data_dir()) / freq
    return canonical if canonical.is_dir() else Path(market_dir()).parent


# Every frequency tier that holds readable bars, sorted.
def frequencies(root: Path | None = None) -> list:
    base = Path(root or symbol_data_dir())
    if not base.is_dir():
        return []
    return sorted(entry.name for entry in base.iterdir()
                  if entry.is_dir() and markets(entry.name, root))


# Every market directory holding .bin files, readable or not, so a damaged one cannot be
# missed by tools that only look at what discovered cleanly.
def market_dirs(freq: str = FREQUENCY, root: Path | None = None) -> list:
    base = frequency_dir(freq, root)
    if not base.is_dir():
        return []
    return sorted(entry.name for entry in base.iterdir() if entry.is_dir() and _bin_files(entry))


# Every market with at least one readable bundle, sorted.
def markets(freq: str = FREQUENCY, root: Path | None = None) -> list:
    base = frequency_dir(freq, root)
    if not base.is_dir():
        return []
    return sorted(entry.name for entry in base.iterdir() if entry.is_dir() and valid_bundles(entry))


# One market's bundles, oldest first, as (first year, path). The year is read from the
# header, not the name, so a bundle called anything at all still lands in the right place.
def bundles(market: str, freq: str = FREQUENCY, root: Path | None = None) -> list:
    return [(int(first[:4]), path) for first, path in valid_bundles(frequency_dir(freq, root) / market)]


# Every .bin in a directory that opens as an intact bundle, as (first session, path),
# oldest first. Anything foreign, truncated or misnamed is left out.
def valid_bundles(directory) -> list:
    return [(first, path) for first, _last, path in spans(directory)]


# The same files with the span each header declares, sorted by first session.
def spans(directory) -> list:
    from sltp.market import index as _index

    found: list = []
    for path in _bin_files(Path(directory)):
        try:
            head = _index.header(path)
        except (OSError, ValueError, struct.error):
            continue
        if _index.problems(path, head):
            continue
        found.append((str(head["first_session"]), str(head["last_session"]), path))
    return sorted(found)


# Files in a directory that look like bundles but do not pass validation, with why.
def rejected(directory) -> list:
    from sltp.market import index as _index

    found: list = []
    for path in _bin_files(Path(directory)):
        problems = _index.problems(path)
        if problems:
            found.append((path, problems))
    return found


# Bundle pairs whose sessions collide, as (earlier, later, first shared, last shared).
# Names are irrelevant, so the only way two files can disagree is to cover a session twice.
def overlaps(directory) -> list:
    found: list = []
    ordered = spans(directory)
    for at, (first, last, path) in enumerate(ordered):
        for other_first, other_last, other in ordered[at + 1:]:
            if other_first <= last and first <= other_last:
                found.append((path, other, max(first, other_first), min(last, other_last)))
    return found


# One market's index path, which sits beside its bundles.
def index_path(market: str, freq: str = FREQUENCY, root: Path | None = None) -> Path:
    return frequency_dir(freq, root) / f"{market}{INDEX_SUFFIX}"


# Every .bin file in one directory, whatever it is called.
def _bin_files(directory: Path) -> list:
    if not directory.is_dir():
        return []
    found: list = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".bin" and entry.stat().st_size > 24:
            found.append(entry)
    return found
