# -------- layout.py --------
# Where the bar archive lives: MARKET_DATA/<frequency>/<market>/*.bin. A bundle's file
# name carries no meaning. Its header says what it holds and which sessions it spans, so
# whatever chunking a user brings is discovered the same way.

# -------- Imports --------
import struct

from pathlib import Path

from sltp.data.paths import FREQUENCY, market_dir, symbol_data_dir


# -------- Constants --------
# FREQUENCY is the tier the engine reads by default; any directory name is a valid one.
INDEX_SUFFIX: str = ".idx"


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
