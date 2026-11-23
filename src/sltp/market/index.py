# -------- index.py --------
# One sidecar index per market, fixed-width and sorted by symbol, so a ticker's rows
# cost a binary search instead of a JSON header parse. It stores row counts only: inside
# a year the rows of one symbol are contiguous, so their start is the sum of the counts
# before them, and the bundle's own column offsets turn that into bytes. Each bundle's
# file size is kept beside its row count so a re-synced bundle is caught, not trusted.

# -------- Imports --------
import json
import struct
import warnings

from pathlib import Path

import numpy as np

from sltp.market import layout


# -------- Format --------
_MAGIC: bytes = b"SLTPIDX1"
_VERSION: int = 4
_EPOCH: np.datetime64 = np.datetime64("1970-01-01")
_HEADER: struct.Struct = struct.Struct("<8sHHIHHI")
# Every bundle stores one contiguous block per column, in this order.
COLUMNS: tuple = (
    ("sym", "<u2"),
    ("date", "<u2"),
    ("open", "<f4"),
    ("high", "<f4"),
    ("low", "<f4"),
    ("close", "<f4"),
    ("vol", "<f4"),
)
ROW_BYTES: int = sum(np.dtype(kind).itemsize for _, kind in COLUMNS)
# A bundle's preamble is magic, format, header length and row count.
BUNDLE_MAGIC: bytes = b"SLTPFLAT"
_PREAMBLE: struct.Struct = struct.Struct("<8sIIQ")
PREAMBLE_SIZE: int = _PREAMBLE.size
# Row counts are u16: the busiest symbol-year measured so far is 262 sessions.
_COUNT_DTYPE = np.dtype("<u2")
_CACHE: dict = {}


# Build the index for one market; the counts-only format rests on per-symbol contiguity.
def write_index(market: str, root: Path | None = None) -> Path:
    found = layout.bundles(market, root=root)
    if not found:
        raise RuntimeError(f"no bundles for market {market!r}")
    require_intact(market, root)
    # Two bundles over the same sessions would put every bar in that span in twice.
    clashes = layout.overlaps(layout.frequency_dir(root=root) / market)
    if clashes:
        earlier, later, first, last = clashes[0]
        raise ValueError(f"{later.name} and {earlier.name} both cover {first}..{last}: "
                         f"those sessions would be stored twice")
    names: list = sorted({name for _, path in found for name in _bundle_header(path)["symbols"]})
    rank: dict = {name: at for at, name in enumerate(names)}
    counts = np.zeros((len(names), len(found)), dtype=_COUNT_DTYPE)
    payloads = np.zeros(len(found), dtype="<u4")
    columns = np.zeros((len(found), len(COLUMNS)), dtype="<u4")
    rows = np.zeros(len(found), dtype="<u4")
    sizes = np.zeros(len(found), dtype="<u8")
    # Bundles hold a span, not necessarily one year, so each one keeps its own label and
    # its first and last session; nothing reads a position back out of a label.
    years = np.zeros(len(found), dtype="<u4")
    firsts = np.zeros(len(found), dtype="<u4")
    lasts = np.zeros(len(found), dtype="<u4")

    for position, (year, path) in enumerate(found):
        header = _bundle_header(path)
        per_symbol = [int(count) for count in header["counts"]]
        starts = [int(start) for start in header["starts"]]
        # The format below cannot express a broken run, so a bundle that has one is refused here.
        if starts != np.concatenate(([0], np.cumsum(per_symbol)[:-1])).tolist():
            raise ValueError(f"{path.name}: rows are not contiguous per symbol")
        if int(header["rows"]) != sum(per_symbol):
            raise ValueError(f"{path.name}: header runs do not cover every row")
        for name, count in zip(header["symbols"], per_symbol):
            counts[rank[name], position] = count
        payloads[position] = _PREAMBLE.size + int(header["header_len"])
        offsets = header["offsets"]
        for slot, (name, kind) in enumerate(COLUMNS):
            at = int(offsets[name])
            if at + int(header["rows"]) * np.dtype(kind).itemsize > path.stat().st_size:
                raise ValueError(f"{path.name}: {name} column runs past the file")
            columns[position, slot] = at
        rows[position] = int(header["rows"])
        # The size the reader re-checks before it trusts any byte range in this bundle.
        sizes[position] = path.stat().st_size
        years[position] = int(year)
        firsts[position] = _code(header["first_session"])
        lasts[position] = _code(header["last_session"])

    blob = _encode(names, counts, payloads, columns, rows, sizes, years, firsts, lasts)
    path = layout.index_path(market, root=root)
    path.write_bytes(blob)
    return path


# A ticker's bars as (year, first row, row count), oldest first.
def lookup(market: str, ticker: str, root: Path | None = None) -> list:
    index = _open(market, root)
    found = _rank(index, str(ticker).strip().upper().encode("utf-8"))
    if found is None:
        return []
    counts = index["counts"]
    row = counts[found]
    years = index["years"]
    return [
        (int(years[position]), int(index["prefix"][found, position]) - int(row[position]),
         int(row[position]))
        for position in np.flatnonzero(row)
    ]


# Where one ticker's rows sit in one bundle's columnar payload, as (offset, rows) per column.
def column_ranges(market: str, year: int, first_row: int, count: int, root: Path | None = None) -> dict:
    index = _open(market, root)
    found = np.flatnonzero(index["years"] == int(year))
    if not found.size:
        return {}
    position = int(found[0])
    base = int(index["payloads"][position])
    return {
        name: (base + int(index["columns"][position, slot]) + int(first_row) * np.dtype(kind).itemsize,
               count)
        for slot, (name, kind) in enumerate(COLUMNS)
    }


# Every symbol the market's bundles hold, sorted.
def symbols(market: str, root: Path | None = None) -> list:
    index = _open(market, root)
    offsets = index["offsets"]
    names = index["names"]
    return [names[offsets[at]:offsets[at + 1]].decode("utf-8") for at in range(int(index["count"]))]


# The row count of every bundle the index covers, oldest first.
def coverage(market: str, root: Path | None = None) -> list:
    index = _open(market, root)
    return [int(value) for value in index["rows"]]


# Refuse a market whose directory holds a damaged bundle. Discovery skips anything that
# fails validation, so without this a half-finished sync would silently drop whole years.
def require_intact(market: str, root: Path | None = None) -> None:
    directory = layout.frequency_dir(root=root) / market
    broken = layout.rejected(directory)
    ours = [(path, problems) for path, problems in broken if not _foreign(path)]
    if ours:
        detail = "; ".join(f"{path.name}: {problems[0]}" for path, problems in ours)
        raise RuntimeError(
            f"{directory} holds {len(ours)} damaged bundle(s) ({detail}). Restore them with "
            f"`sltp data update` or move them out of the folder; see `sltp data check`."
        )
    for path, problems in broken:
        warnings.warn(f"{path.name} is not a bar bundle and is ignored ({problems[0]})",
                      RuntimeWarning, stacklevel=3)


# True when a file is not one of ours, so a stray .bin is skipped instead of fatal.
def _foreign(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(BUNDLE_MAGIC)) != BUNDLE_MAGIC
    except OSError:
        return True


# A bundle's header, for tools that inspect the archive without building an index.
def header(path) -> dict:
    return _bundle_header(Path(path))


# Everything wrong with one bundle, empty when it is readable end to end. Discovery uses
# this as its definition of a usable file, so there is only one validation to keep right.
def problems(path, head: dict | None = None) -> list:
    try:
        header = head if head is not None else _bundle_header(Path(path))
    except (OSError, ValueError, struct.error) as error:
        return [str(error)]
    found: list = []
    rows = int(header["rows"])
    starts = [int(value) for value in header.get("starts", [])]
    counts = [int(value) for value in header.get("counts", [])]
    if rows <= 0 or not (len(header.get("symbols", [])) == len(starts) == len(counts)):
        found.append("bad symbol table")
    elif sum(counts) != rows or starts[0] != 0:
        found.append("runs do not cover all rows")
    for field in ("first_session", "last_session"):
        if not header.get(field):
            found.append(f"header carries no {field}")
    want = rows * ROW_BYTES
    have = Path(path).stat().st_size - PREAMBLE_SIZE - int(header["header_len"])
    if have != want:
        found.append(f"payload is {have} bytes, want {want}")
    return found


# The parsed index for one market, shared by every caller in this process.
# Drop the parsed books so the next read rebuilds them from disk.
def forget() -> None:
    _CACHE.clear()


def book(market: str, root: Path | None = None) -> dict:
    return _open(market, root)


# Bundles whose size no longer matches the index, as (year, indexed, on disk); -1 on disk
# means the bundle set itself changed, which no byte range recorded here survives.
def verify(market: str, root: Path | None = None) -> list:
    index = _open(market, root)
    found = layout.bundles(market, root=root)
    sizes = [int(value) for value in index["sizes"]]
    if len(found) != len(sizes):
        return [(int(index["year_first"]), sum(sizes), -1)]
    return [(year, sizes[at], path.stat().st_size)
            for at, (year, path) in enumerate(found)
            if path.stat().st_size != sizes[at]]


# The size the index recorded for one bundle's year, or None when it holds no such year.
def bundle_size(market: str, year: int, root: Path | None = None) -> int | None:
    index = _open(market, root)
    found = np.flatnonzero(index["years"] == int(year))
    return int(index["sizes"][int(found[0])]) if found.size else None


# A bundle label or first session as days since 1970-01-01.
def _code(label: str) -> int:
    return int((np.datetime64(str(label)) - _EPOCH).astype("int64"))


# The index laid out end to end: header, offset tables, counts, then the name block.
def _encode(names: list, counts: np.ndarray, payloads: np.ndarray, columns: np.ndarray,
            rows: np.ndarray, sizes: np.ndarray, years: np.ndarray, firsts: np.ndarray,
            lasts: np.ndarray) -> bytes:
    encoded: list = [name.encode("utf-8") for name in names]
    offsets = np.zeros(len(encoded) + 1, dtype="<u4")
    offsets[1:] = np.cumsum([len(item) for item in encoded])
    head: bytes = _HEADER.pack(
        _MAGIC, _VERSION, 0, len(encoded), int(years[0]), len(rows), int(offsets[-1])
    )
    return b"".join([
        head,
        offsets.tobytes(),
        payloads.tobytes(),
        columns.tobytes(),
        rows.tobytes(),
        sizes.tobytes(),
        years.tobytes(),
        firsts.tobytes(),
        lasts.tobytes(),
        counts.tobytes(),
        b"".join(encoded),
    ])


# The parsed index, read once per file version; every array is a view over one buffer.
def _open(market: str, root: Path | None = None) -> dict:
    path = layout.index_path(market, root=root)
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; build the index once with write_index()")
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns)
    entry = _CACHE.get(market)
    if entry is not None and entry[0] == signature:
        return entry[1]

    blob = path.read_bytes()
    magic, version, _flags, count, year_first, year_count, name_bytes = _HEADER.unpack_from(blob, 0)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError(f"{path}: not an index this build wrote")
    if count < 1 or year_count < 1:
        raise ValueError(f"{path}: empty index")

    at: int = _HEADER.size
    offsets = np.frombuffer(blob, dtype="<u4", count=count + 1, offset=at)
    at += (count + 1) * 4
    payloads = np.frombuffer(blob, dtype="<u4", count=year_count, offset=at)
    at += year_count * 4
    columns = np.frombuffer(blob, dtype="<u4", count=year_count * len(COLUMNS), offset=at)
    at += year_count * len(COLUMNS) * 4
    rows = np.frombuffer(blob, dtype="<u4", count=year_count, offset=at)
    at += year_count * 4
    sizes = np.frombuffer(blob, dtype="<u8", count=year_count, offset=at)
    at += year_count * 8
    years = np.frombuffer(blob, dtype="<u4", count=year_count, offset=at)
    at += year_count * 4
    firsts = np.frombuffer(blob, dtype="<u4", count=year_count, offset=at)
    at += year_count * 4
    lasts = np.frombuffer(blob, dtype="<u4", count=year_count, offset=at)
    at += year_count * 4
    counts = np.frombuffer(blob, dtype=_COUNT_DTYPE, count=count * year_count, offset=at)
    at += count * year_count * _COUNT_DTYPE.itemsize
    if len(blob) < at + name_bytes:
        raise ValueError(f"{path}: truncated name block")

    shape = (count, year_count)
    index = {
        "offsets": offsets,
        "payloads": payloads,
        "columns": columns.reshape(year_count, len(COLUMNS)),
        "rows": rows,
        "sizes": sizes,
        "years": years,
        "firsts": firsts,
        "lasts": lasts,
        "counts": counts.reshape(shape),
        "prefix": np.cumsum(counts.reshape(shape), axis=0, dtype="<i8"),
        "names": blob[at:at + name_bytes],
        "count": count,
        "year_first": year_first,
        "year_count": year_count,
    }
    _CACHE[market] = (signature, index)
    return index


# Binary search over the name block; None when the market does not hold that symbol.
def _rank(index: dict, wanted: bytes):
    offsets = index["offsets"]
    names = index["names"]
    low, high = 0, int(index["count"]) - 1
    while low <= high:
        middle = (low + high) // 2
        found = names[offsets[middle]:offsets[middle + 1]]
        if found == wanted:
            return middle
        if found < wanted:
            low = middle + 1
        else:
            high = middle - 1
    return None


# A bundle's header as a dict, with its preamble and column offsets folded in.
def _bundle_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        magic, _format, header_len, rows = _PREAMBLE.unpack(handle.read(_PREAMBLE.size))
        if magic != BUNDLE_MAGIC:
            raise ValueError(f"{path.name}: not a flat bundle")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    header["rows"] = int(rows)
    header["header_len"] = int(header_len)
    header["offsets"] = {name: int(at) for name, at in header.get("offsets", {}).items()}
    return header
