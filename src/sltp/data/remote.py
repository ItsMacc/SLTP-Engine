# -------- remote.py --------
# Fetch market chunks from the public R2 bucket into a local cache.
# No credentials, no new dependencies: plain HTTPS with urllib, reads with pyarrow.
# The browser passes its own fetch (JS fetch bridge); the desktop uses the default.
#
#   from sltp.data.remote import ensure_local, sync_bars, sync_catalog

# -------- Imports --------
import json
import os
import urllib.request

from collections.abc import Callable
from pathlib import Path


# -------- Constants --------
# Public bucket URL. Readers never authenticate; upload tokens stay on your machine.
REMOTE_BASE_URL: str = "https://pub-8fe125f9d291408d9a41fa5228a663e2.r2.dev"
# The bucket refuses the stock urllib user agent, so every request names us.
USER_AGENT: str = "sltp"
# Small file naming which chunks exist and what data version they belong to.
MANIFEST_KEY: str = "META/manifest.json"
VERSION_KEY: str = "META/version.txt"
# The two files a backtest always needs, whatever the date range.
CATALOG_KEY: str = "SYMBOL_DATA/symbol_catalog.parquet"
ACTIONS_KEY: str = "MARKET_DAILY/market_actions.parquet"
# Where downloaded chunks live. Separate from ~/.sltp/data so cached R2 files
# never mix with a user's own bar files.
CACHE_ROOT: Path = Path.home() / ".sltp" / "cache"
CACHE_MARKET_DIR: Path = CACHE_ROOT / "MARKET_DAILY"
CACHE_SYMBOL_DIR: Path = CACHE_ROOT / "SYMBOL_DATA"
# One process checks the published version once; later calls trust the marker.
_VERSION_CHECKED: bool = False
# The manifest, parsed once per process.
_MANIFEST: dict | None = None
# The browser's download function, set once at startup. The desktop never sets it,
# so the CLI always uses the urllib default below.
_FETCH: Callable | None = None


# -------- Fetching --------
# Use this download function for every later sync instead of urllib. The browser
# passes its JS fetch bridge here once at startup; the desktop never calls this.
def set_fetch(fetch: Callable) -> None:
    global _FETCH
    _FETCH = fetch


# Forget the override and go back to urllib.
def reset_fetch() -> None:
    global _FETCH
    _FETCH = None


# The explicit hook first, then the browser override, then the urllib default.
def _resolve_fetch(fetch: Callable | None) -> Callable:
    return fetch or _FETCH or _default_fetch


# Download one URL straight to a file: single GET, atomic rename, resumable
# nowhere because a killed download restarts from zero on the next call.
def _default_fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(tmp, dest)


# One small GET returning decoded text, through the same hook as everything else.
def _get_text(key: str, fetch=None) -> str:
    fetch = _resolve_fetch(fetch)
    if fetch is _default_fetch:
        request = urllib.request.Request(f"{REMOTE_BASE_URL}/{key}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response:
            return response.read().decode("utf-8")
    tmp = CACHE_ROOT / ".tmp-text"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fetch(f"{REMOTE_BASE_URL}/{key}", tmp)
    try:
        return tmp.read_text(encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)


# The manifest: chunk names, byte sizes, row counts and session ranges.
def read_manifest(fetch=None) -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads(_get_text(MANIFEST_KEY, fetch))
    return _MANIFEST


# Drop the parsed manifest so the next read fetches it again.
def reset_manifest() -> None:
    global _MANIFEST
    _MANIFEST = None


# The data version currently published, e.g. "2026-11-03".
def read_remote_version(fetch=None) -> str:
    return _get_text(VERSION_KEY, fetch).strip()


# -------- Cache --------
# The version the local cache was downloaded against, or "" when never fetched.
def cached_version(cache_root: Path = CACHE_ROOT) -> str:
    marker = cache_root / "version.txt"
    if not marker.is_file():
        return ""
    return marker.read_text(encoding="utf-8").strip()


# True when the cache matches what R2 publishes, using one tiny version request.
def cache_is_fresh(cache_root: Path = CACHE_ROOT, fetch=None) -> bool:
    remote = read_remote_version(fetch)
    return bool(remote) and cached_version(cache_root) == remote


# Download one chunk with a single GET, skipping it when the local copy already
# matches the manifest size. A custom fetch (the browser's JS bridge) swaps in
# through the same parameter; the desktop default uses urllib.
def ensure_local(key: str, cache_root: Path = CACHE_ROOT, fetch=None, size: int | None = None) -> Path:
    dest = cache_root / key
    if dest.is_file() and (size is None or dest.stat().st_size == size):
        return dest
    _resolve_fetch(fetch)(f"{REMOTE_BASE_URL}/{key}", dest)
    return dest


# Once per process, compare the cache marker against the published version and
# forget anything stale: the manifest is re-read and unknown files are pruned.
def _check_version(fetch=None, cache_root: Path = CACHE_ROOT) -> None:
    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    _VERSION_CHECKED = True
    if cache_is_fresh(cache_root, fetch):
        return
    reset_manifest()
    manifest = read_manifest(fetch)
    wanted = {entry["key"] for entry in manifest["files"]}
    for path in sorted(cache_root.rglob("*.parquet")):
        if str(path.relative_to(cache_root)) not in wanted:
            path.unlink()


# True when an entry's session range touches [start, end]; open bounds match all.
def _overlaps(entry: dict, start: str | None, end: str | None) -> bool:
    first, last = entry.get("first_session"), entry.get("last_session")
    if first is None or last is None:
        return False
    if start is not None and last < start:
        return False
    if end is not None and first > end:
        return False
    return True


# The catalog, needed by every run whatever the range.
def sync_catalog(fetch=None, cache_root: Path = CACHE_ROOT) -> Path:
    _check_version(fetch, cache_root)
    manifest = read_manifest(fetch)
    entry = next(e for e in manifest["files"] if e["key"] == CATALOG_KEY)
    dest = ensure_local(CATALOG_KEY, cache_root, fetch, entry.get("bytes"))
    (cache_root / "version.txt").write_text(manifest["version"], encoding="utf-8")
    return dest


# The corporate-actions file; tiny, so it always comes down whole.
def sync_actions(fetch=None, cache_root: Path = CACHE_ROOT) -> Path:
    _check_version(fetch, cache_root)
    manifest = read_manifest(fetch)
    entry = next(e for e in manifest["files"] if e["key"] == ACTIONS_KEY)
    dest = ensure_local(ACTIONS_KEY, cache_root, fetch, entry.get("bytes"))
    (cache_root / "version.txt").write_text(manifest["version"], encoding="utf-8")
    return dest


# Only the MARKET_DAILY chunks a date range can touch. Open bounds mean the
# whole archive, which is what latest-date and full-universe queries need.
def sync_bars(start: str | None, end: str | None, fetch=None, cache_root: Path = CACHE_ROOT) -> list[Path]:
    _check_version(fetch, cache_root)
    manifest = read_manifest(fetch)
    synced: list[Path] = []
    for entry in manifest["files"]:
        if not entry["key"].startswith("MARKET_DAILY/") or entry["key"] == ACTIONS_KEY:
            continue
        if _overlaps(entry, start, end):
            synced.append(ensure_local(entry["key"], cache_root, fetch, entry.get("bytes")))
    (cache_root / "version.txt").write_text(manifest["version"], encoding="utf-8")
    return synced
