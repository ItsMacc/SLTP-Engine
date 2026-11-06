# -------- paths.py --------
# Where the bar files live. The data is not shipped with the package, so lookup order:
#   1. SLTP_DATA_DIR, naming the market directory or its parent.
#   2. Walking up from this file for <root>/data/market/market_data.parquet.
#   3. ~/.sltp/data/market.
# The resolver returns a path even when the files are absent; the read layer raises.

# -------- Imports --------
import os

from pathlib import Path

from sltp.data.remote import CACHE_MARKET_DIR


# -------- Constants --------
# A directory only counts as the market directory when it holds this file.
MASTER_NAME: str = "market_data.parquet"
ENV_VAR: str = "SLTP_DATA_DIR"
MAX_PARENT_DEPTH: int = 6


# -------- Resolution --------
# Every directory that could be the market directory, best first.
def _candidates() -> list[Path]:
    roots: list[Path] = []

    configured: str | None = os.environ.get(ENV_VAR)
    if configured:
        chosen = Path(configured).expanduser()
        roots.append(chosen / "market")
        roots.append(chosen)

    for parent in list(Path(__file__).resolve().parents)[:MAX_PARENT_DEPTH]:
        roots.append(parent / "data" / "market")

    roots.append(Path.home() / ".sltp" / "data" / "market")
    return roots


# True when a directory holds bars: one master file or any year chunk.
def _has_data(root: Path) -> bool:
    if (root / MASTER_NAME).is_file():
        return True
    return any(path.name.startswith("market_data_") and path.suffix == ".parquet" for path in root.glob("*.parquet")) if root.is_dir() else False


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


# The directory holding the symbol catalog.
def symbol_data_dir() -> Path:
    market: Path = market_dir()
    inside: Path = market / "symbol_data"
    if inside.is_dir():
        return inside
    return market.parent / "symbol_data"


# True when the bar files are present, so callers can fail early with a good message.
def data_available() -> bool:
    return (market_dir() / MASTER_NAME).is_file()
