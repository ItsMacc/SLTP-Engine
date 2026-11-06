# -------- symbol_categories.py --------
# Writes a Category column (stock | etf | gold) into symbol_catalog.parquet, so the read
# layer never has to guess. Evidence in order: the gold funds list, the exchange ETF
# flags, then name keywords for listings the US files miss. Whatever is left is a stock.
# Rerun whenever the catalog changes: python -m sltp.data.symbol_categories

# -------- Imports --------
from __future__ import annotations

import io
import json
import os

from datetime import date

from sltp.data.catalog import (
    CATEGORY_COLUMN, COMPANY_MARKERS, ETF, GOLD, GOLD_FUNDS, STOCK,
    STRONG_FUND_WORDS, WEAK_FUND_WORDS,
)
from sltp.data.paths import symbol_data_dir


# -------- Paths & Constants --------
SYMBOLS_FILE: str = os.path.join(str(symbol_data_dir()), "symbol_catalog.parquet")
LEGACY_SYMBOLS_FILE: str = os.path.join(str(symbol_data_dir()), "symbols.parquet")
ETF_CACHE_FILE: str = os.path.join(str(symbol_data_dir()), "etf_symbols.json")

EXCHANGE_FILES = (
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
)


# -------- Evidence Sources --------
# Return every US-listed symbol the exchanges flag as an ETF.
def download_etf_symbols() -> list[str]:
    import pandas as pd
    import requests

    headers = {"User-Agent": "Mozilla/5.0"}
    symbols: set[str] = set()
    for url, symbol_column in EXCHANGE_FILES:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), sep="|")
        # Both files end with a "File Creation Time" footer row
        frame = frame[frame.iloc[:, 0].astype(str).str.strip() != "File Creation Time"]
        flagged = frame.loc[frame["ETF"] == "Y", symbol_column]
        symbols.update(str(symbol).strip() for symbol in flagged.dropna())
    return sorted(symbols)


# Read the cached exchange ETF flags, downloading them when missing or stale.
def load_etf_symbols(refresh: bool = False) -> set[str]:
    if not refresh and os.path.exists(ETF_CACHE_FILE):
        try:
            with open(ETF_CACHE_FILE, encoding="utf-8") as f:
                return set(json.load(f)["symbols"])
        except (OSError, ValueError, KeyError):
            pass

    symbols: list[str] = download_etf_symbols()
    if symbols:
        os.makedirs(os.path.dirname(ETF_CACHE_FILE), exist_ok=True)
        with open(ETF_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"updated": str(date.today()), "symbols": symbols}, f, indent=0)
    return set(symbols)


# -------- Classification --------
# Return the category for one symbol.
def classify_symbol(symbol: str, name: str, etf_symbols: set[str]) -> str:
    if symbol in GOLD_FUNDS:
        return GOLD
    if symbol in etf_symbols:
        return ETF

    lowered: str = (name or "").lower()
    if any(word in lowered for word in STRONG_FUND_WORDS):
        return ETF
    if any(word in lowered for word in WEAK_FUND_WORDS) and not any(marker in lowered for marker in COMPANY_MARKERS):
        return ETF
    return STOCK


# Return the frame with a fresh Category column.
def classify_frame(frame, etf_symbols: set[str]):
    frame = frame.copy()
    frame[CATEGORY_COLUMN] = [
        classify_symbol(str(symbol), str(name or ""), etf_symbols)
        for symbol, name in zip(frame["Symbol"], frame.get("Name", ""))
    ]
    return frame


# -------- Build --------
# Write Category into symbols.parquet and return the per-category counts.
def build_symbol_categories(*, refresh_etf_flags: bool = False) -> dict[str, int]:
    import pandas as pd

    source: str = SYMBOLS_FILE if os.path.exists(SYMBOLS_FILE) else LEGACY_SYMBOLS_FILE
    frame = pd.read_parquet(source)
    etf_symbols: set[str] = load_etf_symbols(refresh=refresh_etf_flags)
    if not etf_symbols:
        raise RuntimeError(
            "No ETF flags available: the exchange files could not be downloaded and no cache exists. "
            "Rerun with network access so categories are not guessed from names alone."
        )

    frame = classify_frame(frame, etf_symbols)

    # Write through a temp file so a failed write cannot truncate the catalog
    temp_path: str = f"{SYMBOLS_FILE}.{os.getpid()}.tmp"
    frame.to_parquet(temp_path, index=False)
    os.replace(temp_path, SYMBOLS_FILE)

    counts: dict[str, int] = frame[CATEGORY_COLUMN].value_counts().to_dict()
    print(f"Categories written to {os.path.relpath(SYMBOLS_FILE)}: {counts}")
    return counts


# -------- Entry Point --------
if __name__ == "__main__":
    build_symbol_categories(refresh_etf_flags=True)
