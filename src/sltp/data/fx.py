# -------- fx.py --------
# Daily FX to USD, read from MARKET_DATA/fx.parquet. One row per session, one column per
# currency holding USD per unit, so 1 GBP = 1.31 USD and 1 JPY = 0.0063 USD. A day with
# no quote uses the last rate at or before it. Years before a currency's first real quote
# repeat that quote: real_since() and is_approximate() say exactly where the join is.

# -------- Imports --------
import json

import numpy as np
import pyarrow.parquet as pq

from pathlib import Path

from sltp.data.paths import CATALOG_NAME, symbol_data_dir


# -------- Constants --------
FX_NAME: str = "fx.parquet"
# The unit every rate is expressed in; converting USD to USD is the identity.
BASE: str = "USD"
EPOCH: np.datetime64 = np.datetime64("1970-01-01")
# Schema metadata the builder writes: {"GBP": "1998-01-01", ...} of first real quotes.
REAL_SINCE_KEY: bytes = b"sltp_fx_real_since"
# Parsed table and catalogue, each keyed by its file signature.
_CACHE: dict = {}


# Where the table lives, next to the symbol catalogue.
def fx_path() -> Path:
    return Path(symbol_data_dir()) / FX_NAME


# USD per one unit of the currency on the given day.
def rate(currency: str, day) -> float:
    code: str = str(currency).strip().upper()
    if code == BASE:
        return 1.0
    days, columns, _starts = _rates()
    values = columns.get(code)
    if values is None:
        raise KeyError(f"no FX rate for {code!r}; known currencies: {', '.join(sorted(columns))}")
    return _asof(days, values, _days(day), code)


# An amount in the given currency, expressed in USD on the given day.
def to_usd(amount: float, currency: str, day) -> float:
    return float(amount) * rate(currency, day)


# The currency a symbol's bars are quoted in, from the catalogue.
def currency_of(symbol: str) -> str:
    currencies = _currencies()
    code = currencies.get(str(symbol).strip().upper())
    if code is None:
        raise KeyError(f"{symbol!r} is not in the symbol catalogue")
    return code


# A whole series converted in one call: prices, dividends, anything in that currency.
def to_usd_series(values: np.ndarray, codes: np.ndarray, currency: str) -> np.ndarray:
    code: str = str(currency).strip().upper()
    amounts = np.asarray(values, dtype="float64")
    if code == BASE:
        return amounts.astype("float32")
    days, columns, _starts = _rates()
    lookup = columns.get(code)
    if lookup is None:
        raise KeyError(f"no FX rate for {code!r}")
    index = np.searchsorted(days, np.asarray(codes, dtype="int64"), side="right") - 1
    return (amounts * lookup[np.clip(index, 0, lookup.size - 1)]).astype("float32")


# The first day a currency had a real quote; earlier days carry it backwards.
def real_since(currency: str) -> str:
    code: str = str(currency).strip().upper()
    days, _columns, starts = _rates()
    if code == BASE:
        return _label(int(days[0]))
    if code not in starts:
        raise KeyError(f"no FX rate for {code!r}; known currencies: {', '.join(sorted(starts))}")
    return starts[code]


# True when the rate for that day was carried back from a later quote.
def is_approximate(currency: str, day) -> bool:
    return _days(day) < _days(real_since(currency))


# -------- Reading --------
# The parsed table: session days, one array per currency, and the first real quotes.
def _rates() -> tuple:
    path = fx_path()
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; run scripts/build_fx.py once to build it")
    signature = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    entry = _CACHE.get("rates")
    if entry is not None and entry[0] == signature:
        return entry[1]
    table = pq.read_table(path)
    days = table.column("Date").to_numpy().astype("int64")
    columns = {name: table.column(name).to_numpy().astype("float64")
               for name in table.column_names if name != "Date"}
    starts = _starts(table.schema.metadata)
    value = (days, columns, starts)
    _CACHE["rates"] = (signature, value)
    return value


# The approximation boundary the builder stamped into the schema; empty when absent.
def _starts(metadata: dict | None) -> dict:
    if not metadata or REAL_SINCE_KEY not in metadata:
        return {}
    return json.loads(metadata[REAL_SINCE_KEY].decode("utf-8"))


# Symbol to currency code, read once from the catalogue.
def _currencies() -> dict:
    path = Path(symbol_data_dir()) / CATALOG_NAME
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; the symbol catalogue is required for FX lookups")
    signature = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    entry = _CACHE.get("currencies")
    if entry is not None and entry[0] == signature:
        return entry[1]
    table = pq.read_table(path, columns=["Symbol", "CurrencyCode"])
    value = dict(zip(table.column("Symbol").to_pylist(), table.column("CurrencyCode").to_pylist()))
    _CACHE["currencies"] = (signature, value)
    return value


# A YYYY-MM-DD (or a day count) as days since the epoch.
def _days(day) -> int:
    if isinstance(day, (int, np.integer)):
        return int(day)
    return int((np.datetime64(str(day)[:10]) - EPOCH).astype("int64"))


# A day count back to YYYY-MM-DD.
def _label(days: int) -> str:
    return str(EPOCH + np.timedelta64(int(days), "D"))


# The last rate at or before a day, never before the first row.
def _asof(days: np.ndarray, values: np.ndarray, day: int, code: str) -> float:
    index = int(np.searchsorted(days, day, side="right")) - 1
    value = float(values[max(index, 0)])
    if not np.isfinite(value):
        raise ValueError(f"{code} has no usable rate at day {day}")
    return value
