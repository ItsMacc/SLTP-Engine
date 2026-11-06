#!/usr/bin/env python3
# -------- chunk_market_data.py --------
# Throwaway tool: split the single bar file into fixed-width date chunks.
#
# The one 427 MB file is awkward to host, stream and cache. Chunking it by date
# keeps the exact schema, so the read layer does not change shape: it discovers
# market_data_<first year>_<last year>.parquet files and treats them as one set.
#
# Rows are filtered out of the source one batch at a time, so this never holds the
# whole file in memory. Order inside a chunk is the source order (Symbol, then
# Date), which is what lets the reader's Symbol filter skip row groups.
#
#   python scripts/chunk_market_data.py
#   python scripts/chunk_market_data.py --years 10 --source data/market/market_data.parquet
#
# The master file is left untouched. Once the chunks are verified, deleting it
# frees 427 MB and changes nothing: the reader prefers chunks when they exist.

# -------- Imports --------
from __future__ import annotations

import argparse
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from sltp.data.paths import market_dir
from sltp.utils import labels_from_codes, to_code


# -------- Constants --------
CHUNK_NAME = "market_data_{first}_{last}.parquet"
CHUNK_PATTERN = re.compile(r"^market_data_(\d{4})_(\d{4})\.parquet$")
FIRST_YEAR = 1998
CHUNK_YEARS = 6
BATCH_ROWS = 1_000_000
FLUSH_ROWS = 500_000
COMPRESSION = "snappy"


# -------- Helpers --------
# The oldest and newest day code, from row-group statistics rather than a full read.
def date_bounds(parquet: pq.ParquetFile) -> tuple[int, int]:
    metadata = parquet.metadata
    index = next(
        group_index
        for group_index in range(metadata.num_columns)
        if metadata.row_group(0).column(group_index).path_in_schema == "Date"
    )
    oldest: int | None = None
    newest: int | None = None
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(index).statistics
        if statistics is None or statistics.min is None or statistics.max is None:
            raise RuntimeError("The Date column has no row-group statistics; cannot chunk by date.")
        low, high = int(statistics.min), int(statistics.max)
        oldest = low if oldest is None else min(oldest, low)
        newest = high if newest is None else max(newest, high)
    if oldest is None or newest is None:
        raise RuntimeError("The source file has no rows.")
    return oldest, newest


# The year of one day code.
def year_of(code: int) -> int:
    return int(str(labels_from_codes([code])[0])[:4])


# Fixed-width windows covering [first, last], inclusive.
def windows(first: int, last: int, span: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    year = first
    while year <= last:
        end = min(year + span - 1, last)
        out.append((year, end))
        year = end + 1
    return out


# -------- Chunking --------
def chunk(source: str, out_dir: str, span: int) -> int:
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    oldest, newest = date_bounds(parquet)
    first_year, last_year = year_of(oldest), year_of(newest)
    spans = windows(first_year, last_year, span)

    # A rebuild must not leave a previous naming behind: two overlapping chunks
    # would make every symbol's history load twice.
    stale = [name for name in os.listdir(out_dir) if CHUNK_PATTERN.match(name)]
    for name in stale:
        os.remove(os.path.join(out_dir, name))
    if stale:
        print(f"removed {len(stale)} previous chunk(s)")

    bounds: dict[str, tuple[int, int]] = {
        CHUNK_NAME.format(first=start, last=end): (int(to_code(f"{start}-01-01")), int(to_code(f"{end}-12-31")))
        for start, end in spans
    }
    paths: dict[str, str] = {label: os.path.join(out_dir, label) for label in bounds}
    writers: dict[str, pq.ParquetWriter] = {
        label: pq.ParquetWriter(path, schema, compression=COMPRESSION) for label, path in paths.items()
    }
    buffers: dict[str, list[pa.Table]] = {label: [] for label in bounds}
    counts: dict[str, int] = {label: 0 for label in bounds}

    print(f"source: {os.path.basename(source)} | {parquet.metadata.num_rows:,} rows | "
          f"{first_year}-{last_year} | {parquet.metadata.num_row_groups} row groups")

    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=schema.names):
        table = pa.Table.from_batches([batch])
        codes = table.column("Date").to_numpy(zero_copy_only=False)

        for label, (low, high) in bounds.items():
            mask = (codes >= low) & (codes <= high)
            taken = int(mask.sum())
            if taken == 0:
                continue
            buffers[label].append(table.filter(pa.array(mask)))
            counts[label] += taken
            if sum(part.num_rows for part in buffers[label]) >= FLUSH_ROWS:
                writers[label].write_table(pa.concat_tables(buffers[label]).combine_chunks())
                buffers[label] = []

    for label, writer in writers.items():
        if buffers[label]:
            writer.write_table(pa.concat_tables(buffers[label]).combine_chunks())
        writer.close()

    # -------- Report --------
    print()
    print(f"{'chunk':<34}{'rows':>12}{'MB':>8}  {'first':>10}  {'last':>10}")
    total = 0
    for label, (low, high) in bounds.items():
        path = paths[label]
        rows = counts[label]
        total += rows
        if rows == 0:
            print(f"{label:<34}{0:>12}{0:>8}  {'-':>10}  {'-':>10}")
            continue
        reader = pq.ParquetFile(path)
        meta = reader.metadata
        index = next(i for i in range(meta.num_columns) if meta.row_group(0).column(i).path_in_schema == "Date")
        first = int(meta.row_group(0).column(index).statistics.min)
        last = int(meta.row_group(meta.num_row_groups - 1).column(index).statistics.max)
        size = os.path.getsize(path) / 1e6
        print(f"{label:<34}{rows:>12,}{size:>8.1f}  {str(labels_from_codes([first])[0]):>10}  "
              f"{str(labels_from_codes([last])[0]):>10}")

    print()
    print(f"rows written: {total:,} of {parquet.metadata.num_rows:,}")
    if total != parquet.metadata.num_rows:
        print("MISMATCH: the chunks do not cover the source exactly.")
        return 1

    print("ok: every row landed in exactly one chunk")
    print(f"the source file {os.path.basename(source)} is untouched; delete it once you are happy "
          "with the chunks, the reader prefers chunks when they exist.")
    return 0


# -------- Entry Point --------
def main() -> int:
    parser = argparse.ArgumentParser(description="Split the bar file into date chunks.")
    parser.add_argument("--source", default=os.path.join(str(market_dir()), "market_data.parquet"),
                        help="the single file to split (default: the master in the data directory)")
    parser.add_argument("--out", default=str(market_dir()), help="where the chunks are written")
    parser.add_argument("--years", type=int, default=CHUNK_YEARS, help="years per chunk (default: 6)")
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR,
                        help="first year of the first chunk (default: 1998)")
    arguments = parser.parse_args()

    if not os.path.exists(arguments.source):
        print(f"source not found: {arguments.source}")
        return 1

    return chunk(arguments.source, arguments.out, arguments.years)


if __name__ == "__main__":
    sys.exit(main())
