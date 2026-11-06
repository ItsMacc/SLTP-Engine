# -------- __init__.py --------
# The market-data layer: a cached Parquet reader and the tool that builds the files.
# Nothing is imported here on purpose, so importing sltp.data costs nothing.
#
#   from sltp.data.market_data_service import get_stock_history, get_all_tickers
#
# The bar files may be one master or a set of chunks; the reader discovers whichever
# is present. scripts/chunk_market_data.py builds the chunks.
