# -------- __init__.py --------
# The market-data layer: a cached Parquet reader and the tool that builds the files.
# Nothing is imported here on purpose, so importing sltp.data costs nothing.
#
#   from sltp.data.market_data_service import get_stock_history, get_all_tickers
#
# The bar files are one chunk per year (market_data_YYYY.parquet) plus the
# recent tail; the reader discovers whatever the range needs.
