# -------- __init__.py --------
# The market-data layer: a cached flat-bundle reader and the tool that builds the files.
# Nothing is imported here on purpose, so importing sltp.data costs nothing.
#
#   from sltp.data.market_data_service import get_stock_history, get_all_tickers
#
# The bar files are one flat bundle per calendar year (1998.bin) plus the recent
# tail; the reader discovers whatever the range needs.
