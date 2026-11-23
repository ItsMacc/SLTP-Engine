# -------- __init__.py --------
# The market-data layer's inputs: the catalogue, the action tables and the bundle writer.
# Nothing is imported here on purpose, so importing sltp.data costs nothing.
#
#   from sltp.market.market_data import get_stock_history, universe
#
# The bar files are five-year flat bundles (1998-2002.bin) with a sidecar index in
# sltp/market; the reader maps only the chunks a request touches.
