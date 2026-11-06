# -------- __init__.py --------
# Charts: pure-SVG, so a strategy run can be looked at without matplotlib.
#
#   from sltp.graphs import ChartSeries, equity_chart
#   svg = equity_chart([ChartSeries.from_equity("Momentum", run["equity"])])
#   write_chart("equity.svg", svg)

# -------- Imports --------
from sltp.graphs.equity import ChartSeries, equity_chart, write_chart
from sltp.graphs.palette import PALETTE, color_at


# -------- Exports --------
__all__ = ["ChartSeries", "PALETTE", "color_at", "equity_chart", "write_chart"]
