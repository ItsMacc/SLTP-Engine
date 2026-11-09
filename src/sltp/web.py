# -------- web.py --------
# The browser contract: one call in, one plain dict out.
# Everything here composes engine pieces the CLI already uses. No engine code lives
# here and no files are written: charts come back as strings, never as files.
#
#   from sltp.web import run
#   payload = run(strategy, "2020-01-01", "2020-12-31")
#   json.dumps(payload)  # always works

# -------- Imports --------
from __future__ import annotations


# -------- Constants --------
# The payload shape, so a frontend can evolve against it.
FACADE_VERSION: int = 1
# The benchmark drawn next to a lone chart, matching the CLI.
BENCHMARK_TICKER: str = "SPY"
BENCHMARK_NAME: str = "Buy & Hold (SPY)"
BENCHMARK_COLOR: str = "#9aa3b2"


# -------- Normalizing --------
# A stray numpy scalar to its native value; plain values pass through untouched.
def _plain(value):
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return value


# An enum member to its value, so LONG reads "long" instead of "PositionType.LONG".
def _enum_value(member) -> str:
    return str(getattr(member, "value", member))


# One trade as plain data. The bars stay out: a trade already names its ticker,
# and the full history would multiply the payload for no chartable gain.
def _trade_to_dict(trade) -> dict:
    exit = trade.exit
    return {
        "id": int(trade.id),
        "ticker": str(trade.ticker),
        "position_type": _enum_value(trade.position_type),
        "entry_date": str(trade.entry_date),
        "entry_price": float(trade.entry_price),
        "allocation": float(trade.allocation),
        "equity_at_entry": float(trade.equity_at_entry),
        "stop_loss": float(trade.stop_loss),
        "take_profit": float(trade.take_profit),
        "fees": float(trade.fees),
        "dividend_income": float(trade.dividend_income),
        "exit": None
        if exit is None
        else {
            "date": str(exit.date),
            "type": _enum_value(exit.type),
            "price": float(exit.price),
        },
    }


# A run() result as JSON-safe data: stats without the duplicated equity curve,
# the curve as date and value pairs, trades as dicts.
def to_jsonable(result: dict) -> dict:
    metrics = result.get("metrics") or {}
    stats = {str(key): _plain(value) for key, value in metrics.items() if key != "equity"}
    equity = [
        {"date": str(point["date"]), "value": float(point["portfolio_value"])}
        for point in result.get("equity") or []
    ]
    trades = [_trade_to_dict(trade) for trade in result.get("trades") or []]
    return {"stats": stats, "equity": equity, "trades": trades, "stamp": result.get("stamp") or {}}


# -------- Chart --------
# The benchmark a lone chart is drawn against, mirroring the CLI: one fund,
# bought and held. A failed benchmark is skipped, never fatal.
def _benchmark(start: str, end: str, dividends: bool) -> list[dict]:
    from sltp.backtesting import Backtester
    from sltp.strategy import Strategy
    from sltp.utils import Order, PositionType

    class _BuyAndHold(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.name = BENCHMARK_NAME
            self.MAX_POSITIONS = 1
            self.MAX_LOTS_PER_TICKER = 1
            self.MIN_BARS_REQUIRED = 1

        def initialize_universe(self, current_date: str, histories: dict) -> bool:
            self.tickers.clear()
            if BENCHMARK_TICKER not in histories:
                return False
            self.add_ticker(BENCHMARK_TICKER)
            return True

        def on_signal(self, ticker: str, data) -> Order | None:
            if ticker in self.open_tickers or float(data.close_price[-1]) <= 0.0:
                return None
            return Order(
                ticker=ticker,
                position_type=PositionType.LONG,
                allocation=1.0,
                order_date=str(data.dates[-1]),
                stop_loss=0.0,
                take_profit=0.0,
            )

    try:
        return Backtester(_BuyAndHold(), start, end, dividends).run().get("equity") or []
    except Exception:
        return []


# The chart for a result, string only. A lone line gets the benchmark when the
# range is known; without one there is nothing to compare against. No equity
# means no chart, so an empty run returns None instead of raising.
def chart_svg(
    result: dict,
    start: str | None = None,
    end: str | None = None,
    title: str | None = None,
    dividends: bool = True,
) -> str | None:
    from sltp.graphs import ChartSeries, equity_chart

    equity = result.get("equity") or []
    if not equity:
        return None

    name: str = title or "Strategy"
    series = [ChartSeries.from_equity(name, equity)]
    if start is not None and end is not None:
        benchmark = _benchmark(start, end, dividends)
        if benchmark:
            series.append(ChartSeries.from_equity(BENCHMARK_NAME, benchmark, color=BENCHMARK_COLOR))

    first: str = start or str(equity[0]["date"])
    last: str = end or str(equity[-1]["date"])
    return equity_chart(
        series,
        title=name,
        subtitle=f"{first} → {last} · {len(equity)} sessions · index starts at 100",
        start_value=100.0,
    )


# -------- Entry Point --------
# One backtest as one plain dict: stats, equity pairs, trades and the SVG.
def run(strategy, start: str, end: str | None = None, dividends: bool = True, chart: bool = True) -> dict:
    from sltp.backtesting import Backtester

    result = Backtester(strategy, start, end, dividends).run()
    payload = to_jsonable(result)
    payload["svg"] = (
        chart_svg(result, start=start, end=end, title=getattr(strategy, "name", None), dividends=dividends)
        if chart
        else None
    )
    payload["facade_version"] = FACADE_VERSION
    return payload
