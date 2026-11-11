# -------- cli.py --------
# The `sltp` command: run a strategy file, print what it did, draw the curve.
#
#   sltp backtest momentum.py
#   sltp backtest momentum.py --chart
#   sltp compare value.py momentum.py --out compare.svg
#
# A strategy is any Python file with a Strategy subclass; the file's directory joins
# sys.path so it can import its siblings. Output is a summary by default, JSON with --json.

# -------- Imports --------
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys

from datetime import datetime


# The installed version without importing the package, so --version never waits
# on numpy and pyarrow. Every other sltp import lives inside the
# function that needs it; module level here stays cheap for --help too.
def _version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version as _dist_version

        return _dist_version("sltp")
    except PackageNotFoundError:
        return "0.0.0"


# -------- Constants --------
# How much history a run covers when no dates are given.
DEFAULT_YEARS: int = 3
# The benchmark added to a chart that would otherwise show a single line.
BENCHMARK_TICKER: str = "SPY"
BENCHMARK_NAME: str = "Buy & Hold (SPY)"
BENCHMARK_COLOR: str = "#9aa3b2"

EPILOG: str = """\
examples:
  sltp backtest momentum.py
  sltp backtest momentum.py --start 2019-01-01 --chart
  sltp compare value.py momentum.py --out compare.svg
  sltp compare a.py b.py c.py --json > results.json

A strategy is a .py file with a Strategy subclass and a no-argument constructor.
Add :ClassName to the path when the file defines more than one.
"""


# -------- Errors --------
# Anything the user can fix, reported without a traceback.
class StrategyLoadError(Exception):
    pass


# -------- Colour --------
# Plain text unless the stream is a terminal and NO_COLOR is unset.
def _styled(text: str, code: str, stream=None) -> str:
    target = stream if stream is not None else sys.stdout
    if not getattr(target, "isatty", lambda: False)() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _styled(text, "1")


def _dim(text: str) -> str:
    return _styled(text, "2")


def _green(text: str) -> str:
    return _styled(text, "32")


def _red(text: str) -> str:
    return _styled(text, "31")


# Colour codes take no room on screen, so a column is measured and padded by the
# characters a person can see; otherwise a green cell widens its column invisibly
# and every uncoloured row in that column drifts away from it.
_ESCAPE = re.compile(r"\033\[[0-9;]*m")


def _visible(text: str) -> str:
    return _ESCAPE.sub("", text)


def _pad_left(text: str, width: int) -> str:
    return text + " " * max(0, width - len(_visible(text)))


def _pad_right(text: str, width: int) -> str:
    return " " * max(0, width - len(_visible(text))) + text


# -------- Formatting --------
# Signed percentage, so a return always says which way it went.
def _signed(value: float) -> str:
    text = f"{value:+.2f}%"
    return _green(text) if value > 0 else _red(text) if value < 0 else text


# Plain percentage, for values that are naturally negative.
def _percent(value: float) -> str:
    return f"{value:.2f}%"


# -------- Strategy Loading --------
# Import a strategy file under a private module name so two files cannot collide.
def _module_from_file(path: str, index: int) -> object:
    if not os.path.isfile(path):
        raise StrategyLoadError(f"no such file: {path}")

    directory = os.path.dirname(os.path.abspath(path))
    if directory not in sys.path:
        sys.path.insert(0, directory)

    name = f"_sltp_user_{index}_{abs(hash(os.path.abspath(path))) % 100_000_000}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise StrategyLoadError(f"{path} failed while importing: {error}") from error
    return module


# The Strategy subclasses a module defines itself, in definition order.
def _strategy_classes(module: object) -> list[type]:
    from sltp.strategy import Strategy

    found: list[type] = []
    for value in vars(module).values():
        if not isinstance(value, type) or not issubclass(value, Strategy) or value is Strategy:
            continue
        if value.__module__ == module.__name__:
            found.append(value)
    return found


# Resolve "file.py", "file.py:Class" or "package.module" to Strategy subclasses.
def load_strategies(spec: str, index: int = 0) -> list[type]:
    target, _, class_name = spec.partition(":")

    if target.endswith(".py") or "/" in target or os.sep in target:
        module = _module_from_file(target, index)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as error:
            raise StrategyLoadError(f"cannot import {target}: {error}") from error

    if class_name:
        candidate = getattr(module, class_name, None)
        if candidate is None:
            raise StrategyLoadError(f"{target} has no {class_name}")
        classes: list[type] = [candidate]
    else:
        classes = _strategy_classes(module)

    from sltp.strategy import Strategy

    for candidate in classes:
        if not isinstance(candidate, type) or not issubclass(candidate, Strategy):
            raise StrategyLoadError(f"{candidate!r} is not a Strategy subclass")
        if candidate.on_signal is Strategy.on_signal:
            raise StrategyLoadError(f"{candidate.__name__} does not implement on_signal()")

    if not classes:
        raise StrategyLoadError(
            f"no Strategy subclass in {spec}. Define one, or name it with {spec}:ClassName"
        )
    return classes


# Build the strategy, explaining the usual cause when construction fails.
def _instantiate(candidate: type) -> Strategy:
    try:
        return candidate()
    except TypeError as error:
        raise StrategyLoadError(
            f"{candidate.__name__}() could not be built without arguments ({error}). "
            f"Give it a no-argument constructor with sensible defaults."
        ) from error


# -------- Period --------
# Resolve the date range, defaulting to the last three years of stored sessions.
def _period(start: str | None, end: str | None) -> tuple[str, str]:
    from dateutil.relativedelta import relativedelta

    from sltp.data.market_data_service import get_latest_trading_date

    latest: str = get_latest_trading_date()
    end = end or latest
    if start is None:
        start = (datetime.strptime(end, "%Y-%m-%d") - relativedelta(years=DEFAULT_YEARS)).strftime("%Y-%m-%d")

    for label, value in (("start", start), ("end", end)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as error:
            raise StrategyLoadError(f"{label} date must be YYYY-MM-DD, got {value!r}") from error

    if start > end:
        raise StrategyLoadError(f"start {start} is after end {end}")
    return start, end


# -------- Running --------
# Run one strategy over the period.
def _run(strategy: Strategy, start: str, end: str, dividends: bool) -> dict:
    from sltp.backtesting import Backtester

    return Backtester(strategy, start, end, dividends).run()


# Pull the handful of numbers the summary shows out of a run.
# Positions still open at the end have no exit, so they have no return to score: a
# win rate over zero closed trades is not 0%, it is unknown, and the report says so.
def _summary(result: dict) -> dict:
    equity: list[dict] = result.get("equity") or []
    trades = result.get("trades") or []
    first: float = float(equity[0]["portfolio_value"]) if equity else 0.0
    last: float = float(equity[-1]["portfolio_value"]) if equity else 0.0
    return {
        "sessions": len(equity),
        "start_value": first,
        "end_value": last,
        "return_pct": (last / first - 1.0) * 100.0 if first else 0.0,
        "trades": len(trades),
        "closed": sum(1 for trade in trades if getattr(trade, "exit", None) is not None),
        "metrics": result.get("metrics") or {},
    }


# The rows every report shares, so backtest and compare describe the same things.
def _metric_rows(names: list[str], summaries: list[dict]) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []

    def add(label: str, values: list[str]) -> None:
        rows.append([label, *values])

    add("Return", [_signed(summary["return_pct"]) for summary in summaries])
    add("CAGR", [_signed(float(summary["metrics"].get("cagr", 0.0))) for summary in summaries])
    add("Sharpe", [f"{float(summary['metrics'].get('sharpe', 0.0)):.2f}" for summary in summaries])
    add("Sortino", [f"{float(summary['metrics'].get('sortino', 0.0)):.2f}" for summary in summaries])
    add("PSR", [f"{float(summary['metrics'].get('psr', 0.0)):.2f}" for summary in summaries])
    add("Max drawdown", [_percent(float(summary["metrics"].get("max_drawdown", 0.0))) for summary in summaries])
    add("Win rate", [
        _percent(float(summary["metrics"].get("win_rate", 0.0))) if summary["closed"] else "—"
        for summary in summaries
    ])
    add("Trades", [str(summary["trades"]) for summary in summaries])
    add("Turnover", [_percent(float(summary["metrics"].get("turnover", 0.0))) for summary in summaries])
    add("Cost drag", [_percent(float(summary["metrics"].get("cost_drag", 0.0))) for summary in summaries])
    return names, rows


# -------- Output --------
# Aligned columns: the first is a label, the rest are one column per strategy.
def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths: list[int] = [
        max([len(_visible(headers[column])), *(len(_visible(row[column])) for row in rows)])
        for column in range(len(headers))
    ]
    lines: list[str] = []

    header: str = "  " + _pad_left(headers[0], widths[0]) + "".join(
        "   " + _pad_right(headers[column], widths[column]) for column in range(1, len(headers))
    )
    lines.append(_bold(header))
    lines.append("  " + "─" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        lines.append(
            "  " + _pad_left(row[0], widths[0]) + "".join(
                "   " + _pad_right(row[column], widths[column]) for column in range(1, len(row))
            )
        )
    return "\n".join(lines)


# The single-strategy report: a heading and one number per line.
def _report(name: str, summary: dict, start: str, end: str) -> str:
    metrics = summary["metrics"]
    pairs: list[tuple[str, str]] = [
        ("Ending value", f"{summary['end_value']:,.2f}   ({_signed(summary['return_pct'])})"),
        ("CAGR", _signed(float(metrics.get("cagr", 0.0)))),
        ("Sharpe", f"{float(metrics.get('sharpe', 0.0)):.2f}"),
        ("Sortino", f"{float(metrics.get('sortino', 0.0)):.2f}"),
        ("PSR", f"{float(metrics.get('psr', 0.0)):.2f}"),
        ("Max drawdown", _percent(float(metrics.get("max_drawdown", 0.0)))),
        ("Win rate", _percent(float(metrics.get("win_rate", 0.0))) if summary["closed"] else "—"),
        ("Trades", str(summary["trades"])),
        ("Turnover", _percent(float(metrics.get("turnover", 0.0)))),
        ("Cost drag", _percent(float(metrics.get("cost_drag", 0.0)))),
    ]
    width: int = max(len(label) for label, _ in pairs)
    lines: list[str] = [
        _bold(name),
        _dim(f"  {start} → {end} · {summary['sessions']} sessions"),
        "",
    ]
    lines.extend(f"  {label.ljust(width)}   {value}" for label, value in pairs)
    return "\n".join(lines)


# JSON for anything that wants to read the result instead of look at it.
def _json_payload(named: list[tuple[str, dict]], start: str, end: str) -> str:
    runs: list[dict] = []
    for name, result in named:
        summary = _summary(result)
        metrics = {key: value for key, value in summary["metrics"].items() if key != "equity"}
        runs.append({
            "strategy": name,
            "sessions": summary["sessions"],
            "trade_count": summary["trades"],
            "ending_value": summary["end_value"],
            "return_pct": round(summary["return_pct"], 6),
            "metrics": metrics,
            "equity": result.get("equity") or [],
        })
    return json.dumps({"start": start, "end": end, "runs": runs}, indent=2, default=str)


# -------- Chart --------
# A file name for a strategy's chart, when the user did not choose one.
def _chart_path(args, name: str) -> str:
    if args.out:
        return args.out
    slug: str = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "chart"
    return f"{slug}.svg"


# Draw every run on one chart, adding the benchmark when there is only one line.
def _emit_chart(args, named: list[tuple[str, dict]], start: str, end: str) -> str:
    from sltp.graphs import ChartSeries, equity_chart, write_chart
    from sltp.strategy import Strategy
    from sltp.utils import Order, PositionType

    # The benchmark a lone chart is drawn against: one fund, bought and held.
    class _BuyAndHold(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.name = BENCHMARK_NAME
            self.MAX_POSITIONS = 1
            self.MAX_LOTS_PER_TICKER = 1
            self.MIN_BARS_REQUIRED = 1

        def initialize_universe(self, current_date: str, histories=None) -> bool:
            self.tickers.clear()
            if BENCHMARK_TICKER not in self.available_universe:
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

    series: list[ChartSeries] = [ChartSeries.from_equity(name, result["equity"]) for name, result in named]

    if len(series) < 2:
        try:
            benchmark = _run(_BuyAndHold(), start, end, not args.no_dividends)
            if benchmark["equity"]:
                series.append(ChartSeries.from_equity(BENCHMARK_NAME, benchmark["equity"], color=BENCHMARK_COLOR))
        except Exception as error:
            print(f"note: no benchmark line ({error})", file=sys.stderr)

    title: str = named[0][0] if len(named) == 1 else "Strategy comparison"
    sessions: int = len(named[0][1]["equity"])
    svg: str = equity_chart(
        series,
        title=title,
        subtitle=f"{start} → {end} · {sessions} sessions · index starts at 100",
        start_value=100.0,
    )
    path: str = write_chart(_chart_path(args, named[0][0]), svg)
    print(f"chart: {path}")

    # Only open a window when the user did not choose a destination.
    if not args.out:
        try:
            import webbrowser

            webbrowser.open(f"file://{os.path.abspath(path)}")
        except Exception:
            pass
    return path


# -------- Commands --------
def _command_backtest(args) -> int:
    classes = load_strategies(args.strategy)
    if len(classes) > 1:
        names: str = ", ".join(candidate.__name__ for candidate in classes)
        raise StrategyLoadError(
            f"{args.strategy} defines several strategies ({names}); "
            f"name one with {args.strategy}:{classes[0].__name__}"
        )

    strategy = _instantiate(classes[0])
    result = _run(strategy, args.start, args.end, not args.no_dividends)
    named: list[tuple[str, dict]] = [(strategy.name, result)]

    if args.json:
        print(_json_payload(named, args.start, args.end))
    else:
        print()
        print(_report(strategy.name, _summary(result), args.start, args.end))
        print()

    if args.chart or args.out:
        _emit_chart(args, named, args.start, args.end)
    return 0


def _command_compare(args) -> int:
    classes: list[type] = []
    for index, spec in enumerate(args.strategies):
        found = load_strategies(spec, index)
        if len(found) > 1:
            names: str = ", ".join(candidate.__name__ for candidate in found)
            raise StrategyLoadError(
                f"{spec} defines several strategies ({names}); name one with {spec}:{found[0].__name__}"
            )
        classes.extend(found)

    strategies: list[Strategy] = [_instantiate(candidate) for candidate in classes]

    # Two files may declare the same name, so disambiguate before they share a legend.
    seen: dict[str, int] = {}
    for strategy in strategies:
        seen[strategy.name] = seen.get(strategy.name, 0) + 1
    renamed: dict[str, int] = {}
    for strategy in strategies:
        if seen[strategy.name] > 1:
            renamed[strategy.name] = renamed.get(strategy.name, 0) + 1
            strategy.name = f"{strategy.name} ({renamed[strategy.name]})"

    named: list[tuple[str, dict]] = []
    for strategy in strategies:
        named.append((strategy.name, _run(strategy, args.start, args.end, not args.no_dividends)))

    if args.json:
        print(_json_payload(named, args.start, args.end))
    else:
        summaries = [_summary(result) for _, result in named]
        headers, rows = _metric_rows([name for name, _ in named], summaries)
        print()
        print(_table(["", *headers], rows))
        print(_dim(f"  {args.start} → {args.end} · {summaries[0]['sessions']} sessions"))
        print()

    if args.chart or args.out:
        _emit_chart(args, named, args.start, args.end)
    return 0


# -------- Parser --------
# The flags every run takes.
def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-s", "--start", metavar="DATE",
        help=f"first session, YYYY-MM-DD (default: {DEFAULT_YEARS} years before the end)",
    )
    parser.add_argument(
        "-e", "--end", metavar="DATE",
        help="last session, YYYY-MM-DD (default: the latest session on disk)",
    )
    parser.add_argument("--no-dividends", action="store_true", help="ignore cash distributions")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the summary")
    parser.add_argument(
        "-g", "--chart", "--graph", action="store_true",
        help="also draw an SVG equity chart and open it",
    )
    parser.add_argument(
        "-o", "--out", metavar="FILE",
        help="write the chart to FILE instead of the default name (implies --chart)",
    )


# The whole CLI, so tests and the entry point share one definition.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sltp",
        description="Backtest trading strategies against local daily bars.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sltp {_version()}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    backtest = subparsers.add_parser(
        "backtest",
        help="run one strategy",
        description="Run one strategy and report what it did.",
    )
    backtest.add_argument("strategy", metavar="STRATEGY", help="a .py file, or file.py:ClassName")
    _add_run_flags(backtest)

    compare = subparsers.add_parser(
        "compare",
        help="run several strategies side by side",
        description="Run two or more strategies over one period and compare them.",
    )
    compare.add_argument("strategies", nargs="+", metavar="STRATEGY", help="a .py file, or file.py:ClassName")
    _add_run_flags(compare)

    return parser


# -------- Entry Point --------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        args.start, args.end = _period(args.start, args.end)
        if args.command == "backtest":
            return _command_backtest(args)
        return _command_compare(args)
    except StrategyLoadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
