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
  sltp data
  sltp data update

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
# A file name for the chart, when the user did not choose one: one strategy
# keeps its own name, a comparison joins every name with _vs_.
def _chart_path(args, named: list[tuple[str, dict]]) -> str:
    if args.out:
        return args.out
    slugs: list[str] = []
    for name, _ in named:
        slugs.append(re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "chart")
    return f"{'_vs_'.join(slugs)[:120]}.svg"


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
    path: str = write_chart(_chart_path(args, named), svg)
    if args.out:
        # A saved chart is the whole result: say where it went and hand over
        # the SVG itself, so there is nothing to open just to copy the image.
        print(f"result saved in '{path}' at {os.path.abspath(path)}")
        print(svg)
    else:
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


# -------- Data --------
# How big a directory is, in MB, counting only parquet files.
def _parquet_mb(directory: str) -> float:
    total: int = 0
    for name in os.listdir(directory):
        if name.endswith(".parquet"):
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                pass
    return round(total / (1 << 20), 1)


# `sltp data` with no action, and `sltp data status`: where bars come from.
def _command_data_status() -> int:
    from sltp.data import config as _config
    from sltp.data.paths import has_bars, source

    info = source()
    kind, directory = info["kind"], info["dir"]
    auto = _config.auto_update()
    print(f"source      {'your own files' if kind == 'own' else 'managed cache'}")
    print(f"directory   {directory}")
    if kind == "cache":
        local = info.get("version") or "unknown"
        print(f"version     {local} (auto-update {'on' if auto else 'off'})")
        if auto:
            try:
                from sltp.data.remote import read_remote_version

                remote = read_remote_version().strip()
                print(f"latest      {remote}" + (" (up to date)" if remote == local else " (run `sltp data update`)"))
            except Exception:
                print("latest      unreachable (offline?)")
    else:
        print(f"version     local files (auto-update {'on' if auto else 'off'}; refresh only affects the cache)")
    try:
        names = [name for name in os.listdir(directory) if name.endswith(".parquet")]
    except OSError:
        names = []
    print(f"files       {len(names)} parquet files, {_parquet_mb(directory)} MB")
    return 0


# `sltp data update`: pull every chunk from R2, whatever auto says.
def _command_data_update() -> int:
    from sltp.data import remote as _remote

    try:
        result = _remote.refresh_all()
    except Exception as error:
        print(f"error: update failed: {error}", file=sys.stderr)
        return 1
    print(f"updated to {result['version']}: {result['files']} files, {round(result['bytes'] / (1 << 20), 1)} MB")
    return 0


# `sltp data check`: every bar file must open with the columns the engine reads.
def _command_data_check(path: str | None) -> int:
    import pyarrow.parquet as pq

    from sltp.data.paths import has_bars, market_dir

    directory = os.path.abspath(os.path.expanduser(path)) if path else str(market_dir())
    if not os.path.isdir(directory):
        print(f"error: no such directory: {directory}", file=sys.stderr)
        return 1
    if not has_bars(directory):
        print(f"error: {directory} holds no bar files (want market_data_YYYY.parquet)", file=sys.stderr)
        return 1
    needed = {"Symbol", "Date", "Open", "High", "Low", "Close", "Volume"}
    bad: int = 0
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("market_data_") and name.endswith(".parquet")):
            continue
        full = os.path.join(directory, name)
        try:
            schema = pq.read_schema(full)
            names = set(schema.names)
            missing = sorted(needed - names)
            rows = pq.ParquetFile(full).metadata.num_rows
            status = "ok" if not missing and rows > 0 else "FAIL"
            detail = f"{rows:,} rows"
            if missing:
                detail += f", missing columns: {', '.join(missing)}"
            if rows == 0:
                detail += ", empty file"
        except Exception as error:
            status, detail = "FAIL", f"unreadable ({error})"
        if status == "FAIL":
            bad += 1
        print(f"  {status:<4} {name}  {detail}")
    print("all files valid" if bad == 0 else f"{bad} file(s) need attention")
    return 1 if bad else 0


def _command_data(args) -> int:
    from sltp.data import config as _config
    from sltp.data.paths import has_bars

    action = args.data_command or "status"
    if action == "status":
        return _command_data_status()
    if action == "update":
        return _command_data_update()
    if action == "auto":
        _config.set_auto_update(args.state == "on")
        print(f"auto-update {args.state}")
        return 0
    if action == "use":
        if args.clear:
            _config.set_data_override(None)
            print("using the default bar source")
            return 0
        directory = os.path.abspath(os.path.expanduser(args.path))
        if not os.path.isdir(directory):
            print(f"error: no such directory: {directory}", file=sys.stderr)
            return 1
        if not has_bars(directory):
            print(f"error: {directory} holds no bar files (want market_data_YYYY.parquet)", file=sys.stderr)
            return 1
        _config.set_data_override(directory)
        print(f"reading bars from {directory}")
        return 0
    if action == "check":
        return _command_data_check(args.path)
    return 0


# `sltp reset`: forget every choice and behave like a fresh install.
def _command_reset() -> int:
    from sltp.data import config as _config

    _config.reset()
    print("settings restored to defaults: managed cache, auto-update on")
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

    data = subparsers.add_parser(
        "data",
        help="where bars come from and how fresh they are",
        description="Show the bar source, refresh it, or point it at your own files.",
    )
    data_sub = data.add_subparsers(dest="data_command", metavar="ACTION")
    data_sub.add_parser("status", help="show the bar source and freshness (the default)")
    data_sub.add_parser("update", help="download the latest bars from R2 now")
    auto = data_sub.add_parser("auto", help="turn background refresh on or off")
    auto.add_argument("state", choices=["on", "off"], help="on: refresh every run (default); off: stay offline")
    use = data_sub.add_parser("use", help="read bars from your own directory")
    use_group = use.add_mutually_exclusive_group(required=True)
    use_group.add_argument("path", nargs="?", metavar="DIR", help="a directory holding the bar files")
    use_group.add_argument("--clear", action="store_true", help="forget the directory and use the default")
    check = data_sub.add_parser("check", help="validate a directory of bar files")
    check.add_argument("path", nargs="?", metavar="DIR", help="the directory to validate (default: the one in use)")

    reset = subparsers.add_parser(
        "reset",
        help="restore all sltp settings to their defaults",
        description="Forget every choice (data directory, auto-update) and behave like a fresh install. Cached bars stay where they are.",
    )

    return parser


# -------- Entry Point --------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "data":
            return _command_data(args)
        if args.command == "reset":
            return _command_reset()
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
