# -------- equity.py --------
# Draw equity curves as an SVG string: no matplotlib, no browser, no dependency.
# Two panels share one time axis: equity on top, drawdown below. Plain SVG markup, so it
# renders in a file, an <img> tag, a notebook or Pyodide.
#
#   from sltp.graphs import ChartSeries, equity_chart
#   svg = equity_chart([ChartSeries.from_equity("Momentum", result["equity"])])

# -------- Imports --------
from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime

from html import escape

from sltp.graphs.palette import color_at


# -------- Style --------
# Fixed geometry in CSS pixels; the SVG scales with its container.
FONT: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
TEXT: str = "#161b26"
MUTED: str = "#69718a"
GRID: str = "#e8ebf2"
AXIS: str = "#c9cfdd"
BACKGROUND: str = "#ffffff"

WIDTH: int = 1100
PAD_LEFT: int = 74
PAD_RIGHT: int = 30
HEADER: int = 128
MAIN_HEIGHT: int = 330
PANEL_GAP: int = 52
PANEL_HEIGHT: int = 96
PAD_BOTTOM: int = 30


# -------- Series --------
# One line on the chart: a name, the dates and the values at those dates.
@dataclass
class ChartSeries:
    name: str
    dates: list[str]
    values: list[float]
    color: str | None = None
    note: str | None = None

    # Reject a half-built series here, where the caller can still see which one it was.
    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError(f"{self.name}: {len(self.dates)} dates but {len(self.values)} values")
        if not self.dates:
            raise ValueError(f"{self.name}: a chart series needs at least one point")

    # Build straight from a backtest's equity records.
    @classmethod
    def from_equity(cls, name: str, equity: list[dict], color: str | None = None, note: str | None = None) -> "ChartSeries":
        return cls(
            name=name,
            dates=[str(point["date"]) for point in equity],
            values=[float(point["portfolio_value"]) for point in equity],
            color=color,
            note=note,
        )

    # The last value, which the legend shows next to the name.
    @property
    def final(self) -> float:
        return self.values[-1]


# -------- Helpers --------
# A 1/2/2.5/5/10 step that covers a span in roughly `target` ticks.
def _nice_step(span: float, target: int = 5) -> float:
    if span <= 0.0:
        return 1.0
    raw: float = span / target
    magnitude: float = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


# Tick values covering [low, high], starting on a step boundary.
def _ticks(low: float, high: float, target: int = 5) -> list[float]:
    step: float = _nice_step(high - low, target)
    start: float = math.floor(low / step) * step
    values: list[float] = []
    value: float = start
    while value <= high + step * 0.001:
        values.append(round(value, 10))
        value += step
    return values


# A number with thousands separators, without trailing noise.
def _number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


# "Jan 2020" for a YYYY-MM-DD date, and the raw string if it is not one.
def _month_label(value: str) -> str:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").strftime("%b %Y")
    except ValueError:
        return value


# Where each date sits along the shared axis.
def _positions(series: list[ChartSeries]) -> tuple[list[str], dict[str, int]]:
    dates: list[str] = sorted({date for item in series for date in item.dates})
    return dates, {date: index for index, date in enumerate(dates)}


# The line path for one series, breaking the path where its dates skip sessions.
def _line_path(item: ChartSeries, x_of, y_of, index_of: dict[str, int]) -> tuple[str, bool]:
    parts: list[str] = []
    previous: int | None = None
    gapped: bool = False
    for date, value in zip(item.dates, item.values):
        index: int = index_of[date]
        if previous is not None and index != previous + 1:
            gapped = True
        parts.append(f"{'M' if previous is None or index != previous + 1 else 'L'}{x_of(date):.2f} {y_of(value):.2f}")
        previous = index
    return " ".join(parts), gapped


# The legend chips, wrapped onto as many rows as the width needs.
def _legend_rows(series: list[ChartSeries], available: float) -> list[list[tuple[str, str, float]]]:
    rows: list[list[tuple[str, str, float]]] = []
    current: list[tuple[str, str, float]] = []
    used: float = 0.0
    for item in series:
        text: str = f"{item.name}  {_number(item.final)}"
        width: float = 26.0 + len(text) * 6.6
        if current and used + width > available:
            rows.append(current)
            current, used = [], 0.0
        current.append((text, item.color or TEXT, width))
        used += width
    if current:
        rows.append(current)
    return rows


# -------- Chart --------
# Render the equity curves and their drawdowns as one SVG document.
def equity_chart(
    series: list[ChartSeries],
    title: str = "Equity",
    subtitle: str | None = None,
    width: int = WIDTH,
    show_drawdown: bool = True,
    start_value: float | None = None,
) -> str:
    series = list(series)
    if not series:
        raise ValueError("A chart needs at least one series")

    for index, item in enumerate(series):
        if item.color is None:
            item.color = color_at(index)

    dates, index_of = _positions(series)
    last_index: int = max(1, len(dates) - 1)
    plot_left: float = PAD_LEFT
    plot_right: float = width - PAD_RIGHT
    plot_width: float = plot_right - plot_left

    def x_of(date: str) -> float:
        return plot_left + index_of[date] / last_index * plot_width

    # The header grows when the legend needs a second row.
    rows = _legend_rows(series, plot_width)
    header: int = HEADER + 24 * (len(rows) - 1)

    main_top: float = header
    main_bottom: float = main_top + MAIN_HEIGHT
    panel_top: float = main_bottom + PANEL_GAP
    panel_bottom: float = panel_top + PANEL_HEIGHT
    height: int = int(panel_bottom + PAD_BOTTOM if show_drawdown else main_bottom + PAD_BOTTOM)

    # Equity scale, padded so a line never touches the frame.
    low: float = min(min(item.values) for item in series)
    high: float = max(max(item.values) for item in series)
    span: float = high - low
    padding: float = span * 0.06 if span > 0 else max(1.0, abs(high) * 0.02)
    low, high = low - padding, high + padding
    # Ticks are filtered into the window instead of the window being snapped out to
    # a tick: snapping reserves up to a whole empty step below the lowest data point,
    # which is a third of the panel on a chart whose range is a bit under two steps.
    y_ticks: list[float] = [tick for tick in _ticks(low, high, 5) if low <= tick <= high]
    if len(y_ticks) < 2:
        y_ticks = [low, high]

    def y_of(value: float) -> float:
        return main_bottom - (value - low) / (high - low) * MAIN_HEIGHT

    # Drawdown scale: 0 at the top, the worst decline at the bottom.
    drawdowns: list[tuple[ChartSeries, list[float]]] = []
    worst: float = 0.0
    for item in series:
        peak: float = item.values[0]
        values: list[float] = []
        for value in item.values:
            peak = max(peak, value)
            drop: float = (value / peak - 1.0) * 100.0 if peak > 0 else 0.0
            values.append(drop)
            worst = min(worst, drop)
        drawdowns.append((item, values))
    worst = worst if worst < 0 else -1.0

    def dd_y_of(value: float) -> float:
        return panel_top + (value / worst) * PANEL_HEIGHT

    parts: list[str] = []
    add = parts.append

    # width/height give an intrinsic size; the style lets it shrink to fit a narrow
    # container instead of being clipped, which is what happens to any fixed width.
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="max-width:100%;height:auto;display:block" '
        f'font-family="{FONT}" role="img" aria-label="{escape(title)}">')
    add(f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>')

    # Fills, so the lines stay the sharpest thing on the chart.
    for item in series:
        path, gapped = _line_path(item, x_of, y_of, index_of)
        if gapped or len(item.values) < 2:
            continue
        first_x: float = x_of(item.dates[0])
        last_x: float = x_of(item.dates[-1])
        add(f'<path d="{path} L{last_x:.2f} {main_bottom:.2f} L{first_x:.2f} {main_bottom:.2f} Z" '
            f'fill="{item.color}" opacity="0.07"/>')

    # Grid and the equity axis.
    for value in y_ticks:
        y: float = y_of(value)
        add(f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>')
        add(f'<text x="{plot_left - 12:.2f}" y="{y + 4:.2f}" fill="{MUTED}" font-size="11" text-anchor="end">{_number(value, 0)}</text>')
    add(f'<line x1="{plot_left:.2f}" y1="{main_bottom:.2f}" x2="{plot_right:.2f}" y2="{main_bottom:.2f}" stroke="{AXIS}" stroke-width="1"/>')

    # Time axis, shared by both panels.
    tick_total: int = min(6, len(dates))
    if tick_total > 1:
        indexes: list[int] = sorted({round(position * last_index / (tick_total - 1)) for position in range(tick_total)})
    else:
        indexes = [0]
    for index in indexes:
        x: float = plot_left + index / last_index * plot_width
        anchor: str = "start" if index == 0 else "end" if index == last_index else "middle"
        add(f'<text x="{x:.2f}" y="{main_bottom + 20:.2f}" fill="{MUTED}" font-size="11" text-anchor="{anchor}">{escape(_month_label(dates[index]))}</text>')

    # A start marker, so "up 18%" is anchored to something visible.
    if start_value is not None and low <= start_value <= high:
        y = y_of(start_value)
        add(f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="{AXIS}" '
            f'stroke-width="1" stroke-dasharray="4 5"/>')

    # The curves themselves.
    for item in series:
        path, _gapped = _line_path(item, x_of, y_of, index_of)
        add(f'<path d="{path}" fill="none" stroke="{item.color}" stroke-width="2.1" stroke-linejoin="round" '
            f'stroke-linecap="round"><title>{escape(item.name)}: {_number(item.final)}</title></path>')
        add(f'<circle cx="{x_of(item.dates[-1]):.2f}" cy="{y_of(item.final):.2f}" r="3" fill="{item.color}"/>')

    # Drawdown panel.
    if show_drawdown:
        add(f'<text x="{plot_left:.2f}" y="{panel_top - 12:.2f}" fill="{MUTED}" font-size="11" '
            f'letter-spacing="0.4">DRAWDOWN</text>')
        for value in (0.0, worst):
            y = dd_y_of(value)
            add(f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" '
                f'stroke="{GRID if value == 0.0 else AXIS}" stroke-width="1"/>')
            add(f'<text x="{plot_left - 12:.2f}" y="{y + 4:.2f}" fill="{MUTED}" font-size="11" '
                f'text-anchor="end">{_number(value, 0)}%</text>')

        for item, values in drawdowns:
            points: list[str] = [f"M{plot_left:.2f} {dd_y_of(0.0):.2f}"]
            for date, value in zip(item.dates, values):
                points.append(f"L{x_of(date):.2f} {dd_y_of(value):.2f}")
            points.append(f"L{plot_right:.2f} {dd_y_of(0.0):.2f} Z")
            add(f'<path d="{" ".join(points)}" fill="{item.color}" opacity="0.14"/>')
            add(f'<path d="{" ".join(points[:-1])}" fill="none" stroke="{item.color}" stroke-width="1.4" '
                f'stroke-linejoin="round"/>')

    # Header, drawn last so nothing overlaps it.
    add(f'<text x="{plot_left:.2f}" y="42" fill="{TEXT}" font-size="22" font-weight="600">{escape(title)}</text>')
    if subtitle:
        add(f'<text x="{plot_left:.2f}" y="66" fill="{MUTED}" font-size="13">{escape(subtitle)}</text>')

    chip_y: float = 96.0
    for row_index, row in enumerate(rows):
        cursor: float = plot_left
        baseline: float = chip_y + row_index * 24
        for text, color, chip_width in row:
            add(f'<rect x="{cursor:.2f}" y="{baseline - 9:.2f}" width="11" height="11" rx="2.5" fill="{color}"/>')
            add(f'<text x="{cursor + 18:.2f}" y="{baseline:.2f}" fill="{TEXT}" font-size="12.5">{escape(text)}</text>')
            cursor += chip_width

    add("</svg>")
    return "\n".join(parts)


# -------- Files --------
# Write the SVG to disk, creating the directory when needed, and return the path.
def write_chart(path: str, svg: str) -> str:
    directory = path.rsplit("/", 1)[0] if "/" in path else ""
    if directory:
        import os

        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return path
