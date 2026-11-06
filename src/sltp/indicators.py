# -------- indicators.py --------
# The TA-Lib functions the engine calls, implemented in NumPy.
# Each one matches TA-Lib's lookback and tuple order, so warm-up bars are NaN and the
# first real value lands on the same bar. Wilder smoothing is used for ATR, RSI, ADX,
# MFI and STOCHRSI. Verified against TA-Lib 0.8.0 by test/compare_indicators.py.
# matype is accepted for compatibility; only 0 (SMA) is implemented and anything else raises.

# -------- Imports --------
from __future__ import annotations

import numpy as np


# -------- Constants --------
# The only smoothing type implemented, and the scale CCI divides by.
MATYPE_SMA: int = 0
CCI_SCALE: float = 0.015


# -------- Input Handling --------
# Bars are float32 and these functions want float64, so widen on the way in.
def _float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


# A sliding window view, or None when the series is shorter than the window.
def _windows(values: np.ndarray, period: int):
    if period <= 0 or values.size < period:
        return None
    return np.lib.stride_tricks.sliding_window_view(values, period)


# -------- Rolling Primitives --------
# Simple moving average over [start, start + period), NaN before the first window.
def _sma(values: np.ndarray, period: int, start: int = 0) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if period <= 0 or values.size - start < period:
        return out
    kernel = np.ones(period, dtype=np.float64)
    out[start + period - 1:] = np.convolve(values[start:], kernel, mode="valid") / period
    return out


# SMA of the finite tail of a series that already carries warm-up NaNs.
def _sma_tail(values: np.ndarray, period: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return np.full(values.size, np.nan)
    return _sma(values, period, int(finite[0]))


# Index of the first finite value, or the length when the series is all NaN.
def _first_finite(values: np.ndarray) -> int:
    finite = np.isfinite(values)
    return int(np.argmax(finite)) if finite.any() else int(values.size)


# Wilder's smoothing: seed with the mean of the first window, then ease toward each value.
def _wilder(values: np.ndarray, period: int, start: int = 0) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if period <= 0:
        return out
    first = start + period - 1
    if first >= values.size:
        return out

    window = values[start:start + period]
    if not np.all(np.isfinite(window)):
        return out

    previous = float(np.mean(window))
    out[first] = previous
    step = 1.0 / period
    for i in range(first + 1, values.size):
        current = values[i]
        if not np.isfinite(current):
            out[i] = previous
            continue
        previous = previous + (current - previous) * step
        out[i] = previous
    return out


# Exponential moving average, seeded with the SMA of the first full window.
def _ema(values: np.ndarray, period: int, start: int | None = None) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if period <= 0:
        return out

    if start is None:
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size == 0:
            return out
        start = int(finite[0])

    first = start + period - 1
    if first >= values.size:
        return out

    window = values[start:start + period]
    if not np.all(np.isfinite(window)):
        return out

    previous = float(np.mean(window))
    out[first] = previous
    weight = 2.0 / (period + 1.0)
    for i in range(first + 1, values.size):
        previous = values[i] * weight + previous * (1.0 - weight)
        out[i] = previous
    return out


# Highest and lowest value over a trailing window.
def _roll_max(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    windows = _windows(values, period)
    if windows is not None:
        out[period - 1:] = windows.max(axis=1)
    return out


def _roll_min(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    windows = _windows(values, period)
    if windows is not None:
        out[period - 1:] = windows.min(axis=1)
    return out


# True range: the day's span, or the gap from the previous close when larger.
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    out = np.full(close.size, np.nan)
    if close.size == 0:
        return out
    out[0] = high[0] - low[0]
    if close.size > 1:
        span = high[1:] - low[1:]
        gap_up = np.abs(high[1:] - close[:-1])
        gap_down = np.abs(low[1:] - close[:-1])
        out[1:] = np.maximum(span, np.maximum(gap_up, gap_down))
    return out


# -------- Moving Averages --------
# Simple moving average.
def SMA(values, timeperiod: int = 30) -> np.ndarray:
    return _sma(_float(values), timeperiod)


# Exponential moving average.
def EMA(values, timeperiod: int = 30) -> np.ndarray:
    return _ema(_float(values), timeperiod)


# Double exponential moving average.
def DEMA(values, timeperiod: int = 30) -> np.ndarray:
    first = _ema(_float(values), timeperiod)
    second = _ema(first, timeperiod)
    return 2.0 * first - second


# Triple exponential moving average.
def TEMA(values, timeperiod: int = 30) -> np.ndarray:
    first = _ema(_float(values), timeperiod)
    second = _ema(first, timeperiod)
    third = _ema(second, timeperiod)
    return 3.0 * first - 3.0 * second + third


# Triple exponential average: the one-period change of a triple EMA.
def TRIX(values, timeperiod: int = 30) -> np.ndarray:
    series = _float(values)
    out = np.full(series.size, np.nan)

    first = _ema(series, timeperiod)
    second = _ema(first, timeperiod)
    third = _ema(second, timeperiod)

    start = 3 * (timeperiod - 1) + 1
    if third.size > start and start >= 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[start:] = 100.0 * (third[start:] / third[start - 1:third.size - 1] - 1.0)
    return out


# -------- Momentum --------
# Relative Strength Index, from 0 to 100.
def RSI(values, timeperiod: int = 14) -> np.ndarray:
    series = _float(values)
    out = np.full(series.size, np.nan)
    if timeperiod <= 0 or series.size <= timeperiod:
        return out

    change = np.diff(series)
    gains = np.where(change > 0.0, change, 0.0)
    losses = np.where(change < 0.0, -change, 0.0)

    average_gain = float(np.mean(gains[:timeperiod]))
    average_loss = float(np.mean(losses[:timeperiod]))
    out[timeperiod] = _rsi_value(average_gain, average_loss)

    for i in range(timeperiod + 1, series.size):
        average_gain = (average_gain * (timeperiod - 1) + gains[i - 1]) / timeperiod
        average_loss = (average_loss * (timeperiod - 1) + losses[i - 1]) / timeperiod
        out[i] = _rsi_value(average_gain, average_loss)
    return out


# RSI from one pair of smoothed averages; a flat series reads 0, not 100.
def _rsi_value(average_gain: float, average_loss: float) -> float:
    if average_loss == 0.0:
        return 100.0 if average_gain > 0.0 else 0.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


# Moving average convergence divergence; returns (macd, signal, histogram) as TA-Lib does.
def MACD(values, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    series = _float(values)

    # TA-Lib seeds both averages on the slow warm-up bar, so match that or the values drift.
    anchor = max(fastperiod, slowperiod) - 1
    fast = _ema(series, fastperiod, anchor - fastperiod + 1)
    slow = _ema(series, slowperiod, anchor - slowperiod + 1)
    macd = fast - slow

    signal = _ema(macd, signalperiod)

    # All three lines start where the signal does, so the histogram has no NaN prefix.
    macd[:_first_finite(signal)] = np.nan
    return macd, signal, macd - signal


# Rate of change: percentage difference from `timeperiod` bars ago.
def ROC(values, timeperiod: int = 10) -> np.ndarray:
    series = _float(values)
    out = np.full(series.size, np.nan)
    if timeperiod <= 0 or series.size <= timeperiod:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[timeperiod:] = 100.0 * (series[timeperiod:] / series[:series.size - timeperiod] - 1.0)
    return out


# Price momentum: absolute difference from `timeperiod` bars ago.
def MOM(values, timeperiod: int = 10) -> np.ndarray:
    series = _float(values)
    out = np.full(series.size, np.nan)
    if timeperiod <= 0 or series.size <= timeperiod:
        return out
    out[timeperiod:] = series[timeperiod:] - series[:series.size - timeperiod]
    return out


# -------- Volatility --------
# Bollinger Bands around a moving average; returns (upper, middle, lower).
def BBANDS(values, timeperiod: int = 5, nbdevup: float = 2.0, nbdevdn: float = 2.0, matype: int = MATYPE_SMA):
    if matype != MATYPE_SMA:
        raise ValueError(f"BBANDS: only matype={MATYPE_SMA} (SMA) is implemented, got {matype}")

    series = _float(values)
    upper = np.full(series.size, np.nan)
    middle = np.full(series.size, np.nan)
    lower = np.full(series.size, np.nan)

    windows = _windows(series, timeperiod)
    if windows is None:
        return upper, middle, lower

    averages = windows.mean(axis=1)
    deviations = windows.std(axis=1)
    middle[timeperiod - 1:] = averages
    upper[timeperiod - 1:] = averages + nbdevup * deviations
    lower[timeperiod - 1:] = averages - nbdevdn * deviations
    return upper, middle, lower


# Average true range, Wilder-smoothed.
def ATR(high, low, close, timeperiod: int = 14) -> np.ndarray:
    return _wilder(_true_range(_float(high), _float(low), _float(close)), timeperiod, 1)


# Average directional index: trend strength either way.
def ADX(high, low, close, timeperiod: int = 14) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    closes = _float(close)
    out = np.full(closes.size, np.nan)
    if timeperiod <= 0 or closes.size < 2 * timeperiod:
        return out

    true_range = _true_range(highs, lows, closes)

    # Only the larger of the two moves counts, and only when it is positive.
    up_move = np.zeros(closes.size)
    down_move = np.zeros(closes.size)
    up_move[1:] = highs[1:] - highs[:-1]
    down_move[1:] = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    # True range starts at bar 1 and the moves at bar 0; seeding both at bar 1 shifts ADX.
    atr = _wilder(true_range, timeperiod, 1)
    plus_smoothed = _wilder(plus_dm, timeperiod, 0)
    minus_smoothed = _wilder(minus_dm, timeperiod, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_smoothed / atr
        minus_di = 100.0 * minus_smoothed / atr
        total = plus_di + minus_di
        directional = np.where(total > 0.0, 100.0 * np.abs(plus_di - minus_di) / total, np.nan)

    # Wilder smoothing of DX, seeded once DX itself has a full window.
    return _wilder(directional, timeperiod, timeperiod)


# -------- Oscillators --------
# Money flow index from price and volume.
def MFI(high, low, close, volume, timeperiod: int = 14) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    closes = _float(close)
    volumes = _float(volume)
    out = np.full(closes.size, np.nan)
    if timeperiod <= 0 or closes.size <= timeperiod:
        return out

    typical = (highs + lows + closes) / 3.0
    flow = typical * volumes
    rising = np.zeros(closes.size)
    falling = np.zeros(closes.size)
    rising[1:] = np.where(typical[1:] > typical[:-1], flow[1:], 0.0)
    falling[1:] = np.where(typical[1:] < typical[:-1], flow[1:], 0.0)

    positive = _sma(rising, timeperiod, 1) * timeperiod
    negative = _sma(falling, timeperiod, 1) * timeperiod

    for i in range(timeperiod, closes.size):
        up = positive[i]
        down = negative[i]
        if not np.isfinite(up) or not np.isfinite(down):
            continue
        if down == 0.0:
            out[i] = 100.0 if up > 0.0 else 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + up / down)
    return out


# Stochastic oscillator: the close relative to its recent range; returns (slow %K, slow %D).
def STOCH(
    high,
    low,
    close,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = MATYPE_SMA,
    slowd_period: int = 3,
    slowd_matype: int = MATYPE_SMA,
):
    if slowk_matype != MATYPE_SMA or slowd_matype != MATYPE_SMA:
        raise ValueError("STOCH: only the SMA smoothing (matype=0) is implemented")

    highs = _float(high)
    lows = _float(low)
    closes = _float(close)

    highest = _roll_max(highs, fastk_period)
    lowest = _roll_min(lows, fastk_period)

    fast_k = np.full(closes.size, np.nan)
    span = highest - lowest
    usable = np.isfinite(span)
    with np.errstate(divide="ignore", invalid="ignore"):
        fast_k[usable & (span != 0.0)] = 100.0 * (
            closes[usable & (span != 0.0)] - lowest[usable & (span != 0.0)]
        ) / span[usable & (span != 0.0)]
    fast_k[usable & (span == 0.0)] = 0.0

    slow_k = _sma_tail(fast_k, slowk_period)
    slow_d = _sma_tail(slow_k, slowd_period)

    # TA-Lib starts both lines at the end of the whole warm-up, not where each one begins.
    slow_k[:_first_finite(slow_d)] = np.nan
    return slow_k, slow_d


# Stochastic RSI; returns (fast %K, fast %D).
def STOCHRSI(
    values,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = MATYPE_SMA,
):
    if fastd_matype != MATYPE_SMA:
        raise ValueError("STOCHRSI: only the SMA smoothing (matype=0) is implemented")

    series = _float(values)
    strength = RSI(series, timeperiod)
    highest = _roll_max(strength, fastk_period)
    lowest = _roll_min(strength, fastk_period)

    fast_k = np.full(series.size, np.nan)
    span = highest - lowest
    usable = np.isfinite(span) & np.isfinite(strength)
    with np.errstate(divide="ignore", invalid="ignore"):
        fast_k[usable & (span != 0.0)] = 100.0 * (
            strength[usable & (span != 0.0)] - lowest[usable & (span != 0.0)]
        ) / span[usable & (span != 0.0)]
    fast_k[usable & (span == 0.0)] = 0.0

    fast_d = _sma_tail(fast_k, fastd_period)

    # %K is padded to the end of %D's warm-up, as TA-Lib does.
    fast_k[:_first_finite(fast_d)] = np.nan
    return fast_k, fast_d


# Williams %R, from 0 to -100.
def WILLR(high, low, close, timeperiod: int = 14) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    closes = _float(close)

    highest = _roll_max(highs, timeperiod)
    lowest = _roll_min(lows, timeperiod)
    span = highest - lowest

    out = np.full(closes.size, np.nan)
    usable = np.isfinite(span)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[usable & (span != 0.0)] = -100.0 * (
            highest[usable & (span != 0.0)] - closes[usable & (span != 0.0)]
        ) / span[usable & (span != 0.0)]
    out[usable & (span == 0.0)] = 0.0
    return out


# Commodity Channel Index.
def CCI(high, low, close, timeperiod: int = 14) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    closes = _float(close)

    typical = (highs + lows + closes) / 3.0
    windows = _windows(typical, timeperiod)
    out = np.full(closes.size, np.nan)
    if windows is None:
        return out

    averages = windows.mean(axis=1)
    deviations = np.abs(windows - averages[:, None]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (typical[timeperiod - 1:] - averages) / (CCI_SCALE * deviations)
    out[timeperiod - 1:] = np.where(deviations > 0.0, scaled, 0.0)
    return out


# -------- Volume --------
# On-balance volume, as TA-Lib's running total.
def OBV(close, volume) -> np.ndarray:
    closes = _float(close)
    volumes = _float(volume)
    # TA-Lib starts the running total at the first bar's volume, not at zero.
    out = np.zeros(closes.size)
    if closes.size:
        out[0] = volumes[0]
    for i in range(1, closes.size):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


# Accumulation/distribution line.
def AD(high, low, close, volume) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    closes = _float(close)
    volumes = _float(volume)

    out = np.zeros(closes.size)
    span = highs - lows
    with np.errstate(divide="ignore", invalid="ignore"):
        multiplier = np.where(span != 0.0, ((closes - lows) - (highs - closes)) / span, 0.0)
    flow = multiplier * volumes
    if closes.size:
        out = np.cumsum(flow)
    return out


# -------- Trend --------
# Parabolic SAR.
def SAR(high, low, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
    highs = _float(high)
    lows = _float(low)
    size = highs.size
    out = np.full(size, np.nan)
    if size < 2 or acceleration <= 0.0 or maximum < acceleration:
        return out

    step = float(acceleration)
    ceiling = float(maximum)

    # The first two bars decide the opening direction, as TA-Lib does.
    long_trend = bool(lows[1] > lows[0])
    if long_trend:
        sar = lows[0]
        extreme = highs[1]
    else:
        sar = highs[0]
        extreme = lows[1]

    factor = step
    out[1] = sar

    for i in range(2, size):
        sar = sar + factor * (extreme - sar)

        if long_trend:
            # Never inside the last two bars' range.
            sar = min(sar, lows[i - 1], lows[i - 2])
            if lows[i] < sar:
                long_trend = False
                sar = max(extreme, highs[i], highs[i - 1])
                extreme = lows[i]
                factor = step
            elif highs[i] > extreme:
                extreme = highs[i]
                factor = min(factor + step, ceiling)
        else:
            sar = max(sar, highs[i - 1], highs[i - 2])
            if highs[i] > sar:
                long_trend = True
                sar = min(extreme, lows[i], lows[i - 1])
                extreme = highs[i]
                factor = step
            elif lows[i] < extreme:
                extreme = lows[i]
                factor = min(factor + step, ceiling)

        out[i] = sar
    return out


# -------- Exports --------
__all__ = [
    "AD", "ADX", "ATR", "BBANDS", "CCI", "DEMA", "EMA", "MACD", "MFI", "MOM",
    "OBV", "ROC", "RSI", "SAR", "SMA", "STOCH", "STOCHRSI", "TEMA", "TRIX", "WILLR",
]
