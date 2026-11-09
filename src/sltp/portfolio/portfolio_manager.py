# -------- portfolio_manager.py --------
# Keeps the book: orders, fills, positions, costs and equity.
# Orders fill at the next open, and equity is an index that starts at 100.

# -------- Imports --------
import numpy as np

from dataclasses import replace
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Engine imports
from sltp.utils.Event import Event
from sltp.utils.Order import Order, OrderSizing
from sltp.utils.Trade import Trade
from sltp.utils.Dates import to_code
from sltp.strategy import Strategy
from sltp.utils.StockData import StockData
from sltp.utils.EventType import EventType
from sltp.utils.PositionType import PositionType
from sltp.utils.RebalanceFrequency import RebalanceFrequency

# Database imports
from sltp.data.market_data_service import get_stock_history


# -------- Rebalance Arithmetic --------
REBALANCE_DELTAS: dict[RebalanceFrequency, relativedelta] = {
    RebalanceFrequency.DAILY: relativedelta(days=1),
    RebalanceFrequency.WEEKLY: relativedelta(weeks=1),
    RebalanceFrequency.MONTHLY: relativedelta(months=1),
    RebalanceFrequency.QUARTERLY: relativedelta(months=3),
    RebalanceFrequency.YEARLY: relativedelta(years=1),
    RebalanceFrequency.NEVER: None
}


# -------- Portfolio Manager --------
class PortfolioManager:
    # Owns the book: orders, fills, positions, costs and equity.
    def __init__(self, strategy: Strategy) -> None:
        # The strategy this book belongs to
        self.strategy: Strategy = strategy

        # Trading costs, applied to every fill
        self.SLIPPAGE_PCT: float = 0.0005
        self.FEE_RATE: float = 0.001
        # One-way spread cost per fill, and the largest share of a bar one fill may take.
        self.SPREAD_PCT: float = 0.0
        self.MAX_PARTICIPATION: float | None = None
        # Annual borrow rate on open shorts, plus tickers that cannot be shorted at all.
        self.BORROW_RATE: float = 0.0
        self.HARD_TO_BORROW: frozenset[str] = frozenset()

        # Index accounting, starting at 100
        self.starting_equity: float = 100.00
        self.portfolio_value: float = 100.00
        self.cash: float = 100.00
        self.realized_pnl: float = 0.0
        self.dividend_income: float = 0.0
        self.total_cost: float = 0.0
        self.total_slippage: float = 0.0
        self.total_spread: float = 0.0
        self.total_borrow_cost: float = 0.0
        self.traded_notional: float = 0.0

        # Records
        self.trades: list[Trade] = []
        self.open_trades: dict[int, Trade] = {}
        self._open_trades_by_ticker: dict[str, list[Trade]] = {}
        self._open_tickers: set[str] = set()
        self._invested_by_ticker: dict[str, float] = {}
        self._invested_notional: float = 0.0
        self._strategy_open_tickers: frozenset[str] = frozenset()
        self._strategy_open_trades: tuple[Trade, ...] = ()
        self._strategy_view_dirty: bool = True
        self._trade_by_id: dict[int, Trade] = {}
        self.rejected_orders: dict[str, int] = {}
        self._histories: dict[str, StockData] = {}
        self._dividend_events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_dividend_code: int | None = None

        # Equity logs, kept as plain lists while running and turned into arrays on read
        self._dates_log: list[str] = []
        self._equity_log: list[float] = []
        self._cash_log: list[float] = []

        # Queues drained at the next open, plus the ids of lots due to close
        self.pending_orders: list[Order] = []
        self.pending_exits: list[int] = []

        # Last traded price per ticker, so a suspended name keeps its mark
        self.last_prices: dict[str, float] = {}

        # Rebalance tracking
        self.last_rebalance_date: str | None = None

        self._next_trade_id: int = 1

        # Hand the strategy the opening book
        self.refresh_view()

    # -------- Strategy View --------
    # Rebuild the immutable snapshots once after book changes, not per fill.
    def refresh_view(self) -> None:
        if self._strategy_view_dirty:
            self._strategy_open_trades = tuple(self.open_trades.values())
            self._strategy_open_tickers = frozenset(self._open_tickers)
            self._strategy_view_dirty = False
        self.strategy._open_trades = self._strategy_open_trades
        self.strategy._open_ticker_set = self._strategy_open_tickers
        self.strategy._equity = self.portfolio_value

    # -------- Read-only State --------
    # Distinct tickers with at least one open lot.
    @property
    def active_positions(self) -> int:
        return len(self._open_tickers)

    # Every ticker currently holding a lot.
    @property
    def open_tickers(self) -> set[str]:
        return set(self._open_tickers)

    # Fees plus slippage plus spread plus borrow, for reporting.
    # Slippage and spread are already inside the fill price; borrow is a daily cash debit.
    @property
    def total_trading_cost(self) -> float:
        return self.total_cost + self.total_slippage + self.total_spread + self.total_borrow_cost

    # The equity curve as (dates, equities, cash) arrays.
    @property
    def equity_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self._dates_log, dtype=str),
            np.array(self._equity_log, dtype=np.float64),
            np.array(self._cash_log, dtype=np.float64),
        )

    # Open lots in one ticker, oldest first.
    def lots(self, ticker: str) -> list[Trade]:
        return self._open_trades_by_ticker.get(ticker, []).copy()

    # Use already-loaded histories instead of querying Parquet on fills.
    def set_history_cache(self, histories: dict[str, StockData]) -> None:
        self._histories = histories

    # Cash distributions per ticker, as sorted (day codes, amount per share) pairs.
    def set_dividend_events(self, events: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._dividend_events = events

    # Pay dividends to lots held from the previous close; runs before the open fills.
    def pay_dividends(self, current_code: int) -> None:
        start: int = current_code if self._last_dividend_code is None else self._last_dividend_code + 1
        self._last_dividend_code = int(current_code)
        if not self._dividend_events or not self.open_trades:
            return

        for trade in self.open_trades.values():
            entry = self._dividend_events.get(trade.ticker)
            if entry is None or trade.entry_price <= 0:
                continue
            codes, amounts = entry
            entry_code: int = int(to_code(trade.entry_date))
            low: int = int(np.searchsorted(codes, max(start, entry_code + 1), side="left"))
            high: int = int(np.searchsorted(codes, current_code, side="right"))
            if high <= low:
                continue

            notional: float = trade.equity_at_entry * trade.allocation
            shares: float = notional / trade.entry_price
            direction: float = -1.0 if trade.position_type is PositionType.SHORT else 1.0
            paid: float = float(np.sum(amounts[low:high])) * shares * direction
            if paid == 0.0:
                continue

            trade.dividend_income += paid
            self.cash += paid
            self.dividend_income += paid
            self.portfolio_value += paid

    # -------- Order Management --------
    # Queue an entry for the next open; it is validated at the fill.
    def submit_order(self, order: Order) -> None:
        self.pending_orders.append(order)

    # Queue one lot to close at the next open, ignoring a duplicate request.
    def queue_exit(self, trade_id: int) -> None:
        if trade_id not in self.pending_exits:
            self.pending_exits.append(trade_id)

    # -------- Filling --------
    # Fill every queued order at today's open. What the bar's volume would not
    # take stays queued for the next open; an order with no bar is still dropped.
    def fill_pending_orders(self, current_date: str, opens: dict[str, float]) -> None:
        carried: list[Order] = []
        for order in self.pending_orders:
            open_price: float | None = self._get_valid_open_price(order.ticker, opens)
            if open_price is not None:
                remainder: Order | None = self._fill_order(
                    order, open_price, current_date, self._day_volume(order.ticker, current_date)
                )
                if remainder is not None:
                    carried.append(remainder)

        self.pending_orders = carried

    # Fill every queued exit at today's open.
    def fill_pending_exits(self, current_date: str, opens: dict[str, float]) -> None:
        for trade_id in self.pending_exits:
            trade: Trade | None = self._find_trade(trade_id)
            if trade is None or not trade.is_open:
                continue

            open_price: float | None = self._get_valid_open_price(trade.ticker, opens)
            if open_price is not None:
                self._fill_exit(trade, open_price, current_date, EventType.SIGNAL)

        self.pending_exits.clear()

    # -------- Exit --------
    # Close lots whose stop or target was hit. Stop wins if both were.
    def check_exits(self, ticker: str, current_date: str, current_open, current_high, current_low):
        for trade in self.lots(ticker):
            filled_price, event_type = self._check_exit(trade, current_open, current_high, current_low)
            if event_type is not None:
                self._fill_exit(trade, filled_price, current_date, event_type)

    # Liquidate every remaining position at the final bar's close.
    def close_all(self, current_date: str, closes: dict[str, float]) -> None:
        for trade in list(self.open_trades.values()):
            close_price: float = closes.get(trade.ticker, trade.entry_price)
            if close_price is None or close_price <= 0:
                close_price = self.last_prices.get(trade.ticker, trade.entry_price)
            self._fill_exit(trade, close_price, current_date, EventType.END_OF_BACKTEST)

    # -------- Marking --------
    # Mark open positions to market at today's close and log daily equity and cash.
    def update_daily_accounting(self, current_date: str, closes: dict[str, float]) -> None:
        # Cache attribute lookups locally, this runs once per trading day
        last_prices = self.last_prices
        starting_equity = self.starting_equity
        realized_pnl = self.realized_pnl
        dividend_income = self.dividend_income
        total_cost = self.total_cost

        # Update last known prices; a ticker without a price today keeps its last mark
        for ticker, price in closes.items():
            if price > 0:
                last_prices[ticker] = price

        unrealized: float = 0.0
        for trade in self.open_trades.values():
            price: float = last_prices.get(trade.ticker, trade.entry_price)
            direction: float = -1.0 if trade.position_type is PositionType.SHORT else 1.0

            notional: float = trade.equity_at_entry * trade.allocation
            pct_return: float = direction * (price - trade.entry_price) / trade.entry_price
            unrealized += notional * pct_return

        # Borrow accrues daily on open shorts, on the marked notional, like a prime broker debit.
        if self.BORROW_RATE > 0.0:
            for borrow_trade in self.open_trades.values():
                if borrow_trade.position_type is not PositionType.SHORT or borrow_trade.entry_price <= 0:
                    continue
                mark: float = last_prices.get(borrow_trade.ticker, borrow_trade.entry_price)
                if mark <= 0:
                    continue
                borrow_shares: float = borrow_trade.equity_at_entry * borrow_trade.allocation / borrow_trade.entry_price
                daily_fee: float = borrow_shares * mark * self.BORROW_RATE / 252.0
                borrow_trade.borrow_cost += daily_fee
                self.total_borrow_cost += daily_fee
                self.cash -= daily_fee

        # Index accounting: starting + realized + dividends - costs - borrow + unrealized
        self.portfolio_value = starting_equity + realized_pnl + dividend_income - total_cost - self.total_borrow_cost + unrealized

        # Append to the primitive logs, overwriting the same day on a second mark
        if self._dates_log and self._dates_log[-1] == current_date:
            self._equity_log[-1] = self.portfolio_value
            self._cash_log[-1] = self.cash
        else:
            self._dates_log.append(current_date)
            self._equity_log.append(self.portfolio_value)
            self._cash_log.append(self.cash)

        # Flag the view dirty so the strategy sees updated equity tomorrow morning
        self._strategy_view_dirty = True

    # -------- Rebalancing --------
    # True when the cadence has elapsed since the last rebuild.
    def rebalance_due(self, current_date: str) -> bool:
        frequency: RebalanceFrequency = self.strategy.REBALANCE_FREQUENCY
        if frequency is RebalanceFrequency.NEVER:
            return False
        if self.last_rebalance_date is None:
            return True

        due_at: datetime = datetime.strptime(self.last_rebalance_date, "%Y-%m-%d")
        due_at += REBALANCE_DELTAS[frequency]
        return current_date >= due_at.strftime("%Y-%m-%d")

    # Remember when the universe was last rebuilt.
    def mark_rebalanced(self, current_date: str) -> None:
        self.last_rebalance_date = current_date

    # The equity curve as a list of dicts, for JSON responses.
    def get_equity_curve_dicts(self) -> list[dict]:
        dates, equities, cashes = self.equity_curve
        return [
            {"date": d, "equity": float(eq), "cash": float(c)}
            for d, eq, c in zip(dates, equities, cashes)
        ]

    # -------- Helpers --------
    # Validate the constraints, price the fill and record the trade.
    # Returns the unfilled remainder to carry, or None when the order is done.
    def _fill_order(self, order: Order, open_price: float, current_date: str, volume: float | None = None) -> Order | None:
        ticker: str = order.ticker
        ticker_lots: list[Trade] = self.lots(ticker)

        if open_price <= 0:
            self._reject(f"Invalid open price for ticker: {ticker} | open: {open_price}")
            return None

        equity_at_entry: float = self.portfolio_value
        resolved: float = order.calculate_weight(equity_at_entry, self.cash)

        # An available-cash order with no cash left is a no-op, not a rejection.
        if resolved <= 0.0 and order.sizing is OrderSizing.AVAILABLE:
            return None

        # A fill pays its own fee out of the notional, so an oversized order
        # downsizes to the cash instead of dying on a hairline shortfall.
        allocation: float = self._affordable_allocation(resolved, equity_at_entry)

        if not self._validate_order_constraints(order, ticker, ticker_lots, allocation):
            return None

        # Participation cap: one fill takes at most this share of the bar's
        # volume, and the rest rejoins the queue for the next open.
        fill_allocation: float = allocation
        if self.MAX_PARTICIPATION is not None and volume is not None and equity_at_entry > 0:
            allowed: float = volume * open_price * self.MAX_PARTICIPATION / equity_at_entry
            if allowed <= 0.0:
                return order
            fill_allocation = min(allocation, allowed)

        # What the bar would not take, in equity fractions. A fill smaller than
        # a cent would print dust lots day after day, so the order waits whole.
        unfilled: float = allocation - fill_allocation
        if unfilled * equity_at_entry >= 0.01 > fill_allocation * equity_at_entry:
            return order

        # Slippage and spread both move against the position. Spread is the
        # one-way cost, half the quoted spread, tracked apart from slippage.
        if order.position_type is PositionType.LONG:
            slipped_price: float = open_price * (1.0 + self.SLIPPAGE_PCT)
            raw_entry: float = slipped_price * (1.0 + self.SPREAD_PCT)
        else:
            slipped_price = open_price * (1.0 - self.SLIPPAGE_PCT)
            raw_entry = slipped_price * (1.0 - self.SPREAD_PCT)
        entry_price: float = round(raw_entry, 2)
        allocation = fill_allocation
        entry_fee: float = round(equity_at_entry * allocation * self.FEE_RATE, 4)

        history: StockData | None = self._histories.get(ticker)
        if history is None:
            history = get_stock_history(ticker, end=current_date)
        else:
            # Slice the live history up to today so the trade never sees the future
            end: int = int(np.searchsorted(history.date_codes, to_code(current_date), side="right"))
            history = StockData(
                ticker=ticker,
                date_codes=history.date_codes[:end],
                open_price=history.open_price[:end],
                high=history.high[:end],
                low=history.low[:end],
                close_price=history.close_price[:end],
                volume=history.volume[:end],
            )

        trade: Trade = Trade(
            id=self._next_trade_id,
            ticker=ticker,
            history=history,
            position_type=order.position_type,
            entry_date=current_date,
            entry_price=entry_price,
            allocation=allocation,
            equity_at_entry=equity_at_entry,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            fees=entry_fee,
        )

        self._next_trade_id += 1
        self._trade_by_id[trade.id] = trade
        self.trades.append(trade)
        self.open_trades[trade.id] = trade
        self._open_trades_by_ticker.setdefault(ticker, []).append(trade)
        self._open_tickers.add(ticker)

        notional: float = trade.equity_at_entry * trade.allocation
        shares: float = notional / entry_price
        self.total_slippage += shares * abs(slipped_price - open_price)
        self.total_spread += shares * abs(raw_entry - slipped_price)
        self.traded_notional += notional
        self._invested_by_ticker[ticker] = self._invested_by_ticker.get(ticker, 0.0) + notional
        self._invested_notional += notional
        self._strategy_view_dirty = True
        self.last_prices[ticker] = entry_price
        self.total_cost += entry_fee
        self.cash -= (notional + entry_fee)

        # Keep later fills in this batch on equity net of earlier entry fees
        self.portfolio_value -= entry_fee

        # Carry what the bar would not take; a remainder under a cent is dust.
        # Fixed-cash orders carry dollars, the rest carry equity fractions.
        if unfilled * equity_at_entry >= 0.01:
            if order.sizing is OrderSizing.FIXED:
                return replace(order, allocation=order.allocation - fill_allocation * equity_at_entry)
            return replace(order, allocation=order.allocation - fill_allocation)
        return None

    # Realize one lot: exit slippage, fee, event and profit.
    def _fill_exit(self, trade: Trade, price: float, current_date: str, event_type: EventType) -> None:
        if price <= 0:
            self._reject("invalid_price")
            return

        if trade.position_type is PositionType.LONG:
            slipped_price: float = price * (1.0 - self.SLIPPAGE_PCT)
            raw_exit: float = slipped_price * (1.0 - self.SPREAD_PCT)
            direction: float = 1.0
        else:
            slipped_price = price * (1.0 + self.SLIPPAGE_PCT)
            raw_exit = slipped_price * (1.0 + self.SPREAD_PCT)
            direction = -1.0
        exit_price: float = round(raw_exit, 2)

        notional: float = trade.equity_at_entry * trade.allocation
        shares: float = notional / trade.entry_price
        self.traded_notional += shares * exit_price
        self.total_slippage += shares * abs(slipped_price - price)
        self.total_spread += shares * abs(raw_exit - slipped_price)
        self._invested_notional -= notional

        ticker_notional: float = self._invested_by_ticker[trade.ticker] - notional
        if ticker_notional > 1e-12:
            self._invested_by_ticker[trade.ticker] = ticker_notional
        else:
            del self._invested_by_ticker[trade.ticker]

        pct_return: float = direction * (exit_price - trade.entry_price) / trade.entry_price
        exit_fee: float = round(notional * (exit_price / trade.entry_price) * self.FEE_RATE, 4)

        trade.fees = round(trade.fees + exit_fee, 4)
        trade.exit = Event(date=current_date, type=event_type, price=exit_price)

        realized_profit: float = notional * pct_return
        self.cash += (notional + realized_profit - exit_fee)
        self.realized_pnl += realized_profit
        self.total_cost += exit_fee
        self.last_prices[trade.ticker] = exit_price
        del self.open_trades[trade.id]

        ticker_lots: list[Trade] = self._open_trades_by_ticker[trade.ticker]
        ticker_lots.remove(trade)
        if not ticker_lots:
            del self._open_trades_by_ticker[trade.ticker]
            self._open_tickers.remove(trade.ticker)
        self._strategy_view_dirty = True

    # Today's traded volume for the participation cap, or None when unknown.
    def _day_volume(self, ticker: str, current_date: str) -> float | None:
        history: StockData | None = self._histories.get(ticker)
        if history is None or len(history.date_codes) == 0:
            return None
        code: int = int(to_code(current_date))
        index: int = int(np.searchsorted(history.date_codes, code, side="left"))
        if index < len(history.date_codes) and int(history.date_codes[index]) == code:
            return float(history.volume[index])
        return None

    # The opening price for a ticker, or None after recording a refusal.
    def _get_valid_open_price(self, ticker: str, opens: dict[str, float]) -> float | None:
        open_price: float | None = opens.get(ticker)
        if open_price is None or open_price <= 0:
            self._reject(f"No valid bar for ticker: {ticker}")
            return None
        return open_price

    # Check every strategy constraint: lots, positions, allocation and cash.
    def _validate_order_constraints(self, order: Order, ticker: str, ticker_lots: list[Trade], allocation: float) -> bool:
        if allocation <= 0:
            self._reject(f"Invalid order allocation for ticker: {ticker} | allocation: {allocation}")
            return False

        # Some names cannot be borrowed at any price.
        if order.position_type is PositionType.SHORT and ticker in self.HARD_TO_BORROW:
            self._reject(f"Hard to borrow: {ticker}")
            return False

        # Maximum lots per ticker
        max_lots: int | None = self.strategy.MAX_LOTS_PER_TICKER
        if max_lots is not None and len(ticker_lots) >= max_lots:
            self._reject(f"Maximum lots reached for ticker: {ticker} | lots: {max_lots}")
            return False

        # Maximum overall positions
        max_positions: int | None = self.strategy.MAX_POSITIONS
        if max_positions is not None and not ticker_lots and self.active_positions >= max_positions:
            self._reject("Strategy has filled maximum number of positions")
            return False

        # Maximum allocation per ticker
        ticker_allocation: float = round(self._invested_by_ticker.get(ticker, 0.0) / self.portfolio_value, 2)
        max_allocation: float | None = self.strategy.MAX_ALLOCATION_PER_TICKER
        if max_allocation is not None and ticker_allocation + allocation > max_allocation:
            self._reject(
                f"Max allocation exceeded for {ticker}: "
                f"Total {ticker_allocation + allocation:.2%} > Limit {max_allocation:.2%}"
            )
            return False

        # Long entries are funded from cash; shorts only have to stay inside total equity.
        invested: float = self._invested_notional / self.portfolio_value
        if order.position_type is PositionType.LONG and self.portfolio_value > 0:
            available_allocation: float = self.cash / self.portfolio_value
        else:
            available_allocation = 1.0 - invested
        if allocation > available_allocation + 1e-12:
            self._reject(f"Not enough equity to allocate {ticker}")
            return False

        return True

    # The fraction of equity a fill can take once its own fee is paid out of the cash.
    def _affordable_allocation(self, wanted: float, equity_at_entry: float) -> float:
        if equity_at_entry <= 0:
            return wanted

        return min(wanted, self.cash / (equity_at_entry * (1.0 + self.FEE_RATE)))

    # One open lot by id.
    def _find_trade(self, trade_id: int) -> Trade | None:
        return self._trade_by_id.get(trade_id)

    # Count a refusal, so a strategy can see why nothing happened.
    def _reject(self, reason: str) -> None:
        self.rejected_orders[reason] = self.rejected_orders.get(reason, 0) + 1

    # Which resting level, if any, today's bar touched, and at what price.
    @staticmethod
    def _check_exit(trade: Trade, current_open: float, current_high: float, current_low: float) -> tuple[float, EventType | None]:
        stop: float = trade.stop_loss
        target: float = trade.take_profit

        if trade.position_type is PositionType.LONG:
            if stop > 0 and current_low <= stop:
                return min(current_open, stop), EventType.STOP_LOSS
            if target > 0 and current_high >= target:
                return max(current_open, target), EventType.TAKE_PROFIT
        else:
            if stop > 0 and current_high >= stop:
                return max(current_open, stop), EventType.STOP_LOSS
            if target > 0 and current_low <= target:
                return min(current_open, target), EventType.TAKE_PROFIT

        return 0.0, None
