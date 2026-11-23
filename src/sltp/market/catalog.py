# -------- catalog.py --------
# The symbol categories, as plain constants, so importing this stays cheap:
# no pandas and no network. symbol_categories.py does the classification work.

# -------- Constants --------
CATEGORY_COLUMN: str = "Category"

STOCK: str = "stock"
ETF: str = "etf"
GOLD: str = "gold"

# Physical-gold and leveraged/inverse gold funds, exposed to strategies as gold proxies.
GOLD_FUNDS = frozenset({
    "GLD",    # SPDR Gold Shares
    "IAU",    # iShares Gold Trust
    "GLDM",   # SPDR Gold MiniShares Trust
    "SGOL",   # abrdn Physical Gold Shares
    "AAAU",   # Goldman Sachs Physical Gold
    "BAR",    # GraniteShares Gold Trust
    "OUNZ",   # VanEck Merk Gold Trust
    "UGL",    # ProShares Ultra Gold (2x)
    "GLL",    # ProShares UltraShort Gold (-1x)
    "DGP",    # DB Gold Double Long
    "DZZ",    # DB Gold Double Short
})

# Fallback for symbols the US listings miss. Weak words need no company marker.
STRONG_FUND_WORDS: tuple[str, ...] = ("etf", " fund", "trust fund")
WEAK_FUND_WORDS: tuple[str, ...] = ("trust", "index", "portfolio")
COMPANY_MARKERS: tuple[str, ...] = (
    " inc", " corp", " company", " ltd", " plc", " reit", " mortgage",
    " realty", " bancorp", " bankshares", " notes", " holdings",
)
