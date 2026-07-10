"""A curated, offline catalogue of popular tickers across HK / US / A-share.

This is intentionally small and hand-picked -- enough to power a pleasant search
box and to let the app run with zero external calls.  It is NOT an exhaustive
security master; any valid yfinance symbol the user types is still accepted by
the resolver in ``market.py``.
"""

from __future__ import annotations

# (symbol, display name, market, sector)
CATALOG: list[tuple[str, str, str, str]] = [
    # ---- US ----
    ("AAPL", "Apple", "US", "Technology"),
    ("MSFT", "Microsoft", "US", "Technology"),
    ("NVDA", "NVIDIA", "US", "Semiconductors"),
    ("GOOGL", "Alphabet (Google)", "US", "Technology"),
    ("AMZN", "Amazon", "US", "Consumer"),
    ("META", "Meta Platforms", "US", "Technology"),
    ("TSLA", "Tesla", "US", "Autos"),
    ("BRK-B", "Berkshire Hathaway", "US", "Financials"),
    ("JPM", "JPMorgan Chase", "US", "Financials"),
    ("V", "Visa", "US", "Financials"),
    ("JNJ", "Johnson & Johnson", "US", "Healthcare"),
    ("WMT", "Walmart", "US", "Consumer"),
    ("XOM", "Exxon Mobil", "US", "Energy"),
    ("KO", "Coca-Cola", "US", "Consumer"),
    ("SPY", "SPDR S&P 500 ETF", "US", "ETF"),
    ("QQQ", "Invesco QQQ (Nasdaq 100)", "US", "ETF"),
    ("TLT", "iShares 20+ Yr Treasury ETF", "US", "Bonds"),
    ("GLD", "SPDR Gold Shares", "US", "Commodity"),
    ("IEF", "iShares 7-10 Yr Treasury ETF", "US", "Bonds"),
    ("VNQ", "Vanguard Real Estate ETF", "US", "REIT"),
    # ---- Hong Kong ----
    ("0700.HK", "Tencent Holdings", "HK", "Technology"),
    ("9988.HK", "Alibaba Group", "HK", "Technology"),
    ("0941.HK", "China Mobile", "HK", "Telecom"),
    ("1299.HK", "AIA Group", "HK", "Insurance"),
    ("0005.HK", "HSBC Holdings", "HK", "Financials"),
    ("3690.HK", "Meituan", "HK", "Technology"),
    ("1810.HK", "Xiaomi", "HK", "Technology"),
    ("9618.HK", "JD.com", "HK", "Technology"),
    ("2318.HK", "Ping An Insurance", "HK", "Insurance"),
    ("0388.HK", "HK Exchanges & Clearing", "HK", "Financials"),
    ("1211.HK", "BYD", "HK", "Autos"),
    ("2020.HK", "ANTA Sports", "HK", "Consumer"),
    ("0883.HK", "CNOOC", "HK", "Energy"),
    ("2800.HK", "Tracker Fund of Hong Kong", "HK", "ETF"),
    # ---- A-share (Shanghai .SS / Shenzhen .SZ) ----
    ("600519.SS", "Kweichow Moutai", "A", "Consumer"),
    ("601318.SS", "Ping An Insurance", "A", "Insurance"),
    ("600036.SS", "China Merchants Bank", "A", "Financials"),
    ("600900.SS", "China Yangtze Power", "A", "Utilities"),
    ("601899.SS", "Zijin Mining", "A", "Materials"),
    ("600276.SS", "Jiangsu Hengrui Pharma", "A", "Healthcare"),
    ("688981.SS", "SMIC", "A", "Semiconductors"),
    ("601012.SS", "LONGi Green Energy", "A", "Energy"),
    ("000858.SZ", "Wuliangye Yibin", "A", "Consumer"),
    ("300750.SZ", "CATL", "A", "Batteries"),
    ("000333.SZ", "Midea Group", "A", "Consumer"),
    ("002594.SZ", "BYD", "A", "Autos"),
    ("000001.SZ", "Ping An Bank", "A", "Financials"),
    ("300760.SZ", "Mindray Medical", "A", "Healthcare"),
    ("510300.SS", "CSI 300 ETF", "A", "ETF"),
]

MARKET_LABELS = {"US": "US", "HK": "Hong Kong", "A": "A-share (China)"}

_BY_SYMBOL = {row[0].upper(): row for row in CATALOG}


def as_records() -> list[dict]:
    return [
        {"symbol": s, "name": n, "market": m, "market_label": MARKET_LABELS[m], "sector": sec}
        for (s, n, m, sec) in CATALOG
    ]


def lookup(symbol: str) -> dict | None:
    row = _BY_SYMBOL.get(symbol.upper())
    if not row:
        return None
    s, n, m, sec = row
    return {"symbol": s, "name": n, "market": m, "market_label": MARKET_LABELS[m], "sector": sec}


def search(query: str, limit: int = 12) -> list[dict]:
    """Rank catalogue entries by a simple relevance score against ``query``."""
    q = query.strip().upper()
    if not q:
        return as_records()[:limit]
    scored = []
    for s, n, m, sec in CATALOG:
        su, nu = s.upper(), n.upper()
        if su == q or su.split(".")[0] == q:
            score = 0
        elif su.startswith(q) or nu.startswith(q):
            score = 1
        elif q in su or q in nu:
            score = 2
        elif q in sec.upper():
            score = 3
        else:
            continue
        scored.append((score, n, {"symbol": s, "name": n, "market": m,
                                   "market_label": MARKET_LABELS[m], "sector": sec}))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [rec for _, _, rec in scored[:limit]]
