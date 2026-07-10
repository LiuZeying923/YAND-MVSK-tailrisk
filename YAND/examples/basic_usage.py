"""Basic usage: optimize a multi-market basket and print the report.

Run from the repo root:
    PYTHONPATH=src python examples/basic_usage.py
"""

from yand_mvsk import EfficientMVSK
from yand_mvsk.data import fetch_prices

# A basket spanning US, Hong Kong and A-share markets.
tickers = ["AAPL", "MSFT", "NVDA", "0700.HK", "9988.HK", "600519.SS", "TLT", "GLD"]

# offline=True uses deterministic synthetic data (no network needed).
# Set offline=False to pull real prices via yfinance.
data = fetch_prices(tickers, offline=True)
print(f"Loaded {data.prices.shape[0]} days for {len(tickers)} assets ({data.source} data).\n")

# Optimize on 21-day horizon returns with gamma=6 (moderate risk aversion),
# capped at 35% per name.
ef = EfficientMVSK.from_prices(
    data.prices, gamma=6.0, horizon=21, tickers=list(data.prices.columns), max_weight=0.35
)
ef.optimize()

print("Cleaned weights:")
for sym, w in sorted(ef.clean_weights().items(), key=lambda kv: -kv[1]):
    print(f"  {sym:12s} {w*100:6.2f}%")
print()

ef.portfolio_performance(verbose=True)
