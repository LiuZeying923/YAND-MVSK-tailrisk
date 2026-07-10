"""Multi-market price data: HK / US / A-share via yfinance, with a fully
offline synthetic fallback so the app is genuinely open-and-run.

Design goals
------------
* **Never hard-fail on the network.**  If yfinance is missing, offline, or
  returns nothing for a symbol, we fall back to a deterministic synthetic
  series so the optimiser and the whole UI still work end-to-end.
* **Deterministic synthetics.**  Each symbol seeds its own generator from a hash
  of the ticker, so "AAPL" always produces the same demo series -- reproducible
  screenshots and tests, realistic-looking fat tails and correlations.
* **Aligned output.**  Always returns a price DataFrame indexed by date with one
  column per requested symbol, inner-joined on common dates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import lookup

__all__ = ["infer_market", "PriceData", "fetch_prices", "MarketDataError"]

_MARKET_META = {
    "US": {"currency": "USD", "ann_drift": 0.09, "ann_vol": 0.22, "label": "US"},
    "HK": {"currency": "HKD", "ann_drift": 0.06, "ann_vol": 0.28, "label": "Hong Kong"},
    "A": {"currency": "CNY", "ann_drift": 0.07, "ann_vol": 0.30, "label": "A-share (China)"},
}


class MarketDataError(RuntimeError):
    pass


def infer_market(symbol: str) -> str:
    """Infer market from the symbol suffix (US default)."""
    s = symbol.upper()
    if s.endswith(".HK"):
        return "HK"
    if s.endswith(".SS") or s.endswith(".SZ") or s.endswith(".SH"):
        return "A"
    return "US"


def currency_of(symbol: str) -> str:
    return _MARKET_META[infer_market(symbol)]["currency"]


@dataclass
class PriceData:
    """Aligned close prices plus provenance."""

    prices: pd.DataFrame           # index = date, columns = symbols
    source: str                    # "yfinance" | "synthetic" | "mixed"
    synthetic_symbols: list[str]
    meta: dict

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)


def _seed_from_symbol(symbol: str) -> int:
    h = hashlib.sha256(symbol.upper().encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _synthetic_series(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """A single deterministic synthetic close-price series with fat tails.

    Market factor + idiosyncratic component, Student-t innovations, and a couple
    of embedded drawdown regimes so tail-risk analytics have something to chew
    on.  Purely for demo/offline use -- clearly labelled as synthetic upstream.
    """
    market = infer_market(symbol)
    meta = _MARKET_META[market]
    n = len(index)
    rng = np.random.default_rng(_seed_from_symbol(symbol))

    dt = 1.0 / 252.0
    sig = meta["ann_vol"] * np.sqrt(dt)

    # Shared market factor (seeded per market so names within a market co-move).
    mrng = np.random.default_rng(_seed_from_symbol("FACTOR::" + market))
    factor = mrng.standard_t(df=5, size=n) * (meta["ann_vol"] * 0.7 * np.sqrt(dt))

    beta = 0.6 + 0.8 * rng.random()
    idio = rng.standard_t(df=4, size=n) * sig * 0.8
    noise = beta * factor + idio  # zero-mean, fat-tailed log-return shocks

    # Inject two crash regimes at deterministic fractions of the window so the
    # tail-risk analytics have realistic drawdowns to detect.
    for frac, depth, length in ((0.35, 0.16, 18), (0.72, 0.11, 12)):
        start = int(frac * n)
        end = min(n, start + length)
        noise[start:end] -= depth / max(1, (end - start))

    # Set the drift so the *geometric* path targets ``ann_drift``, compensating
    # for volatility drag (0.5 * var) from the fat-tailed shocks.  Without this,
    # exponentiating high-variance noise would pull every asset persistently
    # downward regardless of intended drift.
    daily_geo = meta["ann_drift"] * dt
    log_ret = noise - noise.mean() + daily_geo + 0.5 * float(np.var(noise))
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=index, name=symbol)


def _synthetic_prices(symbols: list[str], n_days: int, end: pd.Timestamp | None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().normalize()
    index = pd.bdate_range(end=end, periods=n_days)
    cols = {s: _synthetic_series(s, index) for s in symbols}
    return pd.DataFrame(cols)


def _try_yfinance(symbols: list[str], period: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download(
            symbols, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None
    # Normalise the (possibly MultiIndex) frame down to a Close-price matrix.
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.levels[0][0]
        close = raw[field]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw
        if len(symbols) == 1:
            close.columns = symbols
    close = close.reindex(columns=symbols)
    close = close.dropna(how="all")
    return close


def fetch_prices(
    symbols: list[str],
    period: str = "3y",
    n_days: int = 520,
    offline: bool = False,
    min_obs: int = 60,
    end=None,
) -> PriceData:
    """Fetch aligned daily close prices for ``symbols``.

    Falls back to a synthetic series per symbol that yfinance cannot supply.
    Always returns at least ``min_obs`` aligned rows (top-up with synthetics if
    a live fetch is too short).
    """
    symbols = [s.strip() for s in symbols if s and s.strip()]
    if not symbols:
        raise MarketDataError("no symbols provided")
    # De-duplicate, preserve order.
    seen: dict[str, None] = {}
    for s in symbols:
        seen.setdefault(s, None)
    symbols = list(seen)

    live: pd.DataFrame | None = None
    if not offline:
        live = _try_yfinance(symbols, period)

    synthetic_symbols: list[str] = []
    series: dict[str, pd.Series] = {}

    if live is not None and live.shape[0] >= min_obs:
        idx = live.index
        for s in symbols:
            col = live[s] if s in live.columns else None
            if col is not None and col.notna().sum() >= min_obs:
                series[s] = col.ffill().bfill()
            else:
                synthetic_symbols.append(s)
                series[s] = _synthetic_series(s, pd.DatetimeIndex(idx))
        source = "yfinance" if not synthetic_symbols else "mixed"
        prices = pd.DataFrame(series).dropna()
    else:
        prices = _synthetic_prices(symbols, n_days, end)
        synthetic_symbols = list(symbols)
        source = "synthetic"

    prices = prices.dropna()
    if prices.shape[0] < min_obs:
        raise MarketDataError(
            f"only {prices.shape[0]} aligned observations (< {min_obs}); "
            "try fewer symbols or a longer period"
        )

    meta = {
        s: {
            "market": infer_market(s),
            "market_label": _MARKET_META[infer_market(s)]["label"],
            "currency": currency_of(s),
            "synthetic": s in synthetic_symbols,
            "catalog": lookup(s),
        }
        for s in prices.columns
    }
    return PriceData(prices=prices, source=source, synthetic_symbols=synthetic_symbols, meta=meta)
