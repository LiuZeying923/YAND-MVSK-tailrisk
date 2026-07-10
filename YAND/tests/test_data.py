"""Market-data layer: catalogue search, market inference, offline synthetics."""

from __future__ import annotations

import numpy as np
import pytest

from yand_mvsk.data import catalog, fetch_prices
from yand_mvsk.data.market import MarketDataError, currency_of, infer_market


@pytest.mark.parametrize("sym,mkt", [
    ("AAPL", "US"), ("SPY", "US"),
    ("0700.HK", "HK"), ("9988.HK", "HK"),
    ("600519.SS", "A"), ("000858.SZ", "A"),
])
def test_infer_market(sym, mkt):
    assert infer_market(sym) == mkt


def test_currency_of():
    assert currency_of("AAPL") == "USD"
    assert currency_of("0700.HK") == "HKD"
    assert currency_of("600519.SS") == "CNY"


def test_catalog_covers_three_markets():
    markets = {r["market"] for r in catalog.as_records()}
    assert {"US", "HK", "A"} <= markets


def test_catalog_search():
    assert any(r["symbol"] == "AAPL" for r in catalog.search("apple"))
    assert any(r["symbol"] == "0700.HK" for r in catalog.search("tencent"))
    assert catalog.search("600519")[0]["symbol"] == "600519.SS"


def test_offline_synthetic_is_deterministic():
    a = fetch_prices(["AAPL", "0700.HK", "600519.SS"], offline=True)
    b = fetch_prices(["AAPL", "0700.HK", "600519.SS"], offline=True)
    assert a.source == "synthetic"
    assert np.allclose(a.prices.to_numpy(), b.prices.to_numpy())
    assert list(a.prices.columns) == ["AAPL", "0700.HK", "600519.SS"]


def test_offline_prices_positive_and_aligned():
    pd = fetch_prices(["AAPL", "MSFT", "TLT", "GLD"], offline=True, n_days=300)
    assert pd.prices.shape == (300, 4)
    assert (pd.prices.to_numpy() > 0).all()
    assert pd.prices.index.is_monotonic_increasing


def test_meta_flags_market_and_currency():
    pd = fetch_prices(["AAPL", "0700.HK"], offline=True)
    assert pd.meta["AAPL"]["currency"] == "USD"
    assert pd.meta["0700.HK"]["market_label"] == "Hong Kong"
    assert pd.meta["AAPL"]["synthetic"] is True


def test_empty_symbols_raises():
    with pytest.raises(MarketDataError):
        fetch_prices([], offline=True)


def test_deduplicates_symbols():
    pd = fetch_prices(["AAPL", "AAPL", "MSFT"], offline=True)
    assert list(pd.prices.columns) == ["AAPL", "MSFT"]
