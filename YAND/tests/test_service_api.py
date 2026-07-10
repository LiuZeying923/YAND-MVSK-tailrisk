"""End-to-end service pipeline and HTTP API (all offline / synthetic)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from yand_mvsk.api.app import app
from yand_mvsk.api.service import optimize_portfolio

client = TestClient(app)

BASKET = ["AAPL", "MSFT", "0700.HK", "600519.SS", "TLT", "GLD"]


# ---- service ---------------------------------------------------------------
def test_optimize_pipeline_offline():
    out = optimize_portfolio(BASKET, gamma=8.0, offline=True, max_weight=0.35)
    assert out["data_source"] == "synthetic"
    w = {r["symbol"]: r["weight"] for r in out["weights"]}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert max(w.values()) <= 0.35 + 1e-6
    assert 0 <= out["risk"]["score"] <= 100
    assert out["risk"]["stress"]
    assert len(out["equity_curve"]) == out["n_observations"] - 1


def test_optimize_with_questionnaire():
    out = optimize_portfolio(BASKET, answers={"Q1": "C", "Q2": "A", "Q3": "C", "Q4": "D", "Q5": "C"},
                             offline=True)
    assert out["gamma_breakdown"]["profile"] == "Conservative"
    assert out["gamma"] == out["gamma_breakdown"]["gamma"]


def test_optimize_infeasible_cap_relaxed():
    # cap too small for the basket -> service relaxes rather than erroring
    out = optimize_portfolio(BASKET, gamma=6.0, offline=True, max_weight=0.05)
    assert out["request"]["max_weight"] > 0.05


def test_recommendation_only_when_it_helps():
    risky = optimize_portfolio(["NVDA", "TSLA", "300750.SZ", "1211.HK"],
                               gamma=2.5, offline=True, max_weight=0.6)
    # elevated basket -> a recommendation should be present
    assert risky["recommendation"] is not None
    robust = optimize_portfolio(["TLT", "GLD", "AAPL", "MSFT"], gamma=15.0, offline=True, max_weight=0.4)
    # if robust, no upsell
    if robust["risk"]["level"] == "robust":
        assert robust["recommendation"] is None


def test_empty_tickers_raises():
    with pytest.raises(ValueError):
        optimize_portfolio([], offline=True)


# ---- API -------------------------------------------------------------------
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_questionnaire_endpoint():
    r = client.get("/api/questionnaire")
    body = r.json()
    assert len(body["questions"]) == 5
    assert len(body["profiles"]) >= 3


def test_gamma_endpoint():
    r = client.post("/api/gamma", json={"answers": {"Q1": "B", "Q2": "B", "Q3": "A", "Q4": "A", "Q5": "A"}})
    assert r.status_code == 200
    assert r.json()["profile"]


def test_gamma_endpoint_invalid():
    r = client.post("/api/gamma", json={"answers": {"Q1": "Z"}})
    assert r.status_code == 422


def test_search_endpoint():
    r = client.get("/api/search", params={"q": "tencent"})
    assert any(x["symbol"] == "0700.HK" for x in r.json()["results"])


def test_optimize_endpoint():
    r = client.post("/api/optimize", json={
        "tickers": BASKET, "gamma": 8.0, "offline": True, "max_weight": 0.35, "horizon": 21,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["weights"] and body["risk"] and body["performance"]


def test_optimize_endpoint_validation():
    r = client.post("/api/optimize", json={"tickers": []})
    assert r.status_code == 422


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "YAND-MVSK" in r.text
