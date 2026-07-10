"""Tail-risk analytics: metrics, monitor, stress, and the guard."""

from __future__ import annotations

import numpy as np
import pytest

from yand_mvsk.risk import (
    RiskMonitor,
    StressTester,
    TailRiskGuard,
    tail_contributions,
    tail_metrics,
)


def test_tail_metrics_basic(rng):
    r = rng.standard_normal(500) * 0.01 + 0.0003
    m = tail_metrics(r, periods_per_year=252)
    assert m.n_obs == 500
    assert m.cvar_95 >= m.var_95 >= 0        # CVaR at least as deep as VaR
    assert m.ann_volatility == pytest.approx(m.volatility * np.sqrt(252), rel=1e-6)
    assert -1 < m.skewness < 1
    assert m.max_drawdown <= 0


def test_tail_metrics_requires_data():
    with pytest.raises(ValueError):
        tail_metrics(np.array([0.1, 0.2]))


def test_cvar_deeper_for_fat_tails(rng):
    thin = rng.standard_normal(2000) * 0.01
    fat = rng.standard_t(df=3, size=2000) * 0.01
    assert tail_metrics(fat).cvar_95 > tail_metrics(thin).cvar_95


def test_monitor_flags_negative_skew_fat_tail():
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0005, 0.008, 120)
    crash = np.concatenate([rng.normal(0.0005, 0.008, 90),
                            np.array([-0.06, -0.09, -0.05]),
                            rng.normal(0.0005, 0.008, 7)])
    mon = RiskMonitor(window=60)
    assert mon.latest(crash).alert
    # a calm window should generally not alert
    assert not mon.latest(calm).alert


def test_stress_synthetic_shock(rng):
    R = rng.standard_normal((252, 5)) * 0.02
    w = np.full(5, 0.2)
    res = StressTester.synthetic_shock(R, w, shock_vol=3.0)
    assert res.portfolio_return < 0
    assert res.n_obs == 1


def test_stress_scenario_by_date():
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    df = pd.DataFrame(np.full((len(idx), 2), -0.01), index=idx, columns=["A", "B"])
    st = StressTester()
    res = st.scenario(df, np.array([0.5, 0.5]),
                      {"name": "COVID", "start": "2020-02-19", "end": "2020-03-23",
                       "description": "x"})
    assert res is not None and res.portfolio_return < 0
    # window outside data -> None
    assert st.scenario(df, np.array([0.5, 0.5]),
                       {"name": "z", "start": "2019-01-01", "end": "2019-02-01"}) is None


def test_guard_score_monotone_in_cvar():
    """A strictly less tail-risky series must not score worse."""
    rng = np.random.default_rng(3)
    mild = rng.normal(0.0004, 0.008, 400)
    wild = rng.standard_t(df=3, size=400) * 0.02 + 0.0004
    guard = TailRiskGuard()
    s_mild = guard.evaluate(mild).score
    s_wild = guard.evaluate(wild).score
    assert s_mild >= s_wild


def test_guard_verdict_fields(rng):
    r = rng.standard_normal(400) * 0.012 + 0.0003
    v = TailRiskGuard().evaluate(r, asset_returns=rng.standard_normal((400, 3)) * 0.02,
                                 weights=np.full(3, 1 / 3))
    d = v.as_dict()
    assert 0 <= d["score"] <= 100
    assert d["level"] in {"robust", "watch", "elevated", "critical"}
    assert d["findings"]
    assert d["stress"]  # synthetic shock fallback guarantees at least one


def test_tail_contributions_sum_to_portfolio_cvar(rng):
    R = rng.standard_t(df=5, size=(600, 4)) * 0.015
    w = np.array([0.4, 0.3, 0.2, 0.1])
    contrib = tail_contributions(R, w)
    port = R @ w
    q = np.quantile(port, 0.05)
    cvar = port[port <= q].mean()
    assert contrib.sum() == pytest.approx(cvar, rel=1e-6)
