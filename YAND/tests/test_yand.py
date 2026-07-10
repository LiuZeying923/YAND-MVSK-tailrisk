"""YAND solver: feasibility, constraints, and agreement with scipy / brute force."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from yand_mvsk.linalg import project_capped_simplex
from yand_mvsk.moments import MVSKObjective, crra_coefficients, horizon_returns
from yand_mvsk.yand import solve_yand


def _simplex_cons(n):
    return [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]


def test_feasibility(returns):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    res = solve_yand(obj, cap=0.4)
    assert res.w.sum() == pytest.approx(1.0, abs=1e-9)
    assert res.w.min() >= -1e-9
    assert res.w.max() <= 0.4 + 1e-9


def test_quadratic_matches_scipy_meanvariance(returns):
    """With c3 = c4 = 0 the objective is a QP; YAND must match SLSQP."""
    obj = MVSKObjective(returns, np.array([1.0, 4.0, 0.0, 0.0]))
    n = obj.n
    res = solve_yand(obj, cap=1.0, multistart=False)
    sp = minimize(obj.value, np.full(n, 1 / n), jac=lambda w: obj.value_grad(w)[1],
                  bounds=[(0, 1)] * n, constraints=_simplex_cons(n),
                  method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    assert res.f <= sp.fun + 1e-8
    assert np.max(np.abs(res.w - sp.x)) < 1e-4


def test_full_mvsk_beats_or_matches_scipy_multistart(returns, rng):
    obj = MVSKObjective(returns, crra_coefficients(9.0))
    n = obj.n
    res = solve_yand(obj, cap=0.35)
    best = np.inf
    for _ in range(25):
        s = project_capped_simplex(rng.random(n), 0.35)
        r = minimize(obj.value, s, jac=lambda w: obj.value_grad(w)[1],
                     bounds=[(0, 0.35)] * n, constraints=_simplex_cons(n),
                     method="SLSQP", options={"maxiter": 800, "ftol": 1e-13})
        if r.success:
            best = min(best, r.fun)
    assert res.f <= best + 1e-6


def test_higher_gamma_lowers_variance(prices):
    """Economic sanity: more risk aversion => less variance."""
    R = horizon_returns(prices, 21)
    variances = []
    for g in (2.0, 6.0, 12.0, 20.0):
        obj = MVSKObjective(R, crra_coefficients(g))
        res = solve_yand(obj, cap=0.5)
        variances.append(obj.central_moments(res.w)[1])
    for a, b in zip(variances, variances[1:]):
        assert b <= a + 1e-9


def test_cap_forces_diversification(returns):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    res = solve_yand(obj, cap=0.2)
    assert (res.w > 1e-4).sum() >= 5  # at least 1/0.2 names must be active


def test_single_asset_degenerate():
    R = np.random.default_rng(0).standard_normal((60, 1)) * 0.01
    obj = MVSKObjective(R, crra_coefficients(6.0))
    res = solve_yand(obj)
    assert res.w[0] == pytest.approx(1.0)


def test_infeasible_cap_raises(returns):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    with pytest.raises(ValueError):
        solve_yand(obj, cap=0.05)  # 0.05 * 10 < 1


def test_result_serializable(returns):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    d = solve_yand(obj, cap=0.4).as_dict()
    assert {"objective", "iterations", "converged", "seconds"} <= d.keys()
