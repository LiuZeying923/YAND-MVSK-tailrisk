"""Oracle correctness: gradients, Hessian-vector products, and the line poly."""

from __future__ import annotations

import numpy as np
import pytest

from yand_mvsk.linalg import project_capped_simplex
from yand_mvsk.moments import (
    MVSKObjective,
    crra_coefficients,
    horizon_returns,
)


def test_crra_coefficients():
    c = crra_coefficients(6.0)
    assert c[0] == 1.0
    assert c[1] == pytest.approx(3.0)          # gamma/2
    assert c[2] == pytest.approx(6 * 7 / 6)    # gamma(gamma+1)/6 = 7
    assert c[3] == pytest.approx(6 * 7 * 8 / 24)  # = 14
    with pytest.raises(ValueError):
        crra_coefficients(0.0)


def test_horizon_returns_buy_and_hold(prices):
    r1 = horizon_returns(prices, 1)
    assert r1.shape == (prices.shape[0] - 1, prices.shape[1])
    # 5-day horizon return equals compounded product of daily minus 1.
    r5 = horizon_returns(prices, 5)
    manual = prices[5:] / prices[:-5] - 1.0
    assert np.allclose(r5, manual)


def test_horizon_returns_validation():
    with pytest.raises(ValueError):
        horizon_returns(np.array([[1.0, 2.0]]), 5)
    with pytest.raises(ValueError):
        horizon_returns(np.full((10, 3), -1.0), 1)  # non-positive prices


def test_gradient_matches_finite_difference(returns, rng):
    obj = MVSKObjective(returns, crra_coefficients(8.0))
    w = project_capped_simplex(rng.random(obj.n), 1.0)
    _, g, _ = obj.value_grad(w)
    eps = 1e-6
    gfd = np.array([
        (obj.value(w + eps * e) - obj.value(w - eps * e)) / (2 * eps)
        for e in np.eye(obj.n)
    ])
    assert np.max(np.abs(g - gfd)) < 1e-6


def test_hessian_vec_matches_finite_difference(returns, rng):
    obj = MVSKObjective(returns, crra_coefficients(8.0))
    w = project_capped_simplex(rng.random(obj.n), 1.0)
    _, _, z = obj.value_grad(w)
    v = rng.standard_normal(obj.n)
    hv = obj.hess_vec(z, v)
    eps = 1e-6
    hvfd = (obj.value_grad(w + eps * v)[1] - obj.value_grad(w - eps * v)[1]) / (2 * eps)
    assert np.max(np.abs(hv - hvfd)) < 1e-5


def test_quartic_line_is_exact(returns, rng):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    w = project_capped_simplex(rng.random(obj.n), 1.0)
    d = rng.standard_normal(obj.n)
    d -= d.mean()  # tangent to the simplex
    line = obj.line(w, d)
    for a in np.linspace(-0.5, 0.5, 25):
        assert line(a) == pytest.approx(obj.value(w + a * d), abs=1e-10)


def test_line_minimize_on_interval():
    from yand_mvsk.moments import QuarticLine
    # phi(a) = (a-2)^2 + 1 embedded in a quartic with A3=A4=0
    line = QuarticLine(A0=5.0, A1=-4.0, A2=1.0, A3=0.0, A4=0.0)
    a, v = line.minimize_on(0.0, 5.0)
    assert a == pytest.approx(2.0, abs=1e-6)
    assert v == pytest.approx(1.0, abs=1e-6)
    # clipped to interval
    a2, _ = line.minimize_on(3.0, 5.0)
    assert a2 == pytest.approx(3.0)


def test_central_moments_sign(returns):
    obj = MVSKObjective(returns, crra_coefficients(6.0))
    w = np.full(obj.n, 1.0 / obj.n)
    m1, m2, m3, m4 = obj.central_moments(w)
    assert m2 > 0 and m4 > 0  # variance & 4th moment positive


def test_rejects_short_or_negative_inputs(rng):
    with pytest.raises(ValueError):
        MVSKObjective(rng.standard_normal((4, 3)), crra_coefficients(6))  # < 8 obs
    with pytest.raises(ValueError):
        MVSKObjective(rng.standard_normal((50, 3)), np.array([1, -1, 0, 0]))  # negative c
