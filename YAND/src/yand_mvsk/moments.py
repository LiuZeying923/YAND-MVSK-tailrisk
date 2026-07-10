"""Tensor-free sample-moment oracles for the MVSK objective.

The whole point of this module is that we never materialise the O(n^3)
coskewness tensor or the O(n^4) cokurtosis tensor.  Every quantity the solver
needs -- objective, gradient, Hessian-vector product, third-order action, and
the exact quartic line-search polynomial -- is obtained from products with the
centred return matrix ``A`` (shape ``T x n``) and elementwise operations on
``z = A w``.  Storage is O(T n).

Objective (Wang/Zhou/Ying/Palomar form, as used by YAND-MVSK):

    f(w) = -c1 * m1(w) + c2 * m2(w) - c3 * m3(w) + c4 * m4(w)

    m1(w) = mu' w
    mp(w) = (1/T) * sum_t (r_t' w - mu' w)^p        for p = 2, 3, 4

which factorises as ``f(w) = -c1 mu'w + (1/T) sum_t psi(z_t)`` with
``psi(s) = c2 s^2 - c3 s^3 + c4 s^4`` and ``z = A w``.

The coefficients come from the fourth-order Taylor expansion of CRRA utility
``u(1+r) = (1+r)^(1-gamma) / (1-gamma)`` around ``1 + E[r]``:

    c = (1, gamma/2, gamma(gamma+1)/6, gamma(gamma+1)(gamma+2)/24)

so a single risk-aversion scalar ``gamma`` fixes the trade-off across all four
moments.  Larger gamma => variance and kurtosis matter more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "crra_coefficients",
    "horizon_returns",
    "MVSKObjective",
    "QuarticLine",
]


def crra_coefficients(gamma: float) -> np.ndarray:
    """Taylor coefficients of CRRA utility at risk aversion ``gamma``."""
    g = float(gamma)
    if g <= 0:
        raise ValueError(f"gamma must be positive, got {g}")
    return np.array(
        [1.0, g / 2.0, g * (g + 1.0) / 6.0, g * (g + 1.0) * (g + 2.0) / 24.0],
        dtype=float,
    )


def horizon_returns(prices: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Overlapping buy-and-hold simple returns over ``horizon`` periods.

    ``prices`` has shape ``(T+horizon, n)``.  The result has shape ``(T, n)``.

    Buy-and-hold matters here: over a multi-period horizon the *portfolio*
    simple return is exactly ``w' r`` where ``r_i = P_i[t+h]/P_i[t] - 1``, so
    the objective stays linear in ``w`` inside ``z = A w`` and the tensor-free
    structure survives.  (Daily-rebalanced returns would not compound this way.)

    Aggregating to a holding horizon is also what makes skewness and kurtosis
    numerically relevant.  On daily decimal returns the ``m3`` and ``m4`` terms
    are ~3 and ~6 orders of magnitude below the ``m2`` term, so they cannot move
    the optimum.  Over a 21-day horizon the four terms are within about one
    order of magnitude of each other.  This is the "frequency mismatch" caveat
    from the project README, handled at the source.
    """
    P = np.asarray(prices, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"prices must be 2-D (T, n), got shape {P.shape}")
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    if P.shape[0] <= h:
        raise ValueError(f"need more than {h} price rows, got {P.shape[0]}")
    if not np.all(np.isfinite(P)) or np.any(P <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return P[h:] / P[:-h] - 1.0


@dataclass
class QuarticLine:
    """``phi(a) = A0 + A1 a + A2 a^2 + A3 a^3 + A4 a^4`` restricted to a ray."""

    A0: float
    A1: float
    A2: float
    A3: float
    A4: float

    def __call__(self, a: float | np.ndarray) -> float | np.ndarray:
        return self.A0 + a * (self.A1 + a * (self.A2 + a * (self.A3 + a * self.A4)))

    def minimize_on(self, lo: float, hi: float) -> tuple[float, float]:
        """Exact minimiser of the quartic on ``[lo, hi]``.

        Checks both endpoints and every real stationary point inside the
        interval.  For a quartic that is all of them, so this is exact, not a
        backtracking approximation.
        """
        cands = [lo, hi]
        # phi'(a) = A1 + 2 A2 a + 3 A3 a^2 + 4 A4 a^3
        deriv = np.array([4.0 * self.A4, 3.0 * self.A3, 2.0 * self.A2, self.A1])
        nz = np.flatnonzero(np.abs(deriv) > 1e-300)
        if nz.size:
            roots = np.roots(deriv[nz[0] :])
            for r in roots:
                if abs(r.imag) < 1e-10 * max(1.0, abs(r.real)):
                    a = float(r.real)
                    if lo < a < hi:
                        cands.append(a)
        vals = [float(self(a)) for a in cands]
        k = int(np.argmin(vals))
        return cands[k], vals[k]


@dataclass
class MVSKObjective:
    """Sample-moment MVSK objective with O(Tn) storage.

    Parameters
    ----------
    R:
        ``(T, n)`` matrix of horizon simple returns (decimal, not percent).
    c:
        Length-4 preference vector ``(c1, c2, c3, c4)``, all non-negative.
    """

    R: np.ndarray
    c: np.ndarray

    mu: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)
    T: int = field(init=False)
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.R = np.ascontiguousarray(self.R, dtype=float)
        if self.R.ndim != 2:
            raise ValueError(f"R must be 2-D, got shape {self.R.shape}")
        self.c = np.asarray(self.c, dtype=float).reshape(4)
        if np.any(self.c < 0):
            raise ValueError("MVSK preference coefficients must be non-negative")
        self.T, self.n = self.R.shape
        if self.T < 8:
            raise ValueError(f"need at least 8 observations, got {self.T}")
        self.mu = self.R.mean(axis=0)
        # Centred returns.  Every oracle below is a product with A or A.T.
        self.A = np.ascontiguousarray(self.R - self.mu)

    # -- moments ---------------------------------------------------------
    def central_moments(self, w: np.ndarray) -> tuple[float, float, float, float]:
        """``(m1, m2, m3, m4)`` of the portfolio return under weights ``w``."""
        z = self.A @ w
        z2 = z * z
        return (
            float(self.mu @ w),
            float(z2.mean()),
            float((z2 * z).mean()),
            float((z2 * z2).mean()),
        )

    # -- first-order oracle ----------------------------------------------
    def value(self, w: np.ndarray) -> float:
        z = self.A @ w
        z2 = z * z
        c = self.c
        return float(
            -c[0] * (self.mu @ w) + (c[1] * z2 - c[2] * z2 * z + c[3] * z2 * z2).mean()
        )

    def value_grad(self, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return ``(f, grad f, z)``.

        ``grad f(w) = -c1 mu + (1/T) A' psi'(z)``, one ``A`` and one ``A.T``
        product.  No tensor is formed.
        """
        c = self.c
        z = self.A @ w
        z2 = z * z
        z3 = z2 * z
        f = float(-c[0] * (self.mu @ w) + (c[1] * z2 - c[2] * z3 + c[3] * z3 * z).mean())
        psi1 = 2.0 * c[1] * z - 3.0 * c[2] * z2 + 4.0 * c[3] * z3
        g = -c[0] * self.mu + (self.A.T @ psi1) / self.T
        return f, g, z

    # -- curvature weights (elementwise, length T) ------------------------
    def psi2(self, z: np.ndarray) -> np.ndarray:
        """``psi''(z) / T`` -- the diagonal weight of the Hessian."""
        c = self.c
        return (2.0 * c[1] - 6.0 * c[2] * z + 12.0 * c[3] * z * z) / self.T

    def psi3(self, z: np.ndarray) -> np.ndarray:
        """``psi'''(z) / T`` -- the diagonal weight of the third-order form."""
        c = self.c
        return (-6.0 * c[2] + 24.0 * c[3] * z) / self.T

    def hess_vec(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``grad^2 f(w) v`` via ``A' (psi''(z) . (A v))``."""
        return self.A.T @ (self.psi2(z) * (self.A @ v))

    # -- exact line-search polynomial -------------------------------------
    def line(self, w: np.ndarray, d: np.ndarray, f0: float | None = None) -> QuarticLine:
        """Coefficients of ``phi(a) = f(w + a d)``, exact, from power sums."""
        c = self.c
        z = self.A @ w
        v = self.A @ d
        if f0 is None:
            f0 = self.value(w)

        def s(r: int, k: int) -> float:
            return float(np.mean((z**r) * (v**k))) if r else float(np.mean(v**k))

        s11, s21, s31 = s(1, 1), s(2, 1), s(3, 1)
        s02, s12, s22 = s(0, 2), s(1, 2), s(2, 2)
        s03, s13 = s(0, 3), s(1, 3)
        s04 = s(0, 4)
        return QuarticLine(
            A0=float(f0),
            A1=float(-c[0] * (self.mu @ d) + 2 * c[1] * s11 - 3 * c[2] * s21 + 4 * c[3] * s31),
            A2=float(c[1] * s02 - 3 * c[2] * s12 + 6 * c[3] * s22),
            A3=float(-c[2] * s03 + 4 * c[3] * s13),
            A4=float(c[3] * s04),
        )
