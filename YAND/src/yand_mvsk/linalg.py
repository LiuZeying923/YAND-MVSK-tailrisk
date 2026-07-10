"""Small linear-algebra helpers used by the YAND solver."""

from __future__ import annotations

import numpy as np

__all__ = [
    "householder_reflector",
    "simplex_tangent_basis",
    "orthogonal_frame",
    "project_capped_simplex",
    "safe_cholesky",
]


def householder_reflector(v: np.ndarray) -> np.ndarray:
    """Symmetric orthogonal ``H`` with ``H v`` parallel to ``e_1``.

    ``v`` must be non-zero.  Uses the standard sign choice to avoid
    cancellation when ``v`` is already close to ``+e_1``.
    """
    v = np.asarray(v, dtype=float)
    m = v.size
    if m == 0:
        return np.zeros((0, 0))
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        raise ValueError("cannot build a reflector from the zero vector")
    u = v / nrm
    sign = 1.0 if u[0] >= 0.0 else -1.0
    u = u.copy()
    u[0] += sign
    un = np.linalg.norm(u)
    if un < 1e-300:  # v was exactly -sign * e_1; reflection is the identity
        return np.eye(m)
    u /= un
    return np.eye(m) - 2.0 * np.outer(u, u)


def simplex_tangent_basis(n: int) -> np.ndarray:
    """Orthonormal ``U`` of shape ``(n, n-1)`` spanning ``{v : 1'v = 0}``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return np.zeros((1, 0))
    H = householder_reflector(np.ones(n))
    # H maps 1/||1|| onto +/- e_1, so columns 1.. are orthogonal to 1.
    return np.ascontiguousarray(H[:, 1:])


def orthogonal_frame(nu: np.ndarray) -> np.ndarray:
    """Orthonormal ``Q`` of shape ``(m, m-1)`` with ``Q' nu = 0``.

    This is the "Householder frame" of the YAND papers: an orthonormal basis of
    the tangent space of the level set at the current point, given the unit
    normal ``nu``.
    """
    m = nu.size
    if m <= 1:
        return np.zeros((m, 0))
    H = householder_reflector(nu)
    # nu is parallel to H[:, 0], hence orthogonal to the remaining columns.
    return np.ascontiguousarray(H[:, 1:])


def project_capped_simplex(v: np.ndarray, cap: float = 1.0, total: float = 1.0) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{w : sum w = total, 0 <= w <= cap}``.

    Bisection on the dual shift ``theta``; ``sum(clip(v - theta, 0, cap))`` is
    non-increasing in ``theta``.
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    if cap * n < total - 1e-12:
        raise ValueError(f"cap={cap} with n={n} cannot reach total={total}")
    lo = float(v.min() - total)
    hi = float(v.max())
    for _ in range(200):
        theta = 0.5 * (lo + hi)
        s = float(np.clip(v - theta, 0.0, cap).sum())
        if abs(s - total) <= 1e-13:
            break
        if s > total:
            lo = theta
        else:
            hi = theta
    w = np.clip(v - 0.5 * (lo + hi), 0.0, cap)
    s = w.sum()
    if s > 0:
        w *= total / s
    return w


def safe_cholesky(H: np.ndarray, lam0: float = 0.0) -> tuple[np.ndarray, float]:
    """Cholesky factor of ``H + lam I``, escalating ``lam`` until it succeeds.

    Returns ``(L, lam)``.  This is the ``H_{T,lambda}`` regularisation of the
    YAND papers: the reduced Hessian of a quartic need not be positive definite
    away from the optimum.
    """
    m = H.shape[0]
    if m == 0:
        return np.zeros((0, 0)), lam0
    scale = float(np.abs(np.diag(H)).max()) or 1.0
    lam = float(lam0)
    for _ in range(60):
        try:
            L = np.linalg.cholesky(H + lam * np.eye(m))
            return L, lam
        except np.linalg.LinAlgError:
            lam = max(2.0 * lam, 1e-10 * scale)
    raise np.linalg.LinAlgError("reduced Hessian could not be regularised")
