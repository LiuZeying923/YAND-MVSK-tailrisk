"""Yau's Affine-Normal Descent (YAND) for MVSK portfolios.

Implements the algorithm of

  * Yau's Affine Normal Descent: Algorithmic Framework and Convergence
    Analysis, arXiv:2603.28448
  * Yau's Affine-Normal Descent for Large-Scale Unrestricted Higher-Moment
    Portfolio Optimization, arXiv:2604.25378

specialised to the long-only capped simplex.

The search direction is the *equi-affine normal* of the current level set,
rather than the Euclidean steepest-descent direction.  Affine-normal directions
are invariant under volume-preserving affine maps of the decision space, so they
adapt to anisotropic curvature automatically; on a strictly convex quadratic
they are collinear with the Newton direction and exact line search converges in
one step.

Per iteration, on the free set F (|F| = nf, tangent-to-level-set dimension
m = nf - 2):

    g       = grad f(w)                         (tensor-free, see moments.py)
    gbar    = g_F - mean(g_F)                   (project onto {1'v = 0})
    nu      = gbar / ||gbar||                   (unit normal of the level set)
    Q       = orthonormal basis of {v : 1'v = 0, nu'v = 0}     (nf x m)
    B       = A_F Q                             (T x m)
    H       = B' diag(psi''(z)/T) B + lam I     (regularised reduced Hessian)
    h       = B' (psi''(z)/T . (A_F nu))        (tangent-normal coupling)
    a_i     = tr(H^-1 d_i H) = [B' (psi'''(z)/T . diag(B H^-1 B'))]_i
              (exact log-determinant correction from the third-order oracle)
    u       = H^-1 (h - ||gbar||/nf * a)
    d       = Q u - nu                          (affine-normal direction)

Two facts worth stating, both checked in the tests:

1. ``g' d = -||gbar|| < 0`` always, because ``nu' Q = 0``.  The affine-normal
   direction is a descent direction by construction -- no orientation flip and
   no sufficient-decrease backtracking is needed.
2. If ``f`` is quadratic the third-order oracle vanishes, ``a = 0``,
   ``u = H^-1 h``, and ``d = Q H^-1 h - nu`` is collinear with the Newton
   direction ``-(grad^2 f)^-1 gbar``.

The step length is the exact minimiser of ``phi(a) = f(w + a d)``, a quartic
polynomial whose five coefficients come from mixed power sums of ``z = A w``
and ``A d``.  No tensor of order 3 or 4 is ever formed; storage is O(Tn).

Bound constraints are handled by an active set: the line search is truncated at
the first coordinate that reaches 0 or ``cap``, that coordinate is pinned, and
descent continues on the exposed face.  A face is left when the KKT multiplier
of a pinned coordinate turns the wrong way.  This yields exact zeros in the
weight vector rather than the 1e-9 dust an interior-point method leaves behind.

Scope note: this is the paper's *direct* configuration, which the paper reports
as preferred below n ~= 80-100 assets.  The large-scale preconditioned-CG
configuration is not implemented here; the direct solver still runs at n = 800
(see ``benchmarks/`` for measured timings), just not at the paper's large-n
speed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .linalg import safe_cholesky
from .moments import MVSKObjective

__all__ = ["YANDResult", "solve_yand"]

_BOUND_TOL = 1e-11


@dataclass
class YANDResult:
    """Outcome of a YAND solve."""

    w: np.ndarray
    f: float
    iterations: int
    converged: bool
    seconds: float
    grad_norm: float
    faces_visited: int = 1
    restarts: int = 1
    history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "objective": float(self.f),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "seconds": float(self.seconds),
            "grad_norm": float(self.grad_norm),
            "faces_visited": int(self.faces_visited),
            "restarts": int(self.restarts),
        }


def _face_frame(nu: np.ndarray) -> np.ndarray:
    """Orthonormal basis ``Q`` (nf x nf-2) of ``{v : 1'v = 0, nu'v = 0}``.

    ``nu`` is a unit vector already orthogonal to ``1``.
    """
    nf = nu.size
    if nf <= 2:
        return np.zeros((nf, 0))
    seed = np.column_stack([np.ones(nf) / np.sqrt(nf), nu])
    Qfull, _ = np.linalg.qr(seed, mode="complete")
    return np.ascontiguousarray(Qfull[:, 2:])


def _logdet_correction(
    B: np.ndarray, p: np.ndarray, chol: tuple[np.ndarray, bool], m: int
) -> np.ndarray:
    """``a_i = tr(H^-1 * d_i H)``, the exact affine-normal log-det term.

    ``d_i H = B' diag(p . B[:, i]) B``, hence
    ``a_i = sum_t p_t B_ti (B H^-1 B')_tt`` and the whole vector is
    ``a = B' (p . diag(B H^-1 B'))``.  Only the *diagonal* of ``B H^-1 B'`` is
    needed, so this costs O(T m^2), never O(T^2).
    """
    if m == 0:
        return np.zeros(0)
    BM = cho_solve(chol, B.T).T  # (T, m) = B H^-1
    diag = np.einsum("tj,tj->t", BM, B)  # diag(B H^-1 B')
    return B.T @ (p * diag)


def _yand_face(
    obj: MVSKObjective,
    w: np.ndarray,
    free: np.ndarray,
    cap: float,
    max_iter: int,
    tol: float,
    max_step_norm: float,
) -> tuple[np.ndarray, float, int, bool, float, int | None]:
    """Run affine-normal descent on one face.

    Returns ``(w, f, iters, converged, grad_norm, blocking_index_or_None)``.
    The blocking index (in *global* coordinates) means a bound was hit and the
    caller should pin it and re-enter on the smaller face.
    """
    idx = np.flatnonzero(free)
    nf = idx.size
    A_F = np.ascontiguousarray(obj.A[:, idx])
    lam = 0.0
    iters = 0

    f, g, z = obj.value_grad(w)
    if nf < 2:
        return w, f, 0, True, 0.0, None

    for iters in range(1, max_iter + 1):
        gF = g[idx]
        gbar = gF - gF.mean()  # reduced gradient == U' g up to isometry
        gnorm = float(np.linalg.norm(gbar))
        if gnorm <= tol:
            return w, f, iters - 1, True, gnorm, None

        nu = gbar / gnorm
        Q = _face_frame(nu)
        m = Q.shape[1]

        if m == 0:
            d_F = -nu  # nf == 2: the level set is a point, descend along -nu
        else:
            B = A_F @ Q
            q = obj.psi2(z)
            p = obj.psi3(z)
            H = (B.T * q) @ B
            L, lam = safe_cholesky(H, lam0=0.5 * lam)
            chol = (L, True)
            h = B.T @ (q * (A_F @ nu))
            a = _logdet_correction(B, p, chol, m)
            rhs = h - (gnorm / nf) * a
            u = cho_solve(chol, rhs)
            d_F = Q @ u - nu

        # d_F is tangent (1'd_F = 0) and g'd = -||gbar|| < 0 by construction.
        dn = float(np.linalg.norm(d_F))
        if not np.isfinite(dn) or dn == 0.0:
            return w, f, iters, True, gnorm, None
        if dn > max_step_norm:
            d_F *= max_step_norm / dn

        # Truncate the ray at the first bound it hits.
        wF = w[idx]
        alpha_max = np.inf
        blocking = -1
        for j in range(nf):
            dj = d_F[j]
            if dj < -1e-16:
                cand = wF[j] / (-dj)
            elif dj > 1e-16 and cap < 1.0:
                cand = (cap - wF[j]) / dj
            else:
                continue
            if cand < alpha_max:
                alpha_max = cand
                blocking = j
        if not np.isfinite(alpha_max):
            alpha_max = 1e6
            blocking = -1
        if alpha_max <= 1e-15:
            # Already pinned against this bound; hand it back to the active set.
            return w, f, iters, False, gnorm, int(idx[blocking]) if blocking >= 0 else None

        d = np.zeros_like(w)
        d[idx] = d_F
        phi = obj.line(w, d, f0=f)
        alpha, f_new = phi.minimize_on(0.0, float(alpha_max))

        if alpha <= 0.0 or not np.isfinite(f_new) or f_new >= f - 1e-16 * max(1.0, abs(f)):
            return w, f, iters, True, gnorm, None

        w = w + alpha * d
        np.clip(w, 0.0, cap, out=w)
        # Re-impose the equality exactly against accumulated rounding.
        s = w[idx].sum()
        target = 1.0 - float(w[~free].sum())
        if s > 0:
            w[idx] *= target / s
        f, g, z = obj.value_grad(w)

        hit = alpha >= alpha_max * (1.0 - 1e-12) and blocking >= 0
        if hit:
            return w, f, iters, False, gnorm, int(idx[blocking])

    gF = g[idx]
    gnorm = float(np.linalg.norm(gF - gF.mean()))
    return w, f, iters, gnorm <= tol, gnorm, None


def _solve_from(
    obj: MVSKObjective,
    w0: np.ndarray,
    cap: float,
    max_iter: int,
    tol: float,
    max_outer: int,
    max_step_norm: float,
) -> YANDResult:
    t0 = time.perf_counter()
    n = obj.n
    w = np.array(w0, dtype=float)
    free = np.ones(n, dtype=bool)
    total_iters = 0
    faces = 1
    converged = False
    gnorm = np.inf
    f = obj.value(w)

    # Pin anything that starts on a bound.
    free &= ~(w <= _BOUND_TOL)
    if cap < 1.0:
        free &= ~(w >= cap - _BOUND_TOL)

    for _ in range(max_outer):
        budget = max(1, max_iter - total_iters)
        w, f, it, conv, gnorm, blocking = _yand_face(
            obj, w, free, cap, budget, tol, max_step_norm
        )
        total_iters += it

        if blocking is not None:
            free[blocking] = False
            faces += 1
            if free.sum() >= 1 and total_iters < max_iter:
                continue

        if total_iters >= max_iter:
            break

        # KKT check: release pinned coordinates whose multiplier points inward.
        _, g, _ = obj.value_grad(w)
        if free.sum() == 0:
            lam_eq = float(g.mean())
        else:
            lam_eq = float(g[free].mean())
        released = False
        scale = max(1.0, float(np.abs(g).max()))
        for i in range(n):
            if free[i]:
                continue
            red = g[i] - lam_eq
            at_lo = w[i] <= _BOUND_TOL
            at_hi = cap < 1.0 and w[i] >= cap - _BOUND_TOL
            if (at_lo and red < -1e-9 * scale) or (at_hi and red > 1e-9 * scale):
                free[i] = True
                released = True
        if released:
            faces += 1
            continue

        converged = conv or gnorm <= tol
        break

    w = np.clip(w, 0.0, cap)
    w /= w.sum()
    return YANDResult(
        w=w,
        f=float(obj.value(w)),
        iterations=total_iters,
        converged=bool(converged),
        seconds=time.perf_counter() - t0,
        grad_norm=float(gnorm),
        faces_visited=faces,
    )


def _starting_points(obj: MVSKObjective, cap: float) -> list[np.ndarray]:
    """Feasible, diverse warm starts.  The MVSK objective is not convex."""
    from .linalg import project_capped_simplex

    n = obj.n
    starts = [project_capped_simplex(np.full(n, 1.0 / n), cap)]
    if n >= 2:
        # Inverse-variance: a good proxy for the variance-dominated optimum.
        var = obj.R.var(axis=0)
        var = np.where(var > 0, var, var[var > 0].min() if np.any(var > 0) else 1.0)
        starts.append(project_capped_simplex(1.0 / var / np.sum(1.0 / var), cap))
        # Mean-tilted: pulls toward the return-seeking corner.
        mu = obj.mu - obj.mu.min()
        if mu.sum() > 0:
            starts.append(project_capped_simplex(0.5 / n + 0.5 * mu / mu.sum(), cap))
    return starts


def solve_yand(
    obj: MVSKObjective,
    cap: float = 1.0,
    w0: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-10,
    max_outer: int = 60,
    multistart: bool = True,
    max_step_norm: float = 1e3,
) -> YANDResult:
    """Minimise ``obj`` over ``{w : sum w = 1, 0 <= w <= cap}``.

    ``multistart`` runs the solver from several feasible warm starts and keeps
    the best objective.  The MVSK objective is a non-convex quartic, so a single
    start can land in a shallow local minimum; the starts are cheap.
    """
    n = obj.n
    if n == 1:
        return YANDResult(np.ones(1), obj.value(np.ones(1)), 0, True, 0.0, 0.0)
    if cap * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight={cap} is infeasible for {n} assets (need >= {1/n:.4f})")

    starts = [np.asarray(w0, float)] if w0 is not None else []
    if multistart or not starts:
        starts.extend(_starting_points(obj, cap))

    best: YANDResult | None = None
    total_iters = 0
    t0 = time.perf_counter()
    for s in starts:
        res = _solve_from(obj, s, cap, max_iter, tol, max_outer, max_step_norm)
        total_iters += res.iterations
        if best is None or res.f < best.f:
            best = res
    assert best is not None
    best.iterations = total_iters
    best.restarts = len(starts)
    best.seconds = time.perf_counter() - t0
    return best
