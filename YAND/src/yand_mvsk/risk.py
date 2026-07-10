"""Tail-risk analytics: point metrics, rolling monitor, stress tests, and the
Tail-Risk Guard that ties them into a single actionable verdict.

The optimiser chooses weights; this module answers the operational question that
comes next -- *how fragile is the resulting portfolio, and should we act?*  It
is deliberately model-light: everything is computed from the realised return
series of the chosen weights, so there is nothing to mis-specify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "tail_metrics",
    "TailMetrics",
    "RiskMonitor",
    "MonitorReading",
    "StressTester",
    "StressResult",
    "TailRiskGuard",
    "GuardVerdict",
    "tail_contributions",
]

# Trading periods per year, used to annualise.
_PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


def _ann_factor(freq: str, periods_per_year: int | None = None) -> int:
    """Resolve an annualisation factor from an explicit count or a ``freq`` label."""
    if periods_per_year is not None:
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be >= 1")
        return int(periods_per_year)
    if freq not in _PERIODS:
        raise ValueError(f"freq must be one of {sorted(_PERIODS)}, got {freq!r}")
    return _PERIODS[freq]


def _sample_skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return 0.0
    m = x - x.mean()
    s = m.std(ddof=0)
    if s < 1e-300:
        return 0.0
    return float((m**3).mean() / s**3)


def _sample_excess_kurt(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 0.0
    m = x - x.mean()
    s = m.std(ddof=0)
    if s < 1e-300:
        return 0.0
    return float((m**4).mean() / s**4 - 3.0)  # excess (Gaussian == 0)


def _max_drawdown(returns: np.ndarray) -> float:
    curve = np.cumprod(1.0 + np.asarray(returns, float))
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min()) if curve.size else 0.0


@dataclass
class TailMetrics:
    n_obs: int
    mean: float
    volatility: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    skewness: float
    excess_kurtosis: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    downside_vol: float
    sortino: float

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def tail_metrics(
    returns: np.ndarray,
    freq: str = "daily",
    rf: float = 0.0,
    periods_per_year: int | None = None,
) -> TailMetrics:
    """Full tail-risk profile of a single return series.

    VaR/CVaR are historical (empirical quantiles), reported as positive loss
    magnitudes.  ``rf`` is the per-annum risk-free rate.  Pass
    ``periods_per_year`` to annualise an aggregated series correctly (e.g. 12
    for 21-trading-day returns); it overrides ``freq``.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        raise ValueError(f"need at least 4 return observations, got {r.size}")
    ann = _ann_factor(freq, periods_per_year)
    mean = float(r.mean())
    vol = float(r.std(ddof=1))
    ann_ret = (1.0 + mean) ** ann - 1.0
    ann_vol = vol * np.sqrt(ann)
    rf_per = (1.0 + rf) ** (1.0 / ann) - 1.0
    sharpe = float((mean - rf_per) / vol * np.sqrt(ann)) if vol > 1e-12 else 0.0

    downside = r[r < rf_per]
    dvol = float(np.sqrt(np.mean((downside - rf_per) ** 2))) if downside.size else 0.0
    sortino = float((mean - rf_per) / dvol * np.sqrt(ann)) if dvol > 1e-12 else 0.0

    def var_cvar(alpha: float) -> tuple[float, float]:
        q = float(np.quantile(r, 1.0 - alpha))
        tail = r[r <= q]
        cvar = float(tail.mean()) if tail.size else q
        return -q, -cvar  # positive loss magnitudes

    var95, cvar95 = var_cvar(0.95)
    var99, cvar99 = var_cvar(0.99)

    return TailMetrics(
        n_obs=int(r.size),
        mean=mean,
        volatility=vol,
        ann_return=ann_ret,
        ann_volatility=ann_vol,
        sharpe=sharpe,
        skewness=_sample_skew(r),
        excess_kurtosis=_sample_excess_kurt(r),
        var_95=var95,
        cvar_95=cvar95,
        var_99=var99,
        cvar_99=cvar99,
        max_drawdown=_max_drawdown(r),
        downside_vol=dvol,
        sortino=sortino,
    )


def tail_contributions(asset_returns: np.ndarray, weights: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Each asset's share of portfolio CVaR (component expected shortfall).

    On the days that make up the worst ``1-alpha`` tail of the *portfolio*, the
    component CVaR of asset i is ``w_i * mean(r_i | portfolio in tail)``.  These
    sum to the portfolio CVaR, so the most negative entry is the name dragging
    the tail down -- exactly what to trim when raising risk aversion can't help.
    """
    R = np.asarray(asset_returns, float)
    w = np.asarray(weights, float)
    port = R @ w
    if port.size < 5:
        return w * R.mean(axis=0)
    q = np.quantile(port, 1.0 - alpha)
    tail = port <= q
    if not tail.any():
        tail = port <= np.quantile(port, 0.1)
    return w * R[tail].mean(axis=0)


@dataclass
class MonitorReading:
    index: int
    skew: float
    excess_kurt: float
    vol: float
    alert: bool
    reasons: list[str] = field(default_factory=list)


class RiskMonitor:
    """Rolling-window skewness/kurtosis early-warning monitor.

    Fires when the window turns *negatively skewed with fat tails*
    (``skew < skew_thresh`` and ``excess_kurt > kurt_thresh``) -- the signature
    of tail risk accumulating -- or when rolling kurtosis blows past a multiple
    of its own historical average, the README's rebalancing trigger.
    """

    def __init__(
        self,
        window: int = 60,
        skew_thresh: float = -0.5,
        kurt_thresh: float = 2.0,
        kurt_spike_mult: float = 2.0,
    ) -> None:
        if window < 8:
            raise ValueError("window must be >= 8")
        self.window = int(window)
        self.skew_thresh = float(skew_thresh)
        self.kurt_thresh = float(kurt_thresh)
        self.kurt_spike_mult = float(kurt_spike_mult)

    def scan(self, returns: np.ndarray) -> list[MonitorReading]:
        r = np.asarray(returns, float)
        readings: list[MonitorReading] = []
        kurt_hist: list[float] = []
        for end in range(self.window, r.size + 1):
            win = r[end - self.window : end]
            sk = _sample_skew(win)
            ku = _sample_excess_kurt(win)
            vol = float(win.std(ddof=1))
            reasons = []
            if sk < self.skew_thresh and ku > self.kurt_thresh:
                reasons.append(
                    f"negative skew {sk:.2f} with fat tail (excess kurtosis {ku:.2f})"
                )
            if kurt_hist:
                base = float(np.mean(kurt_hist))
                if base > 0 and ku > self.kurt_spike_mult * base:
                    reasons.append(
                        f"kurtosis {ku:.2f} is {ku / base:.1f}× its rolling mean — rebalance trigger"
                    )
            readings.append(
                MonitorReading(end - 1, sk, ku, vol, bool(reasons), reasons)
            )
            kurt_hist.append(ku)
        return readings

    def latest(self, returns: np.ndarray) -> MonitorReading | None:
        readings = self.scan(returns)
        return readings[-1] if readings else None


@dataclass
class StressResult:
    name: str
    description: str
    portfolio_return: float
    worst_period: float
    max_drawdown: float
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "portfolio_return": round(self.portfolio_return, 6),
            "worst_period": round(self.worst_period, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "n_obs": self.n_obs,
        }


# Named historical crash windows for date-indexed stress testing.
HISTORICAL_SCENARIOS: tuple[dict, ...] = (
    {"name": "COVID-19 crash", "start": "2020-02-19", "end": "2020-03-23",
     "description": "Fastest-ever 30%+ S&P drawdown."},
    {"name": "2022 bear market", "start": "2022-01-03", "end": "2022-10-12",
     "description": "Rate-hike-driven regime, growth stocks hit hardest."},
    {"name": "2018 Q4 selloff", "start": "2018-10-01", "end": "2018-12-24",
     "description": "Liquidity-driven ~20% correction."},
    {"name": "2015 China / A-share crash", "start": "2015-06-12", "end": "2015-08-26",
     "description": "CSI 300 lost ~40% in weeks."},
)


class StressTester:
    """Replay a portfolio through historical crash windows.

    Two modes:
      * ``scenario`` -- slice a date-indexed return DataFrame to a named window,
      * ``synthetic_shock`` -- apply a parametric single-period shock when history
        is unavailable (the offline/synthetic-data path).
    """

    def __init__(self, scenarios: tuple[dict, ...] = HISTORICAL_SCENARIOS) -> None:
        self.scenarios = scenarios

    def scenario(self, returns_by_date, weights: np.ndarray, spec: dict) -> StressResult | None:
        """``returns_by_date`` is a pandas DataFrame indexed by datetime,
        columns aligned to ``weights``.  Returns ``None`` if the window has no
        overlap with the data."""
        import pandas as pd

        w = np.asarray(weights, float)
        sub = returns_by_date.loc[
            (returns_by_date.index >= pd.Timestamp(spec["start"]))
            & (returns_by_date.index <= pd.Timestamp(spec["end"]))
        ]
        if sub.shape[0] == 0:
            return None
        port = sub.to_numpy() @ w
        total = float(np.prod(1.0 + port) - 1.0)
        return StressResult(
            name=spec["name"],
            description=spec.get("description", ""),
            portfolio_return=total,
            worst_period=float(port.min()),
            max_drawdown=_max_drawdown(port),
            n_obs=int(port.size),
        )

    def run_all(self, returns_by_date, weights: np.ndarray) -> list[StressResult]:
        out = []
        for spec in self.scenarios:
            res = self.scenario(returns_by_date, weights, spec)
            if res is not None:
                out.append(res)
        return out

    @staticmethod
    def synthetic_shock(
        asset_returns: np.ndarray, weights: np.ndarray, shock_vol: float = 3.0
    ) -> StressResult:
        """Stress a portfolio by scaling each asset's worst historical moves.

        Approximates "what if every asset simultaneously realised a
        ``shock_vol``-sigma down move sized by its own tail" -- a conservative,
        distribution-free shock used when we lack aligned crash-window history.
        """
        R = np.asarray(asset_returns, float)
        w = np.asarray(weights, float)
        mu = R.mean(axis=0)
        sd = R.std(axis=0, ddof=1)
        shocked = mu - shock_vol * sd
        total = float(w @ shocked)
        return StressResult(
            name=f"Synthetic {shock_vol:g}σ shock",
            description="Simultaneous per-asset tail move (distribution-free).",
            portfolio_return=total,
            worst_period=total,
            max_drawdown=total,
            n_obs=1,
        )


@dataclass
class GuardVerdict:
    """The Tail-Risk Guard's overall assessment."""

    score: float  # 0 (fragile) .. 100 (robust)
    level: str  # "robust" | "watch" | "elevated" | "critical"
    headline: str
    findings: list[str]
    recommended_gamma_multiplier: float
    metrics: TailMetrics
    monitor: MonitorReading | None
    stress: list[StressResult]

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "headline": self.headline,
            "findings": self.findings,
            "recommended_gamma_multiplier": round(self.recommended_gamma_multiplier, 3),
            "metrics": self.metrics.as_dict(),
            "monitor": self.monitor.__dict__ if self.monitor else None,
            "stress": [s.as_dict() for s in self.stress],
        }


class TailRiskGuard:
    """The project's headline risk-control feature.

    It fuses the point tail metrics, the rolling monitor, and stress results
    into a single 0–100 robustness score, a severity level, plain-language
    findings, and -- crucially -- a *recommended gamma multiplier* that feeds
    straight back into a re-optimisation.  That closes the loop the README
    describes: detect tail-risk accumulation, then act by raising risk aversion.

    Scoring starts at 100 and subtracts penalties for the specific fragilities
    that matter for tail risk: negative skew, excess kurtosis, deep CVaR, deep
    drawdown, and an active monitor alert.  The penalties are transparent and
    bounded; this is a decision aid, not a guarantee.
    """

    def __init__(self, monitor: RiskMonitor | None = None) -> None:
        self.monitor = monitor or RiskMonitor()

    def evaluate(
        self,
        portfolio_returns: np.ndarray,
        asset_returns: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        freq: str = "daily",
        returns_by_date=None,
    ) -> GuardVerdict:
        r = np.asarray(portfolio_returns, float)
        m = tail_metrics(r, freq=freq)
        reading = self.monitor.latest(r) if r.size >= self.monitor.window else None

        # Continuous, bounded penalties anchored to interpretable thresholds.  Each
        # is monotone in the underlying metric, so a genuinely less tail-risky
        # portfolio always scores at least as high -- the earlier design let a
        # binary monitor alert swamp the continuous signal and made the score
        # non-monotone.  Frequency scales the loss-based anchors.
        fscale = {"daily": 1.0, "weekly": 2.2, "monthly": 4.0}.get(freq, 1.0)

        def band(x, lo, hi, weight):
            """0 penalty at ``lo``, full ``weight`` at ``hi`` (linear between)."""
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * weight)

        score = 100.0
        findings: list[str] = []

        pen_skew = band(-m.skewness, 0.0, 1.0, 18.0)           # left tail
        pen_kurt = band(m.excess_kurtosis, 0.0, 5.5, 22.0)      # fat tails
        pen_cvar = band(m.cvar_95, 0.015 * fscale, 0.035 * fscale, 22.0)
        pen_dd = band(-m.max_drawdown, 0.08, 0.35, 20.0)
        score -= pen_skew + pen_kurt + pen_cvar + pen_dd

        if m.skewness < -0.2:
            findings.append(f"Left-skewed returns ({m.skewness:.2f}): losses cluster larger than gains.")
        if m.excess_kurtosis > 1.0:
            findings.append(f"Fat tails (excess kurtosis {m.excess_kurtosis:.2f}): outsized moves more likely than a normal distribution implies.")
        if m.cvar_95 > 0.02 * fscale:
            findings.append(f"Deep expected shortfall: the average loss on the worst 5% of periods is {m.cvar_95*100:.1f}%.")
        if m.max_drawdown < -0.15:
            findings.append(f"Peak-to-trough drawdown reached {m.max_drawdown*100:.1f}% in-sample.")
        # A live monitor alert is informative but must not dominate the score.
        if reading is not None and reading.alert:
            score -= 6.0
            findings.extend(reading.reasons)

        stress: list[StressResult] = []
        if returns_by_date is not None and weights is not None:
            stress = StressTester().run_all(returns_by_date, weights)
        # Always guarantee at least one stress result: if no historical crash
        # window overlaps the sample (common for short/synthetic histories),
        # fall back to a distribution-free simultaneous tail shock.
        if not stress and asset_returns is not None and weights is not None:
            stress = [StressTester.synthetic_shock(asset_returns, weights)]
        for s in stress:
            if s.portfolio_return < -0.25:
                findings.append(f"{s.name}: this basket would have lost {abs(s.portfolio_return)*100:.0f}%.")

        score = float(max(0.0, min(100.0, score)))
        if score >= 75:
            level, mult, head = "robust", 1.0, "Portfolio looks resilient to tail risk."
        elif score >= 55:
            level, mult, head = "watch", 1.2, "Minor tail-risk signals — worth watching."
        elif score >= 35:
            level, mult, head = "elevated", 1.5, "Elevated tail risk — safer reconfiguration recommended."
        else:
            level, mult, head = "critical", 1.9, "Critical tail-risk exposure — reconfigure before committing."

        if not findings:
            findings.append("No significant tail-risk flags detected in the sample window.")

        return GuardVerdict(
            score=score,
            level=level,
            headline=head,
            findings=findings,
            recommended_gamma_multiplier=mult,
            metrics=m,
            monitor=reading,
            stress=stress,
        )
