"""Optimisation orchestration shared by the API and the CLI/examples.

Keeping the heavy lifting here (not in the route handlers) means the exact same
pipeline is exercised by the tests and the notebooks.
"""

from __future__ import annotations

import numpy as np

from ..behavioral import BehavioralGammaOptimizer
from ..data import fetch_prices
from ..data.market import PriceData
from ..optimizer import EfficientMVSK
from ..risk import (
    RiskMonitor,
    StressTester,
    TailRiskGuard,
    tail_contributions,
    tail_metrics,
)

_TRADING_DAYS = 252


def resolve_gamma(gamma, answers, base_gamma):
    """Return ``(gamma, breakdown_or_None)`` from either an explicit value or VBQ-5."""
    if gamma is not None:
        return float(gamma), None
    if answers:
        bd = BehavioralGammaOptimizer(base_gamma).calculate(answers)
        return bd.gamma, bd.as_dict()
    return float(base_gamma), None


def _equity_curve(daily_returns: np.ndarray) -> list[float]:
    return (100.0 * np.cumprod(1.0 + daily_returns)).round(3).tolist()


def _weights_payload(tickers, weights, meta, cutoff=1e-4):
    rows = []
    for t, w in zip(tickers, weights):
        if w <= cutoff:
            continue
        info = meta.get(t, {})
        rows.append({
            "symbol": t,
            "weight": round(float(w), 4),
            "name": (info.get("catalog") or {}).get("name", t),
            "market": info.get("market", ""),
            "market_label": info.get("market_label", ""),
            "currency": info.get("currency", ""),
            "sector": (info.get("catalog") or {}).get("sector", ""),
            "synthetic": bool(info.get("synthetic", False)),
        })
    rows.sort(key=lambda r: r["weight"], reverse=True)
    return rows


def _run_one(prices_df, tickers, gamma, max_weight, horizon, rf, ppy=_TRADING_DAYS):
    ef = EfficientMVSK.from_prices(
        prices_df, gamma=gamma, horizon=horizon, tickers=tickers,
        max_weight=max_weight, periods_per_year=ppy,
    )
    w = ef.optimize()
    perf = ef.portfolio_performance()
    daily = ef.report_returns @ w
    return ef, w, perf, daily


def optimize_portfolio(
    tickers: list[str],
    gamma: float | None = None,
    answers: dict | None = None,
    base_gamma: float = 6.0,
    max_weight: float = 0.35,
    horizon: int = 21,
    lookback: str = "3y",
    offline: bool = False,
    apply_guard: bool = True,
    rf: float = 0.02,
) -> dict:
    """Full pipeline: gamma -> data -> optimise -> risk guard -> (maybe) re-optimise."""
    tickers = [t.strip() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("provide at least one ticker")
    n = len(tickers)
    if max_weight * n < 1.0 - 1e-9:
        max_weight = min(1.0, 1.0 / n * 1.5)  # relax an infeasible cap sensibly

    gamma, gamma_breakdown = resolve_gamma(gamma, answers, base_gamma)

    data: PriceData = fetch_prices(tickers, period=lookback, offline=offline)
    used = list(data.prices.columns)
    prices_df = data.prices

    ef, w, perf, daily = _run_one(prices_df, used, gamma, max_weight, horizon, rf)

    # --- risk guard on the daily portfolio series --------------------------
    asset_daily = prices_df.pct_change().dropna().to_numpy()
    returns_by_date = prices_df.pct_change().dropna()
    guard = TailRiskGuard(RiskMonitor(window=min(60, max(20, len(daily) // 4))))
    verdict = guard.evaluate(
        daily, asset_returns=asset_daily, weights=w,
        freq="daily", returns_by_date=returns_by_date,
    )

    # --- guard-driven recommendation ---------------------------------------
    # Rather than blindly multiplying gamma (which, on a fat-tailed basket, can
    # concentrate the portfolio and make tails *worse*), search a few safer
    # reconfigurations -- higher risk aversion and/or a tighter concentration
    # cap -- and only surface the one that materially lifts the resilience
    # score.  If nothing helps, the tail risk is structural to these assets, so
    # we instead name the biggest contributor to expected shortfall.
    reopt = None
    recommendation = None
    if apply_guard and verdict.recommended_gamma_multiplier > 1.0:
        candidates = []
        for gmul in (1.4, 1.8):
            candidates.append(("gamma", min(25.0, gamma * gmul), max_weight))
        for cap in (max_weight * 0.7, max(1.2 / len(used), max_weight * 0.5)):
            if cap < max_weight - 1e-6 and cap * len(used) >= 1.0:
                candidates.append(("cap", min(25.0, gamma * 1.4), round(cap, 4)))

        best = None
        for kind, g2, cap2 in candidates:
            _, w2, perf2, daily2 = _run_one(prices_df, used, g2, cap2, horizon, rf)
            v2 = guard.evaluate(daily2, asset_returns=asset_daily, weights=w2,
                                freq="daily", returns_by_date=returns_by_date)
            delta = v2.score - verdict.score
            if best is None or delta > best["score_delta"]:
                best = {
                    "kind": kind, "gamma": round(g2, 3), "max_weight": cap2,
                    "weights": _weights_payload(used, w2, data.meta),
                    "performance": perf2, "risk": v2.as_dict(),
                    "equity_curve": _equity_curve(daily2),
                    "score_delta": round(delta, 1),
                }
        if best is not None and best["score_delta"] >= 2.0:
            label = ("raising risk aversion to γ = %s" % best["gamma"]) if best["kind"] == "gamma" \
                else ("tightening the per-name cap to %d%%" % round(best["max_weight"] * 100))
            recommendation = {
                "action": "reconfigure", "how": label,
                "score_delta": best["score_delta"], "new_score": best["risk"]["score"],
            }
            reopt = best
        else:
            # Structural tail risk: name the worst expected-shortfall contributor.
            contrib = tail_contributions(asset_daily, w)
            order = np.argsort(contrib)  # most negative first
            worst = [used[i] for i in order[:2] if contrib[i] < 0]
            recommendation = {
                "action": "structural",
                "how": ("Risk aversion and diversification are already near-optimal for this basket; "
                        "the residual tail risk is intrinsic to " + (", ".join(worst) if worst else "these assets")
                        + ". Consider trimming or replacing " + (worst[0] if worst else "the fattest-tailed name") + "."),
                "worst_contributors": worst,
            }

    # --- equal-weight benchmark for context --------------------------------
    ew = np.full(len(used), 1.0 / len(used))
    ew_daily = prices_df.pct_change().dropna().to_numpy() @ ew
    ew_metrics = tail_metrics(ew_daily, periods_per_year=_TRADING_DAYS, rf=rf)

    dates = [d.strftime("%Y-%m-%d") for d in returns_by_date.index]

    return {
        "request": {
            "tickers": used,
            "gamma": round(gamma, 3),
            "max_weight": max_weight,
            "horizon": horizon,
            "offline": offline,
        },
        "gamma": round(gamma, 3),
        "gamma_breakdown": gamma_breakdown,
        "data_source": data.source,
        "synthetic_symbols": data.synthetic_symbols,
        "n_observations": int(prices_df.shape[0]),
        "date_range": [dates[0], dates[-1]] if dates else [],
        "weights": _weights_payload(used, w, data.meta),
        "performance": perf,
        "risk": verdict.as_dict(),
        "equity_curve": _equity_curve(daily),
        "equity_dates": dates,
        "benchmark": {
            "name": "Equal weight",
            "equity_curve": _equity_curve(ew_daily),
            "ann_return": round(ew_metrics.ann_return, 6),
            "ann_volatility": round(ew_metrics.ann_volatility, 6),
            "sharpe": round(ew_metrics.sharpe, 4),
            "max_drawdown": round(ew_metrics.max_drawdown, 6),
        },
        "reoptimized": reopt,
        "recommendation": recommendation,
        "asset_meta": data.meta,
    }
