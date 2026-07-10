"""``EfficientMVSK`` -- the user-facing optimizer wrapper.

Thin, ergonomic layer over :mod:`yand_mvsk.moments` and :mod:`yand_mvsk.yand`
that mirrors the API sketched in the project README::

    ef = EfficientMVSK.from_prices(prices, gamma=6)
    weights = ef.optimize()
    cleaned = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
"""

from __future__ import annotations

import numpy as np

from .moments import MVSKObjective, crra_coefficients, horizon_returns
from .risk import tail_metrics
from .yand import YANDResult, solve_yand

__all__ = ["EfficientMVSK"]


class EfficientMVSK:
    """Higher-moment (mean-variance-skewness-kurtosis) portfolio optimizer.

    Parameters
    ----------
    returns:
        ``(T, n)`` array of horizon simple returns (decimal).
    gamma:
        CRRA risk-aversion scalar.  Sets the four MVSK preference coefficients
        via a fourth-order Taylor expansion; higher gamma => more tail-averse.
    tickers:
        Optional labels for reporting.
    max_weight:
        Per-asset cap (``1.0`` = uncapped).  A cap below 1 forces
        diversification and is strongly recommended for small baskets.
    report_returns:
        Optional separate ``(T2, n)`` return series used *only* for the
        performance/annualisation report.  When the optimisation runs on
        multi-period horizon returns (so skew/kurtosis are meaningful), pass the
        underlying daily returns here so Sharpe, VaR and annualised figures come
        out on the conventional daily basis.  Defaults to ``returns``.
    periods_per_year:
        Annualisation factor for ``report_returns`` (252 for daily).
    """

    def __init__(
        self,
        returns: np.ndarray,
        gamma: float = 6.0,
        tickers: list[str] | None = None,
        max_weight: float = 1.0,
        coefficients: np.ndarray | None = None,
        report_returns: np.ndarray | None = None,
        periods_per_year: int = 252,
    ) -> None:
        self.returns = np.ascontiguousarray(returns, dtype=float)
        if self.returns.ndim != 2:
            raise ValueError("returns must be 2-D (T, n)")
        self.gamma = float(gamma)
        self.n = self.returns.shape[1]
        self.tickers = list(tickers) if tickers is not None else [f"A{i}" for i in range(self.n)]
        if len(self.tickers) != self.n:
            raise ValueError("tickers length must match number of assets")
        self.max_weight = float(max_weight)
        self.periods_per_year = int(periods_per_year)
        self.report_returns = (
            self.returns if report_returns is None
            else np.ascontiguousarray(report_returns, dtype=float)
        )
        if self.report_returns.shape[1] != self.n:
            raise ValueError("report_returns must have the same number of assets as returns")
        self.c = np.asarray(coefficients, float) if coefficients is not None else crra_coefficients(self.gamma)
        self.objective = MVSKObjective(self.returns, self.c)
        self._result: YANDResult | None = None
        self.weights: np.ndarray | None = None

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_prices(
        cls,
        prices,
        gamma: float = 6.0,
        horizon: int = 1,
        tickers: list[str] | None = None,
        max_weight: float = 1.0,
        periods_per_year: int = 252,
    ) -> "EfficientMVSK":
        """Build from a price matrix or a pandas DataFrame of close prices.

        Optimisation runs on ``horizon``-period buy-and-hold returns (where the
        higher moments are numerically meaningful); the performance report is
        computed on single-period (daily) returns and annualised with
        ``periods_per_year``.
        """
        if hasattr(prices, "columns"):  # pandas DataFrame
            if tickers is None:
                tickers = [str(c) for c in prices.columns]
            price_arr = prices.to_numpy(dtype=float)
        else:
            price_arr = np.asarray(prices, dtype=float)
        opt_rets = horizon_returns(price_arr, horizon=horizon)
        report_rets = horizon_returns(price_arr, horizon=1)
        return cls(
            opt_rets,
            gamma=gamma,
            tickers=tickers,
            max_weight=max_weight,
            report_returns=report_rets,
            periods_per_year=periods_per_year,
        )

    # ---------------------------------------------------------------- solve
    def optimize(self, **kwargs) -> np.ndarray:
        """Run YAND and return the optimal weight vector."""
        self._result = solve_yand(self.objective, cap=self.max_weight, **kwargs)
        self.weights = self._result.w
        return self.weights

    def clean_weights(self, cutoff: float = 1e-4, decimals: int = 4) -> dict[str, float]:
        """Round tiny weights to zero and renormalise; return ``{ticker: w}``."""
        if self.weights is None:
            self.optimize()
        w = np.where(self.weights < cutoff, 0.0, self.weights)
        s = w.sum()
        if s > 0:
            w = w / s
        w = np.round(w, decimals)
        return {t: float(wi) for t, wi in zip(self.tickers, w) if wi > 0}

    # ---------------------------------------------------------------- reporting
    @property
    def result(self) -> YANDResult:
        if self._result is None:
            self.optimize()
        assert self._result is not None
        return self._result

    def portfolio_returns(self, weights: np.ndarray | None = None, report: bool = True) -> np.ndarray:
        """Realised portfolio return series.

        ``report=True`` uses the reporting (daily) series; ``False`` uses the
        optimisation (horizon) series.
        """
        w = self.weights if weights is None else np.asarray(weights, float)
        if w is None:
            w = self.optimize()
        src = self.report_returns if report else self.returns
        return src @ w

    def portfolio_performance(self, verbose: bool = False, weights: np.ndarray | None = None) -> dict:
        """Return (and optionally print) the performance & moment summary.

        Moments used by the optimiser (variance/skew/kurtosis of the horizon
        returns) are reported alongside conventional daily-basis performance.
        """
        if self.weights is None and weights is None:
            self.optimize()
        w = self.weights if weights is None else np.asarray(weights, float)
        m1, m2, m3, m4 = self.objective.central_moments(w)  # on horizon returns
        port = self.report_returns @ w  # daily basis for reporting
        tm = tail_metrics(port, periods_per_year=self.periods_per_year)
        perf = {
            "expected_return": m1,
            "variance": m2,
            "volatility": float(np.sqrt(m2)),
            "skewness": tm.skewness,
            "excess_kurtosis": tm.excess_kurtosis,
            "ann_return": tm.ann_return,
            "ann_volatility": tm.ann_volatility,
            "sharpe": tm.sharpe,
            "sortino": tm.sortino,
            "var_95": tm.var_95,
            "cvar_95": tm.cvar_95,
            "max_drawdown": tm.max_drawdown,
            "gamma": self.gamma,
            "n_assets_held": int(np.sum(w > 1e-4)),
            "solver": self.result.as_dict(),
        }
        if verbose:
            print(f"Gamma (risk aversion):     {self.gamma:.2f}")
            print(f"Expected return (period):  {m1*100:+.3f}%")
            print(f"Annualised return:         {tm.ann_return*100:+.2f}%")
            print(f"Annualised volatility:     {tm.ann_volatility*100:.2f}%")
            print(f"Sharpe ratio:              {tm.sharpe:.2f}")
            print(f"Skewness:                  {tm.skewness:+.3f}")
            print(f"Excess kurtosis:           {tm.excess_kurtosis:+.3f}")
            print(f"95% VaR / CVaR:            {tm.var_95*100:.2f}% / {tm.cvar_95*100:.2f}%")
            print(f"Max drawdown:              {tm.max_drawdown*100:.2f}%")
            print(f"Assets held:               {perf['n_assets_held']} / {self.n}")
            print(f"Solve time:                {self.result.seconds*1000:.1f} ms "
                  f"({self.result.iterations} iters)")
        return perf
