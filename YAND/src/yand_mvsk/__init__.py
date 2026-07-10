"""YAND-MVSK: tail-risk-aware higher-moment portfolio optimization.

Public API::

    from yand_mvsk import EfficientMVSK, BehavioralGammaOptimizer, RiskMonitor
"""

from __future__ import annotations

from .behavioral import BehavioralGammaOptimizer, VBQ5, calculate_behavioral_gamma
from .moments import MVSKObjective, crra_coefficients, horizon_returns
from .optimizer import EfficientMVSK
from .risk import RiskMonitor, StressTester, TailRiskGuard, tail_metrics
from .yand import YANDResult, solve_yand

__version__ = "0.1.0"

__all__ = [
    "EfficientMVSK",
    "MVSKObjective",
    "YANDResult",
    "solve_yand",
    "crra_coefficients",
    "horizon_returns",
    "BehavioralGammaOptimizer",
    "VBQ5",
    "calculate_behavioral_gamma",
    "RiskMonitor",
    "StressTester",
    "TailRiskGuard",
    "tail_metrics",
    "__version__",
]
