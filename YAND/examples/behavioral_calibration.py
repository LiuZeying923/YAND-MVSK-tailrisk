"""Behavioral calibration: map VBQ-5 answers to gamma, then optimize + guard.

Run from the repo root:
    PYTHONPATH=src python examples/behavioral_calibration.py
"""

from yand_mvsk import BehavioralGammaOptimizer
from yand_mvsk.api.service import optimize_portfolio

# Three client personas with different behavioral answers.
personas = {
    "Long-horizon, holds through crashes": {"Q1": "A", "Q2": "B", "Q3": "A", "Q4": "A", "Q5": "A"},
    "Typical retail investor":              {"Q1": "B", "Q2": "B", "Q3": "B", "Q4": "B", "Q5": "B"},
    "Loss-averse, panic-sold before":       {"Q1": "C", "Q2": "A", "Q3": "C", "Q4": "D", "Q5": "C"},
}

bgo = BehavioralGammaOptimizer()
print("VBQ-5 behavioral gamma calibration")
print("=" * 60)
for name, answers in personas.items():
    bd = bgo.calculate(answers)
    print(f"\n{name}")
    print(f"  gamma = {bd.gamma:.2f}   profile: {bd.profile}")
    for c in bd.contributions:
        print(f"    {c['question']} ({c['bias']}): {c['answer_label']} -> {c['effect']}")

# Run the full pipeline for the loss-averse persona on a real-ish basket.
print("\n" + "=" * 60)
print("Optimizing for the loss-averse persona:\n")
out = optimize_portfolio(
    tickers=["NVDA", "0700.HK", "600519.SS", "TLT", "GLD"],
    answers=personas["Loss-averse, panic-sold before"],
    offline=True,
    max_weight=0.35,
)
print(f"  Calibrated gamma: {out['gamma']}  ({out['gamma_breakdown']['profile']})")
print("  Weights:", {r["symbol"]: round(r["weight"], 3) for r in out["weights"]})
print(f"  Tail-Risk Guard: {out['risk']['score']}/100 ({out['risk']['level']})")
if out.get("recommendation"):
    print(f"  Recommendation: {out['recommendation']['how']}")
