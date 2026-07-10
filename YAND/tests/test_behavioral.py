"""VBQ-5 behavioral gamma calibration."""

from __future__ import annotations

import pytest

from yand_mvsk.behavioral import (
    GAMMA_CAP,
    GAMMA_FLOOR,
    BehavioralGammaOptimizer,
    calculate_behavioral_gamma,
    questionnaire_schema,
)


def test_neutral_answers_near_base():
    g = calculate_behavioral_gamma({"Q1": "B", "Q2": "B", "Q3": "B", "Q4": "B", "Q5": "B"})
    assert 6.0 <= g <= 9.0  # middle answers -> moderate band


def test_brave_answers_reach_aggressive():
    # Long horizon, buys dips, humble forecast, held through crash, sells loser.
    g = calculate_behavioral_gamma({"Q1": "A", "Q2": "B", "Q3": "A", "Q4": "A", "Q5": "A"})
    assert g < 3.5  # aggressive band reachable


def test_fearful_answers_hit_cap():
    bd = BehavioralGammaOptimizer().calculate(
        {"Q1": "C", "Q2": "A", "Q3": "C", "Q4": "D", "Q5": "C"}
    )
    assert bd.raw_gamma > GAMMA_CAP
    assert bd.gamma == GAMMA_CAP
    assert bd.capped is True
    assert bd.profile == "Conservative"


def test_disposition_is_additive():
    base = calculate_behavioral_gamma({"Q2": "B"})
    sold_winner = calculate_behavioral_gamma({"Q2": "A"})
    assert sold_winner == pytest.approx(base + 2.0)


def test_gamma_stays_in_bounds_for_all_combos():
    from itertools import product
    from yand_mvsk.behavioral import VBQ5

    opt = BehavioralGammaOptimizer()
    keys = [[c.key for c in q.choices] for q in VBQ5]
    for combo in product(*keys):
        ans = {q.id: k for q, k in zip(VBQ5, combo)}
        g = opt.calculate_gamma(ans)
        assert GAMMA_FLOOR <= g <= GAMMA_CAP


def test_breakdown_is_explainable():
    bd = BehavioralGammaOptimizer().calculate({"Q1": "C", "Q3": "C"})
    assert len(bd.contributions) == 2
    assert all("effect" in c and "bias" in c for c in bd.contributions)
    d = bd.as_dict()
    assert d["gamma"] == pytest.approx(bd.gamma) and "profile" in d


def test_invalid_answer_raises():
    with pytest.raises(ValueError):
        BehavioralGammaOptimizer().calculate({"Q1": "Z"})


def test_questionnaire_schema_shape():
    qs = questionnaire_schema()
    assert len(qs) == 5
    assert {q["id"] for q in qs} == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert all(q["choices"] for q in qs)
