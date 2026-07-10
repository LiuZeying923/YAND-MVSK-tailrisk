"""VBQ-5: Visual Behavioral Questionnaire and behavioral-gamma calibration.

Traditional risk questionnaires ask "how much loss can you tolerate?" and get
answers people don't live up to when a market actually falls.  VBQ-5 instead
probes three well-documented biases with relative, scenario-based prompts:

  * loss aversion (Q1, Q5) -- pain of a loss outweighs the pleasure of a gain,
  * disposition effect (Q2) -- selling winners while clinging to losers,
  * overconfidence (Q3)     -- prediction intervals that are far too narrow,

plus time horizon (Q4), which is preference rather than bias but drives risk
capacity directly.  Detected biases *raise* the effective risk-aversion
``gamma`` fed to the optimiser (so a client who behaves fearfully under stress
gets a more tail-averse portfolio than their stated appetite implies), while
genuinely risk-tolerant answers -- buying dips, a decade-long horizon, having
actually held through a crash -- *lower* it.  This two-sided calibration is
what lets the questionnaire reach the aggressive (gamma 2-3), moderate (6-8)
and conservative (12-20) bands the README describes; a purely upward adjustment
could never produce an aggressive profile.

The upward multipliers match the table published in the project README
(loss-aversion C -> x1.4, narrow-band C -> x1.5, panic-sell C -> x1.6, the
disposition +2.0); the downward multipliers for the brave answers are the
symmetric extension.  Everything stays deliberately simple and transparent --
a calibration heuristic a human advisor can read and sanity-check, not a fitted
black box.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "VBQ5",
    "BehavioralGammaOptimizer",
    "calculate_behavioral_gamma",
    "questionnaire_schema",
    "GammaBreakdown",
    "PROFILE_BANDS",
]

GAMMA_CAP = 20.0
GAMMA_FLOOR = 1.5
DEFAULT_BASE_GAMMA = 6.0

# Client profile bands (gamma range -> label + blurb), from the README.
PROFILE_BANDS: tuple[tuple[float, float, str, str], ...] = (
    (12.0, GAMMA_CAP, "Conservative", "Capital preservation focus; retirees and the loss-sensitive."),
    (8.0, 12.0, "Moderately conservative", "Prefers stability, tolerates mild drawdowns."),
    (5.0, 8.0, "Moderate", "The typical retail investor; balanced growth and protection."),
    (3.0, 5.0, "Moderately aggressive", "Growth-oriented, comfortable with volatility."),
    (GAMMA_FLOOR, 3.0, "Aggressive", "Long horizon, seeks upside, rides out tail events."),
)


@dataclass(frozen=True)
class Choice:
    key: str
    label: str


@dataclass(frozen=True)
class Question:
    id: str
    bias: str
    prompt: str
    choices: tuple[Choice, ...]
    # effect: choice key -> ("mul", factor) or ("add", amount)
    effect: dict


# The five questions, verbatim in intent with the README's VBQ-5 table.
VBQ5: tuple[Question, ...] = (
    Question(
        id="Q1",
        bias="Loss aversion",
        prompt=(
            "The market falls, but your stock drops 10% more than the market "
            "(e.g. market −5%, your stock −15%). What do you do?"
        ),
        choices=(
            Choice("A", "Buy more — “it’s on sale”"),
            Choice("B", "Hold and wait it out"),
            Choice("C", "Sell / cut losses"),
        ),
        # Buying the dip signals genuine risk appetite -> discount gamma; cutting
        # losses signals loss aversion -> raise it (the README's C -> x1.4).
        effect={"A": ("mul", 0.8), "B": ("mul", 1.05), "C": ("mul", 1.4)},
    ),
    Question(
        id="Q2",
        bias="Disposition effect",
        prompt=(
            "You urgently need cash and must sell exactly one holding: "
            "Stock A is up +20% (a winner), Stock B is down −20% (a loser). "
            "Which do you sell?"
        ),
        choices=(
            Choice("A", "Sell the winner (Stock A)"),
            Choice("B", "Sell the loser (Stock B)"),
        ),
        # Hard additive penalty: selling winners & holding losers is the classic
        # disposition mistake and it compounds tail risk, so it adds rather than
        # scales.
        effect={"A": ("add", 2.0), "B": ("add", 0.0)},
    ),
    Question(
        id="Q3",
        bias="Overconfidence",
        prompt=(
            "Predict next year’s level of your main index (S&P 500 / CSI 300). "
            "You are 80% sure it lands within which range?"
        ),
        choices=(
            Choice("A", "Very wide (±30% or more)"),
            Choice("B", "Medium (±15%)"),
            Choice("C", "Very narrow (±5%)"),
        ),
        # A narrow band signals overconfidence -> under-estimated risk -> raise gamma.
        # A wide, humble band signals calibrated uncertainty -> mild discount.
        effect={"A": ("mul", 0.9), "B": ("mul", 1.1), "C": ("mul", 1.5)},
    ),
    Question(
        id="Q4",
        bias="Time horizon",
        prompt="How long until you need this money?",
        choices=(
            Choice("A", "10+ years"),
            Choice("B", "3–10 years"),
            Choice("C", "1–3 years"),
            Choice("D", "Less than 1 year"),
        ),
        # A long horizon can ride out tail events -> discount; a short one cannot.
        effect={"A": ("mul", 0.75), "B": ("mul", 1.0), "C": ("mul", 1.3), "D": ("mul", 1.6)},
    ),
    Question(
        id="Q5",
        bias="Realized panic response",
        prompt=(
            "Recall a time your portfolio actually fell a lot (say 10–20% of total "
            "assets). What did you really do?"
        ),
        choices=(
            Choice("A", "Held, or bought more"),
            Choice("B", "Anxious, but did nothing"),
            Choice("C", "Panic-sold near the bottom"),
        ),
        # Actually holding through a crash is the strongest signal of true risk
        # tolerance -> discount; panic-selling -> raise (the README's C -> x1.6).
        effect={"A": ("mul", 0.8), "B": ("mul", 1.1), "C": ("mul", 1.6)},
    ),
)

_QUESTION_BY_ID = {q.id: q for q in VBQ5}


@dataclass
class GammaBreakdown:
    """Explainable trace of how the behavioral gamma was produced."""

    base_gamma: float
    multiplier: float
    additive: float
    raw_gamma: float
    gamma: float
    capped: bool
    profile: str
    profile_blurb: str
    contributions: list[dict]

    def as_dict(self) -> dict:
        return {
            "base_gamma": round(self.base_gamma, 4),
            "multiplier": round(self.multiplier, 4),
            "additive": round(self.additive, 4),
            "raw_gamma": round(self.raw_gamma, 4),
            "gamma": round(self.gamma, 4),
            "capped": self.capped,
            "profile": self.profile,
            "profile_blurb": self.profile_blurb,
            "contributions": self.contributions,
        }


def _profile_for(gamma: float) -> tuple[str, str]:
    for lo, hi, name, blurb in PROFILE_BANDS:
        if lo <= gamma < hi or (hi >= GAMMA_CAP and gamma >= lo):
            return name, blurb
    return PROFILE_BANDS[-1][2], PROFILE_BANDS[-1][3]


class BehavioralGammaOptimizer:
    """Map VBQ-5 answers to a behaviorally-calibrated risk-aversion ``gamma``.

    ``final_gamma = clip(base_gamma * product(multipliers) + sum(additions),
    floor, cap)``.  Multiplicative factors model biases that scale fear;
    additive terms model discrete, hard mistakes (the disposition effect).
    """

    def __init__(self, base_gamma: float = DEFAULT_BASE_GAMMA) -> None:
        if not GAMMA_FLOOR <= base_gamma <= GAMMA_CAP:
            raise ValueError(f"base_gamma must be in [{GAMMA_FLOOR}, {GAMMA_CAP}]")
        self.base_gamma = float(base_gamma)

    def calculate(self, answers: dict[str, str]) -> GammaBreakdown:
        multiplier = 1.0
        additive = 0.0
        contributions: list[dict] = []

        for q in VBQ5:
            ans = answers.get(q.id)
            if ans is None:
                continue
            ans = str(ans).strip().upper()
            if ans not in q.effect:
                raise ValueError(f"{q.id}: invalid answer {ans!r}; expected one of {sorted(q.effect)}")
            kind, amount = q.effect[ans]
            chosen = next((c.label for c in q.choices if c.key == ans), ans)
            if kind == "mul":
                multiplier *= amount
                delta_desc = f"×{amount:g}" if amount != 1.0 else "no change"
            else:  # add
                additive += amount
                delta_desc = f"+{amount:g}" if amount != 0.0 else "no change"
            contributions.append(
                {"question": q.id, "bias": q.bias, "answer": ans,
                 "answer_label": chosen, "effect": delta_desc}
            )

        raw = self.base_gamma * multiplier + additive
        gamma = min(max(raw, GAMMA_FLOOR), GAMMA_CAP)
        name, blurb = _profile_for(gamma)
        return GammaBreakdown(
            base_gamma=self.base_gamma,
            multiplier=multiplier,
            additive=additive,
            raw_gamma=raw,
            gamma=gamma,
            capped=raw > GAMMA_CAP or raw < GAMMA_FLOOR,
            profile=name,
            profile_blurb=blurb,
            contributions=contributions,
        )

    def calculate_gamma(self, answers: dict[str, str]) -> float:
        """Convenience: return just the calibrated gamma scalar."""
        return self.calculate(answers).gamma


def calculate_behavioral_gamma(answers: dict[str, str], base_gamma: float = DEFAULT_BASE_GAMMA) -> float:
    """Functional shortcut matching the README's ``calculate_behavioral_gamma``."""
    return BehavioralGammaOptimizer(base_gamma).calculate_gamma(answers)


def questionnaire_schema() -> list[dict]:
    """JSON-serialisable VBQ-5 description for the frontend."""
    return [
        {
            "id": q.id,
            "bias": q.bias,
            "prompt": q.prompt,
            "choices": [{"key": c.key, "label": c.label} for c in q.choices],
        }
        for q in VBQ5
    ]
