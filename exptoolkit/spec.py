"""Load and validate a decision plan (see plans/*.yaml)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

PLAN_DIR = Path(__file__).resolve().parent.parent / "plans"


@dataclass
class Economics:
    gross_margin: float
    cost_per_email: float
    min_profit_per_1000: float

    def profit_per_1000(self, revenue_lift_per_customer: float) -> float:
        """Incremental profit per 1,000 emails for a given revenue lift per customer."""
        return 1000 * (self.gross_margin * revenue_lift_per_customer - self.cost_per_email)

    @property
    def break_even_lift(self) -> float:
        """Revenue lift per customer that exactly pays for the email."""
        return self.cost_per_email / self.gross_margin

    @property
    def worthwhile_lift(self) -> float:
        """Revenue lift per customer needed to clear the minimum worthwhile profit."""
        return (self.cost_per_email + self.min_profit_per_1000 / 1000) / self.gross_margin


@dataclass
class DecisionSpec:
    name: str
    question: str
    owner: str
    arms: Dict[str, str]
    control_arm: str
    primary_metric: dict
    secondary_metrics: List[dict]
    guardrails: List[dict]
    economics: Economics
    decision_rule: dict
    statistics: dict
    source: Optional[Path] = field(default=None, repr=False)

    @property
    def treatment_arms(self) -> List[str]:
        return [a for a in self.arms if a != self.control_arm]

    def validate_against(self, df: pd.DataFrame, arm_col: str = "arm") -> List[str]:
        """Check that the data can answer this plan. Returns a list of problems."""
        problems = []
        missing_arms = set(self.arms) - set(df[arm_col].unique())
        if missing_arms:
            problems.append(f"Arms in the plan but not in the data: {sorted(missing_arms)}")
        columns = [self.primary_metric["column"]] + [m["column"] for m in self.secondary_metrics]
        cov = self.statistics.get("variance_reduction_covariate")
        if cov:
            columns.append(cov)
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            problems.append(f"Columns in the plan but not in the data: {missing_cols}")
        return problems

    def summary(self) -> pd.DataFrame:
        """One readable table of what was decided up front."""
        e, r = self.economics, self.decision_rule
        rows = [
            ("Question", self.question.strip()),
            ("Decision owner", self.owner),
            ("Options", ", ".join(self.arms.values())),
            ("Primary metric", self.primary_metric["label"]),
            ("Secondary metrics", ", ".join(m["label"] for m in self.secondary_metrics)),
            ("Guardrails", "; ".join(
                g["label"] + ("" if g.get("measured", True) else " (not measured)")
                for g in self.guardrails)),
            ("Gross margin", f"{e.gross_margin:.0%}"),
            ("Cost per email", f"${e.cost_per_email:.2f}"),
            ("Minimum worthwhile profit", f"${e.min_profit_per_1000:,.0f} per 1,000 emails"),
            ("Choose", r["choose"]),
            ("Ship if", f"probability best at least {r['ship_if']['probability_best_at_least']:.0%} "
                        f"and expected loss at most ${r['ship_if']['expected_loss_per_1000_at_most']:,.0f} per 1,000 emails"),
            ("Otherwise", r["otherwise"]),
        ]
        return pd.DataFrame(rows, columns=["", "Decided up front"]).set_index("")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid decision plan: {msg}")


def load_spec(path) -> DecisionSpec:
    """Read a plan file and check it is complete and sensible."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PLAN_DIR / path
    raw = yaml.safe_load(path.read_text())

    for key in ["name", "question", "arms", "control_arm", "primary_metric",
                "economics", "decision_rule", "statistics"]:
        _require(key in raw, f"missing '{key}'")
    _require(raw["control_arm"] in raw["arms"], "control_arm must be one of the arms")
    _require(len(raw["arms"]) >= 2, "need at least two arms")

    econ = Economics(**raw["economics"])
    _require(0 < econ.gross_margin <= 1, "gross_margin must be between 0 and 1")
    _require(econ.cost_per_email >= 0, "cost_per_email cannot be negative")
    _require(econ.min_profit_per_1000 >= 0, "min_profit_per_1000 cannot be negative")

    stats = raw["statistics"]
    _require(0 < stats.get("alpha", 0) < 0.5, "alpha must be between 0 and 0.5")
    _require(0.5 <= stats.get("power", 0) < 1, "power must be between 0.5 and 1")

    ship_if = raw["decision_rule"].get("ship_if", {})
    _require(0.5 <= ship_if.get("probability_best_at_least", 0) <= 1,
             "probability_best_at_least must be between 0.5 and 1")

    return DecisionSpec(
        name=raw["name"],
        question=raw["question"],
        owner=raw.get("owner", ""),
        arms=raw["arms"],
        control_arm=raw["control_arm"],
        primary_metric=raw["primary_metric"],
        secondary_metrics=raw.get("secondary_metrics", []),
        guardrails=raw.get("guardrails", []),
        economics=econ,
        decision_rule=raw["decision_rule"],
        statistics=stats,
        source=path,
    )
