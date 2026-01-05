# src/metacognition/signals.py
"""
Defines and computes the operational signals for metacognitive control.
Corresponds to the concepts in Section 6.1 and 8.4 of the paper.

UPDATED FOR PoC: Replaced hardcoded mock values with dynamic heuristics
that align with the manuscript's descriptions:
- Uncertainty (u) is now based on the completeness of the plan's grounded parameters.
- Violation Probability (p_viol) is now based on the "precondition gap" - the
  fraction of preconditions in the plan that are not currently met.

MINIMUM DEFENSIBILITY UPDATE (2026-01):
- p_viol now looks ahead h steps (configurable) instead of only the first step.
- Uses a simple geometric decay so near-term violations matter more than far-term ones.
- Optionally mixes in a small "recent failures" prior from state when present, improving
  arbitration stability without adding new dependencies or learning code.
"""
from typing import Dict, Any, List
from ..knowledge.rule_pool import RulePool
from ..core.state import SymbolicState


class SignalGenerator:
    """
    Computes uncertainty and violation probability to guide arbitration.
    """

    def __init__(self, config: Dict[str, Any], rule_pool: RulePool):
        """
        Initializes the SignalGenerator.

        Args:
            config: Metacognition configuration.
            rule_pool: A reference to the RulePool to check preconditions.
        """
        self.config = config
        self.rule_pool = rule_pool

    def compute_uncertainty(self, plan: List[Dict[str, Any]]) -> float:
        """
        Estimates uncertainty (u) based on how well parameters are grounded in the plan.
        This serves as a PoC-friendly proxy for the token-level entropy described
        in Section 8.4.1 of the paper. A plan with missing or default parameters
        indicates the planner was uncertain.

        Args:
            plan: The current symbolic plan.

        Returns:
            Uncertainty score in [0, 1].
        """
        if not plan:
            return 1.0  # Maximum uncertainty if no plan could be formed

        total_params = 0
        ungrounded_params = 0

        for step in plan:
            op = step.get("operator")
            params = step.get("params", {}) or {}
            if not op:
                continue

            for expected_param in getattr(op, "params", []):
                total_params += 1
                if params.get(expected_param) is None:
                    ungrounded_params += 1

        uncertainty = (ungrounded_params / total_params) if total_params > 0 else 0.0
        uncertainty = max(0.0, min(1.0, float(uncertainty)))
        print(
            f"SIGNALS: Computed uncertainty u = {uncertainty:.2f} ({ungrounded_params}/{total_params} ungrounded params)"
        )
        return uncertainty

    def predict_violation_probability(self, plan: List[Dict[str, Any]], state: SymbolicState) -> float:
        """
        Predicts the probability of a constraint violation (p_viol) using a heuristic.
        This implements a lookahead "precondition gap" proxy:
        - checks up to `violation_horizon` upcoming operators
        - applies geometric decay so earlier failures weigh more

        Args:
            plan: The current symbolic plan.
            state: The current symbolic state.

        Returns:
            Predicted violation probability in [0, 1].
        """
        if not plan:
            return 1.0  # High probability of violation if there is no valid plan

        h = int(self.config.get("violation_horizon", 3))
        h = max(1, min(h, len(plan)))
        decay = float(self.config.get("violation_decay", 0.7))
        decay = max(0.0, min(1.0, decay))

        weighted_unmet = 0.0
        weighted_total = 0.0

        for i in range(h):
            step = plan[i]
            op = step.get("operator")
            params = step.get("params", {}) or {}
            if not op:
                continue

            preconds = getattr(op, "preconditions", []) or []
            total_preconditions = len(preconds)
            if total_preconditions == 0:
                continue

            unmet = 0
            for pred in preconds:
                try:
                    if not bool(pred(state, params)):
                        unmet += 1
                except Exception:
                    unmet += 1  # treat check errors as potential violations

            w = (decay ** i)
            weighted_unmet += w * unmet
            weighted_total += w * total_preconditions

        p_gap = (weighted_unmet / weighted_total) if weighted_total > 0 else 0.0

        # Optional "recent failures" prior from state (no new machinery required).
        # Any numeric in [0,1] is accepted; otherwise ignored.
        prior = 0.0
        try:
            if hasattr(state, "get"):
                val = state.get("recent_failure_rate")
                if val is None:
                    val = state.get("recent_failure_prior")
                if isinstance(val, (int, float)):
                    prior = max(0.0, min(1.0, float(val)))
        except Exception:
            prior = 0.0

        prior_w = float(self.config.get("recent_failure_prior_weight", 0.15))
        prior_w = max(0.0, min(1.0, prior_w))

        # Combine as a conservative union of risks (keeps bounds and is monotone).
        p_viol = 1.0 - ((1.0 - p_gap) * (1.0 - prior_w * prior))
        p_viol = max(0.0, min(1.0, float(p_viol)))

        print(
            f"SIGNALS: Predicted violation probability p_viol = {p_viol:.2f} "
            f"(gap={p_gap:.2f}, h={h}, decay={decay:.2f}, prior={prior:.2f})"
        )
        return p_viol
