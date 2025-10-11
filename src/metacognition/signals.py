# src/metacognition/signals.py
"""
Defines and computes the operational signals for metacognitive control.
Corresponds to the concepts in Section 6.1 and 8.4 of the paper.

UPDATED FOR PoC: Replaced hardcoded mock values with dynamic heuristics
that align with the manuscript's descriptions:
- Uncertainty (u) is now based on the completeness of the plan's grounded parameters.
- Violation Probability (p_viol) is now based on the "precondition gap" - the
  fraction of preconditions in the plan that are not currently met.
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
            params = step.get("params", {})
            if not op:
                continue

            # Check each expected parameter for the operator
            for expected_param in op.params:
                total_params += 1
                if params.get(expected_param) is None:
                    ungrounded_params += 1

        uncertainty = (ungrounded_params / total_params) if total_params > 0 else 0.0
        print(
            f"SIGNALS: Computed uncertainty u = {uncertainty:.2f} ({ungrounded_params}/{total_params} ungrounded params)")
        return uncertainty

    def predict_violation_probability(self, plan: List[Dict[str, Any]], state: SymbolicState) -> float:
        """
        Predicts the probability of a constraint violation (p_viol) using a heuristic.
        This implements the "precondition gap" feature described in Section 8.4.2
        of the paper. It checks what fraction of the immediate preconditions in the
        plan are currently not satisfied by the state.

        Args:
            plan: The current symbolic plan.
            state: The current symbolic state.

        Returns:
            Predicted violation probability in [0, 1].
        """
        if not plan:
            return 1.0  # High probability of violation if there is no valid plan

        # For a PoC, we'll just check the first step of the plan for simplicity.
        # A full implementation would check the next `h` operators.
        first_step = plan[0]
        op = first_step.get("operator")
        params = first_step.get("params", {})

        if not op:
            return 1.0

        total_preconditions = len(op.preconditions)
        unmet_preconditions = 0

        if total_preconditions == 0:
            return 0.0  # No preconditions means no chance of violation

        for pred in op.preconditions:
            try:
                # Check each precondition against the current state
                if not bool(pred(state, params)):
                    unmet_preconditions += 1
            except Exception:
                unmet_preconditions += 1  # Treat errors during check as potential violations

        p_viol = unmet_preconditions / total_preconditions
        print(
            f"SIGNALS: Predicted violation probability p_viol = {p_viol:.2f} ({unmet_preconditions}/{total_preconditions} unmet preconditions)")
        return p_viol