# src/metacognition/monitor.py
"""
Implements the monitoring component of the metacognitive cycle.
Aggregates signals like uncertainty, novelty, and violations for evaluation (Section VI-A).
"""
from typing import Dict, Any, List
from .signals import SignalGenerator
import numpy as np  # For entropy/novelty calculations


class Monitor:
    """
    Monitors the system by aggregating signals (uncertainty, novelty, violations)
    to inform evaluation and regulation in the metacognitive loop.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            signal_gen: SignalGenerator,
            experience_pool=None,  # Optional for novelty detection via trace similarity
    ):
        self.config = config
        self.signal_gen = signal_gen
        self.experience_pool = experience_pool
        self.novelty_threshold = config.get('novelty_threshold', 0.7)  # High similarity = low novelty
        print("MONITOR: Initialized with novelty_threshold={:.2f}".format(self.novelty_threshold))

    def monitor(self, plan: List[Dict], state: Dict[str, Any]) -> Dict[str, float]:
        """
        Aggregates key signals for metacognitive evaluation.
        Returns a dict with uncertainty (u), violation prob (p_viol), and novelty.
        """
        signals = {}

        # 1) Uncertainty from parser/plan (Eq. 12)
        signals['uncertainty'] = self.signal_gen.compute_uncertainty(plan)

        # 2) Violation probability (Eq. 13)
        signals['p_viol'] = self.signal_gen.predict_violation_probability(plan, state)

        # 3) Novelty: Measure based on similarity to historical traces
        signals['novelty'] = self._compute_novelty(plan, state)

        print(
            f"MONITOR: Aggregated signals -> u={signals['uncertainty']:.2f}, p_viol={signals['p_viol']:.2f}, novelty={signals['novelty']:.2f}")

        # Optional: Aggregate constraint violations or invariants
        # e.g., signals['invariant_distance'] = self._check_invariants(state)

        return signals

    def _compute_novelty(self, plan: List[Dict], state: Dict[str, Any]) -> float:
        """
        Computes novelty as 1 - max_similarity to historical traces.
        Uses experience_pool for retrieval; mock low novelty if unavailable.
        """
        if not self.experience_pool:
            return 0.4  # Moderate novelty mock

        # Mock failure_info for retrieval (based on plan/state)
        failure_info = {'operator': plan[0]['operator'].name if plan else 'Unknown'}  # First op as proxy
        similar = self.experience_pool.retrieve_similar(failure_info, k=5)

        if not similar:
            return 1.0  # High novelty if no matches

        # Mock similarity: e.g., cosine-like on trace lengths (placeholder)
        similarities = [1.0 / (1 + abs(len(plan) - len(s['trace']))) for s in similar]
        max_sim = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_sim

        # Threshold: If novelty > threshold, flag for S2 escalation
        return novelty

    def _check_invariants(self, state: Dict[str, Any]) -> float:
        """
        Placeholder for invariant distance checks (e.g., state consistency).
        Returns a distance metric (0 = no violation).
        """
        # Example: Check if required fields are set
        required = ['payment_method', 'travel_dates']
        missing = [k for k in required if k not in state]
        return len(missing) / len(required) if required else 0.0