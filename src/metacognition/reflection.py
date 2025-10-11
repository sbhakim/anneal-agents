# src/metacognition/reflection.py
"""
Implements post-execution reflection for updating policies and thresholds.
Corresponds to the reflection process in Section VI-C of the paper.
"""
from typing import Dict, Any, List
from ..utils.metrics import MetricsCollector
from .arbitrator import Arbitrator


class Reflection:
    """
    Performs post-execution analysis to update metacognitive policies,
    thresholds, and retrieval libraries based on observed outcomes.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        arbitrator: Arbitrator,
        metrics: MetricsCollector,
        experience_pool=None,  # Optional for updating retrieval indices
    ):
        self.config = config
        self.arbitrator = arbitrator
        self.metrics = metrics
        self.experience_pool = experience_pool
        self.enable_reflection = config.get('enable_reflection', True)
        self.tau_adjust_step = config.get('tau_adjust_step', 0.05)  # Increment for threshold tuning
        self.window_size = config.get('reflection_window', 10)  # Tasks to consider for analysis
        print("REFLECTION: Initialized with enable_reflection={}".format(self.enable_reflection))

    def reflect(self, trace: List[Dict], success: bool, task_id: int) -> None:
        """
        Post-execution reflection: Analyzes outcome and updates thresholds/policies.
        Based on observed successes/failures, tunes risk/speed trade-offs (e.g., adjust τ_u, τ_p).
        """
        if not self.enable_reflection:
            print("REFLECTION: Disabled. Skipping.")
            return

        print("\n--- REFLECTION INITIATED ---")

        # 1) Analyze recent performance using metrics
        recent_rfr = self.metrics.calculate_rfr(window_size=self.window_size)
        recent_csr = self.metrics.calculate_csr()
        print(f"REFLECTION: Recent RFR={recent_rfr:.1%}, CSR={recent_csr:.1%}")

        # 2) Detect patterns: e.g., high RFR -> more caution (increase thresholds to trigger S2 more)
        if recent_rfr > 0.1:  # High repeat failures -> be more deliberative
            self.arbitrator.tau_u += self.tau_adjust_step
            self.arbitrator.tau_p += self.tau_adjust_step
            self.arbitrator.tau_u = min(self.arbitrator.tau_u, 0.8)  # Cap to avoid over-caution
            self.arbitrator.tau_p = min(self.arbitrator.tau_p, 0.6)
            print(f"REFLECTION: High RFR detected. Increased τ_u to {self.arbitrator.tau_u:.2f}, τ_p to {self.arbitrator.tau_p:.2f}")

        elif recent_csr > 0.9 and recent_rfr < 0.05:  # High success -> more aggressive (lower thresholds for faster S1)
            self.arbitrator.tau_u -= self.tau_adjust_step
            self.arbitrator.tau_p -= self.tau_adjust_step
            self.arbitrator.tau_u = max(self.arbitrator.tau_u, 0.2)  # Floor to maintain safety
            self.arbitrator.tau_p = max(self.arbitrator.tau_p, 0.1)
            print(f"REFLECTION: High success detected. Decreased τ_u to {self.arbitrator.tau_u:.2f}, τ_p to {self.arbitrator.tau_p:.2f}")

        # 3) Update retrieval libraries if experience_pool available
        if self.experience_pool and not success:
            # Example: Index the failure trace for better future retrieval
            failure_info = trace[-1] if trace else {}
            metadata = {
                'operator': failure_info.get('operator'),
                'error_type': failure_info.get('error'),
                'task_id': task_id,
            }
            self.experience_pool.add_trace(trace, success=False, metadata=metadata)
            print("REFLECTION: Added failure trace to experience pool for improved retrieval.")

        # 4) Check for adaptation and log
        failure_key = f"{failure_info.get('operator')}:{failure_info.get('error')}" if not success else None
        if failure_key and self.metrics.check_adaptation(failure_key, window_size=self.window_size):
            print("REFLECTION: Adaptation confirmed for failure class '{}'".format(failure_key))

        print("--- REFLECTION COMPLETED ---\n")