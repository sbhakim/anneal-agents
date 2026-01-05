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
        self.tau_adjust_step = config.get('tau_adjust_step', 0.05)  # Step for threshold tuning
        self.window_size = config.get('reflection_window', 10)  # Tasks to consider for analysis
        print("REFLECTION: Initialized with enable_reflection={}".format(self.enable_reflection))

    def _safe_metric(self, fn_name: str, default: float, **kwargs) -> float:
        fn = getattr(self.metrics, fn_name, None)
        if not callable(fn):
            return default
        try:
            return float(fn(**kwargs))
        except Exception:
            return default

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
        # Use terminal RFR for safety-critical tuning; observed RFR is still useful for diagnostics.
        observed_rfr = self._safe_metric("calculate_observed_rfr", default=self.metrics.calculate_rfr(window_size=self.window_size),
                                        window_size=self.window_size)
        terminal_rfr = self._safe_metric("calculate_terminal_rfr", default=observed_rfr, window_size=self.window_size)
        recent_csr = self.metrics.calculate_csr()
        print(f"REFLECTION: Recent terminal RFR={terminal_rfr:.1%}, observed RFR={observed_rfr:.1%}, CSR={recent_csr:.1%}")

        # 2) Detect patterns: terminal failures -> more caution (increase thresholds to trigger S2 more)
        if terminal_rfr > 0.1:
            self.arbitrator.tau_u += self.tau_adjust_step
            self.arbitrator.tau_p += self.tau_adjust_step
            self.arbitrator.tau_u = min(self.arbitrator.tau_u, 0.8)  # Cap to avoid over-caution
            self.arbitrator.tau_p = min(self.arbitrator.tau_p, 0.6)
            print(
                f"REFLECTION: High terminal RFR detected. Increased τ_u to {self.arbitrator.tau_u:.2f}, τ_p to {self.arbitrator.tau_p:.2f}"
            )

        elif recent_csr > 0.9 and terminal_rfr < 0.05:
            self.arbitrator.tau_u -= self.tau_adjust_step
            self.arbitrator.tau_p -= self.tau_adjust_step
            self.arbitrator.tau_u = max(self.arbitrator.tau_u, 0.2)  # Floor to maintain safety
            self.arbitrator.tau_p = max(self.arbitrator.tau_p, 0.1)
            print(
                f"REFLECTION: High success detected. Decreased τ_u to {self.arbitrator.tau_u:.2f}, τ_p to {self.arbitrator.tau_p:.2f}"
            )

        # 3) Update retrieval libraries if experience_pool available
        failure_info: Dict[str, Any] = trace[-1] if trace else {}
        err = failure_info.get('error_type') or failure_info.get('error')  # tolerate legacy key
        if self.experience_pool and not success:
            metadata = {
                'operator': failure_info.get('operator'),
                'error_type': err,
                'task_id': task_id,
            }
            self.experience_pool.add_trace(trace, success=False, metadata=metadata)
            print("REFLECTION: Added failure trace to experience pool for improved retrieval.")

        # 4) Check for adaptation and log
        failure_key = f"{failure_info.get('operator')}:{err}" if not success and err else None
        if failure_key and self.metrics.check_adaptation(failure_key, window_size=self.window_size):
            print("REFLECTION: Adaptation confirmed for failure class '{}'".format(failure_key))

        print("--- REFLECTION COMPLETED ---\n")
