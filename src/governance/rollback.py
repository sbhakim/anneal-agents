# src/governance/rollback.py
"""
Implements patch rollback mechanisms for handling regressions.
Corresponds to Section IX-D of the paper (semantic versioning and safe rollback).
"""
import time
from typing import Dict, Any, Optional, List
from ..knowledge.rule_pool import RulePool
from ..utils.metrics import MetricsCollector
from .provenance import ProvenanceTracker
from .trust import TrustScorer


class RollbackManager:
    """
    Detects regressions post-patch and initiates rollbacks with versioning.
    Ties into trust scoring: rollback if ρ < threshold after failures.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rule_pool: RulePool,
        metrics: MetricsCollector,
        provenance: ProvenanceTracker,
        trust: TrustScorer,
        experience_pool=None,  # Optional for replay-based regression checks
    ):
        self.config = config
        self.rule_pool = rule_pool
        self.metrics = metrics
        self.provenance = provenance
        self.trust = trust
        self.experience_pool = experience_pool
        self.rollback_threshold = config.get('rollback', {}).get('trust_threshold', 0.5)
        self.window_size = config.get('rollback', {}).get('window_size', 10)  # Tasks to check post-patch
        # Store operator versions for rollback (simple dict for PoC)
        self.operator_versions: Dict[str, List[Dict]] = {}  # op_name -> list of {'version': str, 'preconditions': list, 'effects': list}
        print("ROLLBACK: Initialized with trust threshold={:.2f}".format(self.rollback_threshold))

    def _safe_metric(self, fn_name: str, default: float, **kwargs) -> float:
        fn = getattr(self.metrics, fn_name, None)
        if not callable(fn):
            return default
        try:
            return float(fn(**kwargs))
        except Exception:
            return default

    def _get_patch_rho(self, patch_id: str) -> float:
        """
        TrustScorer may store either rho directly or just (s,f). Compute if needed.
        """
        rep = getattr(self.trust, "patch_reputations", {}).get(patch_id, {}) or {}
        rho = rep.get("rho", None)
        if isinstance(rho, (int, float)):
            return float(rho)

        s = rep.get("s", None)
        f = rep.get("f", None)
        if isinstance(s, (int, float)) and isinstance(f, (int, float)):
            alpha = float(getattr(self.trust, "alpha", 1.0))
            beta = float(getattr(self.trust, "beta", 1.0))
            denom = alpha + beta + float(s) + float(f)
            return (alpha + float(s)) / denom if denom > 0 else 1.0

        return 1.0

    def version_operator(self, op_name: str) -> None:
        """
        Snapshots the current operator state for versioning before a patch.
        Called pre-commit in FDKA.
        """
        op = self.rule_pool.get_operator(op_name)
        if op:
            if op_name not in self.operator_versions:
                self.operator_versions[op_name] = []
            snapshot = {
                'version': op.metadata.get('version', '1.0'),
                'preconditions': op.preconditions.copy(),
                'effects': op.effects.copy(),
            }
            self.operator_versions[op_name].append(snapshot)
            print(f"ROLLBACK: Versioned {op_name} at v{snapshot['version']}")

    def check_for_regression(self, patch_id: str, failure_key: Optional[str] = None) -> bool:
        """
        Checks if a patch caused regressions (e.g., increased failure rate in window).
        Returns True if rollback is warranted.
        """
        # Use terminal failures when available; observed failures include recoveries/injections.
        fallback = self.metrics.calculate_rfr(window_size=self.window_size)
        recent_rfr = self._safe_metric("calculate_terminal_rfr", default=fallback, window_size=self.window_size)
        if recent_rfr > 0.05:  # Paper-style target: terminal RFR < 5%
            print(f"ROLLBACK: High terminal RFR ({recent_rfr:.1%}) detected post-patch {patch_id}")
            return True

        # Check trust score (ρ)
        rho = self._get_patch_rho(patch_id)
        if rho < self.rollback_threshold:
            print(f"ROLLBACK: Low trust ρ={rho:.2f} for patch {patch_id}")
            return True

        # Optional: Replay on experience pool to detect regressions
        if self.experience_pool and failure_key:
            similar = self.experience_pool.retrieve_similar({'error': failure_key}, k=5)
            if similar:
                regression_count = sum(1 for t in similar if not t.get('success', True))
                if (regression_count / len(similar)) > 0.2:
                    print(f"ROLLBACK: Regressions in {regression_count}/{len(similar)} similar traces")
                    return True

        return False

    def initiate_rollback(self, patch_id: str, op_name: str, justified: bool = True) -> bool:
        """
        Reverts the operator to the most recent snapshotted version (pre-patch).
        Logs the rollback and updates metrics/trust.
        """
        snapshot = None

        # Prefer explicit pre-commit snapshots.
        if op_name in self.operator_versions and self.operator_versions[op_name]:
            snapshot = self.operator_versions[op_name].pop()

        # Fallback: use RulePool operator_history if version_operator wasn't called.
        if snapshot is None:
            hist = getattr(self.rule_pool, "operator_history", {}).get(op_name, [])
            if hist:
                last = hist[-1]
                snapshot = {
                    'version': last.get('version', '1.0'),
                    'preconditions': list(last.get('preconditions', [])),
                    'effects': list(last.get('effects', [])),
                }

        if snapshot is None:
            print(f"ROLLBACK: No prior snapshot for {op_name}. Cannot rollback.")
            return False

        curr_op = self.rule_pool.get_operator(op_name)
        if curr_op:
            curr_op.preconditions = snapshot['preconditions']
            curr_op.effects = snapshot['effects']
            curr_op.metadata['version'] = snapshot['version']
            print(f"ROLLBACK: Reverted {op_name} to v{snapshot['version']}")

            # Log provenance
            prov_tuple = {
                "patch_id": patch_id,
                "event": "rollback",
                "justified": justified,
                "timestamp": time.time(),
            }
            self.provenance.log(prov_tuple)

            # Update trust (mark as failure)
            self.trust.update_trust_score(patch_id, success=False)

            # Record in metrics
            self.metrics.record_rollback(patch_id, justified=justified)

            return True
        return False
