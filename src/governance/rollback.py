# src/governance/rollback.py
"""
Implements patch rollback mechanisms for handling regressions.
Corresponds to Section IX-D of the paper (semantic versioning and safe rollback).
"""
import time
from typing import Dict, Any, Optional
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
        Checks if a patch caused regressions (e.g., increased RFR in window).
        Returns True if rollback is warranted.
        """
        # Use metrics to check RFR in recent window
        recent_rfr = self.metrics.calculate_rfr(window_size=self.window_size)
        if recent_rfr > 0.05:  # Threshold from paper's adaptation check (RFR < 5%)
            print(f"ROLLBACK: High RFR ({recent_rfr:.1%}) detected post-patch {patch_id}")
            return True

        # Check trust score
        rho = self.trust.patch_reputations.get(patch_id, {}).get('rho', 1.0)
        if rho < self.rollback_threshold:
            print(f"ROLLBACK: Low trust ρ={rho:.2f} for patch {patch_id}")
            return True

        # Optional: Replay on experience pool to detect regressions
        if self.experience_pool and failure_key:
            similar = self.experience_pool.retrieve_similar({'error': failure_key}, k=5)
            regression_count = sum(1 for t in similar if not t['success'])
            if regression_count / len(similar) > 0.2:
                print(f"ROLLBACK: Regressions in {regression_count}/{len(similar)} similar traces")
                return True

        return False

    def initiate_rollback(self, patch_id: str, op_name: str, justified: bool = True) -> bool:
        """
        Reverts the operator to the previous version if available.
        Logs the rollback and updates metrics/trust.
        """
        if op_name not in self.operator_versions or len(self.operator_versions[op_name]) < 2:
            print(f"ROLLBACK: No prior version for {op_name}. Cannot rollback.")
            return False

        # Pop the latest (faulty) version and restore previous
        prev_snapshot = self.operator_versions[op_name].pop()
        curr_op = self.rule_pool.get_operator(op_name)
        if curr_op:
            curr_op.preconditions = prev_snapshot['preconditions']
            curr_op.effects = prev_snapshot['effects']
            curr_op.metadata['version'] = prev_snapshot['version']
            print(f"ROLLBACK: Reverted {op_name} to v{prev_snapshot['version']}")

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