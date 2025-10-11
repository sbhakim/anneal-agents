# src/baselines/verify_only.py
"""
Verify-Before-Act Only Baseline (Verify-Only)
Corresponds to manuscript Section XII.2 baseline definition.

This baseline represents traditional neurosymbolic systems with precondition checking:
- Has symbolic state and operators (neurosymbolic core)
- Includes verify-before-act for precondition checking
- Catches constraint violations BEFORE execution
- NO FDKA: Cannot learn new preconditions or adapt operators
- NO reflection: No memory of past failures
- NO metacognitive arbitration: Direct execution only

Key difference from SelfEvolve:
- Static preconditions: Can only check what was initially encoded
- If initial preconditions miss a constraint, task fails repeatedly
- Better than Static-NS (early violation detection)
- Worse than SelfEvolve (cannot learn missing constraints)

Expected behavior: ~70% success rate, TTA = ∞
- Prevents failures where preconditions are already encoded
- Fails repeatedly on novel constraints (e.g., policy changes)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.state import SymbolicState
from ..core.planner import Planner
from ..knowledge.rule_pool import RulePool
from ..scenarios.travel_planning import TravelPlanningScenario
from ..utils.metrics import MetricsCollector
from ..utils.logger import setup_logger


class VerifyOnlyAgent:
    """
    Verify-Before-Act baseline without self-evolution.

    Architecture:
    1. Plan using HTN compilation (symbolic)
    2. Verify preconditions BEFORE each action
    3. If verification passes, execute
    4. If verification fails, halt (no learning)
    5. If execution fails despite passing verification, halt (no adaptation)

    Key limitation: Preconditions are STATIC
    - Cannot add new preconditions discovered from failures
    - If a constraint is missing from initial encoding, system fails indefinitely
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Verify-Only baseline.

        Args:
            config: System configuration
        """
        self.config = config

        # Setup logger
        log_config = config.get('logging', {})
        log_config['level'] = log_config.get('level', 'WARNING')
        self.logger = setup_logger(log_config, name="verify_only")

        self.logger.info("=" * 70)
        self.logger.info("Initializing Verify-Only Baseline...")
        self.logger.info("=" * 70)

        # Core neurosymbolic components (KEEP)
        self.state = SymbolicState()
        self.rule_pool = RulePool(config['knowledge']['rule_pool_path'])
        self.planner = Planner(config['planner'], self.rule_pool)
        self.scenario = TravelPlanningScenario(config['scenario'])

        # Metrics collector
        self.metrics = MetricsCollector()

        # Enable verification flag (always True for this baseline)
        self.enable_verification = True

        # REMOVED: FDKA pipeline
        # REMOVED: Metacognitive arbitration (no S1/S2 routing)
        # REMOVED: Reflection or memory
        # REMOVED: Governance (no provenance, trust, canary)

        # Track verification statistics
        self.verification_stats = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "caught_violations": []
        }

        self.logger.info("✅ Verify-Only baseline ready (static preconditions)")
        self.logger.info("=" * 70)

    def run_evaluation(self) -> MetricsCollector:
        """
        Run evaluation with verify-before-act only.

        Returns:
            MetricsCollector with results
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 VERIFY-ONLY BASELINE EVALUATION")
        print(f"{'=' * 70}")

        for task_id in range(self.scenario.num_tasks):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self._run_task_with_verification(task_id, instruction)

        print(f"\n{'=' * 70}")
        print(f"✅ VERIFY-ONLY EVALUATION COMPLETE")
        print(f"{'=' * 70}")

        # Print verification statistics
        self._print_verification_stats()

        # Print summary
        self.metrics.print_summary()

        return self.metrics

    def _run_task_with_verification(self, task_id: int, instruction: str) -> None:
        """
        Run a single task with verify-before-act.

        Process:
        1. Parse instruction and update state
        2. Generate plan
        3. For each step:
           a. Verify preconditions
           b. If pass: execute
           c. If fail: halt with verification error
        4. No retry or learning on failure

        Args:
            task_id: Task identifier
            instruction: Natural language instruction
        """
        print(f"\n{'=' * 70}")
        print(f"Task {task_id + 1}: {instruction}")
        print(f"{'=' * 70}")

        # Update state from instruction
        self.state.update_from_instruction(instruction)
        self._reset_world_for_task()

        # Compile plan
        plan = self.planner.compile(instruction, self.state.to_dict())

        if not plan:
            print("❌ Planning failed (no operators matched)")
            self.metrics.record_task(task_id, False, [{"error": "PlanningFailed"}])
            return

        # Execute with verification (single attempt - no retry)
        print("\n--- Execution with Verification ---")
        trace, success, failure_type = self._execute_with_verification(
            plan, task_id, instruction
        )

        if success:
            print("✅ Task Succeeded")
            self.metrics.record_task(task_id, True, trace)
        else:
            print(f"❌ Task Failed: {failure_type}")
            print("   [Verify-Only: Cannot adapt - preconditions are static]")
            self.metrics.record_task(task_id, False, trace)

            # Track failure for RFR
            failure_info = self._extract_failure_info(trace)
            if failure_info:
                failure_key = f"{failure_info.get('operator')}:{failure_info.get('error')}"
                self.metrics.check_adaptation(failure_key, window_size=20)

    def _execute_with_verification(
            self,
            plan: List[Dict[str, Any]],
            task_id: int,
            instruction: str
    ) -> Tuple[List[Dict], bool, Optional[str]]:
        """
        Execute plan with verify-before-act.

        Key behavior:
        - Check preconditions before each step
        - If preconditions fail: halt immediately (early detection)
        - If preconditions pass but execution fails: still halt (cannot adapt)

        Returns:
            (trace, success, failure_type)
        """
        trace = []

        for i, step in enumerate(plan):
            op = step["operator"]
            params = step.get("params", {})

            print(f"  Step {i + 1}: {op.name}")

            # Sync state (for payment method visibility)
            if "payment" in params:
                self.state.set("payment_method", params["payment"])

            # Apply world defaults
            self._apply_world_defaults()

            trace.append({
                "step": i,
                "operator": op.name,
                "params": params,
                "state_before": self.state.to_dict()
            })

            # VERIFY PRECONDITIONS (key feature of this baseline)
            self.verification_stats["total_checks"] += 1

            ok, violated = self._verify_preconditions(op, params)

            if not ok:
                # Precondition failed - caught early!
                self.verification_stats["failed"] += 1
                self.verification_stats["caught_violations"].append({
                    "task_id": task_id,
                    "operator": op.name,
                    "violated": violated
                })

                print(f"    ❌ Verification FAILED: {violated}")
                print(f"    [Caught constraint violation before execution]")

                trace.append({
                    "error": "PreconditionUnmet",
                    "operator": op.name,
                    "violated_precondition": violated,
                    "caught_by_verification": True
                })

                return trace, False, "PreconditionUnmet"

            # Preconditions passed
            self.verification_stats["passed"] += 1
            print(f"    ✓ Verification PASSED")

            # Execute (simulate with failure injection)
            # Even with verification, execution can fail (e.g., network errors)
            should_fail = self.scenario.should_inject_failure(
                task_id, op.name, params, self.state
            )

            if should_fail:
                # Execution failed despite passing verification
                failure_info = self.scenario.get_failure_details(
                    task_id, op.name, params, self.state
                )

                print(f"    ❌ Execution FAILED: {failure_info.get('error')}")
                print(f"    [Failure not caught by static preconditions]")

                trace.append({
                    "error": failure_info.get("error"),
                    "operator": op.name,
                    "message": failure_info.get("message"),
                    "caught_by_verification": False
                })

                return trace, False, failure_info.get("error")

            # Success: apply effects
            self.state = op.apply_effects(self.state, params)

            trace.append({
                "step": i,
                "operator": op.name,
                "state_after": self.state.to_dict(),
                "success": True
            })

            print(f"    ✓ Execution SUCCEEDED")

        return trace, True, None

    def _verify_preconditions(
            self,
            op: Any,
            params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check preconditions using rule pool.

        This is the core mechanism: formal verification before execution.

        Args:
            op: Operator to verify
            params: Grounded parameters

        Returns:
            (ok, violated_predicate_name)
        """
        if not self.enable_verification:
            return True, None

        return self.rule_pool.evaluate_preconditions(op.name, params, self.state)

    def _apply_world_defaults(self) -> None:
        """Apply world defaults from scenario."""
        if not hasattr(self.scenario, "world_defaults"):
            return

        defaults = self.scenario.world_defaults()
        for k, v in defaults.items():
            if self.state.get(k) is None:
                self.state.set(k, v)

    def _reset_world_for_task(self) -> None:
        """Reset world state for new task."""
        if not hasattr(self.scenario, "world_defaults"):
            return

        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults:
                self.state.set(k, defaults[k])
        for k in ("corporate_card_policy", "blackout_dates"):
            if k in defaults and self.state.get(k) is None:
                self.state.set(k, defaults[k])

    def _extract_failure_info(self, trace: List[Dict]) -> Optional[Dict]:
        """Extract failure information from trace."""
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return None

    def _print_verification_stats(self) -> None:
        """Print verification statistics."""
        stats = self.verification_stats

        print("\n" + "=" * 70)
        print("📋 VERIFICATION STATISTICS")
        print("=" * 70)

        print(f"Total Verification Checks: {stats['total_checks']}")
        print(f"  Passed: {stats['passed']} ({stats['passed'] / stats['total_checks'] * 100:.1f}%)")
        print(f"  Failed: {stats['failed']} ({stats['failed'] / stats['total_checks'] * 100:.1f}%)")

        if stats['caught_violations']:
            print(f"\nViolations Caught by Verification: {len(stats['caught_violations'])}")
            for i, v in enumerate(stats['caught_violations'][:5], 1):  # Show first 5
                print(f"  {i}. Task {v['task_id']}: {v['operator']} - {v['violated']}")
            if len(stats['caught_violations']) > 5:
                print(f"  ... and {len(stats['caught_violations']) - 5} more")

        print("=" * 70)

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics for analysis."""
        stats = self.verification_stats.copy()

        if stats['total_checks'] > 0:
            stats['pass_rate'] = stats['passed'] / stats['total_checks']
            stats['fail_rate'] = stats['failed'] / stats['total_checks']
        else:
            stats['pass_rate'] = 0.0
            stats['fail_rate'] = 0.0

        stats['num_caught_violations'] = len(stats['caught_violations'])

        return stats


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test Verify-Only baseline in isolation.
    Should show better performance than Static-NS (early violation detection)
    but worse than SelfEvolve (cannot learn new constraints).
    """
    print("=" * 70)
    print("Testing Verify-Only Baseline")
    print("=" * 70)

    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config_loader import load_config

    config = load_config("config.yaml")

    # Override for quick test
    config['scenario']['num_tasks'] = 10
    config['logging']['level'] = 'INFO'

    # Run baseline
    agent = VerifyOnlyAgent(config)
    metrics = agent.run_evaluation()

    # Print results
    print("\n" + "=" * 70)
    print("VERIFY-ONLY TEST RESULTS")
    print("=" * 70)

    summary = metrics.get_summary()
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"RFR: {summary['repeat_failure_rate']:.1%}")
    print(f"TTA: {summary['time_to_adapt']}")

    # Verification-specific metrics
    v_stats = agent.get_verification_stats()
    print(f"\nVerification Pass Rate: {v_stats['pass_rate']:.1%}")
    print(f"Violations Caught: {v_stats['num_caught_violations']}")

    # Expected: Better than Static-NS, worse than SelfEvolve
    print("\nExpected behavior:")
    print("  - Success rate: ~70-75% (catches some errors early)")
    print("  - RFR: ~12-15% (prevents violations but can't learn new ones)")
    print("  - TTA: ∞ (no learning - static preconditions)")
    print("  - Verification catches ~30% of potential failures")

    print("\n✅ Verify-Only baseline test complete")