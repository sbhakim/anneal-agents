# src/baselines/static_ns.py
"""
Static Neurosymbolic Baseline (Static-NS)
Corresponds to manuscript Section XII.2 baseline definition.

This baseline represents a traditional neurosymbolic agent WITHOUT self-evolution:
- Has planner, parser, executor with explicit state (neurosymbolic core)
- Includes verify-before-act for precondition checking
- NO FDKA: Operators remain fixed, no learning from failures
- NO metacognitive reflection or threshold tuning
- NO governance (provenance, trust scoring, canary)

Expected behavior: Initial failures repeat indefinitely (RFR ~30%, TTA = ∞)
This demonstrates the necessity of self-evolution for open-world adaptability.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import core components (reuse existing infrastructure)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.state import SymbolicState
from ..core.planner import Planner
from ..core.executor import Executor
from ..knowledge.rule_pool import RulePool
from ..scenarios.travel_planning import TravelPlanningScenario
from ..utils.metrics import MetricsCollector
from ..utils.logger import setup_logger
from ..metacognition.signals import SignalGenerator
from ..metacognition.arbitrator import Arbitrator


class StaticNSAgent:
    """
    Static Neurosymbolic Agent - Baseline without self-evolution.

    Key differences from SelfEvolve:
    1. No FDKA pipeline: failures logged but no patches generated
    2. No reflection: thresholds remain fixed
    3. No governance: no provenance, trust, or canary
    4. Operators frozen: rule pool never updated

    Keeps:
    - Neurosymbolic planning (HTN compilation)
    - Verify-before-act (precondition checking)
    - Metacognitive arbitration (for fair comparison)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Static-NS baseline.

        Args:
            config: System configuration (same format as SelfEvolve)
        """
        self.config = config

        # Setup logger
        log_config = config.get('logging', {})
        log_config['level'] = log_config.get('level', 'WARNING')  # Quiet for batch runs
        self.logger = setup_logger(log_config, name="static_ns")

        self.logger.info("=" * 70)
        self.logger.info("Initializing Static-NS Baseline...")
        self.logger.info("=" * 70)

        # Core components (KEEP)
        self.state = SymbolicState()
        self.rule_pool = RulePool(config['knowledge']['rule_pool_path'])
        self.planner = Planner(config['planner'], self.rule_pool)
        self.scenario = TravelPlanningScenario(config['scenario'])

        # Metacognition (KEEP for fair comparison)
        self.signal_gen = SignalGenerator(config['metacognition'], self.rule_pool)
        self.arbitrator = Arbitrator(config['metacognition'])

        # Executor (KEEP verify-before-act)
        self.executor = Executor(
            config['executor'],
            failure_injector=self.scenario.failure_injector,
            rule_pool=self.rule_pool,
            signal_gen=self.signal_gen,
            arbitrator=self.arbitrator,
            scenario=self.scenario
        )

        # Metrics collector
        self.metrics = MetricsCollector()

        # REMOVED: FDKA pipeline
        # REMOVED: Governance (provenance, trust, canary)
        # REMOVED: Reflection

        self.logger.info("✅ Static-NS baseline ready (NO learning)")
        self.logger.info("=" * 70)

    def run_evaluation(self) -> MetricsCollector:
        """
        Run evaluation without self-evolution.

        Returns:
            MetricsCollector with results
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 STATIC-NS BASELINE EVALUATION")
        print(f"{'=' * 70}")

        for task_id in range(self.scenario.num_tasks):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self._run_task_static(task_id, instruction)

        print(f"\n{'=' * 70}")
        print(f"✅ STATIC-NS EVALUATION COMPLETE")
        print(f"{'=' * 70}")

        # Print summary
        self.metrics.print_summary()

        return self.metrics

    def _run_task_static(self, task_id: int, instruction: str) -> None:
        """
        Run a single task WITHOUT self-evolution.

        Key difference: On failure, log it but do NOT trigger FDKA.
        No patches are generated or applied.

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

        # Execute with limited retries (NO FDKA repair)
        max_attempts = 1  # Only 1 attempt since we can't learn
        final_trace = []

        for attempt in range(max_attempts):
            print(f"\n--- Execution Attempt #{attempt + 1} ---")
            self._reset_world_for_attempt()

            trace, success, failure_type = self.executor.execute(plan, self.state, task_id)
            final_trace = trace or final_trace

            if success:
                print("✅ Task Succeeded")
                self.metrics.record_task(task_id, True, trace)
                return

            # CRITICAL DIFFERENCE: On failure, just log it (no FDKA)
            print(f"❌ Task Failed: {failure_type}")
            print("   [Static-NS: No learning - operator remains unchanged]")

            # No retry since we can't fix the issue
            break

        # Record failure
        print("❌ Task Failed (static operators cannot adapt)")
        self.metrics.record_task(task_id, False, final_trace)

        # Check if this is a repeat failure (for RFR tracking)
        failure_info = self._extract_failure_info(final_trace)
        if failure_info:
            failure_key = f"{failure_info.get('operator')}:{failure_info.get('error')}"
            self.metrics.check_adaptation(failure_key, window_size=20)

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

    def _reset_world_for_attempt(self) -> None:
        """Reset world state for retry attempt."""
        if not hasattr(self.scenario, "world_defaults"):
            return
        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults:
                self.state.set(k, defaults[k])

    def _extract_failure_info(self, trace: List[Dict]) -> Optional[Dict]:
        """Extract failure information from trace."""
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return None


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test Static-NS baseline in isolation.
    Should show high RFR and no adaptation.
    """
    print("=" * 70)
    print("Testing Static-NS Baseline")
    print("=" * 70)

    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ..utils.config_loader import load_config

    config = load_config("config.yaml")

    # Override for quick test
    config['scenario']['num_tasks'] = 10
    config['logging']['level'] = 'INFO'

    # Run baseline
    agent = StaticNSAgent(config)
    metrics = agent.run_evaluation()

    # Print results
    print("\n" + "=" * 70)
    print("STATIC-NS TEST RESULTS")
    print("=" * 70)

    summary = metrics.get_summary()
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"RFR: {summary['repeat_failure_rate']:.1%}")
    print(f"TTA: {summary['time_to_adapt']}")

    # Expected: Low success rate, high RFR, no adaptation
    print("\nExpected behavior:")
    print("  - Success rate: ~50-60% (fails repeatedly on same errors)")
    print("  - RFR: ~25-35% (no learning → repeat failures)")
    print("  - TTA: ∞ for all failure classes (never adapts)")

    print("\n✅ Static-NS baseline test complete")