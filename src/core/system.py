# src/core/system.py
"""
SelfEvolveSystem orchestrator (PoC).
This final version integrates all architectural components from the manuscript:
- The Metacognitive loop is now fully dynamic.
- The FDKA pipeline now includes the crucial "Canary Deployment" stage
  before any patch is permanently committed, ensuring a final safety check.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import uuid

# Core components
from .state import SymbolicState
from .planner import Planner
from .executor import Executor

# FDKA components
from ..fdka.propose_edit import FDKAPipeline
from ..fdka.scoring import Scorer

# Governance
from ..governance.guard import Guard
from ..governance.provenance import ProvenanceTracker
from ..governance.trust import TrustScorer
from ..governance.canary import CanaryRunner

# Metacognition
from ..metacognition.signals import SignalGenerator
from ..metacognition.arbitrator import Arbitrator

# Knowledge
from ..knowledge.rule_pool import RulePool
from ..knowledge.experience_pool import ExperiencePool

# Utils
from ..utils.logger import setup_logger
from ..utils.metrics import MetricsCollector

# Scenario
from ..scenarios.travel_planning import TravelPlanningScenario


class SelfEvolveSystem:
    """
    Main SELFEVOLVE system orchestrator that integrates all components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(config['logging'])
        self.logger.info("=" * 70)
        self.logger.info("Initializing SELFEVOLVE system...")
        self.logger.info("=" * 70)

        self.state = SymbolicState()
        self.rule_pool = RulePool(config['knowledge']['rule_pool_path'])
        self.experience_pool = ExperiencePool(max_size=1000)
        self.signal_gen = SignalGenerator(config['metacognition'], self.rule_pool)
        self.arbitrator = Arbitrator(config['metacognition'])
        self.planner = Planner(config['planner'], self.rule_pool)
        self.scenario = TravelPlanningScenario(config['scenario'])
        self.executor = Executor(
            config['executor'], failure_injector=self.scenario.failure_injector,
            rule_pool=self.rule_pool, signal_gen=self.signal_gen,
            arbitrator=self.arbitrator, scenario=self.scenario,
        )
        self.fdka = FDKAPipeline(config['fdka'])
        self.scorer = Scorer(config['fdka'], experience_pool=self.experience_pool)
        self.guard = Guard(config.get('governance', {}))
        self.provenance = ProvenanceTracker(config['governance']['provenance'])
        self.trust_scorer = TrustScorer(config['governance']['trust'])
        self.canary_runner = CanaryRunner(config.get('governance', {}).get('canary', {}))
        self.fdka_threshold = config.get('fdka', {}).get('threshold', 0.5)
        self.metrics = MetricsCollector()
        self._last_committed_patch_id: Optional[str] = None
        self.logger.info("✅ SELFEVOLVE system ready")
        self.logger.info("=" * 70)

    def _reset_world_for_task(self) -> None:
        if not hasattr(self.scenario, "world_defaults"): return
        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults: self.state.set(k, defaults[k])
        for k in ("corporate_card_policy", "blackout_dates"):
            if k in defaults and self.state.get(k) is None: self.state.set(k, defaults[k])

    def _reset_world_for_attempt(self) -> None:
        if not hasattr(self.scenario, "world_defaults"): return
        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults: self.state.set(k, defaults[k])

    def run_task(self, task_id: int, instruction: str):
        print(f"\n{'=' * 70}\nTask {task_id + 1}: {instruction}\n{'=' * 70}")
        self.state.update_from_instruction(instruction)
        self._reset_world_for_task()
        plan = self.planner.compile(instruction, self.state.to_dict())
        if not plan:
            self.metrics.record_task(task_id, False, [{"error": "PlanningFailed"}])
            return

        max_attempts = 3
        final_trace = []
        for attempt in range(max_attempts):
            print(f"\n--- Execution Attempt #{attempt + 1} ---")
            self._reset_world_for_attempt()
            trace, success, failure_type = self.executor.execute(plan, self.state, task_id)
            final_trace = trace or final_trace

            if success:
                print("✅ Task Succeeded on this attempt.")
                self.metrics.record_task(task_id, True, trace)
                self.experience_pool.add_trace(trace, True, metadata={'instruction': instruction, 'task_id': task_id})
                if self._last_committed_patch_id:
                    self.trust_scorer.update_trust_score(self._last_committed_patch_id, success=True)
                self._last_committed_patch_id = None
                return

            print(f"⚠️ Execution failed with type: {failure_type}")
            if failure_type in ("PreconditionUnmet", "ToolError"):
                print("🧠 Escalating to FDKA for governed self-edit...")
                patch_applied, committed_patch = self._handle_failure(trace, task_id, instruction)
                if patch_applied and committed_patch:
                    operator_name = committed_patch.get("operator")
                    if operator_name: self.scenario.mark_operator_patched(operator_name)
                    # Re-compile the plan to ensure the new patch is used
                    print("🔁 Re-compiling plan and retrying execution with the patched operator...")
                    plan = self.planner.compile(instruction, self.state.to_dict())
                    continue
                print("❌ FDKA did not commit a patch. Halting attempts for this task.")
                break
            print("❌ Non-recoverable failure type; stopping attempts.")
            break

        print("❌ Task Failed after all attempts.")
        if self._last_committed_patch_id: self.trust_scorer.update_trust_score(self._last_committed_patch_id,
                                                                               success=False)
        self._last_committed_patch_id = None
        self.metrics.record_task(task_id, False, final_trace)
        # Pass failure metadata to the experience pool
        failure_info = self._extract_failure_info(final_trace)
        if failure_info:
            metadata = {
                'instruction': instruction, 'task_id': task_id,
                'operator': failure_info.get('operator'),
                'error_type': failure_info.get('error')
            }
            self.experience_pool.add_trace(final_trace, False, metadata=metadata)
            self.metrics.check_adaptation(f"{failure_info.get('operator')}:{failure_info.get('error')}")

    def _handle_failure(self, trace: list, task_id: int, instruction: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        patch_applied = False
        patch_dict = None
        proposed_patch = self.fdka.propose_edit(trace, self.rule_pool)

        # --- UPDATED LOGIC ---
        # The scorer now returns a dictionary of all scores.
        scores = self.scorer.score(proposed_patch, trace)
        agg_score = scores.get("aggregate", 0.0)
        # --- END OF UPDATED LOGIC ---

        if agg_score >= self.fdka_threshold:
            guard_result = self.guard.check(proposed_patch, context={"scores": scores})
            decision = guard_result.get("decision")
            reason = guard_result.get("reason")

            if decision == 'allow':
                print(" FDKA: Guardrails passed. Proceeding to canary test...")

                # --- UPDATED LOGIC ---
                # Pass the entire 'scores' dictionary to the canary context.
                canary_context = {
                    "rule_pool": self.rule_pool, "stage_fn": self.rule_pool.update_operator,
                    "simulator": self._run_canary_simulation,
                    "examples": self.experience_pool.get_failure_traces(
                        operator=proposed_patch.get("operator")) or self.experience_pool.traces[-10:],
                    "scorer": self.scorer,
                    "scores": scores  # Pass the full dictionary
                }
                # --- END OF UPDATED LOGIC ---

                canary_result = self.canary_runner.run(proposed_patch, canary_context)

                if canary_result.get("passed"):
                    print(" FDKA: Canary test passed. Committing patch permanently.")
                    if self.rule_pool.update_operator(proposed_patch):
                        patch_applied = True
                        patch_dict = proposed_patch
                else:
                    print(f" FDKA: ❌ Canary test failed: {canary_result.get('reason')}")
            else:
                print(f" FDKA: ❌ Guardrails blocked patch: {reason}")
        else:
            print(f" FDKA: ❌ Score {agg_score:.2f} below threshold {self.fdka_threshold}. Patch rejected.")

        patch_id = (patch_dict or proposed_patch).get("id", f"patch-{uuid.uuid4().hex[:8]}")
        self.provenance.log(
            {"patch_id": patch_id, "task_id": task_id, "applied": patch_applied, "patch": patch_dict or proposed_patch})
        self.metrics.record_patch(patch_dict or proposed_patch, success=patch_applied, committed=patch_applied,
                                  scores=scores)
        if patch_applied:
            self._last_committed_patch_id = patch_id
            self.trust_scorer.initialize_trust(patch_id)
            print("✅ FDKA: Patch was successfully committed.")
        else:
            print("❌ FDKA: Patch was rejected.")

        return patch_applied, patch_dict

    def _run_canary_simulation(self, example: Dict, patched_rule_pool: RulePool) -> Dict:
        instruction = example.get("metadata", {}).get("instruction")
        if not instruction: return {"ok": False, "violations": 1}
        sim_state = SymbolicState()
        sim_state.update_from_instruction(instruction)
        sim_plan = self.planner.compile(instruction, sim_state.to_dict())
        sim_executor = Executor(self.config['executor'], None, patched_rule_pool, self.signal_gen, self.arbitrator,
                                self.scenario)
        _, success, _ = sim_executor.execute(sim_plan, sim_state, task_id=-1)
        return {"ok": success, "violations": 0 if success else 1}

    def _extract_failure_info(self, trace: list) -> dict:
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return {}

    def run_evaluation(self) -> MetricsCollector:
        print(f"\n{'=' * 70}\n🚀 STARTING SELFEVOLVE EVALUATION\n{'=' * 70}")
        for task_id in range(self.config.get('scenario', {}).get('num_tasks', 10)):
            instruction = self.scenario.get_task(task_id)
            if instruction: self.run_task(task_id, instruction)
        print(f"\n{'=' * 70}\n✅ EVALUATION COMPLETE\n{'=' * 70}")
        self.metrics.print_summary()
        results_dir = Path(self.config['output']['results_dir'])
        self.metrics.save(results_dir / "metrics.json")
        self.experience_pool.save(results_dir / "experience_pool.json")
        print(f"\n💾 Results saved to: {results_dir}")
        return self.metrics