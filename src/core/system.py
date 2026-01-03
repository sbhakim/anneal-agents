# src/core/system.py
"""
SelfEvolveSystem orchestrator (PoC).
This version integrates all architectural components from the manuscript:
- Metacognitive arbitration (S1/S2/VERIFY/DEFER)
- Executor with Verify-Before-Act
- FDKA pipeline with scoring + governance + canary + commit
- Metrics + Reflection hooks
- Manuscript-style exports (Table 3/4 CSVs)

CRITICAL BEHAVIORAL GUARANTEE (Reliability):
- We do NOT mark an operator as "patched" immediately on commit.
  We only mark it patched AFTER observing real success post-patch.
  This prevents inflated results due to failure-injector disabling.

MINIMUM PLANNING NORMALIZATION:
- Normalize task-transient state before planning/replanning to prevent
  "(invalid) (invalid)" leakage into params/state and compiled plans.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import uuid
import json

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
from ..metacognition.reflection import Reflection

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

        # Metrics first so we can wire it into dependent subsystems (FDKA)
        self.metrics = MetricsCollector()

        self.state = SymbolicState()
        self.rule_pool = RulePool(config['knowledge']['rule_pool_path'])
        self.experience_pool = ExperiencePool(max_size=1000)
        self.signal_gen = SignalGenerator(config['metacognition'], self.rule_pool)
        self.arbitrator = Arbitrator(config['metacognition'])
        self.planner = Planner(config['planner'], self.rule_pool)
        self.scenario = TravelPlanningScenario(config['scenario'])

        self.executor = Executor(
            config['executor'],
            failure_injector=self.scenario.failure_injector,
            rule_pool=self.rule_pool,
            signal_gen=self.signal_gen,
            arbitrator=self.arbitrator,
            scenario=self.scenario,
        )

        # Pass metrics collector to FDKA to capture efficiency stats from LLM calls
        self.fdka = FDKAPipeline(config['fdka'], metrics_collector=self.metrics)
        self.scorer = Scorer(config['fdka'], experience_pool=self.experience_pool)
        self.guard = Guard(config.get('governance', {}))
        self.provenance = ProvenanceTracker(config['governance']['provenance'])
        self.trust_scorer = TrustScorer(config['governance']['trust'])
        self.canary_runner = CanaryRunner(config.get('governance', {}).get('canary', {}))

        # Explicit enable flag; threshold is for accept gating only
        self.fdka_enabled: bool = bool(config.get('fdka', {}).get('enabled', True))
        self.fdka_threshold: float = float(config.get('fdka', {}).get('threshold', 0.5))

        self._last_committed_patch_id: Optional[str] = None

        # In-memory record of committed patches for downstream analysis
        self._committed_patches: list[Dict[str, Any]] = []

        self.reflection = Reflection(
            config['metacognition'],
            self.arbitrator,
            self.metrics,
            self.experience_pool
        )

        # Log effective runtime switches for diagnosability
        pe = (config.get('fdka', {}) or {}).get('propose_edit', {}) or {}
        self.logger.info(f"FDKA enabled: {self.fdka_enabled}  | threshold: {self.fdka_threshold:.3f}")
        self.logger.info(f"FDKA provider/model: {pe.get('llm_provider', 'N/A')} / {pe.get('model', 'N/A')}")
        self.logger.info("✅ SELFEVOLVE system ready")
        self.logger.info("=" * 70)

    # ---------------------------------------------------------------------
    # State normalization / resets
    # ---------------------------------------------------------------------

    def _reset_world_for_task(self) -> None:
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
        if not hasattr(self.scenario, "world_defaults"):
            return
        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults:
                self.state.set(k, defaults[k])

    def _normalize_state_for_planning(self) -> None:
        """
        Minimum normalization to prevent transient failure markers from polluting replans.

        Key issue observed: injected constraint logic can tag payment_method with '(invalid)',
        and replanning can propagate '(invalid) (invalid) ...' into tool params across attempts.
        We keep task-level semantics but ensure planning uses a clean default baseline.
        """
        # Normalize payment-related fields
        pm = self.state.get("payment_method")
        if isinstance(pm, str) and "invalid" in pm.lower():
            # Strip all "(invalid)" markers
            cleaned = pm
            while "(invalid)" in cleaned.lower():
                cleaned = cleaned.replace("(invalid)", "").replace("  ", " ").strip()
            # If stripping produces empty, restore base
            if not cleaned:
                cleaned = "CorporateCard:CC-5512"
            self.state.set("payment_method", cleaned)

        # Reset explicit invalid flags for planning (the injector/executor will re-set when needed)
        if self.state.get("payment_invalid") is True:
            self.state.set("payment_invalid", False)
        if self.state.get("payment_valid") is False:
            self.state.set("payment_valid", True)

    # ---------------------------------------------------------------------
    # Task runner
    # ---------------------------------------------------------------------

    def run_task(self, task_id: int, instruction: str):
        print(f"\n{'=' * 70}\nTask {task_id + 1}: {instruction}\n{'=' * 70}")

        # Track operators whose patches were committed during this task;
        # only mark as patched upon observed post-patch success.
        pending_patched_operators: set[str] = set()

        self.state.update_from_instruction(instruction)
        self._reset_world_for_task()
        self._normalize_state_for_planning()

        plan = self.planner.compile(instruction, self.state.to_dict())

        # Treat planning failure as learnable and route through FDKA
        if not plan:
            print("PLANNER: ❌ Plan compilation failed.")
            planning_trace = [{"error": "PlanningFailed", "operator": "UNKNOWN", "instruction": instruction}]
            self.metrics.record_task(task_id, False, planning_trace)
            print("🧠 Escalating to FDKA for governed self-edit (PlanningFailed)...")

            patch_applied, committed_patch = self._handle_failure(planning_trace, task_id, instruction)

            # Defer mark_operator_patched until success
            if patch_applied and committed_patch:
                op = committed_patch.get("operator")
                if op:
                    pending_patched_operators.add(op)

            if patch_applied:
                print("🔁 Re-compiling plan after FDKA patch...")
                self._normalize_state_for_planning()
                plan = self.planner.compile(instruction, self.state.to_dict())
                if not plan:
                    print("PLANNER: ❌ Still failed after patch; aborting task.")
                    return
            else:
                print("❌ No patch committed; aborting task.")
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

                # ✅ RELIABILITY: only now mark patched operators as "patched" (disables future injection honestly)
                if pending_patched_operators:
                    for op_name in sorted(pending_patched_operators):
                        self.scenario.mark_operator_patched(op_name)
                    pending_patched_operators.clear()

                if task_id > 0 and task_id % 5 == 0:
                    print(f"  🔍 Checking adaptation progress at task {task_id}...")
                    for failure_key in list(self.metrics.failure_classes.keys()):
                        if self.metrics.check_adaptation(failure_key, window_size=min(10, task_id)):
                            print(f"  ✅ Adaptation confirmed for '{failure_key}'")

                if task_id > 0 and task_id % 5 == 0:
                    print(f"  💭 Running reflection (task {task_id})...")
                    try:
                        self.reflection.reflect(trace, success, task_id)
                        print(f"     Updated thresholds: τ_u={self.arbitrator.tau_u:.2f}, τ_p={self.arbitrator.tau_p:.2f}")
                    except Exception as e:
                        print(f"     ⚠️ Reflection failed: {e}")

                if self._last_committed_patch_id:
                    self.trust_scorer.update_trust_score(self._last_committed_patch_id, success=True)
                self._last_committed_patch_id = None
                return

            print(f"⚠️ Execution failed with type: {failure_type}")
            if failure_type in ("PreconditionUnmet", "ToolError"):
                print("🧠 Escalating to FDKA for governed self-edit...")
                patch_applied, committed_patch = self._handle_failure(trace, task_id, instruction)

                # Defer mark_operator_patched until success
                if patch_applied and committed_patch:
                    operator_name = committed_patch.get("operator")
                    if operator_name:
                        pending_patched_operators.add(operator_name)

                    print("🔁 Re-compiling plan and retrying execution with the patched operator...")
                    self._normalize_state_for_planning()
                    plan = self.planner.compile(instruction, self.state.to_dict())
                    continue

                print("❌ FDKA did not commit a patch. Halting attempts for this task.")
                break

            print("❌ Non-recoverable failure type; stopping attempts.")
            break

        print("❌ Task Failed after all attempts.")
        if self._last_committed_patch_id:
            self.trust_scorer.update_trust_score(self._last_committed_patch_id, success=False)
        self._last_committed_patch_id = None
        self.metrics.record_task(task_id, False, final_trace)

        failure_info = self._extract_failure_info(final_trace)
        if failure_info:
            metadata = {
                'instruction': instruction,
                'task_id': task_id,
                'operator': failure_info.get('operator'),
                'error_type': failure_info.get('error'),
            }
            if not any(t['metadata'].get('task_id') == task_id and not t['success'] for t in self.experience_pool.traces):
                self.experience_pool.add_trace(final_trace, False, metadata=metadata)
            self.metrics.check_adaptation(f"{failure_info.get('operator')}:{failure_info.get('error')}")

    # ---------------------------------------------------------------------
    # FDKA handling
    # ---------------------------------------------------------------------

    def _handle_failure(self, trace: list, task_id: int, instruction: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        # Explicit boolean gate, not threshold-as-off.
        if not self.fdka_enabled:
            print("  DEBUG_FDKA: disabled by config. Skipping self-edit.")
            return False, None

        patch_applied = False
        patch_dict: Optional[Dict[str, Any]] = None

        failure_info = self._extract_failure_info(trace)
        if failure_info:
            metadata = {
                'instruction': instruction,
                'task_id': task_id,
                'operator': failure_info.get('operator'),
                'error_type': failure_info.get('error'),
            }
            # Always add the failure trace so utility/pre-filter can query it
            self.experience_pool.add_trace(trace, False, metadata=metadata)
            print(f"  📝 Added failure trace to experience pool (operator={metadata['operator']})")

        # Propose a candidate patch (Stage 2)
        proposed_patch = self.fdka.propose_edit(trace, self.rule_pool) or {}

        # Guard against provider/LLM failures returning non-dict/empty payloads
        if not isinstance(proposed_patch, dict) or not proposed_patch:
            print(" FDKA: ❌ No patch proposed (provider error or empty response).")
            self.metrics.record_value_check(vetoed=False, reason="no_patch_proposed")
            self.metrics.record_causal_check(escalated=False, reason="no_patch_proposed")
            return False, None

        # Rejected during schema/typing validation
        if proposed_patch.get("action") == "REJECTED":
            print(f" FDKA: ❌ Patch rejected during validation: {proposed_patch.get('details')}")
            self.metrics.record_value_check(vetoed=False, reason="validation_reject")
            self.metrics.record_causal_check(escalated=False, reason="validation_reject")
            return False, None

        # Score (plausibility/consistency/utility/risk) + budget penalties
        scores = self.scorer.score(proposed_patch, trace) or {}
        agg_score = scores.get("aggregate", 0.0)

        # Accept score gate before governance/canary
        if agg_score < self.fdka_threshold:
            print(f" FDKA: ❌ Score {agg_score:.2f} below threshold {self.fdka_threshold:.2f}. Patch rejected.")
            self.metrics.record_patch(proposed_patch, success=False, committed=False, scores=scores)
            self.metrics.record_value_check(vetoed=False, reason="below_threshold")
            self.metrics.record_causal_check(escalated=False, reason="below_threshold")
            return False, None

        # Governance: value + causal + invariants
        guard_result = self.guard.check(
            proposed_patch,
            context={
                "scores": scores,
                "trace": trace,
                "state": self.state.to_dict(),
            }
        )

        # Governance metrics
        self.metrics.record_value_check(
            vetoed=(guard_result['decision'] == 'veto'),
            reason=guard_result.get('reason', '')
        )
        self.metrics.record_causal_check(
            escalated=(guard_result['decision'] == 'request_human'),
            reason=guard_result.get('reason', '')
        )

        if guard_result['decision'] == 'allow':
            print(" FDKA: Guardrails passed. Proceeding to canary test...")
            canary_context = {
                "rule_pool": self.rule_pool,
                "stage_fn": self.rule_pool.update_operator,
                "simulator": self._run_canary_simulation,
                "examples": self.experience_pool.get_failure_traces(operator=proposed_patch.get("operator"))
                            or self.experience_pool.traces[-10:],
                "scorer": self.scorer,
                "scores": scores
            }

            canary_result = self.canary_runner.run(proposed_patch, canary_context) or {}
            self.metrics.record_canary_test(passed=bool(canary_result.get("passed")))

            if canary_result.get("passed"):
                print(" FDKA: Canary test passed. Committing patch permanently.")
                if self.rule_pool.update_operator(proposed_patch):
                    patch_applied = True
                    patch_dict = proposed_patch
            else:
                print(f" FDKA: ❌ Canary test failed: {canary_result.get('reason')}")
        elif guard_result['decision'] == 'request_human':
            print(" FDKA: ⚠️ Causal guard requests human escalation; not committing.")
        else:
            print(f" FDKA: ❌ Guardrails blocked patch: {guard_result.get('reason')}")

        patch_id = (patch_dict or proposed_patch).get("id", f"patch-{uuid.uuid4().hex[:8]}")
        self.provenance.log({
            "patch_id": patch_id,
            "task_id": task_id,
            "applied": patch_applied,
            "patch": patch_dict or proposed_patch
        })
        self.metrics.record_patch(patch_dict or proposed_patch, success=patch_applied, committed=patch_applied, scores=scores)

        if patch_applied:
            self._last_committed_patch_id = patch_id
            self.trust_scorer.initialize_trust(patch_id)
            try:
                self._committed_patches.append({
                    "patch_id": patch_id,
                    "operator": patch_dict.get("operator"),
                    "action": patch_dict.get("action"),
                    "scores": {
                        "plausibility": scores.get("plausibility"),
                        "consistency": scores.get("consistency"),
                        "utility": scores.get("utility"),
                        "risk": scores.get("risk"),
                        "aggregate": scores.get("aggregate"),
                    }
                })
            except Exception:
                pass
            print("✅ FDKA: Patch was successfully committed.")
        else:
            print("❌ FDKA: Patch was rejected.")

        return patch_applied, patch_dict

    # ---------------------------------------------------------------------
    # Canary simulation
    # ---------------------------------------------------------------------

    def _run_canary_simulation(self, example: Dict, patched_rule_pool: RulePool) -> Dict:
        instruction = example.get("metadata", {}).get("instruction")
        if not instruction:
            return {"ok": False, "violations": 1}

        sim_state = SymbolicState()
        sim_state.update_from_instruction(instruction)

        sim_plan = self.planner.compile(instruction, sim_state.to_dict())
        if not sim_plan:
            return {"ok": False, "violations": 1}

        sim_executor = Executor(
            self.config['executor'],
            None,
            patched_rule_pool,
            self.signal_gen,
            self.arbitrator,
            self.scenario
        )

        _, success, _ = sim_executor.execute(sim_plan, sim_state, task_id=-1)
        return {"ok": success, "violations": 0 if success else 1}

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _extract_failure_info(self, trace: list) -> dict:
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return {}

    # ---------------------------------------------------------------------
    # Evaluation runner
    # ---------------------------------------------------------------------

    def run_evaluation(self) -> MetricsCollector:
        print(f"\n{'=' * 70}\n🚀 STARTING SELFEVOLVE EVALUATION\n{'=' * 70}")

        for task_id in range(self.config.get('scenario', {}).get('num_tasks', 10)):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self.run_task(task_id, instruction)

        print(f"\n{'=' * 70}\n✅ EVALUATION COMPLETE\n{'=' * 70}")
        self.metrics.print_summary()

        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.save(results_dir / "metrics.json")
        self.experience_pool.save(results_dir / "experience_pool.json")

        # Write a compact list of committed patches
        try:
            (results_dir / "patches_committed.json").write_text(
                json.dumps(self._committed_patches, indent=2)
            )
        except Exception:
            pass

        # Write a minimal guardrail summary
        try:
            guard_summary = self.guard.get_statistics()
            (results_dir / "governance_summary.json").write_text(
                json.dumps(guard_summary, indent=2)
            )
        except Exception:
            pass

        print(f"\n📊 Generating manuscript data tables...")
        try:
            self.metrics.export_failure_analysis_csv(results_dir / "table3_per_failure.csv")
            print(f"   ✅ Table 3: {results_dir / 'table3_per_failure.csv'}")
        except Exception as e:
            print(f"   ⚠️ Table 3 export failed: {e}")

        try:
            self.metrics.export_governance_csv(results_dir / "table4_governance.csv")
            print(f"   ✅ Table 4: {results_dir / 'table4_governance.csv'}")
        except Exception as e:
            print(f"   ⚠️ Table 4 export failed: {e}")

        print(f"\n💾 Results saved to: {results_dir}")
        return self.metrics
