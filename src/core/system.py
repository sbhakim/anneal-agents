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


MINIMUM SMT/DEMO WIRING (2026-01):
- Pass synthetic SMT sanity patches into canary_context when available so CanaryRunner can
  log SAT/UNSAT evidence (consistency-only) without affecting the main canary decision.

INTEGRATION NOTE (2026-01):
- Executor now owns Verify→Repair (h=1–2). System-level local precondition repair is optional and
  gated behind config to avoid competing repair controllers.

MINIMUM REPRODUCIBILITY BUNDLE (2026-01):
- Emit a run_manifest.json + config_resolved.json capturing seeds, config snapshot, environment,
  and output paths for auditability and exact reruns.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
import uuid
import json
from copy import deepcopy
import sys
import platform
import subprocess
from datetime import datetime, timezone

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
from ..scenarios.test_cases import (
    get_canary_suite_for_operator,
    get_smt_sanity_patches_for_operator,
)


class _NoOpProvenance:
    """No-op provenance sink (used when governance.provenance.enable is false)."""

    def log(self, provenance_tuple: Dict):
        return

    def log_patch_event(self, *args, **kwargs):
        return


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

        # Reproducibility: stable run identifier for manifests + artifacts
        self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

        # Normalize potentially-mis-typed config blocks (some runners set booleans for ablations)
        fdka_cfg = config.get("fdka", {})
        if not isinstance(fdka_cfg, dict):
            fdka_cfg = {}
        gov_cfg = config.get("governance", {})
        if not isinstance(gov_cfg, dict):
            gov_cfg = {}

        # Metrics first so we can wire it into dependent subsystems (FDKA)
        self.metrics = MetricsCollector()

        # Attach minimal run metadata for defensible multi-LLM attribution (saved in metrics.json)
        pe_cfg = (fdka_cfg.get("propose_edit", {}) or {}) if isinstance(fdka_cfg, dict) else {}
        scenario_cfg = config.get("scenario", {}) or {}
        self.metrics.set_run_metadata({
            "run_id": self.run_id,
            "results_dir": (config.get("output", {}) or {}).get("results_dir"),
            "difficulty": scenario_cfg.get("difficulty"),
            "seed": scenario_cfg.get("failure_injector_seed"),
            "num_tasks": scenario_cfg.get("num_tasks"),
            "llm_provider": pe_cfg.get("llm_provider"),
            "model": pe_cfg.get("model"),
            "fdka_enabled": bool(fdka_cfg.get("enabled", True)),
            "fdka_threshold": float(fdka_cfg.get("threshold", 0.5)),
        })

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
        self.fdka = FDKAPipeline(fdka_cfg, metrics_collector=self.metrics)
        self.scorer = Scorer(fdka_cfg, experience_pool=self.experience_pool)
        self.guard = Guard(gov_cfg)

        # Provenance can be disabled in ablations; keep a no-op sink to avoid None checks everywhere.
        prov_cfg = (gov_cfg.get("provenance", {}) or {}) if isinstance(gov_cfg, dict) else {}
        if bool(prov_cfg.get("enable", True)):
            self.provenance = ProvenanceTracker(prov_cfg)
        else:
            self.provenance = _NoOpProvenance()

        trust_cfg = (gov_cfg.get("trust", {}) or {}) if isinstance(gov_cfg, dict) else {}
        self.trust_scorer = TrustScorer(trust_cfg)
        self.canary_runner = CanaryRunner((gov_cfg.get('canary', {}) or {}) if isinstance(gov_cfg, dict) else {})

        # Explicit enable flag; threshold is for accept gating only
        self.fdka_enabled: bool = bool(fdka_cfg.get('enabled', True))
        self.fdka_threshold: float = float(fdka_cfg.get('threshold', 0.5))

        self._last_committed_patch_id: Optional[str] = None

        # In-memory record of committed patches for downstream analysis
        self._committed_patches: List[Dict[str, Any]] = []

        # Operators committed but not yet validated by a real post-patch success
        self._pending_patch_success_ops: Set[str] = set()

        self.reflection = Reflection(
            config['metacognition'],
            self.arbitrator,
            self.metrics,
            self.experience_pool
        )

        # Log effective runtime switches for diagnosability
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"FDKA enabled: {self.fdka_enabled}  | threshold: {self.fdka_threshold:.3f}")
        self.logger.info(f"FDKA provider/model: {pe_cfg.get('llm_provider', 'N/A')} / {pe_cfg.get('model', 'N/A')}")
        self.logger.info("✅ SELFEVOLVE system ready")
        self.logger.info("=" * 70)

    # ---------------------------------------------------------------------
    # Reproducibility helpers
    # ---------------------------------------------------------------------

    def _get_git_commit(self) -> Optional[str]:
        """Best-effort git commit hash for provenance; returns None when unavailable."""
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            return out if out else None
        except Exception:
            return None

    def _write_run_bundle(self, results_dir: Path) -> None:
        """
        Emit a minimal run bundle to support exact reruns and reviewer audit.
        Writes:
        - run_manifest.json
        - config_resolved.json
        """
        results_dir.mkdir(parents=True, exist_ok=True)

        scenario_cfg = self.config.get("scenario", {}) or {}
        output_cfg = self.config.get("output", {}) or {}
        fdka_cfg = self.config.get("fdka", {}) or {}
        if not isinstance(fdka_cfg, dict):
            fdka_cfg = {}
        pe = (fdka_cfg.get("propose_edit", {}) or {})

        manifest = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "agent": "selfevolve",
            "python_version": sys.version.split(" ")[0],
            "platform": platform.platform(),
            "git_commit": self._get_git_commit(),
            "trace_schema_version": 1,
            "scenario": {
                "difficulty": getattr(self.scenario, "difficulty", scenario_cfg.get("difficulty")),
                "num_tasks": getattr(self.scenario, "num_tasks", scenario_cfg.get("num_tasks")),
                "failure_rate": getattr(self.scenario, "failure_rate", scenario_cfg.get("failure_rate")),
                "task_generation_seed": getattr(self.scenario, "task_seed", scenario_cfg.get("task_generation_seed")),
                "failure_injector_seed": getattr(self.scenario, "failure_seed", scenario_cfg.get("failure_injector_seed")),
            },
            "fdka": {
                "enabled": self.fdka_enabled,
                "threshold": self.fdka_threshold,
                "provider": pe.get("llm_provider"),
                "model": pe.get("model"),
            },
            "outputs": {
                "results_dir": str(results_dir),
                "metrics_json": str(results_dir / "metrics.json"),
                "experience_pool_json": str(results_dir / "experience_pool.json"),
                "patches_committed_json": str(results_dir / "patches_committed.json"),
                "governance_summary_json": str(results_dir / "governance_summary.json"),
                "table3_per_failure_csv": str(results_dir / "table3_per_failure.csv"),
                "table4_governance_csv": str(results_dir / "table4_governance.csv"),
            },
            "counts": {
                "committed_patches": len(self._committed_patches),
            },
            # Keep a compact copy of run attribution so manifest and metrics.json align.
            "run_metadata": (self.metrics.get_run_metadata() if hasattr(self.metrics, "get_run_metadata") else {}),
        }

        try:
            (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            self.logger.info(f"⚠️ Failed to write run_manifest.json: {e}")

        # Save the resolved config as executed (single-file snapshot).
        try:
            (results_dir / "config_resolved.json").write_text(json.dumps(self.config, indent=2))
        except Exception as e:
            self.logger.info(f"⚠️ Failed to write config_resolved.json: {e}")

        # Convenience pointer for quick copying into manuscripts.
        try:
            (results_dir / "run_id.txt").write_text(self.run_id + "\n")
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # State normalization / resets
    # ---------------------------------------------------------------------

    def _reset_world_for_task(self) -> None:
        if not hasattr(self.scenario, "world_defaults"):
            return
        defaults = self.scenario.world_defaults()

        for k in ("hotel_status", "flight_status"):
            if k in defaults:
                self.state.set(k, deepcopy(defaults[k]))

        # Always apply scenario world defaults so state matches injector/scenario truth.
        for k in ("corporate_card_policy", "blackout_dates"):
            if k in defaults:
                self.state.set(k, deepcopy(defaults[k]))

    def _reset_world_for_attempt(self) -> None:
        if not hasattr(self.scenario, "world_defaults"):
            return
        defaults = self.scenario.world_defaults()
        for k in ("hotel_status", "flight_status"):
            if k in defaults:
                self.state.set(k, deepcopy(defaults[k]))

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
            cleaned = pm
            while "(invalid)" in cleaned.lower():
                cleaned = cleaned.replace("(invalid)", "").replace("  ", " ").strip()
            if not cleaned:
                cleaned = "CorporateCard:CC-5512"
            self.state.set("payment_method", cleaned)

        # Reset explicit invalid flags for planning (the injector/executor will re-set when needed)
        if self.state.get("payment_invalid") is True:
            self.state.set("payment_invalid", False)
        if self.state.get("payment_valid") is False:
            self.state.set("payment_valid", True)

    def _plan_signature(self, plan: list) -> tuple:
        sig = []
        for step in plan or []:
            op = step.get("operator")
            name = getattr(op, "name", str(op))
            params = step.get("params", {}) or {}
            items = tuple(sorted((str(k), repr(v)) for k, v in params.items()))
            sig.append((name, items))
        return tuple(sig)

    def _find_instability_hint(self, trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return instability hint payload when executor flags repeated recoveries."""
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and entry.get("event_type") == "instability_hint" and entry.get("suggest_fdka") is True:
                return entry
        return None

    # ---------------------------------------------------------------------
    # Task runner
    # ---------------------------------------------------------------------

    def run_task(self, task_id: int, instruction: str):
        print(f"\n{'=' * 70}\nTask {task_id + 1}: {instruction}\n{'=' * 70}")

        aggregated_trace: List[Dict[str, Any]] = []
        pending_patched_operators: Set[str] = set()

        self.state.update_from_instruction(instruction)
        self._reset_world_for_task()
        self._normalize_state_for_planning()

        plan = self.planner.compile(instruction, self.state.to_dict())

        if not plan:
            print("PLANNER: ❌ Plan compilation failed.")
            planning_trace = [{"error": "PlanningFailed", "operator": "PLANNER", "instruction": instruction}]
            aggregated_trace.extend(planning_trace)

            print("🧠 Escalating to FDKA for governed self-edit (PlanningFailed)...")
            patch_applied, committed_patch = self._handle_failure(planning_trace, task_id, instruction)

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
                    self.metrics.record_task(task_id, False, aggregated_trace)
                    if not any(t['metadata'].get('task_id') == task_id and not t['success'] for t in self.experience_pool.traces):
                        self.experience_pool.add_trace(
                            aggregated_trace,
                            False,
                            metadata={"instruction": instruction, "task_id": task_id, "operator": "PLANNER", "error_type": "PlanningFailed"},
                        )
                    return
            else:
                print("❌ No patch committed; aborting task.")
                self.metrics.record_task(task_id, False, aggregated_trace)
                if not any(t['metadata'].get('task_id') == task_id and not t['success'] for t in self.experience_pool.traces):
                    self.experience_pool.add_trace(
                        aggregated_trace,
                        False,
                        metadata={"instruction": instruction, "task_id": task_id, "operator": "PLANNER", "error_type": "PlanningFailed"},
                    )
                return

        max_attempts = 3
        final_trace = []

        # Optional legacy controller; keep off by default so executor Repair loop is authoritative.
        enable_local_repair = bool((self.config.get("system", {}) or {}).get("enable_local_repair", False))
        did_local_repair = False

        for attempt in range(max_attempts):
            print(f"\n--- Execution Attempt #{attempt + 1} ---")
            self._reset_world_for_attempt()

            trace, success, failure_type = self.executor.execute(plan, self.state, task_id)

            if trace:
                aggregated_trace.extend(trace)
            final_trace = trace or final_trace

            if success:
                print("✅ Task Succeeded on this attempt.")

                task_trace_for_metrics = aggregated_trace or trace
                self.metrics.record_task(task_id, True, task_trace_for_metrics)
                self.experience_pool.add_trace(
                    task_trace_for_metrics,
                    True,
                    metadata={'instruction': instruction, 'task_id': task_id}
                )

                # Mark operators patched only after observing success post-commit
                pending = set(pending_patched_operators) | set(self._pending_patch_success_ops)
                if pending:
                    for op_name in sorted(pending):
                        self.scenario.mark_operator_patched(op_name)
                    pending_patched_operators.clear()
                    self._pending_patch_success_ops.clear()

                # Optional: trigger FDKA even on success when executor flags repeated recoveries
                # NOTE: do not wrap config in bool(); we need the mapping for .get(...)
                fdka_cfg = self.config.get("fdka", {})
                if not isinstance(fdka_cfg, dict):
                    fdka_cfg = {}
                if fdka_cfg.get("enable_on_instability", True):
                    hint = self._find_instability_hint(task_trace_for_metrics)
                    if hint:
                        print("🧠 Instability hint detected (recoveries repeated). Escalating to FDKA (soft) ...")
                        patch_applied, committed_patch = self._handle_failure(
                            task_trace_for_metrics,
                            task_id,
                            instruction,
                            record_failure_trace=False  # do not label success-run trace as terminal failure
                        )
                        if patch_applied and committed_patch:
                            op = committed_patch.get("operator")
                            if op:
                                # validate on next real success (do not mark patched now)
                                self._pending_patch_success_ops.add(op)

                if task_id > 0 and task_id % 5 == 0:
                    print(f"  🔍 Checking adaptation progress at task {task_id}...")
                    for failure_key in list(self.metrics.failure_event_counts.keys()) or list(self.metrics.failure_classes.keys()):
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

            # Optional: legacy local repair controller. Prefer executor Repair loop; keep off by default.
            if enable_local_repair and failure_type == "PreconditionUnmet" and not did_local_repair:
                before_sig = self._plan_signature(plan)
                repaired_plan = self.planner.replan(instruction, self.state.to_dict(), plan)
                after_sig = self._plan_signature(repaired_plan)

                if after_sig != before_sig:
                    try:
                        for st in repaired_plan:
                            p = (st.get("params") or {}).get("payment")
                            if p:
                                self.state.set("payment_method", p)
                                break
                    except Exception:
                        pass

                    self._normalize_state_for_planning()
                    plan = repaired_plan
                    did_local_repair = True
                    print("🔧 Local repair applied. Retrying execution without FDKA...")
                    continue

            if failure_type in ("PreconditionUnmet", "ToolError"):
                print("🧠 Escalating to FDKA for governed self-edit...")
                patch_applied, committed_patch = self._handle_failure(trace, task_id, instruction)

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

        task_trace_for_metrics = aggregated_trace or final_trace
        self.metrics.record_task(task_id, False, task_trace_for_metrics)

        failure_info = self._extract_failure_info(task_trace_for_metrics)
        if failure_info:
            metadata = {
                'instruction': instruction,
                'task_id': task_id,
                'operator': failure_info.get('operator'),
                'error_type': failure_info.get('error'),
            }
            if not any(t['metadata'].get('task_id') == task_id and not t['success'] for t in self.experience_pool.traces):
                self.experience_pool.add_trace(task_trace_for_metrics, False, metadata=metadata)
            self.metrics.check_adaptation(f"{failure_info.get('operator')}:{failure_info.get('error')}")

    # ---------------------------------------------------------------------
    # FDKA handling
    # ---------------------------------------------------------------------

    def _handle_failure(
        self,
        trace: list,
        task_id: int,
        instruction: str,
        record_failure_trace: bool = True
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if not self.fdka_enabled:
            print("  DEBUG_FDKA: disabled by config. Skipping self-edit.")
            return False, None

        patch_applied = False
        patch_dict: Optional[Dict[str, Any]] = None

        # Only persist as a failure example when the task truly failed (avoid poisoning with recovered runs)
        if record_failure_trace:
            failure_info = self._extract_failure_info(trace)
            if failure_info:
                metadata = {
                    'instruction': instruction,
                    'task_id': task_id,
                    'operator': failure_info.get('operator'),
                    'error_type': failure_info.get('error'),
                }
                self.experience_pool.add_trace(trace, False, metadata=metadata)
                print(f"  📝 Added failure trace to experience pool (operator={metadata['operator']})")

        proposed_patch = self.fdka.propose_edit(trace, self.rule_pool) or {}

        if not isinstance(proposed_patch, dict) or not proposed_patch:
            print(" FDKA: ❌ No patch proposed (provider error or empty response).")
            self.metrics.record_pipeline_rejection("PIPELINE:NO_PATCH_PROPOSED")
            # Audit: proposed_patch is empty; still record a consistent event
            self.provenance.log_patch_event(
                {"id": f"patch-{uuid.uuid4().hex[:8]}", "decision": "rejected", "reason": "no_patch_proposed"},
                decision="rejected",
                reason="no_patch_proposed",
                trace={"trace_id": f"task-{task_id}"},
                context={"task_id": task_id, "instruction": instruction},
            )
            return False, None

        # Ensure a stable patch id exists for joinability across metrics/provenance/rule_pool.
        if not proposed_patch.get("id"):
            proposed_patch["id"] = f"patch-{uuid.uuid4().hex[:8]}"

        if proposed_patch.get("action") == "REJECTED":
            print(f" FDKA: ❌ Patch rejected during validation: {proposed_patch.get('details')}")
            self.metrics.record_pipeline_rejection("PIPELINE:VALIDATION_REJECT")
            self.provenance.log_patch_event(
                proposed_patch,
                decision="rejected",
                reason="validation_reject",
                trace={"trace_id": f"task-{task_id}"},
                context={"task_id": task_id, "instruction": instruction},
            )
            return False, None

        scores = self.scorer.score(proposed_patch, trace) or {}
        agg_score = scores.get("aggregate", 0.0)

        if agg_score < self.fdka_threshold:
            print(f" FDKA: ❌ Score {agg_score:.2f} below threshold {self.fdka_threshold:.2f}. Patch rejected.")
            proposed_patch["reason_code"] = "PIPELINE:BELOW_THRESHOLD"
            proposed_patch["decision"] = "rejected"
            self.metrics.record_patch(proposed_patch, success=False, committed=False, scores=scores)
            self.metrics.record_pipeline_rejection("PIPELINE:BELOW_THRESHOLD")
            self.provenance.log_patch_event(
                proposed_patch,
                decision="rejected",
                reason="below_threshold",
                trace={"trace_id": f"task-{task_id}"},
                context={"task_id": task_id, "instruction": instruction, "scores": scores},
            )
            return False, None

        guard_result = self.guard.check(
            proposed_patch,
            context={
                "scores": scores,
                "trace": trace,
                "state": self.state.to_dict(),
            }
        )

        value_reason_code = (guard_result.get("value", {}) or {}).get("reason_code", "")
        causal_reason_code = (guard_result.get("causal", {}) or {}).get("reason_code", "")

        self.metrics.record_value_check(
            vetoed=(guard_result['decision'] == 'veto'),
            reason=guard_result.get('reason', ''),
            reason_code=value_reason_code or guard_result.get("reason_code", ""),
        )
        self.metrics.record_causal_check(
            escalated=(guard_result['decision'] == 'request_human'),
            reason=guard_result.get('reason', ''),
            reason_code=causal_reason_code or guard_result.get("reason_code", ""),
        )

        canary_result: Dict[str, Any] = {}

        if guard_result['decision'] == 'allow':
            print(" FDKA: Guardrails passed. Proceeding to canary test...")

            op_name = proposed_patch.get("operator")
            fail_examples = self.experience_pool.get_failure_traces(operator=op_name) or []
            succ_examples = self.experience_pool.get_success_traces(operator=op_name) or []
            examples = (fail_examples[-5:] + succ_examples[-5:]) or self.experience_pool.traces[-10:]

            canary_context = {
                "rule_pool": self.rule_pool,
                "stage_fn": self.rule_pool.update_operator,
                "simulator": self._run_canary_simulation,
                "examples": examples,
                "canary_suite": get_canary_suite_for_operator(op_name),
                "scorer": self.scorer,
                "scores": scores,
                "synthetic_patches": get_smt_sanity_patches_for_operator(op_name),
            }

            canary_result = self.canary_runner.run(proposed_patch, canary_context) or {}
            self.metrics.record_canary_test(
                passed=bool(canary_result.get("passed")),
                mode=canary_result.get("mode"),
                reason=canary_result.get("reason", "")
            )

            if canary_result.get("passed"):
                print(" FDKA: Canary test passed. Committing patch permanently.")

                # IMPORTANT: PatchUpdateResult truthiness includes "skipped"; use .applied for commit semantics.
                update_res = self.rule_pool.update_operator(proposed_patch)

                patch_applied = bool(getattr(update_res, "applied", False))
                patch_dict = proposed_patch if patch_applied else None

                # If canary staging accidentally mutated the live pool, commit may show "skipped".
                if (not patch_applied) and bool(getattr(update_res, "skipped", False)):
                    print(
                        f"  ⚠️ FDKA: Patch commit returned skipped ({getattr(update_res, 'reason', '')}). "
                        f"Not counting as committed."
                    )

                # Attach audit fields from RulePool (signature/edit_key/polarity/conflict)
                try:
                    proposed_patch["signature"] = getattr(update_res, "signature", "") or proposed_patch.get("signature", "")
                    proposed_patch["edit_key"] = getattr(update_res, "edit_key", "") or proposed_patch.get("edit_key", "")
                    proposed_patch["polarity"] = getattr(update_res, "polarity", "") or proposed_patch.get("polarity", "")
                    proposed_patch["reason"] = getattr(update_res, "reason", "") or proposed_patch.get("reason", "")
                    proposed_patch["conflict_with"] = getattr(update_res, "conflict_with", "") or proposed_patch.get("conflict_with", "")
                except Exception:
                    pass
            else:
                print(f" FDKA: ❌ Canary test failed: {canary_result.get('reason')}")
        elif guard_result['decision'] == 'request_human':
            print(" FDKA: ⚠️ Causal guard requests human escalation; not committing.")
        else:
            print(f" FDKA: ❌ Guardrails blocked patch: {guard_result.get('reason')}")

        # Provenance: record the decision consistently (applied vs skipped vs rejected).
        decision = "applied" if patch_applied else "rejected"
        if canary_result.get("passed") and (not patch_applied) and proposed_patch.get("reason") in ("duplicate_cached", "no_change_already_present"):
            decision = "skipped"

        reason = ""
        if guard_result.get("decision") == "veto":
            reason = guard_result.get("reason", "guard_veto")
        elif guard_result.get("decision") == "request_human":
            reason = guard_result.get("reason", "request_human")
        elif guard_result.get("decision") == "allow":
            if not canary_result.get("passed"):
                reason = canary_result.get("reason", "canary_fail")
            else:
                reason = proposed_patch.get("reason", "committed" if patch_applied else "skipped")

        # Stable reason_code for audit counters (independent of human-readable reason).
        reason_code = guard_result.get("reason_code", "") or ""
        if guard_result.get("decision") == "veto":
            reason_code = value_reason_code or reason_code or "VALUE:VETO"
        elif guard_result.get("decision") == "request_human":
            reason_code = causal_reason_code or reason_code or "CAUSAL:REQUEST_HUMAN"
        elif guard_result.get("decision") == "allow":
            if not canary_result.get("passed"):
                reason_code = "CANARY:FAIL"
            else:
                reason_code = proposed_patch.get("reason") or ("PATCH:SKIPPED" if decision == "skipped" else "PATCH:APPLIED")

        # Write one audit record with joinable ids and key governance outcomes.
        try:
            proposed_patch["decision"] = decision
            proposed_patch["reason_code"] = reason_code
            self.provenance.log_patch_event(
                proposed_patch,
                decision=decision,
                reason=reason,
                trace={"trace_id": f"task-{task_id}"},
                context={
                    "task_id": task_id,
                    "instruction": instruction,
                    "scores": scores,
                    "guard": guard_result,
                    "canary": canary_result,
                },
            )
        except Exception:
            # Fall back to legacy log to avoid breaking runs.
            self.provenance.log({
                "patch_id": proposed_patch.get("id"),
                "task_id": task_id,
                "applied": patch_applied,
                "patch": patch_dict or proposed_patch
            })

        self.metrics.record_patch(patch_dict or proposed_patch, success=patch_applied, committed=patch_applied, scores=scores)

        if patch_applied:
            patch_id = proposed_patch.get("id")
            self._last_committed_patch_id = patch_id
            self.trust_scorer.initialize_trust(patch_id)
            try:
                self._committed_patches.append({
                    "patch_id": patch_id,
                    "operator": proposed_patch.get("operator"),
                    "action": proposed_patch.get("action"),
                    "scores": {
                        "plausibility": scores.get("plausibility"),
                        "consistency": scores.get("consistency"),
                        "utility": scores.get("utility"),
                        "risk": scores.get("risk"),
                        "aggregate": scores.get("aggregate"),
                        "z3": scores.get("z3"),  # keep for audit joins
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

        sim_planner = Planner(self.config['planner'], patched_rule_pool)
        sim_plan = sim_planner.compile(instruction, sim_state.to_dict())
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
            if not isinstance(entry, dict):
                continue
            if "error" in entry:
                return entry
            if "error_type" in entry:
                normalized = dict(entry)
                normalized["error"] = normalized.get("error_type")
                return normalized
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

        # Save core artifacts
        self.metrics.save(results_dir / "metrics.json")
        self.experience_pool.save(results_dir / "experience_pool.json")

        try:
            (results_dir / "patches_committed.json").write_text(
                json.dumps(self._committed_patches, indent=2)
            )
        except Exception:
            pass

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

        try:
            self.metrics.export_governance_detail_csv(results_dir / "table4_governance_detail.csv")
            print(f"   ✅ Table 4 detail: {results_dir / 'table4_governance_detail.csv'}")
        except Exception as e:
            print(f"   ⚠️ Table 4 detail export failed: {e}")

        # Reproducibility bundle (manifest + config snapshot)
        self._write_run_bundle(results_dir)

        print(f"\n💾 Results saved to: {results_dir}")
        return self.metrics
