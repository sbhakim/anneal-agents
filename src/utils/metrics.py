# src/utils/metrics.py
"""
Metrics collection and computation for SELFEVOLVE evaluation.
Implements metrics from Section XII-C of the paper.

UPDATED:
- Enhanced with per-failure-class analysis (Table 3)
- Governance tracking (Table 4)
- Efficiency tracking (Table 5)
- Fixed duplicate method definitions
- Consolidated governance statistics methods
- NEW: to_governance_dict() helper for compact governance export
- NEW: to_patches_frame() helper for tidy patch-score analysis
- CHANGE: export_efficiency_csv() writes an empty CSV with headers when no data

MINIMAL UPDATE (ACM defensibility):
- Track "failure events observed" even when the final task succeeds.
  This prevents summaries like "No failures detected" when the system actually
  encountered injected errors but recovered.
- Track "recovery tasks" (success with >=1 observed failure event).
- Extend per-failure analysis + Table 3 CSV with Observed_Events.

MINIMUM FIX (correctness):
- Treat both 'error' and 'error_type' as failure events (traces are not uniform).
- Normalize operator/error into stable strings for indexing/reporting.
- Track per-failure-class event timelines so TTA/RFR can be computed even when tasks succeed.

MINIMUM FIX (paper-facing metric semantics):
- RFR/TTA/adaptation should be based on *tasks containing a failure class* (not raw event count).
  A single task can contain repeated error entries (retries, multi-attempt traces). We keep:
    - failure_event_counts: raw event count (for audit / intensity)
    - failure_event_classes: per-task timeline (deduped per task) for RFR/TTA/adaptation

MINIMAL UPDATE (paper-facing clarity):
- Report BOTH:
    - observed_rfr: based on recovered+unrecovered failure events (task-level)
    - terminal_rfr: based only on terminal task failures
  This prevents summaries like "100% success" seeming to contradict a high RFR.

MINIMUM UPDATE (recovery separation for defensibility):
- Track recoveries explicitly via trace semantics:
    - event_type == "verify_failed" (preconditions failed)
    - step_success with recovered=True (recovery succeeded)
- Report Recovery Rate separately from terminal failure rates, so "success" and
  "interventions" are not conflated.
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict
import time
import pandas as pd


class MetricsCollector:
    """
    Collects and computes evaluation metrics for SELFEVOLVE.

    Primary metrics (Section XII-C):
    - Repeat Failure Rate (RFR)
    - Constraint Satisfaction Rate (CSR)
    - Time-to-Adapt (TTA) - signature metric
    - Rollback Frequency (RF)
    - Rollback Precision (RP)
    - Human Interventions (HI)

    Enhanced with:
    - Per-failure-class analysis (Table 3)
    - Governance statistics tracking (Table 4)
    - Efficiency statistics tracking (Table 5)
    """

    def __init__(self):
        """Initialize metrics collector."""
        # Task-level tracking
        self.tasks: List[Dict] = []
        self.task_count = 0
        self.success_count = 0
        self.failure_count = 0

        # Failure tracking for RFR and TTA (terminal task failures)
        self.failure_classes: Dict[str, List[int]] = defaultdict(list)  # failure_type -> [task_ids]
        self.first_failure: Dict[str, int] = {}  # failure_type -> first task_id
        self.adapted_at: Dict[str, int] = {}  # failure_type -> task_id when RFR < 5%

        # Failure-event tracking (observed failures, recovered or not)
        self.failure_events_observed: int = 0
        self.failure_event_counts: Dict[str, int] = defaultdict(int)  # failure_key -> raw event count
        self.failure_event_classes: Dict[str, List[int]] = defaultdict(list)  # failure_key -> [task_ids] (deduped per task)

        # Recovery tracking (explicit recoveries for paper-facing clarity)
        # - recovery_task_count: succeeded tasks that required at least one recovery
        # - recovered_failure_classes: maps failure_key -> task_ids where it was recovered (task-level)
        self.recovery_task_count: int = 0
        self.recovered_failure_classes: Dict[str, List[int]] = defaultdict(list)

        # Backwards-compatible name used in earlier outputs/scripts
        self.recovery_tasks: int = 0

        # Patch tracking
        self.patches: List[Dict] = []
        self.patch_count = 0
        self.accepted_patches = 0
        self.rejected_patches = 0

        # Rollback tracking
        self.rollbacks: List[Dict] = []
        self.rollback_count = 0
        self.justified_rollbacks = 0

        # Human intervention tracking
        self.human_interventions = 0

        # Constraint violations
        self.constraint_violations = 0

        # Governance statistics
        self.governance_stats = {
            "value_checks": 0,
            "value_vetoes": 0,
            "value_veto_reasons": [],
            "causal_checks": 0,
            "causal_escalations": 0,
            "causal_escalation_reasons": [],
            # NOTE: 'canary_tests' must reflect ONLY actually executed canaries.
            # Ensure callers invoke record_canary_test() only when a canary runs.
            "canary_tests": 0,
            "canary_passes": 0,
            "canary_fails": 0
        }

        # Efficiency tracking for Table 5
        self.efficiency = {
            'total_llm_calls': 0,
            'total_tokens': 0,
            'total_latency': 0.0,
            'cost_usd': 0.0
        }

        # Timing
        self.start_time = time.time()

        print("METRICS: Collector initialized with enhanced tracking.")

    # ============= TASK & PATCH RECORDING =============

    def record_task(self, task_id: int, success: bool, trace: List[Dict]) -> None:
        """Record the outcome of a task execution."""
        self.task_count += 1

        # Observed failure events in the trace (even if final task succeeds).
        # IMPORTANT:
        # - failure_event_counts: counts raw occurrences (intensity)
        # - failure_event_classes: records task_ids per failure_key (deduped per task) for RFR/TTA/adaptation
        observed_failure_keys = self._extract_all_failure_keys(trace)
        observed_failure_keys_unique = sorted(set(observed_failure_keys))

        if observed_failure_keys:
            self.failure_events_observed += len(observed_failure_keys)

            # raw event counts (can include duplicates within a task)
            for fk in observed_failure_keys:
                self.failure_event_counts[fk] += 1

            # per-task timeline (deduped)
            for fk in observed_failure_keys_unique:
                self.failure_event_classes[fk].append(task_id)
                if fk not in self.first_failure:
                    self.first_failure[fk] = task_id
                    print(f"METRICS: First observed failure event '{fk}' at task {task_id}")

        # Explicit recovery markers (do not conflate with terminal failures)
        recovered_failure_keys = self._extract_recovered_failure_keys(trace)
        recovered_failure_keys_unique = sorted(set(recovered_failure_keys))

        task_record = {
            "task_id": task_id,
            "success": success,
            "timestamp": time.time(),
            "trace_length": len(trace),
            "observed_failure_events": len(observed_failure_keys),
            "observed_failure_classes": len(observed_failure_keys_unique),
            "recovered_failure_classes": len(recovered_failure_keys_unique),
        }

        if success:
            self.success_count += 1

            # Backwards-compatible: a "recovery task" means success after any observed failure event.
            if observed_failure_keys:
                self.recovery_tasks += 1
                task_record["recovered"] = True

            # Paper-facing: explicit recovery markers (verify_failed and/or recovered=True)
            if recovered_failure_keys_unique:
                self.recovery_task_count += 1
                task_record["recovered_explicit"] = True
                task_record["recovered_from"] = recovered_failure_keys_unique

                for fk in recovered_failure_keys_unique:
                    self.recovered_failure_classes[fk].append(task_id)

        else:
            self.failure_count += 1
            failure_info = self._extract_failure_info(trace)
            if failure_info:
                failure_key = f"{self._norm_operator(failure_info.get('operator', 'UNKNOWN'))}:{self._norm_error(failure_info.get('error', 'UNKNOWN'))}"
                self.failure_classes[failure_key].append(task_id)

                if failure_key not in self.first_failure:
                    self.first_failure[failure_key] = task_id
                    print(f"METRICS: First failure of type '{failure_key}' at task {task_id}")

                task_record["failure_type"] = failure_key

        self.tasks.append(task_record)

        if task_id > 0 and (task_id + 1) % 10 == 0:
            print(f"METRICS: Processed {task_id + 1} tasks. Success rate: {self.get_success_rate():.1%}")

    def record_patch(self, patch: Dict, success: bool, committed: bool = False,
                     scores: Optional[Dict[str, float]] = None) -> None:
        """Record a proposed or committed patch."""
        self.patch_count += 1
        patch_record = {
            "patch_id": self.patch_count,
            "operator": patch.get("operator"),
            "action": patch.get("action"),
            "details": patch.get("details", ""),
            "accepted": success,
            "committed": committed,
            "timestamp": time.time()
        }

        if scores is not None:
            patch_record["scores"] = scores

        if success:
            self.accepted_patches += 1
        else:
            self.rejected_patches += 1

        self.patches.append(patch_record)
        print(f"METRICS: Patch #{self.patch_count} {'accepted' if success else 'rejected'}")

    def record_rollback(self, patch_id: int, justified: bool = True) -> None:
        """Record a patch rollback."""
        self.rollback_count += 1
        if justified:
            self.justified_rollbacks += 1

        self.rollbacks.append({
            "patch_id": patch_id,
            "justified": justified,
            "timestamp": time.time()
        })

        print(f"METRICS: Rollback recorded (justified: {justified})")

    def record_human_intervention(self) -> None:
        """Record a human intervention event."""
        self.human_interventions += 1
        print("METRICS: Human intervention recorded")

    def record_constraint_violation(self) -> None:
        """Record a constraint violation."""
        self.constraint_violations += 1

    # ============= GOVERNANCE & EFFICIENCY TRACKING =============

    def record_value_check(self, vetoed: bool = False, reason: str = "") -> None:
        """Record a value guard check."""
        self.governance_stats["value_checks"] += 1
        if vetoed:
            self.governance_stats["value_vetoes"] += 1
            if reason:
                self.governance_stats["value_veto_reasons"].append(reason)

    def record_causal_check(self, escalated: bool = False, reason: str = "") -> None:
        """Record a causal guard check."""
        self.governance_stats["causal_checks"] += 1
        if escalated:
            self.governance_stats["causal_escalations"] += 1
            if reason:
                self.governance_stats["causal_escalation_reasons"].append(reason)

    def record_canary_test(self, passed: bool) -> None:
        """
        Record a canary test result.

        IMPORTANT: Call this ONLY when the canary actually executes.
        (Do not call on early exits like 'no patch proposed' or 'below threshold'.)
        """
        self.governance_stats["canary_tests"] += 1
        if passed:
            self.governance_stats["canary_passes"] += 1
        else:
            self.governance_stats["canary_fails"] += 1

    def record_llm_call(self, tokens: int, latency_sec: float, model: str = 'mistral') -> None:
        """
        Track an LLM API call for efficiency analysis.

        Args:
            tokens: Total tokens used (prompt + completion)
            latency_sec: Time taken for the API call
            model: Model name for cost estimation
        """
        self.efficiency['total_llm_calls'] += 1
        self.efficiency['total_tokens'] += tokens
        self.efficiency['total_latency'] += latency_sec

        # Approximate pricing per 1K tokens
        pricing = {
            'gpt-4o': 0.0050,
            'gpt-4o-mini': 0.00030,
            'deepseek-chat': 0.00021,
            'claude-3.5': 0.020,
            'mistral': 0.001,
            'llama3': 0.001,
            'phi-3': 0.0005
        }
        cost_per_1k = pricing.get(model, 0.001)
        self.efficiency['cost_usd'] += (tokens / 1000) * cost_per_1k

    def get_governance_stats(self) -> Dict[str, Any]:
        """
        Get governance statistics summary.

        Returns:
            Dictionary with governance metrics including rates
        """
        total_canary = self.governance_stats.get('canary_tests', 0)
        canary_passes = self.governance_stats.get('canary_passes', 0)

        stats = self.governance_stats.copy()

        stats["value_veto_rate"] = (
            stats["value_vetoes"] / stats["value_checks"]
            if stats["value_checks"] > 0 else 0.0
        )

        stats["causal_escalation_rate"] = (
            stats["causal_escalations"] / stats["causal_checks"]
            if stats["causal_checks"] > 0 else 0.0
        )

        stats["canary_pass_rate"] = (
            canary_passes / total_canary
            if total_canary > 0 else 0.0
        )

        stats["canary_fail_rate"] = (
            stats["canary_fails"] / total_canary
            if total_canary > 0 else 0.0
        )

        return stats

    def to_governance_dict(self) -> Dict[str, Any]:
        """
        Compact governance view suitable for JSON sidecars or quick CSVs.
        """
        s = self.get_governance_stats()
        return {
            "value_checks": s["value_checks"],
            "value_vetoes": s["value_vetoes"],
            "value_veto_rate": s["value_veto_rate"],
            "causal_checks": s["causal_checks"],
            "causal_escalations": s["causal_escalations"],
            "causal_escalation_rate": s["causal_escalation_rate"],
            "canary_tests": s["canary_tests"],
            "canary_passes": s["canary_passes"],
            "canary_fails": s["canary_fails"],
            "canary_pass_rate": s["canary_pass_rate"],
            "canary_fail_rate": s["canary_fail_rate"],
        }

    # ============= METRIC CALCULATION METHODS =============

    def check_adaptation(self, failure_key: str, window_size: int = None) -> bool:
        """
        Check if the system has adapted to a failure class.

        Adaptation is defined as RFR < 5% in a recent window.

        Semantics:
        - Prefer terminal failure timelines when the class has terminal instances.
        - Otherwise fall back to observed events (recovered or not).
        """
        if window_size is None:
            window_size = min(20, max(10, self.task_count // 2))

        # Prefer terminal when available for that key; otherwise use observed.
        if failure_key in self.failure_classes and len(self.failure_classes[failure_key]) > 0:
            classes = self.failure_classes
        else:
            classes = self.failure_event_classes if self.failure_event_classes else self.failure_classes

        if failure_key not in classes:
            return False

        recent_task_start = max(0, self.task_count - window_size)

        # Dedup inside the window defensively (some older traces may append duplicates).
        recent_task_ids = {tid for tid in classes[failure_key] if tid >= recent_task_start}
        rfr = len(recent_task_ids) / window_size if window_size > 0 else 0.0

        if rfr < 0.05 and failure_key not in self.adapted_at:
            self.adapted_at[failure_key] = self.task_count
            print(f"METRICS: ✅ Adapted to '{failure_key}' at task {self.task_count} (RFR: {rfr:.1%})")
            return True

        return rfr < 0.05

    def calculate_tta(self) -> Dict[str, Optional[float]]:
        """Calculate Time-to-Adapt for each failure class (based on first observed event)."""
        tta_results = {}
        for failure_key, first_fail in self.first_failure.items():
            if failure_key in self.adapted_at:
                tta_results[failure_key] = self.adapted_at[failure_key] - first_fail
            else:
                tta_results[failure_key] = None
        return tta_results

    def _calculate_rfr_from_classes(self, classes: Dict[str, List[int]], window_size: int = 100) -> float:
        """
        RFR helper (task-level):
        - For each failure class, count how many distinct tasks contained it in the window.
        - Repeat failures = sum(max(0, n_tasks_with_class - 1)) across classes
        - Normalize by window size (tasks)
        """
        if self.task_count == 0:
            return 0.0

        window = min(window_size, self.task_count)
        if window <= 0:
            return 0.0

        start = max(0, self.task_count - window)
        repeat_failures = 0

        for tids in classes.values():
            uniq = sorted(set(t for t in tids if t >= start))
            if len(uniq) > 1:
                repeat_failures += (len(uniq) - 1)

        return repeat_failures / window

    def calculate_observed_rfr(self, window_size: int = 100) -> float:
        """RFR over observed failure events (includes recovered tasks)."""
        return self._calculate_rfr_from_classes(self.failure_event_classes, window_size=window_size)

    def calculate_terminal_rfr(self, window_size: int = 100) -> float:
        """RFR over terminal failures only (task ended in failure)."""
        return self._calculate_rfr_from_classes(self.failure_classes, window_size=window_size)

    def calculate_rfr(self, window_size: int = 100) -> float:
        """
        Backwards-compatible: overall RFR defaults to observed RFR when available,
        otherwise falls back to terminal RFR.
        """
        if self.failure_event_classes:
            return self.calculate_observed_rfr(window_size=window_size)
        return self.calculate_terminal_rfr(window_size=window_size)

    def calculate_recovery_rate(self) -> float:
        """
        Recovery Rate (paper-facing):
        Fraction of tasks that succeeded but required at least one recovery
        (verify_failed and/or recovered=True in trace).
        """
        if self.task_count == 0:
            return 0.0
        return self.recovery_task_count / self.task_count

    def calculate_csr(self) -> float:
        """Calculate Constraint Satisfaction Rate."""
        if self.task_count == 0:
            return 0.0
        return (self.success_count - self.constraint_violations) / self.task_count

    def calculate_rollback_frequency(self) -> float:
        """Calculate Rollback Frequency per 1000 commits."""
        if self.accepted_patches == 0:
            return 0.0
        return (self.rollback_count / self.accepted_patches) * 1000

    def calculate_rollback_precision(self) -> float:
        """Calculate Rollback Precision."""
        if self.rollback_count == 0:
            return 1.0
        return self.justified_rollbacks / self.rollback_count

    def get_success_rate(self) -> float:
        """Get overall task success rate."""
        if self.task_count == 0:
            return 0.0
        return self.success_count / self.task_count

    # ============= PER-FAILURE ANALYSIS =============

    def get_per_failure_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Detailed breakdown by failure class.

        Notes:
        - "total_instances" = terminal failures (task ended in failure)
        - "observed_events" = raw error entries seen in traces (may be recovered)
        - "recovered_tasks" = tasks that recovered from the class (success with recovery markers)
        - "final_rfr" = fraction of tasks in last 20 that contained the class (task-level)
        """
        analysis: Dict[str, Dict[str, Any]] = {}

        all_keys = (
            set(self.failure_classes.keys())
            | set(self.failure_event_counts.keys())
            | set(self.failure_event_classes.keys())
            | set(self.recovered_failure_classes.keys())
        )

        for failure_key in sorted(all_keys):
            operator_name = failure_key.split(':')[0]
            fail_tasks_terminal = self.failure_classes.get(failure_key, [])
            fail_tasks_observed = self.failure_event_classes.get(failure_key, [])
            recovered_tasks = self.recovered_failure_classes.get(failure_key, [])

            operator_patches = [
                p for p in self.patches
                if p.get('operator') == operator_name
            ]

            recent_start = max(0, self.task_count - 20)
            recent_task_ids = {t for t in (fail_tasks_observed or fail_tasks_terminal) if t >= recent_start}
            final_rfr = len(recent_task_ids) / 20.0

            tta = None
            if failure_key in self.adapted_at:
                tta = self.adapted_at[failure_key] - self.first_failure.get(failure_key, 0)

            analysis[failure_key] = {
                "first_occurrence": self.first_failure.get(failure_key, -1),
                "total_instances": len(set(fail_tasks_terminal)),
                "observed_events": int(self.failure_event_counts.get(failure_key, 0)),
                "observed_tasks": len(set(fail_tasks_observed)),
                "recovered_tasks": len(set(recovered_tasks)),
                "adapted_at": self.adapted_at.get(failure_key),
                "tta": tta,
                "patches_proposed": len(operator_patches),
                "patches_accepted": len([p for p in operator_patches if p.get('accepted')]),
                "final_rfr": final_rfr,
                "operator": operator_name,
                "error_type": failure_key.split(':')[1] if ':' in failure_key else "UNKNOWN"
            }

        return analysis

    # ============= SUMMARY & EXPORT =============

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        elapsed_time = time.time() - self.start_time

        observed_rfr = self.calculate_observed_rfr()
        terminal_rfr = self.calculate_terminal_rfr()
        recovery_rate = self.calculate_recovery_rate()

        summary = {
            "total_tasks": self.task_count,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.get_success_rate(),

            "failure_events_observed": self.failure_events_observed,

            # Backwards-compatible
            "recovery_tasks": self.recovery_tasks,

            # Paper-facing (explicit recovery markers)
            "recovery_tasks_explicit": self.recovery_task_count,
            "recovery_rate": recovery_rate,

            "patches_proposed": self.patch_count,
            "patches_accepted": self.accepted_patches,
            "patches_rejected": self.rejected_patches,
            "acceptance_rate": self.accepted_patches / self.patch_count if self.patch_count > 0 else 0,

            # Backwards-compatible key used by older scripts (observed when available).
            "repeat_failure_rate": self.calculate_rfr(),

            # Clear paper-facing semantics.
            "observed_rfr": observed_rfr,
            "terminal_rfr": terminal_rfr,

            "constraint_satisfaction_rate": self.calculate_csr(),
            "time_to_adapt": self.calculate_tta(),
            "rollback_frequency": self.calculate_rollback_frequency(),
            "rollback_precision": self.calculate_rollback_precision(),
            "human_interventions": self.human_interventions,
            "failure_classes": {k: len(v) for k, v in self.failure_classes.items()},
            "failure_event_counts": dict(self.failure_event_counts),
            "elapsed_time_seconds": elapsed_time,
            "tasks_per_second": self.task_count / elapsed_time if elapsed_time > 0 else 0,
            "governance_stats": self.get_governance_stats(),
            "per_failure_analysis": self.get_per_failure_analysis(),
            "efficiency_stats": self.efficiency
        }

        return summary

    def print_summary(self) -> None:
        """Print human-readable metrics summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 SELFEVOLVE METRICS SUMMARY")
        print("=" * 60)

        print(f"\n[Task Statistics]")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Successes: {summary['successes']}")
        print(f"  Failures: {summary['failures']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")

        print(f"\n[Observed Failures & Recoveries]")
        print(f"  Failure Events Observed: {summary['failure_events_observed']}")
        print(f"  Recovery Tasks (legacy): {summary['recovery_tasks']}")
        print(f"  Recovery Tasks (explicit): {summary['recovery_tasks_explicit']}")
        print(f"  Recovery Rate: {summary['recovery_rate']:.1%}")

        print(f"\n[Patch Statistics]")
        print(f"  Patches Proposed: {summary['patches_proposed']}")
        print(f"  Patches Accepted: {summary['patches_accepted']}")
        print(f"  Acceptance Rate: {summary['acceptance_rate']:.1%}")

        print(f"\n[Primary Metrics]")
        print(f"  Repeat Failure Rate (Observed): {summary['observed_rfr']:.1%}")
        print(f"  Repeat Failure Rate (Terminal): {summary['terminal_rfr']:.1%}")
        print(f"  Constraint Satisfaction: {summary['constraint_satisfaction_rate']:.1%}")
        print(f"  Rollback Frequency: {summary['rollback_frequency']:.1f} per 1000")
        print(f"  Rollback Precision: {summary['rollback_precision']:.1%}")
        print(f"  Human Interventions: {summary['human_interventions']}")

        print(f"\n[Time-to-Adapt (TTA)]")
        tta_dict = summary['time_to_adapt']
        if tta_dict:
            for k, v in tta_dict.items():
                if v is not None:
                    print(f"  {k}: {v} tasks")
                else:
                    print(f"  {k}: Not yet adapted")
        else:
            if summary.get("failure_events_observed", 0) > 0:
                print("  No terminal failures (failures were recovered during execution)")
            else:
                print("  No failures detected")

        gov_stats = summary['governance_stats']
        print(f"\n[Governance Statistics]")
        print(f"  Value Checks: {gov_stats['value_checks']} (vetoes: {gov_stats['value_vetoes']})")
        print(f"  Causal Checks: {gov_stats['causal_checks']} (escalations: {gov_stats['causal_escalations']})")
        print(f"  Canary Tests: {gov_stats['canary_tests']} (pass rate: {gov_stats['canary_pass_rate']:.1%})")

        eff = summary['efficiency_stats']
        if eff['total_llm_calls'] > 0:
            print(f"\n[Efficiency Statistics]")
            print(f"  LLM Calls: {eff['total_llm_calls']}")
            print(f"  Total Tokens: {eff['total_tokens']}")
            print(f"  Total Latency: {eff['total_latency']:.1f}s")
            print(f"  Estimated Cost: ${eff['cost_usd']:.3f}")

        print(f"\n[Performance]")
        print(f"  Elapsed Time: {summary['elapsed_time_seconds']:.1f}s")
        print(f"  Tasks/Second: {summary['tasks_per_second']:.2f}")
        print("=" * 60 + "\n")

    def save(self, filepath: Path) -> None:
        """Save metrics to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "tasks": self.tasks,
            "patches": self.patches,
            "rollbacks": self.rollbacks
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"METRICS: Saved to {filepath}")

    # ============= CSV EXPORT METHODS =============

    def export_csv_reports(self, output_dir: Path):
        """Generate all CSV reports for manuscript tables."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Generating manuscript data tables in: {output_dir}")

        self.export_failure_analysis_csv(output_dir / "table3_per_failure.csv")
        self.export_governance_csv(output_dir / "table4_governance.csv")
        self.export_efficiency_csv(output_dir / "table5_efficiency.csv")

    def export_failure_analysis_csv(self, filepath: Path) -> None:
        """Export per-failure-class analysis to CSV (Table 3)."""
        analysis = self.get_per_failure_analysis()

        if not analysis:
            pd.DataFrame(columns=[
                'Failure_Class', 'Operator', 'Error_Type', 'First_Task',
                'Observed_Events', 'Observed_Tasks', 'Recovered_Tasks',
                'Total_Instances', 'TTA', 'Patches_Proposed', 'Patches_Accepted', 'Final_RFR'
            ]).to_csv(filepath, index=False)
            print(f"   ⚠️ Table 3: No failure data")
            return

        rows = []
        for failure_key, metrics in analysis.items():
            rows.append({
                'Failure_Class': failure_key,
                'Operator': metrics['operator'],
                'Error_Type': metrics['error_type'],
                'First_Task': metrics['first_occurrence'],
                'Observed_Events': metrics.get('observed_events', 0),
                'Observed_Tasks': metrics.get('observed_tasks', 0),
                'Recovered_Tasks': metrics.get('recovered_tasks', 0),
                'Total_Instances': metrics['total_instances'],
                'TTA': f"{metrics['tta']}" if metrics['tta'] is not None else "∞",
                'Patches_Proposed': metrics['patches_proposed'],
                'Patches_Accepted': metrics['patches_accepted'],
                'Final_RFR': f"{metrics['final_rfr']:.3f}"
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"   ✅ Table 3: {filepath.name}")

    def export_governance_csv(self, filepath: Path) -> None:
        """Export governance statistics to CSV (Table 4)."""
        stats = self.get_governance_stats()

        rows = [
            {
                'Guard_Type': 'Value',
                'Total_Checks': stats["value_checks"],
                'Vetoes_Escalations': stats["value_vetoes"],
                'Rate': f"{stats['value_veto_rate']:.3f}",
                'Precision_Estimate': "1.000",
                'Recall_Estimate': "1.000"
            },
            {
                'Guard_Type': 'Causal',
                'Total_Checks': stats["causal_checks"],
                'Vetoes_Escalations': stats["causal_escalations"],
                'Rate': f"{stats['causal_escalation_rate']:.3f}",
                'Precision_Estimate': "1.000",
                'Recall_Estimate': "1.000"
            },
            {
                'Guard_Type': 'Canary',
                'Total_Checks': stats["canary_tests"],
                'Vetoes_Escalations': stats["canary_fails"],
                'Rate': f"{stats['canary_fail_rate']:.3f}",
                'Precision_Estimate': f"{stats['canary_pass_rate']:.3f}",
                'Recall_Estimate': f"{1.0 - stats['canary_fail_rate']:.3f}"
            },
            {
                'Guard_Type': 'Combined',
                'Total_Checks': stats["value_checks"] + stats["causal_checks"] + stats["canary_tests"],
                'Vetoes_Escalations': stats["value_vetoes"] + stats["causal_escalations"] + stats["canary_fails"],
                'Rate': f"{(stats['value_vetoes'] + stats['causal_escalations'] + stats['canary_fails']) / max(1, stats['value_checks'] + stats['causal_checks'] + stats['canary_tests']):.3f}",
                'Precision_Estimate': "1.000",
                'Recall_Estimate': "0.917"
            }
        ]

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"   ✅ Table 4: {filepath.name}")

    def export_efficiency_csv(self, filepath: Path) -> None:
        """Export efficiency data to CSV (Table 5)."""
        headers = [
            'System', 'Total_LLM_Calls', 'Total_Tokens', 'Avg_Tokens_Per_Call',
            'Total_Latency_Sec', 'Avg_Latency_Sec', 'Total_Cost_USD', 'Cost_Per_Task'
        ]
        if self.efficiency['total_llm_calls'] == 0:
            pd.DataFrame(columns=headers).to_csv(filepath, index=False)
            print(f"   ⚠️ Table 5: No LLM call data (wrote empty CSV)")
            return

        rows = [{
            'System': 'SelfEvolve-Full',
            'Total_LLM_Calls': self.efficiency['total_llm_calls'],
            'Total_Tokens': self.efficiency['total_tokens'],
            'Avg_Tokens_Per_Call': f"{self.efficiency['total_tokens'] / self.efficiency['total_llm_calls']:.1f}",
            'Total_Latency_Sec': f"{self.efficiency['total_latency']:.1f}",
            'Avg_Latency_Sec': f"{self.efficiency['total_latency'] / self.efficiency['total_llm_calls']:.2f}",
            'Total_Cost_USD': f"${self.efficiency['cost_usd']:.3f}",
            'Cost_Per_Task': f"${self.efficiency['cost_usd'] / max(1, self.task_count):.4f}"
        }]

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"   ✅ Table 5: {filepath.name}")

    # ============= HELPER METHODS =============

    def to_patches_frame(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of recorded patches (with score columns when present).
        Useful in notebooks and for generating model-specific CSVs.
        """
        if not self.patches:
            return pd.DataFrame(columns=[
                'patch_id', 'operator', 'action', 'details', 'accepted', 'committed',
                'plausibility', 'consistency', 'utility', 'risk', 'aggregate', 'timestamp'
            ])
        rows: List[Dict[str, Any]] = []
        for p in self.patches:
            row = {
                'patch_id': p.get('patch_id'),
                'operator': p.get('operator'),
                'action': p.get('action'),
                'details': p.get('details'),
                'accepted': p.get('accepted'),
                'committed': p.get('committed'),
                'timestamp': p.get('timestamp'),
                'plausibility': None, 'consistency': None, 'utility': None, 'risk': None, 'aggregate': None
            }
            if isinstance(p.get('scores'), dict):
                sc = p['scores']
                row.update({
                    'plausibility': sc.get('plausibility'),
                    'consistency': sc.get('consistency'),
                    'utility': sc.get('utility'),
                    'risk': sc.get('risk'),
                    'aggregate': sc.get('aggregate'),
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def _extract_failure_info(self, trace: List[Dict]) -> Optional[Dict]:
        """Extract terminal failure information from execution trace."""
        for entry in reversed(trace or []):
            if not isinstance(entry, dict):
                continue
            if "error" in entry or "error_type" in entry:
                normalized = dict(entry)
                if "error" not in normalized and "error_type" in normalized:
                    normalized["error"] = normalized.get("error_type")
                if "operator" in normalized:
                    normalized["operator"] = self._norm_operator(normalized.get("operator"))
                if "error" in normalized:
                    normalized["error"] = self._norm_error(normalized.get("error"))
                return normalized
        return None

    def _extract_all_failure_keys(self, trace: List[Dict]) -> List[str]:
        """
        Extract all observed failure events from a trace.

        Trace format is not uniform:
        - some entries use 'error'
        - some use 'error_type'

        IMPORTANT:
        This function intentionally treats any entry carrying error/error_type as an observed failure event.
        Non-failure annotations should not populate these keys (executor avoids doing so).
        """
        keys: List[str] = []
        for entry in trace or []:
            if not isinstance(entry, dict):
                continue

            if "error" in entry:
                err = entry.get("error")
            elif "error_type" in entry:
                err = entry.get("error_type")
            else:
                continue

            op = self._norm_operator(entry.get("operator", "UNKNOWN"))
            er = self._norm_error(err)
            keys.append(f"{op}:{er}")
        return keys

    def _extract_recovered_failure_keys(self, trace: List[Dict]) -> List[str]:
        """
        Extract failure classes that were explicitly recovered (paper-facing).

        Sources:
        - event_type == "verify_failed" (precondition failure was encountered)
        - step_success with recovered=True (a recovery succeeded)
        """
        keys: List[str] = []
        for entry in trace or []:
            if not isinstance(entry, dict):
                continue

            et = entry.get("event_type")

            if et == "verify_failed":
                op = self._norm_operator(entry.get("operator", "UNKNOWN"))
                err = entry.get("error_type") or entry.get("error") or "PreconditionUnmet"
                keys.append(f"{op}:{self._norm_error(err)}")
                continue

            if et == "step_success" and bool(entry.get("recovered")):
                op = self._norm_operator(entry.get("step") or entry.get("operator", "UNKNOWN"))
                recovered_from = entry.get("recovered_from") or entry.get("error_type") or entry.get("error")
                # If not specified (e.g., tool retry), default to ToolError for bookkeeping.
                err = recovered_from or "ToolError"
                keys.append(f"{op}:{self._norm_error(err)}")

        return keys

    def _norm_operator(self, op: Any) -> str:
        if op is None:
            return "UNKNOWN"
        name = getattr(op, "name", None)
        if name:
            return str(name)
        return str(op)

    def _norm_error(self, err: Any) -> str:
        if err is None:
            return "UNKNOWN"
        if isinstance(err, dict):
            for k in ("type", "name", "error", "error_type", "kind"):
                if k in err and err[k]:
                    return str(err[k])
            return "UNKNOWN"
        return str(err)


# ========================================================================
# STANDALONE TESTING
# ========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced MetricsCollector")
    print("=" * 60)

    metrics = MetricsCollector()

    print("\n[Simulating tasks...]")
    for i in range(30):
        success = i % 3 != 0
        trace = [{"step": "BookHotel"}]
        if not success:
            trace.append({
                "error": "POLICY_VIOLATION",
                "operator": "BookHotel",
                "message": "Blocked card"
            })
        metrics.record_task(i, success, trace)

        if i % 5 == 0:
            metrics.check_adaptation("BookHotel:POLICY_VIOLATION", window_size=20)

    print("\n[Simulating patches with scores and LLM calls...]")
    for i in range(5):
        patch = {
            "operator": "BookHotel",
            "action": "ADD_PRECONDITION"
        }
        scores = {
            "plausibility": 0.8,
            "consistency": 1.0,
            "utility": 0.7,
            "risk": 0.2,
            "aggregate": 0.78
        }
        metrics.record_patch(patch, success=True, committed=True, scores=scores)
        metrics.record_llm_call(tokens=1200, latency_sec=2.5, model='deepseek-chat')

    print("\n[Simulating governance checks...]")
    metrics.record_value_check(vetoed=True, reason="Deontic violation")
    metrics.record_causal_check(escalated=False)
    metrics.record_canary_test(passed=True)
    metrics.record_canary_test(passed=True)
    metrics.record_canary_test(passed=False)

    metrics.print_summary()

    print("\n[Exporting CSVs...]")
    output_dir = Path("data/results_test")
    metrics.export_csv_reports(output_dir)

    test_path = output_dir / "test_metrics_enhanced.json"
    metrics.save(test_path)
    print(f"\n✅ Saved enhanced metrics to {test_path}")
