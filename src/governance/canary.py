# src/governance/canary.py
"""
Implements the Canary Deployment stage of the governance pipeline.
Corresponds to Section 9.5 of the paper.

MINIMUM ROBUSTNESS UPDATES (targeted):
- Never "PASS by default" when there is no behavioral evidence.
- If examples are insufficient, run LOW-POWER canary on what exists.
- If no examples exist, optionally run a deterministic micro-suite if provided in context.
- Heuristic-only mode is now conservative: only monotonic-safe, low-risk patches can pass.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random


@dataclass
class CanaryConfig:
    """Configuration for the CanaryRunner."""
    sample_size: int = 8
    max_fail_rate: float = 0.10
    min_tests_for_stats: int = 5
    heuristic_risk_threshold: float = 0.6

    # Conservative cold-start defaults: only monotonic-safe actions may pass with no examples.
    coldstart_allowed_actions: tuple = ("ADD_PRECONDITION",)
    coldstart_min_aggregate: float = 0.55


class CanaryRunner:
    """
    Executes a sandboxed test run for a proposed patch.

    Modes:
      - strict: enough examples for statistical gate + risk gate
      - low_power: some examples but too few for stats; reject on any failure + risk gate
      - deterministic_only: no examples, but a deterministic suite exists; reject on any failure + risk gate
      - coldstart_no_examples: no examples and no suite; conservative heuristic-only gate
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        self.cfg = CanaryConfig(
            sample_size=int(cfg.get("num_tests", 8)),
            max_fail_rate=float(cfg.get("statistical_max_fail_rate", 0.10)),
            min_tests_for_stats=int(cfg.get("min_tests_for_stats", 5)),
            heuristic_risk_threshold=float(cfg.get("heuristic_risk_threshold", 0.6)),
            coldstart_allowed_actions=tuple(cfg.get("coldstart_allowed_actions", ("ADD_PRECONDITION",))),
            coldstart_min_aggregate=float(cfg.get("coldstart_min_aggregate", 0.55)),
        )

        print("CANARY_RUNNER: Initialized")
        print(f"  - Sample size: {self.cfg.sample_size}")
        print(f"  - Max fail rate (statistical): {self.cfg.max_fail_rate}")
        print(f"  - Min tests for statistics: {self.cfg.min_tests_for_stats}")
        print(f"  - Heuristic risk threshold: {self.cfg.heuristic_risk_threshold}")
        print(f"  - Cold-start allowed actions: {list(self.cfg.coldstart_allowed_actions)}")
        print(f"  - Cold-start min aggregate: {self.cfg.coldstart_min_aggregate}")

    def run(self, patch: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'=' * 70}")
        print("CANARY TEST STARTING")
        print(f"{'=' * 70}")
        print(f"Patch ID: {patch.get('id', 'unknown')}")
        print(f"Action: {patch.get('action')}")
        print(f"Operator: {patch.get('operator')}")
        print(f"Details: {str(patch.get('details', 'N/A'))[:60]}...")

        examples: List[Any] = context.get("examples", []) or []
        suite: List[Any] = context.get("canary_suite", []) or []

        print("\nTest Configuration:")
        print(f"  Examples available: {len(examples)}")
        print(f"  Deterministic suite size: {len(suite)}")
        print(f"  Sample size config: {self.cfg.sample_size}")

        # If there are no examples but a deterministic suite exists, run it.
        if not examples and suite:
            print("  No examples available -> using deterministic micro-suite")
            return self._run_simulations(
                patch=patch,
                context=context,
                test_inputs=self._sample_examples(suite, min(self.cfg.sample_size, len(suite))),
                mode="deterministic_only",
                reject_on_any_failure=True,
            )

        # If nothing to test, fall back to conservative heuristics.
        if not examples and not suite:
            print("  No examples and no suite -> conservative heuristic-only mode")
            return self._heuristic_only_decision(patch, context)

        # We have at least some examples.
        low_power = len(examples) < self.cfg.min_tests_for_stats
        mode = "low_power" if low_power else "strict"

        if low_power:
            print(
                f"  LOW-POWER canary: only {len(examples)} examples "
                f"(need {self.cfg.min_tests_for_stats}+ for statistical test)"
            )

        # Always include deterministic suite inputs if provided, because it strengthens coverage.
        # Keep it bounded by sample_size.
        num_from_examples = min(self.cfg.sample_size, len(examples))
        sampled_examples = self._sample_examples(examples, num_from_examples)

        combined = list(sampled_examples)
        if suite:
            remaining = max(0, self.cfg.sample_size - len(combined))
            if remaining > 0:
                combined.extend(self._sample_examples(suite, min(remaining, len(suite))))

        print(f"  Testing with: {len(combined)} inputs (examples+suite)")
        return self._run_simulations(
            patch=patch,
            context=context,
            test_inputs=combined,
            mode=mode,
            reject_on_any_failure=low_power,  # low-power must be strict: any failure => reject
        )

    def _run_simulations(
        self,
        patch: Dict[str, Any],
        context: Dict[str, Any],
        test_inputs: List[Any],
        mode: str,
        reject_on_any_failure: bool,
    ) -> Dict[str, Any]:
        rule_pool = context.get("rule_pool")
        stage_fn = context.get("stage_fn")
        simulator = context.get("simulator")

        if not rule_pool or not stage_fn:
            print("  FAILED: Missing rule_pool or stage_fn")
            return self._result(False, "Missing canary context (rule_pool or stage_fn)", mode=mode)

        if not simulator:
            print("  FAILED: No simulator provided")
            return self._result(False, "No simulator available", mode=mode)

        print("\nStaging patch...")
        snapshot = self._safe_call(rule_pool, "snapshot")

        try:
            staged = stage_fn(patch)
            if not staged:
                print("  FAILED: Could not stage patch")
                return self._result(False, "Patch staging failed", mode=mode)

            print("  Patch staged successfully")
            print(f"\nRunning {len(test_inputs)} simulations...")
            violations = 0

            for i, example in enumerate(test_inputs, 1):
                print(f"  Test {i}/{len(test_inputs)}: ", end="", flush=True)
                try:
                    result = simulator(example, rule_pool)
                    if not result.get("ok", False):
                        violations += 1
                        print("FAILED")
                        print(f"    Reason: {result.get('reason', 'Unknown')}")
                    else:
                        print("PASSED")
                except Exception as e:
                    violations += 1
                    print("ERROR")
                    print(f"    Exception: {str(e)[:120]}")

            tested = len(test_inputs)
            fail_rate = violations / tested if tested > 0 else 0.0
            pass_rate = (tested - violations) / tested if tested > 0 else 0.0

            print("\nSimulation Results:")
            print(f"  Passed: {tested - violations}/{tested}")
            print(f"  Failed: {violations}/{tested}")
            print(f"  Pass Rate: {pass_rate:.1%}")
            print(f"  Fail Rate: {fail_rate:.1%}")

            scores = context.get("scores", {}) or {}
            risk_score = float(scores.get("risk", 0.5))

            print("\nRisk Assessment:")
            print(f"  Risk score: {risk_score:.3f}")
            print(f"  Threshold: {self.cfg.heuristic_risk_threshold:.3f}")

            if risk_score > self.cfg.heuristic_risk_threshold:
                print("  FAILED: Risk too high")
                return self._result(
                    False,
                    f"Risk score {risk_score:.3f} exceeds threshold {self.cfg.heuristic_risk_threshold:.3f}",
                    tested=tested,
                    fail_rate=fail_rate,
                    mode=mode,
                )

            if reject_on_any_failure and violations > 0:
                print("  FAILED: Reject-on-any-failure mode")
                return self._result(
                    False,
                    f"{mode}: {violations}/{tested} failures",
                    tested=tested,
                    fail_rate=fail_rate,
                    mode=mode,
                )

            if mode in ("low_power", "deterministic_only"):
                print(f"  PASSED: {mode} (no failures observed; risk acceptable)")
                return self._result(
                    True,
                    f"{mode} passed (no failures observed; risk acceptable)",
                    tested=tested,
                    fail_rate=fail_rate,
                    mode=mode,
                )

            max_allowed_failures = int(self.cfg.max_fail_rate * tested)
            statistical_pass = violations <= max_allowed_failures

            print("\nStatistical Test:")
            print(f"  Max allowed failures: {max_allowed_failures} ({self.cfg.max_fail_rate:.1%})")
            print(f"  Actual failures: {violations}")
            print(f"  Result: {'PASS' if statistical_pass else 'FAIL'}")

            if not statistical_pass:
                return self._result(
                    False,
                    f"Statistical test failed: {violations} > {max_allowed_failures}",
                    tested=tested,
                    fail_rate=fail_rate,
                    mode=mode,
                )

            passed = (violations == 0) or (pass_rate >= 0.8 and statistical_pass)
            return self._result(
                passed,
                "Canary test passed" if passed else f"Too many failures: {violations}/{tested}",
                tested=tested,
                fail_rate=fail_rate,
                mode=mode,
            )

        finally:
            if snapshot is not None:
                self._restore(rule_pool, snapshot)

    def _heuristic_only_decision(self, patch: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        print("\nHeuristic-Only Decision Mode (no examples)")

        scores = context.get("scores", {}) or {}
        risk_score = float(scores.get("risk", 0.5))
        plausibility = float(scores.get("plausibility", 0.5))
        aggregate = float(scores.get("aggregate", 0.5))
        action = str(patch.get("action", ""))

        print("Score Analysis:")
        print(f"  Action: {action}")
        print(f"  Plausibility: {plausibility:.3f}")
        print(f"  Risk: {risk_score:.3f}")
        print(f"  Aggregate: {aggregate:.3f}")

        # Hard risk gate.
        if risk_score > self.cfg.heuristic_risk_threshold:
            return self._result(
                False,
                f"Cold start: Risk {risk_score:.3f} exceeds threshold",
                mode="coldstart_no_examples",
            )

        # Conservative allow-list: only monotonic-safe actions may pass without evidence.
        if action not in self.cfg.coldstart_allowed_actions:
            return self._result(
                False,
                f"Cold start: No behavioral evidence; action '{action}' not allowed without canary examples",
                mode="coldstart_no_examples",
            )

        # Require a minimum aggregate (prevents passing on weak scoring).
        if aggregate < self.cfg.coldstart_min_aggregate:
            return self._result(
                False,
                f"Cold start: Aggregate {aggregate:.3f} below minimum {self.cfg.coldstart_min_aggregate:.3f}",
                mode="coldstart_no_examples",
            )

        # Plausibility is still relevant even for monotonic actions.
        if plausibility < 0.3:
            return self._result(
                False,
                f"Cold start: Plausibility {plausibility:.3f} too low",
                mode="coldstart_no_examples",
            )

        return self._result(
            True,
            "Cold start: Allowed monotonic-safe patch under conservative heuristic gate",
            tested=0,
            fail_rate=0.0,
            mode="coldstart_no_examples",
        )

    def _restore(self, rule_pool: Any, snapshot: Any) -> None:
        try:
            rule_pool.restore(snapshot)
            print("  RulePool restored to pre-canary state")
        except Exception as e:
            print(f"  CRITICAL: Failed to restore RulePool snapshot: {e}")

    def _sample_examples(self, examples: List[Any], k: int) -> List[Any]:
        if not examples or k <= 0:
            return []
        if len(examples) <= k:
            return list(examples)
        return random.Random(1337).sample(examples, k)

    def _safe_call(self, obj: Any, method: str, *args, **kwargs) -> Any:
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"  Error calling '{method}': {e}")
        return None

    def _result(
        self,
        passed: bool,
        reason: str,
        tested: int = 0,
        fail_rate: float = 0.0,
        mode: str = "strict",
    ) -> Dict[str, Any]:
        status = "PASS" if passed else "FAIL"
        print(f"CANARY_RUNNER: Result -> {status}")
        print(f"               Reason: {reason}")

        return {
            "passed": passed,
            "tested": tested,
            "fail_rate": round(float(fail_rate), 4),
            "reason": reason,
            "mode": mode,
        }
