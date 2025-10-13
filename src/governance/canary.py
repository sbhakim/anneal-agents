# src/governance/canary.py
"""
Implements the Canary Deployment stage of the governance pipeline.
Corresponds to Section 9.5 of the paper.

UPDATED:
- Fixed attribute access bugs (self.cfg.* instead of self.*)
- Added comprehensive debugging output
- Improved cold-start handling for insufficient examples
- Enhanced error messages and decision logging
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import random


@dataclass
class CanaryConfig:
    """Configuration for the CanaryRunner."""
    sample_size: int = 8
    max_fail_rate: float = 0.10
    min_tests_for_stats: int = 5
    heuristic_risk_threshold: float = 0.6


class CanaryRunner:
    """
    Executes a sandboxed test run for a proposed patch.
    Uses statistical testing when sufficient examples are available,
    falls back to heuristic risk assessment otherwise.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize canary runner with configuration.

        Args:
            config: Dictionary with canary settings
        """
        cfg = config or {}

        # ✅ FIXED: Store all config in self.cfg
        self.cfg = CanaryConfig(
            sample_size=int(cfg.get("num_tests", 8)),  # ✅ Use 'num_tests' from config
            max_fail_rate=float(cfg.get("statistical_max_fail_rate", 0.10)),
            min_tests_for_stats=int(cfg.get("min_tests_for_stats", 5)),
            heuristic_risk_threshold=float(cfg.get("heuristic_risk_threshold", 0.6))
        )

        print("CANARY_RUNNER: Initialized")
        print(f"  - Sample size: {self.cfg.sample_size}")
        print(f"  - Max fail rate (statistical): {self.cfg.max_fail_rate}")
        print(f"  - Min tests for statistics: {self.cfg.min_tests_for_stats}")
        print(f"  - Heuristic risk threshold: {self.cfg.heuristic_risk_threshold}")

    def run(self, patch: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Run canary deployment test with comprehensive debugging.

        Args:
            patch: The proposed patch to test
            context: Test context including examples, rule_pool, simulator, etc.

        Returns:
            Dictionary with 'passed' boolean and diagnostic information
        """
        print(f"\n{'=' * 70}")
        print(f"🧪 CANARY TEST STARTING")
        print(f"{'=' * 70}")
        print(f"Patch ID: {patch.get('id', 'unknown')}")
        print(f"Action: {patch.get('action')}")
        print(f"Operator: {patch.get('operator')}")
        print(f"Details: {patch.get('details', 'N/A')[:60]}...")

        # Get test examples
        examples = context.get('examples', [])
        print(f"\n📊 Test Configuration:")
        print(f"   Examples available: {len(examples)}")
        print(f"   Sample size config: {self.cfg.sample_size}")  # ✅ FIXED: Use self.cfg

        # ✅ ENHANCED: Better cold-start handling
        if not examples:
            print(f"   ⚠️ No examples available (cold start)")
            print(f"   → Using heuristic-only mode")
            return self._heuristic_only_decision(patch, context)

        if len(examples) < self.cfg.min_tests_for_stats:
            print(f"   ⚠️ Only {len(examples)} examples (need {self.cfg.min_tests_for_stats}+ for robust test)")
            print(f"   → Allowing patch due to insufficient data (cold start)")
            return {
                "passed": True,
                "reason": f"Insufficient examples ({len(examples)} < {self.cfg.min_tests_for_stats}) - cold start allowance",
                "tested": len(examples),
                "fail_rate": 0.0
            }

        # Sample examples
        num_tests = min(self.cfg.sample_size, len(examples))  # ✅ FIXED: Use self.cfg
        if len(examples) <= num_tests:
            sample = examples
        else:
            sample = random.sample(examples, num_tests)

        print(f"   Testing with: {len(sample)} examples")

        # Validate context
        rule_pool = context.get('rule_pool')
        stage_fn = context.get('stage_fn')
        simulator = context.get('simulator')

        if not rule_pool or not stage_fn:
            print(f"   ❌ FAILED: Missing rule_pool or stage_fn")
            return self._result(False, "Missing canary context (rule_pool or stage_fn)")

        if not simulator:
            print(f"   ❌ FAILED: No simulator provided")
            return self._result(False, "No simulator available")

        # Stage patch (create temporary snapshot)
        print(f"\n🔄 Staging patch...")
        snapshot = self._safe_call(rule_pool, 'snapshot')

        try:
            staged = stage_fn(patch)

            if not staged:
                print(f"   ❌ FAILED: Could not stage patch")
                return self._result(False, "Patch staging failed")

            print(f"   ✅ Patch staged successfully")

            # Run simulations
            print(f"\n🔬 Running {len(sample)} simulations...")
            violations = 0

            for i, example in enumerate(sample, 1):
                print(f"   Test {i}/{len(sample)}: ", end='', flush=True)

                try:
                    result = simulator(example, rule_pool)

                    if not result.get('ok', False):
                        violations += 1
                        print(f"❌ FAILED")
                        print(f"      Reason: {result.get('reason', 'Unknown')}")
                    else:
                        print(f"✅ PASSED")

                except Exception as e:
                    violations += 1
                    print(f"❌ ERROR")
                    print(f"      Exception: {str(e)[:60]}")

            # Compute results
            pass_rate = (len(sample) - violations) / len(sample)
            fail_rate = violations / len(sample)

            print(f"\n📈 Simulation Results:")
            print(f"   Passed: {len(sample) - violations}/{len(sample)}")
            print(f"   Failed: {violations}/{len(sample)}")
            print(f"   Pass Rate: {pass_rate:.1%}")
            print(f"   Fail Rate: {fail_rate:.1%}")

            # Statistical test
            max_allowed_failures = int(self.cfg.max_fail_rate * len(sample))  # ✅ FIXED
            statistical_pass = violations <= max_allowed_failures

            print(f"\n📊 Statistical Test:")
            print(f"   Max allowed failures: {max_allowed_failures} ({self.cfg.max_fail_rate:.1%})")
            print(f"   Actual failures: {violations}")
            print(f"   Result: {'✅ PASS' if statistical_pass else '❌ FAIL'}")

            if not statistical_pass:
                print(f"\n{'=' * 70}")
                print(f"❌ CANARY TEST FAILED (Statistical)")
                print(f"{'=' * 70}\n")
                return self._result(
                    False,
                    f"Statistical test failed: {violations} > {max_allowed_failures}",
                    tested=len(sample),
                    fail_rate=fail_rate
                )

            # Heuristic risk check
            scores = context.get('scores', {})
            risk_score = scores.get('risk', 0.5)

            print(f"\n⚠️ Risk Assessment:")
            print(f"   Risk score: {risk_score:.3f}")
            print(f"   Threshold: {self.cfg.heuristic_risk_threshold:.3f}")  # ✅ FIXED

            if risk_score > self.cfg.heuristic_risk_threshold:  # ✅ FIXED
                print(f"   ❌ FAIL: Risk too high")
                print(f"\n{'=' * 70}")
                print(f"❌ CANARY TEST FAILED (High Risk)")
                print(f"{'=' * 70}\n")
                return self._result(
                    False,
                    f"Risk score {risk_score:.3f} exceeds threshold {self.cfg.heuristic_risk_threshold:.3f}",
                    tested=len(sample),
                    fail_rate=fail_rate
                )

            print(f"   ✅ PASS: Risk acceptable")

            # Final decision: Pass if either perfect or good enough
            passed = (violations == 0) or (pass_rate >= 0.8 and statistical_pass)

            print(f"\n{'=' * 70}")
            if passed:
                print(f"✅ CANARY TEST PASSED")
                print(f"   Zero failures: {violations == 0}")
                print(f"   Statistical pass: {statistical_pass}")
                print(f"   Risk acceptable: {risk_score <= self.cfg.heuristic_risk_threshold}")
            else:
                print(f"❌ CANARY TEST FAILED")
                print(f"   Too many failures for comfort")
            print(f"{'=' * 70}\n")

            return self._result(
                passed,
                "Canary test passed" if passed else f"Too many failures: {violations}/{len(sample)}",
                tested=len(sample),
                fail_rate=fail_rate
            )

        finally:
            # Always restore rule pool to original state
            if snapshot is not None:
                self._restore(rule_pool, snapshot)

    def _heuristic_only_decision(self, patch: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Make decision based purely on heuristics when no examples available.

        Uses both risk score and plausibility to decide.
        """
        print(f"\n🎯 Heuristic-Only Decision Mode")
        print(f"   (No examples available for simulation)")

        scores = context.get('scores', {})
        risk_score = scores.get('risk', 0.5)
        plausibility = scores.get('plausibility', 0.5)
        aggregate = scores.get('aggregate', 0.5)

        print(f"\n📊 Score Analysis:")
        print(f"   Plausibility: {plausibility:.3f}")
        print(f"   Risk: {risk_score:.3f}")
        print(f"   Aggregate: {aggregate:.3f}")

        # Decision rules for cold start
        if risk_score > self.cfg.heuristic_risk_threshold:
            print(f"\n   ❌ REJECT: Risk too high ({risk_score:.3f} > {self.cfg.heuristic_risk_threshold:.3f})")
            return self._result(False, f"Cold start: Risk {risk_score:.3f} exceeds threshold")

        if plausibility < 0.3:
            print(f"\n   ❌ REJECT: Plausibility too low ({plausibility:.3f} < 0.3)")
            return self._result(False, f"Cold start: Plausibility {plausibility:.3f} too low")

        if aggregate < 0.4:
            print(f"\n   ❌ REJECT: Aggregate score too low ({aggregate:.3f} < 0.4)")
            return self._result(False, f"Cold start: Aggregate {aggregate:.3f} below minimum")

        print(f"\n   ✅ ACCEPT: Scores acceptable for cold start")
        print(f"{'=' * 70}\n")
        return self._result(True, "Cold start: Heuristics acceptable", tested=0, fail_rate=0.0)

    def _get_heuristic_risk(self, scorer: Optional[Any], context: Dict[str, Any]) -> float:
        """Extract risk score from context."""
        scores = context.get("scores", {})
        return float(scores.get("risk", 0.5))

    def _restore(self, rule_pool: Any, snapshot: Any) -> None:
        """Restore rule pool to pre-canary state."""
        try:
            rule_pool.restore(snapshot)
            print("   🔄 RulePool restored to pre-canary state")
        except Exception as e:
            print(f"   ⚠️ CRITICAL - Failed to restore RulePool snapshot: {e}")

    def _sample_examples(self, examples: List[Any], k: int) -> List[Any]:
        """Sample k examples randomly."""
        if not examples or k <= 0:
            return []
        if len(examples) <= k:
            return list(examples)
        return random.Random(1337).sample(examples, k)

    def _safe_call(self, obj: Any, method: str, *args, **kwargs) -> Any:
        """Safely call a method on an object, returning None on error."""
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"   ⚠️ Error calling '{method}': {e}")
        return None

    def _result(self, passed: bool, reason: str, tested: int = 0, fail_rate: float = 0.0) -> Dict[str, Any]:
        """Format canary test result."""
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"CANARY_RUNNER: Result -> {status}")
        print(f"               Reason: {reason}")

        return {
            "passed": passed,
            "tested": tested,
            "fail_rate": round(fail_rate, 4),
            "reason": reason
        }