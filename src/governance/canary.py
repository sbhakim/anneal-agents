# src/governance/canary.py
"""
Implements the Canary Deployment stage of the governance pipeline.
Corresponds to Section 9.5 of the paper.

UPDATED: The fallback heuristic is now more intelligent. When no statistical
simulation is possible (due to a lack of examples), it now uses a two-gate
check on both the patch's risk and its plausibility score, ensuring that
only high-confidence or very low-risk patches can pass without simulation.
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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.cfg = CanaryConfig(
            sample_size=int(cfg.get("canary_sample_size", 8)),
            max_fail_rate=float(cfg.get("max_fail_rate", 0.10)),
            min_tests_for_stats=int(cfg.get("min_tests", 5)),
            heuristic_risk_threshold=float(cfg.get("tau_impact", 0.6))
        )
        print("CANARY_RUNNER: Initialized")
        print(f"  - Sample size: {self.cfg.sample_size}")
        print(f"  - Max fail rate (statistical): {self.cfg.max_fail_rate}")
        print(f"  - Heuristic risk threshold: {self.cfg.heuristic_risk_threshold}")

    def run(self, patch: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a canary test for the given patch."""
        rule_pool = context.get("rule_pool")
        stage_fn = context.get("stage_fn")
        simulator = context.get("simulator")
        examples = list(context.get("examples", []))
        scorer = context.get("scorer")

        if not all([rule_pool, stage_fn, simulator]):
            return self._result(passed=False, reason="Missing rule_pool, stage_fn, or simulator.")

        snapshot = self._safe_call(rule_pool, "snapshot")
        if snapshot is None:
            return self._result(passed=False, reason="RulePool does not support snapshot().")

        try:
            if not stage_fn(patch):
                return self._result(passed=False, reason="stage_fn returned False")

            tested, fails = 0, 0
            if examples:
                sample = self._sample_examples(examples, self.cfg.sample_size)
                for ex in sample:
                    try:
                        out = simulator(ex, rule_pool) or {}
                        tested += 1
                        if not out.get("ok", True) or out.get("violations", 0) > 0:
                            fails += 1
                    except Exception as e:
                        tested += 1
                        fails += 1

            if tested >= self.cfg.min_tests_for_stats:
                fail_rate = fails / tested
                passed = fail_rate <= self.cfg.max_fail_rate
                reason = f"Statistical: fail_rate {fail_rate:.2f} {'<=' if passed else '>'} {self.cfg.max_fail_rate}"
                return self._result(passed=passed, tested=tested, fail_rate=fail_rate, reason=reason)
            else:
                # **FIX**: Tighten heuristic with a two-gate check on risk and plausibility.
                risk = self._get_heuristic_risk(scorer, context)
                plausibility = context.get("scores", {}).get("plausibility", 0.0)

                # Pass if: (risk is medium AND plausibility is high) OR risk is very low.
                if (risk < 0.45 and plausibility >= 0.7) or risk < 0.3:
                    passed = True
                    reason = f"Heuristic PASS: risk={risk:.2f} and/or plaus={plausibility:.2f} meet criteria"
                else:
                    passed = False
                    reason = f"Heuristic FAIL: risk={risk:.2f} or plaus={plausibility:.2f} insufficient"

                return self._result(passed=passed, tested=tested, fail_rate=risk, reason=reason)

        except Exception as e:
            return self._result(passed=False, reason=f"Exception: {e}")
        finally:
            self._restore(rule_pool, snapshot)

    def _get_heuristic_risk(self, scorer: Optional[Any], context: Dict[str, Any]) -> float:
        scores = context.get("scores", {})
        return float(scores.get("risk", 0.5))

    def _restore(self, rule_pool: Any, snapshot: Any) -> None:
        try:
            rule_pool.restore(snapshot)
            print("CANARY_RUNNER: RulePool restored to pre-canary state.")
        except Exception as e:
            print(f"CANARY_RUNNER: CRITICAL - Failed to restore RulePool snapshot: {e}")

    def _sample_examples(self, examples: List[Any], k: int) -> List[Any]:
        if not examples or k <= 0: return []
        if len(examples) <= k: return list(examples)
        return random.Random(1337).sample(examples, k)

    def _safe_call(self, obj: Any, method: str, *args, **kwargs) -> Any:
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"CANARY_RUNNER: Error calling '{method}': {e}")
        return None

    def _result(self, passed: bool, reason: str, tested: int = 0, fail_rate: float = 0.0) -> Dict[str, Any]:
        print(f"CANARY_RUNNER: Result -> {'PASS' if passed else 'FAIL'}. Reason: {reason}")
        return {"passed": passed, "tested": tested, "fail_rate": round(fail_rate, 4), "reason": reason}