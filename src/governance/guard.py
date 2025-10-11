# src/governance/guard.py

"""
Guard implements the manuscript’s lexicographic gate: Score → Guard (value, causal) → Stage/Canary → Commit/Rollback.
The value guard can hard-veto high-impact/risk patches; the causal guard escalates to human when ambiguity is high.
It accepts a patch and context (optional ValueKG/CausalKG, scores, trace), with thresholds configurable; when KGs are absent it falls back to simple heuristics.
Returns a minimal decision dict: {'allow' | 'veto' | 'request_human'} plus brief reasons—lightweight PoC, no external deps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class GuardConfig:
    enable_value_guard: bool = True
    enable_causal_guard: bool = True
    tau_impact: float = 0.6  # veto if value impact >= tau_impact
    tau_ambiguity: float = 0.5  # escalate to human if ambiguity >= tau_ambiguity


class Guard:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.cfg = GuardConfig(
            enable_value_guard=cfg.get("value_guard", True),
            enable_causal_guard=cfg.get("causal_guard", True),
            # Use tau_impact from gates section for consistency
            tau_impact=cfg.get("gates", {}).get("tau_impact", 0.6),
            tau_ambiguity=cfg.get("ambiguity_threshold", 0.5),
        )
        # Log-style prints keep parity with your current run output
        print("GUARDRAILS: Initialized")
        print(f"  - Value guard: {'ENABLED' if self.cfg.enable_value_guard else 'DISABLED'}")
        print(f"  - Causal guard: {'ENABLED' if self.cfg.enable_causal_guard else 'DISABLED'}")
        print(f"  - Impact threshold (tau_impact): {self.cfg.tau_impact}")
        print(f"  - Ambiguity threshold (tau_ambiguity): {self.cfg.tau_ambiguity}")

    # ------------------------------- Public API ------------------------------- #

    def check(self, patch: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Lexicographic guard:
          1) Value guard (safety/policy/ethics) -> 'veto' if high impact/risk
          2) Causal guard (ambiguity/impact of root cause) -> 'request_human' if high
          3) Otherwise -> 'allow'
        """
        context = context or {}

        v_decision, v_payload = self._check_value_guardrail(patch, context)
        if self.cfg.enable_value_guard and v_decision == "veto":
            return {
                "decision": "veto",
                "reason": v_payload.get("explanation", "Value guard veto"),
                "value": v_payload,
                "causal": {"ambiguity": 0.0, "impact": 0.0, "explanation": "Skipped due to value veto"},
            }

        c_decision, c_payload = self._check_causal_guardrail(patch, context)
        if self.cfg.enable_causal_guard and c_decision == "request_human":
            return {
                "decision": "request_human",
                "reason": c_payload.get("explanation", "Causal guard requested human review"),
                "value": v_payload,
                "causal": c_payload,
            }

        return {
            "decision": "allow",
            "reason": "Guards passed",
            "value": v_payload,
            "causal": c_payload,
        }

    # ------------------------------ Guard Stages ------------------------------ #

    def _check_value_guardrail(self, patch: Any, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Estimate the *impact/risk* of committing this patch under value/policy constraints.
        If `value_kg` is present, delegate to it; otherwise use heuristics from scoring.
        """
        if not self.cfg.enable_value_guard:
            return "allow", {"impact": 0.0, "explanation": "Value guard disabled"}

        # Try ValueKG if available
        value_kg = ctx.get("value_kg")
        impact = None
        explanation = ""

        if value_kg and hasattr(value_kg, "assess_patch"):
            try:
                res = value_kg.assess_patch(patch) or {}
                impact = _clamp01(res.get("impact"))
                explanation = res.get("explanation", "ValueKG assessment")
            except Exception as e:
                explanation = f"ValueKG error: {e}. Falling back to heuristics."

        # Heuristic fallback: use scorer risk (q_val/scope) if provided
        if impact is None:
            scores = ctx.get("scores", {})
            action = _get(patch, "action")
            details = _get(patch, "details", "")

            # UPDATED: More nuanced heuristic for risk
            # Check if a REFINE_EFFECT patch is adding a safety guard
            is_safety_refinement = (action == "REFINE_EFFECT" and "IfThen" in details)

            if is_safety_refinement:
                # Assign a lower base risk for safety-enhancing patches
                impact = _clamp01(scores.get("risk", 0.25))
                explanation = "Heuristic impact (lowered for safety refinement)"
            else:
                # Use the original, more conservative proxy for other cases
                impact = _clamp01(scores.get("risk", _risk_proxy_from_action(action)))
                if not explanation:
                    explanation = "Heuristic impact from scores/risk proxy"

        # Decision
        if impact >= self.cfg.tau_impact:
            return "veto", {"impact": impact, "explanation": explanation}
        return "allow", {"impact": impact, "explanation": explanation}

    def _check_causal_guardrail(self, patch: Any, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Estimate *ambiguity* of the causal story and potential *impact* if wrong.
        If `causal_kg` is present, delegate; else use simple ambiguity heuristics from trace.
        """
        if not self.cfg.enable_causal_guard:
            return "allow", {"ambiguity": 0.0, "impact": 0.0, "explanation": "Causal guard disabled"}

        causal_kg = ctx.get("causal_kg")
        trace = ctx.get("trace", {})

        ambiguity = None
        impact = None
        explanation = ""

        if causal_kg and hasattr(causal_kg, "assess_trace"):
            try:
                res = causal_kg.assess_trace(trace) or {}
                ambiguity = _clamp01(res.get("ambiguity"))
                impact = _clamp01(res.get("impact"))
                explanation = res.get("explanation", "CausalKG assessment")
            except Exception as e:
                explanation = f"CausalKG error: {e}. Falling back to heuristics."

        # Heuristic fallback when KG absent/unavailable:
        if ambiguity is None:
            candidates = _get(trace, "root_cause_candidates") or []
            ambiguity = _clamp01(min(1.0, 0.2 * max(0, len(candidates) - 1)))
            if not explanation:
                explanation = "Heuristic ambiguity from candidate count"
        if impact is None:
            scores = ctx.get("scores", {})
            impact = _clamp01(scores.get("risk", 0.3))

        if ambiguity >= self.cfg.tau_ambiguity:
            return "request_human", {
                "ambiguity": ambiguity,
                "impact": impact,
                "explanation": explanation,
            }
        return "allow", {"ambiguity": ambiguity, "impact": impact, "explanation": explanation}


# ------------------------------- Helper utils -------------------------------- #

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Safe dict/obj attribute access."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp01(x: Optional[float]) -> float:
    """Clamp to [0,1] with sensible defaults."""
    try:
        return max(0.0, min(1.0, float(x)))
    except (ValueError, TypeError):
        return 0.0


def _risk_proxy_from_action(action: Optional[str]) -> float:
    """
    Very small heuristic to avoid requiring scorer in PoC:
    - Adding/weakening effects is riskier than adding preconditions.
    - Unknown actions default to medium risk (0.3).
    """
    if not action:
        return 0.3
    a = action.upper()
    if "REFINE_EFFECT" in a or "UPDATE_EFFECT" in a or "ADD_EFFECT" in a:
        return 0.6
    if "UPDATE_TOOL" in a or "UPDATE_SCHEMA" in a:
        return 0.5
    if "ADD_PRECONDITION" in a:
        return 0.2
    return 0.3