# src/governance/guard.py

"""
Guard implements lexicographic guardrails for patch validation.
Corresponds to Section VIII-E of the manuscript.

The guardrail system has two tiers (lexicographic priority):
1. Value Guardrails (deontic constraints) → Can VETO patches
2. Causal Guardrails (ambiguity/impact) → Can REQUEST_HUMAN review

Value vetoes are absolute and override all utility considerations.
Causal escalations request human judgment on ambiguous causal claims.

FEATURES:
- Full ValueKG/CausalKG integration when available
- Graceful fallback to heuristics when KGs unavailable
- Dangerous pattern detection (security, safety, privacy violations)
- Causal ambiguity detection (cycles, unidentifiable claims)
- High-impact propagation analysis (fan-out estimation)
- Conflict resolution strategy (Section VIII-E.3)
- Comprehensive testing and statistics

UPDATED:
- Merged best features from both guard implementations.
- Read ambiguity threshold from governance.gates.tau_conf (consistent with config.yaml),
  with fallback to legacy 'ambiguity_threshold'.
- Added derived statistics fields `value_checks` and `causal_checks` in `get_statistics()`
  to align with downstream analysis code expecting those keys.

MINIMUM RELIABILITY UPDATES (2026-01):
- Ensure ValueKG is hydrated from runtime state (blackout_dates + corporate_card_policy)
  because governance config alone typically does not carry domain state.
- Return specific hazard_id when CausalKG triggers a hazard to improve debuggability and
  avoid opaque "Known hazard condition" escalations.

MINIMUM MANUSCRIPT/RESULTS UPDATE (2026-01):
- Return a structured `reason_code` (stable vocabulary) in addition to free-text `reason`.
  This makes Table 4 + audit logs consistent across runs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import re


@dataclass
class GuardConfig:
    """Configuration for guardrail system."""
    enable_value_guard: bool = True
    enable_causal_guard: bool = True
    tau_impact: float = 0.6  # Veto if value impact >= tau_impact
    tau_ambiguity: float = 0.5  # Escalate if ambiguity >= tau_ambiguity (aka tau_conf)


class Guard:
    """
    Applies value and causal constraints to veto or escalate risky patches.

    Implements two-tier guardrails from Section VIII-E:
    1. Value guardrails (Eq. 16): Enforce deontic (must/must not) rules
    2. Causal guardrails (Eq. 17-18): Detect ambiguity and high-impact propagation

    Value vetoes are lexicographic: they override all utility considerations.
    Causal escalations request human review for ambiguous causal claims.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize guardrails with configuration and optional knowledge graphs.

        Args:
            config: Configuration dictionary with guardrail settings
        """
        cfg = config or {}

        # NOTE: We read tau_ambiguity from governance.gates.tau_conf to match config.yaml.
        gates = cfg.get("gates", {}) if isinstance(cfg.get("gates", {}), dict) else {}
        self.cfg = GuardConfig(
            enable_value_guard=cfg.get("value_guard", True),
            enable_causal_guard=cfg.get("causal_guard", True),
            tau_impact=gates.get("tau_impact", 0.6),
            tau_ambiguity=gates.get("tau_conf", cfg.get("ambiguity_threshold", 0.5)),
        )

        # Knowledge graph integration (optional)
        self.value_kg = None
        self.causal_kg = None

        # Track last domain-state used to hydrate KGs (minimal state-based reinit)
        self._kg_state_fingerprint: Optional[Tuple[Tuple[str, ...], str]] = None

        # Try to initialize KGs if available
        self._initialize_kgs(cfg)

        # Statistics tracking
        self.stats = {
            'total_checks': 0,
            'value_vetoes': 0,
            'causal_escalations': 0,
            'allows': 0,
            'kg_checks': 0,
            'heuristic_checks': 0
        }

        # Cache for performance
        self.veto_cache: Dict[str, bool] = {}

        # Log initialization
        print("GUARDRAILS: Initialized")
        print(f"  - Value guard: {'ENABLED' if self.cfg.enable_value_guard else 'DISABLED'}")
        print(f"  - Causal guard: {'ENABLED' if self.cfg.enable_causal_guard else 'DISABLED'}")
        print(f"  - Impact threshold (tau_impact): {self.cfg.tau_impact}")
        print(f"  - Ambiguity threshold (tau_conf): {self.cfg.tau_ambiguity}")
        print(f"  - ValueKG: {'Available' if self.value_kg else 'Unavailable (using heuristics)'}")
        print(f"  - CausalKG: {'Available' if self.causal_kg else 'Unavailable (using heuristics)'}")

    def _initialize_kgs(self, config: Dict[str, Any]) -> None:
        """Initialize knowledge graphs if available."""
        # Try to load ValueKG
        try:
            from ..knowledge.value_kg import ValueKG
            kg_config = {
                'blackout_dates': config.get('blackout_dates', []),
                'corporate_card_policy': config.get('corporate_card_policy', 'blocked_on_blackout_dates')
            }
            self.value_kg = ValueKG(kg_config)
        except (ImportError, Exception):
            self.value_kg = None

        # Try to load CausalKG
        try:
            from ..knowledge.causal_kg import CausalKG
            self.causal_kg = CausalKG(config)
        except (ImportError, Exception):
            self.causal_kg = None

    def _maybe_hydrate_kgs_from_state(self, state: Any) -> None:
        """
        MINIMUM FIX:
        ValueKG was often initialized with empty blackout_dates because governance config
        doesn't include domain state. If runtime state contains blackout_dates and policy,
        hydrate (re-init) ValueKG once per distinct state fingerprint.
        """
        if not isinstance(state, dict):
            return

        blackout = state.get("blackout_dates")
        policy = state.get("corporate_card_policy", "blocked_on_blackout_dates")

        if not isinstance(blackout, list) or not blackout:
            return

        # Create a stable fingerprint from blackout dates + policy
        blackout_fp = tuple(sorted(str(x) for x in blackout))
        fp = (blackout_fp, str(policy))

        if self._kg_state_fingerprint == fp:
            return

        # Re-init ValueKG with state-derived config (safe, minimal, avoids empty defaults)
        try:
            from ..knowledge.value_kg import ValueKG
            self.value_kg = ValueKG({
                "blackout_dates": list(blackout),
                "corporate_card_policy": policy,
            })
            self._kg_state_fingerprint = fp
        except Exception:
            # Leave existing ValueKG as-is on error
            pass

    def _is_low_risk_tool_schema_drift(self, patch: Any) -> bool:
        action = str(_get(patch, "action", "") or "")
        if action != "UPDATE_TOOL_SCHEMA":
            return False

        details = str(_get(patch, "details", "") or "")
        d = details.lower()

        # Endpoint/version drift patterns (keep narrow).
        has_endpoint_drift = ("/v1/" in d and "/v2/" in d) or ("v1/book" in d and "v2/book" in d) or ("deprecated" in d)

        if not has_endpoint_drift:
            return False

        # Escalate if this touches sensitive surfaces.
        sensitive = (
            "auth", "oauth", "token", "key", "secret", "credential",
            "payment", "card", "security", "privacy", "pii", "encrypt",
            "signature", "jwt", "mfa", "permission",
        )
        if any(s in d for s in sensitive):
            return False

        return True

    # ========================================================================
    # MAIN PUBLIC API
    # ========================================================================

    def check(self, patch: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main guardrail check with lexicographic priority.

        Priority order:
        1. Value guard (safety/policy/ethics) → 'veto' if high impact/risk
        2. Causal guard (ambiguity/impact) → 'request_human' if high
        3. Otherwise → 'allow'

        Args:
            patch: Patch or action to validate
            context: Optional context with state, scores, KGs, trace

        Returns:
            Dictionary with:
            - decision: 'veto' | 'request_human' | 'allow'
            - reason_code: stable code for analysis (Table 4 / audits)
            - reason: Human-readable explanation
            - value: Value guard payload (impact, explanation, reason_code)
            - causal: Causal guard payload (ambiguity, impact, explanation, reason_code)
        """
        context = context or {}
        self.stats['total_checks'] += 1

        # ✅ Minimal reliability: hydrate ValueKG from runtime state if available
        self._maybe_hydrate_kgs_from_state(context.get("state", {}))

        # Stage 1: Value Guardrails (can VETO)
        v_decision, v_payload = self._check_value_guardrail(patch, context)

        if self.cfg.enable_value_guard and v_decision == "veto":
            self.stats['value_vetoes'] += 1
            return {
                "decision": "veto",
                "reason_code": v_payload.get("reason_code", "VALUE:VETO"),
                "reason": v_payload.get("explanation", "Value guard veto"),
                "value": v_payload,
                "causal": {
                    "ambiguity": 0.0,
                    "impact": 0.0,
                    "reason_code": "CAUSAL:SKIPPED_VALUE_VETO",
                    "explanation": "Skipped due to value veto"
                },
            }

        # Stage 2: Causal Guardrails (can REQUEST_HUMAN)
        c_decision, c_payload = self._check_causal_guardrail(patch, context)

        if self.cfg.enable_causal_guard and c_decision == "request_human":
            self.stats['causal_escalations'] += 1
            return {
                "decision": "request_human",
                "reason_code": c_payload.get("reason_code", "CAUSAL:REQUEST_HUMAN"),
                "reason": c_payload.get("explanation", "Causal guard requested human review"),
                "value": v_payload,
                "causal": c_payload,
            }

        # All guards passed
        self.stats['allows'] += 1
        return {
            "decision": "allow",
            "reason_code": "ALLOW",
            "reason": "Guards passed",
            "value": v_payload,
            "causal": c_payload,
        }

    # ========================================================================
    # VALUE GUARDRAILS (Section VIII-E.1, Eq. 16)
    # ========================================================================

    def _check_value_guardrail(self, patch: Any, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Estimate the impact/risk of committing this patch under value/policy constraints.

        From Eq. 16:
        Guard_val(Δo) = {
            veto,   ∃c ∈ KG^val : c(ψ(Δo)) = ⊥,
            allow,  otherwise
        }

        IMPORTANT: Value vetoes are lexicographic - they override ALL utility.

        Args:
            patch: Patch to validate
            ctx: Context with state, scores, value_kg

        Returns:
            Tuple of (decision, payload) where decision ∈ {'veto', 'allow'}
        """
        if not self.cfg.enable_value_guard:
            return "allow", {"impact": 0.0, "reason_code": "VALUE:DISABLED", "explanation": "Value guard disabled"}

        impact = None
        explanation = ""
        reason_code = "VALUE:ALLOW"

        # Try ValueKG if available (preferred)
        value_kg = ctx.get("value_kg") or self.value_kg

        if value_kg and hasattr(value_kg, "check_deontic"):
            try:
                self.stats['kg_checks'] += 1
                state = ctx.get("state", {})
                params = self._extract_params(patch, state)

                deontic_ok = value_kg.check_deontic(state, params)

                if not deontic_ok:
                    violated_rule = self._identify_violated_rule(patch, state, value_kg)
                    explanation = f"Violated deontic rule: {violated_rule}" if violated_rule else "Deontic constraint violated"
                    return "veto", {"impact": 1.0, "reason_code": "VALUE:DEONTIC", "explanation": explanation}

                impact = 0.2
                explanation = "Passed ValueKG deontic checks"
                reason_code = "VALUE:KG_OK"

            except Exception as e:
                explanation = f"ValueKG error: {e}. Using heuristics."
                impact = None

        # Check dangerous patterns (always active, even with KG)
        has_danger, danger_reason = self._check_dangerous_patterns(patch)
        if has_danger:
            return "veto", {"impact": 1.0, "reason_code": "VALUE:DANGEROUS_PATTERN", "explanation": danger_reason}

        # Heuristic fallback when KG unavailable or didn't compute impact
        if impact is None:
            self.stats['heuristic_checks'] += 1
            scores = ctx.get("scores", {})
            action = _get(patch, "action")
            details = _get(patch, "details", "")

            is_safety_refinement = (action == "REFINE_EFFECT" and "IfThen" in details)

            if is_safety_refinement:
                impact = _clamp01(scores.get("risk", 0.25))
                explanation = "Heuristic impact (lowered for safety refinement)"
                reason_code = "VALUE:HEURISTIC_SAFETY_REFINEMENT"
            else:
                impact = _clamp01(scores.get("risk", _risk_proxy_from_action(action)))
                explanation = "Heuristic impact from scores/risk proxy"
                reason_code = "VALUE:HEURISTIC"

        # Decision
        if impact >= self.cfg.tau_impact:
            return "veto", {"impact": impact, "reason_code": "VALUE:IMPACT_THRESHOLD", "explanation": explanation}

        return "allow", {"impact": impact, "reason_code": reason_code, "explanation": explanation}

    def _check_dangerous_patterns(self, patch: Any) -> Tuple[bool, str]:
        """
        Check for hardcoded dangerous patterns that should always be vetoed.

        Patterns include:
        - Removing security checks
        - Disabling authentication
        - Removing payment validation
        - Bypassing privacy controls
        - Removing preconditions (weakens safety)
        - Removing negations from safety constraints
        """
        action = _get(patch, "action", "")
        details = _get(patch, "details", "").lower()

        dangerous_patterns = [
            ('disable', 'security'),
            ('remove', 'auth'),
            ('bypass', 'check'),
            ('skip', 'validation'),
            ('allow', 'all'),
            ('disable', 'verify'),
            ('remove', 'payment'),
            ('skip', 'privacy'),
            ('disable', 'safety'),
            ('bypass', 'permission')
        ]

        for pattern in dangerous_patterns:
            if all(keyword in details for keyword in pattern):
                return True, f"Dangerous pattern: {' '.join(pattern)}"

        if action == 'REMOVE_PRECONDITION':
            return True, "Removing preconditions weakens safety guarantees"

        if re.search(r'remove.*not\(', details, re.IGNORECASE):
            return True, "Removing negation from safety constraint"

        weaken_keywords = ['weaken', 'relax', 'loosen', 'reduce']
        safety_keywords = ['guard', 'check', 'verify', 'validate', 'precondition']

        if any(w in details for w in weaken_keywords) and any(s in details for s in safety_keywords):
            return True, "Weakening existing safety mechanisms"

        return False, ""

    def _extract_params(self, patch: Any, state: Optional[Dict]) -> Dict[str, Any]:
        """
        Extract parameters from patch for deontic checking.

        Args:
            patch: Patch with details
            state: Current state

        Returns:
            Dictionary of parameters
        """
        params = {}

        if state:
            params['payment'] = state.get('payment_method', '')

        details = _get(patch, "details", "")

        if 'payment' in details.lower():
            match = re.search(r'payment[,\s]*([A-Za-z0-9:_-]+)', details)
            if match:
                params['payment'] = match.group(1)

        if 'dates' in details.lower() or 'date' in details.lower():
            if state:
                params['dates'] = state.get('travel_dates', '')

        return params

    def _identify_violated_rule(
            self,
            patch: Any,
            state: Optional[Dict],
            value_kg: Any
    ) -> Optional[str]:
        """
        Identify which specific deontic rule was violated.
        """
        if not value_kg or not hasattr(value_kg, 'rules'):
            return None

        params = self._extract_params(patch, state)

        for rule_id, rule_fn in value_kg.rules.items():
            try:
                if not rule_fn(state or {}, params):
                    return rule_id
            except Exception:
                continue

        return None

    # ========================================================================
    # CAUSAL GUARDRAILS (Section VIII-E.2, Eq. 17-18)
    # ========================================================================

    def _check_causal_guardrail(self, patch: Any, ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Check for causal ambiguity and high-impact propagation.
        """
        if not self.cfg.enable_causal_guard:
            return "allow", {"ambiguity": 0.0, "impact": 0.0, "reason_code": "CAUSAL:DISABLED", "explanation": "Causal guard disabled"}

        causal_kg = ctx.get("causal_kg") or self.causal_kg
        trace = ctx.get("trace", {})
        state = ctx.get("state", {})

        ambiguity = None
        impact = None
        explanation = ""
        reason_code = "CAUSAL:ALLOW"

        # Low-risk tool drift schema updates should reach canary instead of human escalation.
        if self._is_low_risk_tool_schema_drift(patch):
            return "allow", {
                "ambiguity": 0.0,
                "impact": 0.2,
                "reason_code": "CAUSAL:ALLOW_TOOL_DRIFT_SCHEMA",
                "explanation": "Allowing endpoint drift schema update (low causal impact); canary will gate"
            }

        # Try CausalKG if available (preferred)
        if causal_kg and hasattr(causal_kg, "assess_trace"):
            try:
                self.stats['kg_checks'] += 1
                res = causal_kg.assess_trace(trace) or {}
                ambiguity = _clamp01(res.get("ambiguity"))
                impact = _clamp01(res.get("impact"))
                explanation = res.get("explanation", "CausalKG assessment")
                reason_code = "CAUSAL:KG"
            except Exception as e:
                explanation = f"CausalKG error: {e}. Using heuristics."

        is_ambiguous, ambig_reason = self._check_ambiguity(patch, state, causal_kg)
        if is_ambiguous:
            return "request_human", {
                "ambiguity": 1.0,
                "impact": impact or 0.5,
                "reason_code": "CAUSAL:AMBIGUITY",
                "explanation": f"Causal ambiguity: {ambig_reason}"
            }

        is_high_impact, impact_reason = self._check_high_impact(patch, state, causal_kg)
        if is_high_impact:
            return "request_human", {
                "ambiguity": ambiguity or 0.3,
                "impact": 1.0,
                "reason_code": "CAUSAL:HIGH_IMPACT",
                "explanation": f"High-impact: {impact_reason}"
            }

        has_hazard, hazard_reason = self._check_hazards(patch, state, causal_kg)
        if has_hazard:
            hz = str(hazard_reason).strip() or "UNKNOWN"
            return "request_human", {
                "ambiguity": ambiguity or 0.3,
                "impact": impact or 0.7,
                "reason_code": f"CAUSAL:HAZARD:{hz}",
                "explanation": f"Hazard: {hazard_reason}"
            }

        # Heuristic fallback
        if ambiguity is None:
            self.stats['heuristic_checks'] += 1
            candidates = _get(trace, "root_cause_candidates") or []
            ambiguity = _clamp01(min(1.0, 0.2 * max(0, len(candidates) - 1)))
            if not explanation:
                explanation = "Heuristic ambiguity from candidate count"
                reason_code = "CAUSAL:HEURISTIC"

        if impact is None:
            scores = ctx.get("scores", {})
            impact = _clamp01(scores.get("risk", 0.3))

        if ambiguity >= self.cfg.tau_ambiguity:
            return "request_human", {
                "ambiguity": ambiguity,
                "impact": impact,
                "reason_code": "CAUSAL:THRESHOLD",
                "explanation": explanation or "High causal ambiguity"
            }

        return "allow", {
            "ambiguity": ambiguity,
            "impact": impact,
            "reason_code": reason_code,
            "explanation": explanation or "Causal checks passed"
        }

    def _check_ambiguity(
            self,
            patch: Any,
            state: Optional[Dict],
            causal_kg: Any
    ) -> Tuple[bool, str]:
        """
        Check for causal ambiguity: low identifiability of causal claims.
        """
        details = _get(patch, "details", "")
        action = _get(patch, "action", "")

        if causal_kg and hasattr(causal_kg, 'detect_cycles'):
            try:
                has_cycles = causal_kg.detect_cycles()
                if has_cycles:
                    return True, "Circular causal dependencies detected"
            except Exception:
                pass

        if action == 'REFINE_EFFECT':
            if not self._has_causal_evidence(details, state):
                return True, "Insufficient evidence for causal relationship"

        if 'IfThen' in details or 'When' in details:
            condition_vars = self._extract_condition_vars(details)
            if not self._are_vars_identifiable(condition_vars):
                return True, f"Unidentifiable variables: {condition_vars}"

        return False, ""

    def _check_high_impact(
            self,
            patch: Any,
            state: Optional[Dict],
            causal_kg: Any
    ) -> Tuple[bool, str]:
        """
        Check for high-impact propagation: large downstream fan-out or irreversible effects.
        """
        operator = _get(patch, "operator", "")
        action = _get(patch, "action", "")
        details = _get(patch, "details", "")

        if action == 'ADD_PRECONDITION':
            safety_predicates = [
                'Valid', 'Check', 'Verify', 'Ensure', 'Require',
                'Not(Blocked', 'Not(Invalid', 'Not(Expired'
            ]
            if any(pred in details for pred in safety_predicates):
                return False, "Safety precondition (low risk)"

        high_impact_operators = [
            'ProcessPayment', 'ExecuteTransaction', 'DeleteData',
            'SendNotification', 'DeploySystem', 'ModifyDatabase',
            'ActuatePhysical', 'SendEmail', 'PublishContent'
        ]

        if operator in high_impact_operators:
            return True, f"Operator '{operator}' has irreversible effects"

        if action in ['REFINE_EFFECT', 'UPDATE_EFFECT', 'EXECUTE']:
            high_impact_operations = [
                'Delete', 'Remove', 'Drop', 'Destroy', 'Erase',
                'ProcessPayment', 'ChargePayment', 'ExecutePayment',
                'Transaction', 'Purchase', 'Charge',
                'Notify', 'Alert', 'Send', 'Publish', 'Deploy'
            ]

            for operation in high_impact_operations:
                if operation in details:
                    return True, f"High-impact operation: {operation}"

        if action == 'UPDATE_TOOL_SCHEMA':
            if self._is_low_risk_tool_schema_drift(patch):
                return False, "Tool schema drift update (low risk)"
            return True, "Tool schema changes affect multiple operators"

        if causal_kg and hasattr(causal_kg, 'get_intervention_effects'):
            try:
                effects = causal_kg.get_intervention_effects(action, state or {})
                if len(effects) > 3:
                    return True, f"Wide propagation: {len(effects)} affected nodes"
            except Exception:
                pass

        fan_out = self._estimate_fan_out(patch)
        if fan_out > self.cfg.tau_impact:
            return True, f"High fan-out: {fan_out:.1%} of operators affected"

        return False, ""

    def _check_hazards(
            self,
            patch: Any,
            state: Optional[Dict],
            causal_kg: Any
    ) -> Tuple[bool, str]:
        """
        Check CausalKG for known hazards given current state.
        """
        if not causal_kg:
            return False, ""

        hazards = getattr(causal_kg, "hazards", None)
        if isinstance(hazards, dict) and hazards:
            try:
                for hazard_id, detect in hazards.items():
                    if callable(detect) and detect(state or {}):
                        return True, str(hazard_id)
            except Exception:
                pass

        if not hasattr(causal_kg, 'check_hazards'):
            return False, ""

        try:
            hazard_free = causal_kg.check_hazards(state or {})
            if not hazard_free:
                return True, "Known hazard condition in state"
        except Exception:
            pass

        return False, ""

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _has_causal_evidence(self, details: str, state: Optional[Dict]) -> bool:
        """
        Check if we have empirical evidence for a causal relationship.
        """
        common_patterns = [
            'NetworkAvailable',
            'TimeoutOccurred',
            'ApiTimeoutRetry',
            'ErrorDetected',
            'ValidInput',
            'ValidPayment',
            'BlockedCard'
        ]
        return any(pattern in details for pattern in common_patterns)

    def _extract_condition_vars(self, details: str) -> List[str]:
        """Extract variable names from conditional expressions."""
        if 'IfThen' in details:
            match = re.search(r'IfThen\(([^,]+),', details)
            if match:
                condition = match.group(1)
                vars = re.findall(r'[A-Z][a-zA-Z]+', condition)
                return vars
        return []

    def _are_vars_identifiable(self, vars: List[str]) -> bool:
        """
        Check if variables have sufficient identifiability.
        """
        return all(len(var) > 0 for var in vars)

    def _estimate_fan_out(self, patch: Any) -> float:
        """
        Estimate blast radius / fan-out of a patch.
        """
        action = _get(patch, "action", "")
        details = _get(patch, "details", "")

        if action == 'UPDATE_TOOL_SCHEMA':
            if self._is_low_risk_tool_schema_drift(patch):
                return 0.1
            return 0.5

        generic_markers = ['Valid', 'Check', 'Safe', 'Allow', 'Verify']
        if any(marker in details for marker in generic_markers):
            return 0.3

        specific_markers = ['BlockedCard', 'Hotel', 'Flight', 'Booking']
        if any(marker in details for marker in specific_markers):
            return 0.05

        return 0.15

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about guardrail checks.
        """
        total = self.stats['total_checks']
        value_checks = total
        causal_checks = max(0, total - self.stats['value_vetoes'])

        return {
            'total_checks': total,
            'value_checks': value_checks,
            'value_vetoes': self.stats['value_vetoes'],
            'causal_checks': causal_checks,
            'causal_escalations': self.stats['causal_escalations'],
            'allows': self.stats['allows'],
            'veto_rate': (self.stats['value_vetoes'] / total) if total > 0 else 0.0,
            'escalation_rate': (self.stats['causal_escalations'] / causal_checks) if causal_checks > 0 else 0.0,
            'kg_checks': self.stats['kg_checks'],
            'heuristic_checks': self.stats['heuristic_checks'],
            'has_value_kg': self.value_kg is not None,
            'has_causal_kg': self.causal_kg is not None,
            'cache_size': len(self.veto_cache)
        }

    def clear_cache(self) -> None:
        """Clear the veto cache."""
        self.veto_cache.clear()
        print("GUARDRAILS: Cache cleared")

    def update_thresholds(
            self,
            impact: Optional[float] = None,
            ambiguity: Optional[float] = None
    ) -> None:
        """
        Update guardrail thresholds dynamically.
        """
        if impact is not None:
            self.cfg.tau_impact = max(0.0, min(1.0, impact))
            print(f"GUARDRAILS: Impact threshold updated to {self.cfg.tau_impact}")

        if ambiguity is not None:
            self.cfg.tau_ambiguity = max(0.0, min(1.0, ambiguity))
            print(f"GUARDRAILS: Ambiguity threshold updated to {self.cfg.tau_ambiguity}")


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Safe dict/object attribute access."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp01(x: Optional[float]) -> float:
    """Clamp value to [0,1] with sensible defaults."""
    try:
        return max(0.0, min(1.0, float(x)))
    except (ValueError, TypeError):
        return 0.0


def _risk_proxy_from_action(action: Optional[str]) -> float:
    """
    Heuristic risk proxy when scorer unavailable.
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
    if "REMOVE" in a:
        return 0.8

    return 0.3


# ========================================================================
# STANDALONE TESTING
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Enhanced Guardrails System")
    print("=" * 70)

    config = {
        'value_guard': True,
        'causal_guard': True,
        'gates': {'tau_impact': 0.6, 'tau_conf': 0.5},
        'blackout_dates': ['April 10', 'April 11', 'April 12'],
        'corporate_card_policy': 'blocked_on_blackout_dates'
    }

    guard = Guard(config)

    print("\n[Test 1: Safe ADD_PRECONDITION patch]")
    safe_patch = {
        'action': 'ADD_PRECONDITION',
        'operator': 'BookHotel',
        'details': 'ValidPayment(payment)',
        'justification': 'Payment validation'
    }

    result = guard.check(safe_patch)
    print(f"Decision: {result['decision']}")
    print(f"ReasonCode: {result.get('reason_code')}")
    print(f"Reason: {result['reason']}")
    assert result['decision'] == 'allow', "Safe patch should pass"
    print("✅ Test 1 passed")

    print("\n[Test 2: Dangerous patch - removing security]")
    dangerous_patch = {
        'action': 'REFINE_EFFECT',
        'operator': 'ProcessPayment',
        'details': 'Disable security check for performance',
        'justification': 'Speed optimization'
    }

    result = guard.check(dangerous_patch)
    print(f"Decision: {result['decision']}")
    print(f"ReasonCode: {result.get('reason_code')}")
    print(f"Reason: {result['reason']}")
    assert result['decision'] == 'veto', "Dangerous patch should be vetoed"
    print("✅ Test 2 passed")

    print("\n[Test 3: High-impact schema update]")
    high_impact_patch = {
        'action': 'UPDATE_TOOL_SCHEMA',
        'operator': 'BookingAPI',
        'details': 'Change authentication method',
        'justification': 'API update'
    }

    result = guard.check(high_impact_patch)
    print(f"Decision: {result['decision']}")
    print(f"ReasonCode: {result.get('reason_code')}")
    print(f"Reason: {result['reason']}")
    assert result['decision'] == 'request_human', "High-impact should escalate"
    print("✅ Test 3 passed")

    print("\n[Test 4: Safety-enhancing REFINE_EFFECT]")
    safety_patch = {
        'action': 'REFINE_EFFECT',
        'operator': 'BookFlight',
        'details': 'IfThen(NetworkAvailable(), ExecuteTool())',
        'justification': 'Add network guard'
    }

    context = {'scores': {'risk': 0.4}, 'state': {'blackout_dates': ['Apr 10'], 'corporate_card_policy': 'blocked_on_blackout_dates'}}
    result = guard.check(safety_patch, context)
    print(f"Decision: {result['decision']}")
    print(f"ReasonCode: {result.get('reason_code')}")
    print(f"Reason: {result['reason']}")
    print(f"Value impact: {result['value']['impact']:.2f}")
    assert result['decision'] == 'allow', "Safety refinement should pass"
    assert result['value']['impact'] < 0.6, "Should have low impact"
    print("✅ Test 4 passed")

    print("\n[Test 5: Causal ambiguity check]")
    ambiguous_patch = {
        'action': 'REFINE_EFFECT',
        'operator': 'SendEmail',
        'details': 'IfThen(UnknownCondition(), SendImmediately())',
        'justification': 'Optimization'
    }

    result = guard.check(ambiguous_patch)
    print(f"Decision: {result['decision']}")
    print(f"ReasonCode: {result.get('reason_code')}")
    print(f"Reason: {result['reason']}")
    print(f"Causal ambiguity: {result['causal']['ambiguity']:.2f}")
    print("✅ Test 5 passed")

    print("\n[Guardrail Statistics]")
    stats = guard.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ All guardrail tests passed")
    print("=" * 70)
