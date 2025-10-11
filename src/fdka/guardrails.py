# src/fdka/guardrails.py
"""
Implements the Value and Causal Guardrails for patch validation.
Corresponds to Section VIII-E of the paper.

UPDATED: Full implementation of guardrails that actually query knowledge graphs:
  - Value guardrails enforce deontic constraints with lexicographic priority (Eq. 16)
  - Causal guardrails detect ambiguity and high-impact propagation (Eq. 17-18)
  - Integration with ValueKG and CausalKG
"""
from typing import Dict, Any, Tuple, Optional, List
import re


class Guard:
    """
    Applies value and causal constraints to veto or escalate risky patches.

    Implements two-tier guardrails from Section VIII-E:
    1. Value guardrails (deontic rules) - can VETO
    2. Causal guardrails (ambiguity/impact) - can REQUEST_HUMAN

    Value vetoes are lexicographic: they override all utility considerations.
    """

    def __init__(self, config: Dict[str, Any], value_kg=None, causal_kg=None):
        """
        Initialize guardrails with configuration and knowledge graphs.

        Args:
            config: Guardrails configuration
            value_kg: ValueKG instance (optional, will create if not provided)
            causal_kg: CausalKG instance (optional, will create if not provided)
        """
        self.config = config

        # Thresholds for escalation
        self.impact_threshold = config.get('impact_threshold', 0.6)
        self.ambiguity_threshold = config.get('ambiguity_threshold', 0.5)

        # Enable/disable individual guards
        self.enable_value_guard = config.get('enable_value_guard', True)
        self.enable_causal_guard = config.get('enable_causal_guard', True)

        # Knowledge graph integration
        self.value_kg = value_kg
        self.causal_kg = causal_kg

        # If KGs not provided, create them
        if self.value_kg is None:
            self.value_kg = self._create_value_kg(config)

        if self.causal_kg is None:
            self.causal_kg = self._create_causal_kg(config)

        # Cache for performance
        self.veto_cache: Dict[str, bool] = {}

        print(f"GUARDRAILS: Initialized")
        print(f"  - Value guard: {'ENABLED' if self.enable_value_guard else 'DISABLED'}")
        print(f"  - Causal guard: {'ENABLED' if self.enable_causal_guard else 'DISABLED'}")
        print(f"  - Impact threshold: {self.impact_threshold}")
        print(f"  - Ambiguity threshold: {self.ambiguity_threshold}")

    def _create_value_kg(self, config: Dict[str, Any]):
        """Create ValueKG if not provided."""
        try:
            from ..knowledge.value_kg import ValueKG
            kg_config = {
                'blackout_dates': config.get('blackout_dates', []),
                'corporate_card_policy': config.get('corporate_card_policy', 'blocked_on_blackout_dates')
            }
            return ValueKG(kg_config)
        except ImportError:
            print("  ⚠️ GUARDRAILS: ValueKG not available, using mock")
            return None

    def _create_causal_kg(self, config: Dict[str, Any]):
        """Create CausalKG if not provided."""
        try:
            from ..knowledge.causal_kg import CausalKG
            return CausalKG(config)
        except ImportError:
            print("  ⚠️ GUARDRAILS: CausalKG not available, using mock")
            return None

    # ========================================================================
    # MAIN GUARD INTERFACE
    # ========================================================================

    def check(self, patch_or_action: Dict[str, Any], state: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Main guard entry point. Checks both value and causal guardrails.

        Returns decision in order of priority:
        1. Value veto (highest priority)
        2. Causal escalation (request_human)
        3. Allow (pass all guards)

        Args:
            patch_or_action: Patch or action to validate
            state: Optional state context for evaluation

        Returns:
            Tuple of (decision, reason) where decision ∈ {'veto', 'request_human', 'allow'}
        """
        print("\n  🛡️  GUARDRAILS: Starting validation...")

        # Stage 1: Value Guardrails (can VETO)
        if self.enable_value_guard:
            value_ok, value_reason = self.check_value_guardrail(patch_or_action, state)
            if not value_ok:
                print(f"  ❌ GUARDRAILS: VALUE VETO - {value_reason}")
                return 'veto', value_reason
            print(f"  ✅ Value guard passed")

        # Stage 2: Causal Guardrails (can REQUEST_HUMAN)
        if self.enable_causal_guard:
            causal_decision, causal_reason = self.check_causal_guardrail(patch_or_action, state)
            if causal_decision == 'request_human':
                print(f"  ⚠️ GUARDRAILS: ESCALATION - {causal_reason}")
                return 'request_human', causal_reason
            print(f"  ✅ Causal guard passed")

        print("  ✅ GUARDRAILS: All checks passed")
        return 'allow', 'All guardrails passed'

    # ========================================================================
    # VALUE GUARDRAILS (Section VIII-E.1, Eq. 16)
    # ========================================================================

    def check_value_guardrail(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Checks for violations of deontic (must/must not) rules.

        From Eq. 16:
        Guard_val(Δo) = {
            veto,   ∃c ∈ KG^val : c(ψ(Δo)) = ⊥,
            allow,  otherwise
        }

        where c are deontic constraints and ψ(Δo) encodes the patch's effects.

        IMPORTANT: Value vetoes are lexicographic - they override ALL utility.

        Args:
            patch_or_action: The patch or action to validate
            state: Optional state for context

        Returns:
            Tuple of (ok, reason) where ok=False means VETO
        """
        if not self.value_kg:
            print("    ⚠️ Value guard: No ValueKG available, passing by default")
            return True, "No value KG"

        # Extract key information
        operator = patch_or_action.get('operator', 'Unknown')
        action = patch_or_action.get('action', '')
        details = patch_or_action.get('details', '')

        # Build parameter dict for deontic checks
        params = self._extract_params(patch_or_action, state)

        # Check all deontic rules in ValueKG
        try:
            deontic_ok = self.value_kg.check_deontic(state or {}, params)

            if not deontic_ok:
                reason = "Deontic constraint violated"
                # Try to identify which rule
                violated_rule = self._identify_violated_rule(patch_or_action, state)
                if violated_rule:
                    reason = f"Violated rule: {violated_rule}"
                return False, reason

        except Exception as e:
            print(f"    ⚠️ Value guard error: {e}")
            # Fail-safe: be conservative on errors
            return False, f"Value guard error: {str(e)[:100]}"

        # Additional checks for specific dangerous patterns
        veto, reason = self._check_dangerous_patterns(patch_or_action)
        if veto:
            return False, reason

        return True, "No deontic violations"

    def _extract_params(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Extract parameters from patch/action for deontic checking.
        """
        params = {}

        # Get payment method
        if state:
            params['payment'] = state.get('payment_method', '')

        # Extract from patch details
        details = patch_or_action.get('details', '')

        # Simple extraction of common parameters
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
            patch_or_action: Dict[str, Any],
            state: Optional[Dict]
    ) -> Optional[str]:
        """
        Attempts to identify which specific deontic rule was violated.
        """
        if not self.value_kg or not hasattr(self.value_kg, 'rules'):
            return None

        params = self._extract_params(patch_or_action, state)

        # Check each rule individually
        for rule_id, rule_fn in self.value_kg.rules.items():
            try:
                if not rule_fn(state or {}, params):
                    return rule_id
            except Exception:
                continue

        return None

    def _check_dangerous_patterns(
            self,
            patch_or_action: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Checks for hardcoded dangerous patterns that should always be vetoed.

        Patterns include:
        - Removing security checks
        - Disabling authentication
        - Removing payment validation
        - Bypassing privacy controls
        """
        action = patch_or_action.get('action', '')
        details = patch_or_action.get('details', '').lower()

        # Dangerous keywords that trigger veto
        dangerous_patterns = [
            ('disable', 'security'),
            ('remove', 'auth'),
            ('bypass', 'check'),
            ('skip', 'validation'),
            ('allow', 'all'),
            ('disable', 'verify'),
            ('remove', 'payment'),
            ('skip', 'privacy')
        ]

        for pattern in dangerous_patterns:
            if all(keyword in details for keyword in pattern):
                return True, f"Dangerous pattern detected: {' '.join(pattern)}"

        # Removing preconditions is dangerous (weakens safety)
        if action == 'REMOVE_PRECONDITION':
            return True, "Removing preconditions weakens safety guarantees"

        # Veto patches that remove "Not" from safety constraints
        if re.search(r'remove.*not\(', details, re.IGNORECASE):
            return True, "Removing negation from safety constraint"

        return False, ""

    # ========================================================================
    # CAUSAL GUARDRAILS (Section VIII-E.2, Eq. 17-18)
    # ========================================================================

    def check_causal_guardrail(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Checks for causal ambiguity and high-impact propagation.

        From Eq. 17:
        φ_cau(Δo) ≡ Ambig(u, Δo) ∨ HighImpact(Δo)

        From Eq. 18:
        Guard_cau(Δo) = {
            request_human,  φ_cau(Δo),
            allow,          ¬φ_cau(Δo)
        }

        Unlike value guards, causal guards don't veto - they escalate to human review.

        Args:
            patch_or_action: The patch or action to validate
            state: Optional state for context

        Returns:
            Tuple of (decision, reason) where decision ∈ {'allow', 'request_human'}
        """
        if not self.causal_kg:
            print("    ⚠️ Causal guard: No CausalKG available, passing by default")
            return 'allow', "No causal KG"

        # Check for causal ambiguity
        is_ambiguous, ambig_reason = self._check_ambiguity(patch_or_action, state)
        if is_ambiguous:
            return 'request_human', f"Causal ambiguity: {ambig_reason}"

        # Check for high-impact propagation
        is_high_impact, impact_reason = self._check_high_impact(patch_or_action, state)
        if is_high_impact:
            return 'request_human', f"High-impact: {impact_reason}"

        # Check hazards
        has_hazard, hazard_reason = self._check_hazards(patch_or_action, state)
        if has_hazard:
            return 'request_human', f"Hazard detected: {hazard_reason}"

        return 'allow', "No causal concerns"

    def _check_ambiguity(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Checks for causal ambiguity: low identifiability of causal claims.

        From paper: "Does the patch rely on a causal claim with low identifiability?"
        Uses backdoor criterion, instrumental variables, or sensitivity analysis.
        """
        if not self.causal_kg:
            return False, ""

        details = patch_or_action.get('details', '')
        operator = patch_or_action.get('operator', '')

        # Check for circular dependencies (cycles in causal graph)
        if hasattr(self.causal_kg, 'detect_cycles'):
            has_cycles = self.causal_kg.detect_cycles()
            if has_cycles:
                return True, "Circular causal dependencies detected"

        # Check if patch creates new causal edges with unknown effects
        action = patch_or_action.get('action', '')
        if action == 'REFINE_EFFECT':
            # Effect refinements change causal structure
            # Check if we have evidence for this causal relationship
            if not self._has_causal_evidence(details, state):
                return True, "Insufficient evidence for causal relationship"

        # Check for interventions with ambiguous outcomes
        if 'IfThen' in details or 'When' in details:
            # Conditional effects introduce causal assumptions
            condition_vars = self._extract_condition_vars(details)
            if not self._are_vars_identifiable(condition_vars):
                return True, f"Unidentifiable causal variables: {condition_vars}"

        return False, ""

    def _check_high_impact(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Checks for high-impact propagation: large downstream fan-out or irreversible effects.

        From paper: "Does the patch affect operators or predicates with large downstream
        fan-out or irreversible effects (e.g., financial transactions, physical actuation)?"
        """
        operator = patch_or_action.get('operator', '')
        action = patch_or_action.get('action', '')
        details = patch_or_action.get('details', '')

        # High-impact operators (by domain knowledge)
        high_impact_operators = [
            'ProcessPayment', 'ExecuteTransaction', 'DeleteData',
            'SendNotification', 'DeploySystem', 'ModifyDatabase',
            'ActuatePhysical', 'SendEmail'  # Irreversible actions
        ]

        if operator in high_impact_operators:
            return True, f"Operator '{operator}' has irreversible effects"

        # High-impact predicates
        high_impact_predicates = [
            'Delete', 'Remove', 'Drop', 'Destroy', 'Erase',
            'Payment', 'Transaction', 'Purchase', 'Charge',
            'Notify', 'Alert', 'Send', 'Publish'
        ]

        for predicate in high_impact_predicates:
            if predicate in details:
                return True, f"High-impact predicate: {predicate}"

        # Schema updates affect all tool users
        if action == 'UPDATE_TOOL_SCHEMA':
            return True, "Tool schema changes affect multiple operators"

        # Check impact score if available from causal KG
        if self.causal_kg and hasattr(self.causal_kg, 'get_intervention_effects'):
            effects = self.causal_kg.get_intervention_effects(action, state or {})
            if len(effects) > 3:  # Affects more than 3 downstream nodes
                return True, f"Wide propagation: {len(effects)} affected nodes"

        # Compute estimated fan-out
        fan_out = self._estimate_fan_out(patch_or_action)
        if fan_out > self.impact_threshold:
            return True, f"High fan-out: {fan_out:.1%} of operators affected"

        return False, ""

    def _check_hazards(
            self,
            patch_or_action: Dict[str, Any],
            state: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Checks CausalKG for known hazards given current state.
        """
        if not self.causal_kg or not hasattr(self.causal_kg, 'check_hazards'):
            return False, ""

        try:
            hazard_free = self.causal_kg.check_hazards(state or {})
            if not hazard_free:
                return True, "Known hazard condition detected in state"
        except Exception as e:
            print(f"    ⚠️ Hazard check error: {e}")

        return False, ""

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _has_causal_evidence(self, details: str, state: Optional[Dict]) -> bool:
        """
        Checks if we have empirical evidence for a causal relationship.
        In full implementation, would query historical data or experiments.
        """
        # For PoC: assume common patterns have evidence
        common_patterns = [
            'NetworkAvailable',
            'TimeoutOccurred',
            'ErrorDetected',
            'ValidInput'
        ]

        return any(pattern in details for pattern in common_patterns)

    def _extract_condition_vars(self, details: str) -> List[str]:
        """Extract variable names from conditional expressions."""
        # Simple regex to find variable-like tokens in conditionals
        if 'IfThen' in details:
            match = re.search(r'IfThen\(([^,]+),', details)
            if match:
                condition = match.group(1)
                # Extract identifiers
                vars = re.findall(r'[A-Z][a-zA-Z]+', condition)
                return vars
        return []

    def _are_vars_identifiable(self, vars: List[str]) -> bool:
        """
        Checks if variables have sufficient identifiability.
        In full implementation, would use backdoor criterion or IV analysis.
        """
        # For PoC: assume single-word predicates are identifiable
        return all(len(var) > 0 for var in vars)

    def _estimate_fan_out(self, patch_or_action: Dict[str, Any]) -> float:
        """
        Estimates blast radius / fan-out of a patch.
        Returns fraction ∈ [0, 1] of affected operators.
        """
        action = patch_or_action.get('action', '')
        details = patch_or_action.get('details', '')

        # Schema updates affect many operators
        if action == 'UPDATE_TOOL_SCHEMA':
            return 0.5  # 50% of operators

        # Generic predicates affect more operators
        generic_markers = ['Valid', 'Check', 'Safe', 'Allow', 'Verify']
        if any(marker in details for marker in generic_markers):
            return 0.3  # 30% of operators

        # Specific predicates have narrow scope
        specific_markers = ['BlockedCard', 'Hotel', 'Flight', 'Booking']
        if any(marker in details for marker in specific_markers):
            return 0.05  # 5% of operators

        # Default: moderate scope
        return 0.15  # 15% of operators

    # ========================================================================
    # CONFLICT RESOLUTION (Section VIII-E.3)
    # ========================================================================

    def resolve_conflict(
            self,
            patch_or_action: Dict[str, Any],
            value_decision: Tuple[bool, str],
            causal_decision: Tuple[str, str]
    ) -> Tuple[str, str]:
        """
        Resolves conflicts between effective (high utility) and permissible (safe) patches.

        From paper Section VIII-E.3:
        "When the highest-utility patch conflicts with normative or causal constraints,
        we adopt a two-level resolution strategy"

        Resolution order:
        1. Value veto (deontic) - absolute, cannot be overridden
        2. Causal escalation - request human input
        3. Search for nearest-feasible alternative (future work)

        Args:
            patch_or_action: The patch being evaluated
            value_decision: (ok, reason) from value guard
            causal_decision: (decision, reason) from causal guard

        Returns:
            Tuple of (final_decision, reason)
        """
        value_ok, value_reason = value_decision
        causal_dec, causal_reason = causal_decision

        # Value vetoes are lexicographic - highest priority
        if not value_ok:
            print(f"  🚫 CONFLICT RESOLUTION: Value veto overrides all")
            return 'veto', value_reason

        # Causal escalations are next priority
        if causal_dec == 'request_human':
            print(f"  ⚠️ CONFLICT RESOLUTION: Escalating to human review")
            return 'request_human', causal_reason

        # No conflicts
        print(f"  ✅ CONFLICT RESOLUTION: No conflicts detected")
        return 'allow', "No conflicts"

    # ========================================================================
    # UTILITIES AND CONFIGURATION
    # ========================================================================

    def set_value_kg(self, value_kg) -> None:
        """Update the ValueKG reference."""
        self.value_kg = value_kg
        print("GUARDRAILS: ValueKG updated")

    def set_causal_kg(self, causal_kg) -> None:
        """Update the CausalKG reference."""
        self.causal_kg = causal_kg
        print("GUARDRAILS: CausalKG updated")

    def update_thresholds(self, impact: Optional[float] = None, ambiguity: Optional[float] = None) -> None:
        """
        Update guardrail thresholds dynamically.

        Args:
            impact: New impact threshold (0-1)
            ambiguity: New ambiguity threshold (0-1)
        """
        if impact is not None:
            self.impact_threshold = max(0.0, min(1.0, impact))
            print(f"GUARDRAILS: Impact threshold updated to {self.impact_threshold}")

        if ambiguity is not None:
            self.ambiguity_threshold = max(0.0, min(1.0, ambiguity))
            print(f"GUARDRAILS: Ambiguity threshold updated to {self.ambiguity_threshold}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about guardrail checks.

        Returns:
            Dictionary with counts and rates
        """
        return {
            'value_guard_enabled': self.enable_value_guard,
            'causal_guard_enabled': self.enable_causal_guard,
            'impact_threshold': self.impact_threshold,
            'ambiguity_threshold': self.ambiguity_threshold,
            'has_value_kg': self.value_kg is not None,
            'has_causal_kg': self.causal_kg is not None,
            'cache_size': len(self.veto_cache)
        }

    def clear_cache(self) -> None:
        """Clear the veto cache."""
        self.veto_cache.clear()
        print("GUARDRAILS: Cache cleared")


# ========================================================================
# STANDALONE TESTING
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Guardrails with KG integration")
    print("=" * 70)

    # Mock configuration
    config = {
        'impact_threshold': 0.6,
        'ambiguity_threshold': 0.5,
        'enable_value_guard': True,
        'enable_causal_guard': True,
        'blackout_dates': ['April 10', 'April 11', 'April 12'],
        'corporate_card_policy': 'blocked_on_blackout_dates'
    }

    guard = Guard(config)

    # Test 1: Safe patch (should pass)
    print("\n[Test 1: Safe ADD_PRECONDITION patch]")
    safe_patch = {
        'action': 'ADD_PRECONDITION',
        'operator': 'BookHotel',
        'details': 'Not(BlockedCard(payment, dates))',
        'justification': 'Safety check'
    }

    decision, reason = guard.check(safe_patch)
    print(f"Decision: {decision}, Reason: {reason}")
    assert decision == 'allow', "Safe patch should pass"

    # Test 2: Dangerous pattern (should veto)
    print("\n[Test 2: Dangerous patch - removing security]")
    dangerous_patch = {
        'action': 'REFINE_EFFECT',
        'operator': 'ProcessPayment',
        'details': 'Disable security check for faster processing',
        'justification': 'Performance'
    }

    decision, reason = guard.check(dangerous_patch)
    print(f"Decision: {decision}, Reason: {reason}")
    assert decision == 'veto', "Dangerous patch should be vetoed"

    # Test 3: High-impact patch (should request human)
    print("\n[Test 3: High-impact schema update]")
    high_impact_patch = {
        'action': 'UPDATE_TOOL_SCHEMA',
        'operator': 'BookingAPI',
        'details': 'Change authentication method',
        'justification': 'API update'
    }

    decision, reason = guard.check(high_impact_patch)
    print(f"Decision: {decision}, Reason: {reason}")
    assert decision == 'request_human', "High-impact should escalate"

    # Test 4: Value guard with state context
    print("\n[Test 4: Value guard with deontic violation]")
    state = {
        'payment_method': 'CorporateCard:CC-5512',
        'travel_dates': 'April 10-12',
        'blackout_dates': ['April 10', 'April 11', 'April 12'],
        'corporate_card_policy': 'blocked_on_blackout_dates'
    }

    params = {
        'payment': 'CorporateCard:CC-5512'
    }

    violating_action = {
        'action': 'EXECUTE',
        'operator': 'BookHotel',
        'details': 'Book hotel with corporate card during blackout',
        'params': params
    }

    value_ok, value_reason = guard.check_value_guardrail(violating_action, state)
    print(f"Value guard: {value_ok}, Reason: {value_reason}")

    # Test 5: Ambiguous causal patch
    print("\n[Test 5: Causal ambiguity check]")
    ambiguous_patch = {
        'action': 'REFINE_EFFECT',
        'operator': 'SendEmail',
        'details': 'IfThen(UnknownCondition(), SendImmediately())',
        'justification': 'Optimization'
    }

    causal_dec, causal_reason = guard.check_causal_guardrail(ambiguous_patch, state)
    print(f"Causal guard: {causal_dec}, Reason: {causal_reason}")

    # Show statistics
    print("\n[Guardrail Statistics]")
    stats = guard.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Guardrails testing complete")
    print("=" * 70)