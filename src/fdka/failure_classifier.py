# src/fdka/failure_classifier.py
"""
Failure Taxonomy Classifier - Distinguishes patchable bugs from business constraints.

This module implements a rule-based + heuristic classifier to determine whether
a failure is:
1. PATCHABLE: Operator bug that FDKA should attempt to fix
2. CONSTRAINT: Business rule/policy that should NOT be patched

Critical for domains with high constraint-to-bug ratios (e.g., e-commerce).
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class FailureClassifier:
    """
    Classifies failures as patchable bugs vs immutable constraints.

    Uses domain-specific heuristics to avoid wasting FDKA effort on
    business rules that cannot/should not be patched.
    """

    def __init__(self, domain: str = "travel"):
        self.domain = domain

        # E-commerce constraint patterns (regex-based detection)
        self.ecommerce_constraints = [
            # Refund window constraints
            (r'refund.*window.*exceeded', 'refund_window'),
            (r'return.*window.*(\d+)\s*days', 'refund_window'),
            (r'purchased.*(\d+)\s*days?\s*ago', 'refund_window'),

            # Shipping restrictions
            (r'APO|FPO|PO_?BOX', 'shipping_restriction'),
            (r'cannot ship to.*addresses', 'shipping_restriction'),
            (r'restricted.*shipping|shipping.*restricted', 'shipping_restriction'),

            # Inventory/stock constraints
            (r'inventory.*insufficient|insufficient.*inventory', 'inventory_limit'),
            (r'out.*of.*stock|stock.*unavailable', 'inventory_limit'),
            (r'only\s+0\s+units?\s+available', 'inventory_limit'),
            (r'insufficient.*stock', 'inventory_limit'),

            # Price changes (dynamic pricing)
            (r'price.*changed|price.*update', 'dynamic_pricing'),
            (r'item price changed', 'dynamic_pricing'),

            # Promotion policies
            (r'promo.*expired|expired.*promo', 'promotion_expired'),
            (r'promo.*conflict|conflicting.*promo', 'promotion_policy'),

            # Security/policy constraints
            (r'employee.*verification.*required', 'security_policy'),
            (r'violates.*company.*policy|company.*policy.*violation', 'security_policy'),
            (r'bulk.*discount.*minimum', 'business_rule'),
        ]

        # Travel domain constraint patterns
        self.travel_constraints = [
            (r'blackout.*date', 'blackout_date'),
            (r'flight.*fully.*booked', 'capacity_limit'),
            (r'hotel.*no.*vacancy', 'capacity_limit'),
        ]

        print(f"CLASSIFIER: Initialized for domain='{domain}'")

    def is_patchable(self, trace: List[Dict[str, Any]], instruction: str = "") -> Tuple[bool, Optional[str]]:
        """
        Determine if failure is patchable (bug) or constraint.

        Args:
            trace: Execution trace with error information
            instruction: Original task instruction (optional context)

        Returns:
            (is_patchable: bool, reason: str or None)
            - (True, None) if patchable bug
            - (False, constraint_type) if immutable constraint
        """
        # Extract error message from trace
        error_info = self._extract_error_info(trace)
        if not error_info:
            # No clear error - assume patchable (conservative)
            return True, None

        error_msg = error_info.get('message', '')
        error_type = error_info.get('error', '')
        operator = error_info.get('operator', '')

        # Combine all text for pattern matching
        search_text = f"{error_msg} {error_type} {operator} {instruction}".lower()

        # Check domain-specific constraint patterns
        patterns = self._get_constraint_patterns()

        for pattern, constraint_type in patterns:
            if re.search(pattern, search_text, re.IGNORECASE):
                print(f"  🔍 CLASSIFIER: Detected constraint '{constraint_type}' (pattern: {pattern[:30]}...)")
                print(f"     Matched in: {search_text[:80]}...")
                return False, constraint_type

        # No constraint detected - assume patchable
        return True, None

    def _get_constraint_patterns(self) -> List[Tuple[str, str]]:
        """Get constraint patterns for current domain."""
        if self.domain == 'ecommerce':
            return self.ecommerce_constraints
        elif self.domain == 'travel' or self.domain == 'travel_planning':
            return self.travel_constraints
        else:
            return []  # Unknown domain - no constraints defined

    def _extract_error_info(self, trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract error information from trace."""
        for entry in reversed(trace or []):
            if not isinstance(entry, dict):
                continue
            if 'error' in entry or 'error_type' in entry or 'message' in entry:
                return {
                    'error': entry.get('error', entry.get('error_type', 'Unknown')),
                    'message': entry.get('message', ''),
                    'operator': entry.get('operator', ''),
                    'policy_ref': entry.get('policy_ref', ''),
                }
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about classifications (for debugging)."""
        return {
            'domain': self.domain,
            'constraint_patterns': len(self._get_constraint_patterns()),
        }


# ============================================================================
# Convenience function for quick checks
# ============================================================================

def should_trigger_fdka(trace: List[Dict[str, Any]], instruction: str = "", domain: str = "travel") -> bool:
    """
    Quick check: Should FDKA be triggered for this failure?

    Returns:
        True if failure is patchable (trigger FDKA)
        False if failure is constraint (skip FDKA)
    """
    classifier = FailureClassifier(domain=domain)
    is_patchable, constraint_type = classifier.is_patchable(trace, instruction)

    if not is_patchable:
        print(f"  ⏭️  CLASSIFIER: Skipping FDKA (immutable constraint: {constraint_type})")

    return is_patchable
