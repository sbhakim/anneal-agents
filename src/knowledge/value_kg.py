# src/knowledge/value_kg.py
"""
Implements the Value Knowledge Graph (KG_val) for deontic and policy constraints.
Encodes rules, policies, and preferences for verification and guardrails (Section II-C).
"""
from typing import Dict, Any, List, Callable


class ValueKG:
    """
    A knowledge graph encoding deontic rules (obligations, prohibitions) and stakeholder preferences.
    Supports adding rules and querying for violations during verification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.rules: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]] = {}  # rule_id -> predicate fn
        self.preferences: Dict[str, float] = {}  # stakeholder_id -> preference_weight
        self.blackout_dates: List[str] = config.get('blackout_dates', [])
        self.corporate_policy: str = config.get('corporate_card_policy', 'blocked_on_blackout_dates')
        self._load_default_rules()
        print(f"VALUE_KG: Initialized with {len(self.rules)} rules and blackout dates: {self.blackout_dates}")

    def _load_default_rules(self) -> None:
        """
        Loads default deontic rules for the travel domain (e.g., prohibitions on card use).
        """
        # Rule: Prohibit corporate card during blackout dates
        def prohibit_blackout_card(state: Dict[str, Any], params: Dict[str, Any]) -> bool:
            card_type = params.get("payment", state.get("payment_method", ""))
            travel_dates = state.get("travel_dates", "")
            is_corporate = "CorporateCard" in card_type
            is_blackout = any(date in travel_dates for date in self.blackout_dates)
            return not (is_corporate and self.corporate_policy == 'blocked_on_blackout_dates' and is_blackout)

        self.rules['no_blackout_corporate'] = prohibit_blackout_card

        # Example preference: Prefer personal cards for low-cost trips (mock weight)
        self.preferences['user_preference'] = 0.8  # Weight for preferring personal cards

    def add_rule(self, rule_id: str, predicate: Callable[[Dict[str, Any], Dict[str, Any]], bool]) -> None:
        """
        Adds a new deontic rule (e.g., from FDKA patches or policy updates).
        """
        if rule_id in self.rules:
            print(f"VALUE_KG: Rule '{rule_id}' already exists. Overwriting.")
        self.rules[rule_id] = predicate
        print(f"VALUE_KG: Added rule '{rule_id}'")

    def check_deontic(self, state: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """
        Checks all deontic rules for violations.
        Returns True if no violations (all rules pass), False otherwise.
        Used in verification/guardrails (Section VIII-E).
        """
        for rule_id, pred in self.rules.items():
            try:
                if not pred(state, params):
                    print(f"VALUE_KG: Violation in rule '{rule_id}'")
                    return False
            except Exception as e:
                print(f"VALUE_KG: Error in rule '{rule_id}': {e}")
                return False  # Fail-safe on error
        return True

    def get_preference_score(self, stakeholder_id: str, action: Dict[str, Any]) -> float:
        """
        Computes a preference score for an action (mock for PoC).
        Higher score if action aligns with preferences (e.g., using preferred card).
        """
        if stakeholder_id not in self.preferences:
            return 0.5  # Neutral default
        # Mock: Higher if personal card used
        card_type = action.get("payment", "")
        if "PersonalCard" in card_type:
            return self.preferences[stakeholder_id]
        return 1.0 - self.preferences[stakeholder_id]  # Lower if not preferred

    def update_policy(self, new_policy: str, new_blackout_dates: List[str] = None) -> None:
        """
        Updates global policies (e.g., after policy flip in scenarios).
        """
        self.corporate_policy = new_policy
        if new_blackout_dates:
            self.blackout_dates.extend([d for d in new_blackout_dates if d not in self.blackout_dates])
        print(f"VALUE_KG: Policy updated to '{new_policy}' with blackout dates: {self.blackout_dates}")