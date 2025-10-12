# src/knowledge/rule_pool.py
"""
Manages the collection of all available operators (R_rules).
UPDATED:
- Added the missing `list_preconditions` method, which is required by the
  FDKA pipeline's serialization stage to construct the LLM prompt.
- Fixed idempotency bug: Effects are now checked by function name to prevent duplicates.
"""
from typing import Dict, Any, Callable, List, Tuple, Optional, Set
import re
import copy
from ..core.operators import Operator
from ..core.state import SymbolicState


# ========================================================================
# BASE PREDICATE LIBRARY (Initial Knowledge)
# ========================================================================

def is_card_valid(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """Checks for the presence of a payment method."""
    payment = params.get("payment", state.get("payment_method"))
    return payment is not None


def is_hotel_available(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """Checks if the hotel status is 'available'."""
    return state.get("hotel_status") == "available"


def book_hotel_effect(state: SymbolicState, params: Dict[str, Any]) -> SymbolicState:
    """Sets the hotel status to 'booked'."""
    if state.get("hotel_status") != "booked":
        state.set("hotel_status", "booked")
        print("EFFECT: Hotel booked.")
    return state


def is_flight_available(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """Checks if the flight status is 'available'."""
    return state.get("flight_status") == "available"


def book_flight_effect(state: SymbolicState, params: Dict[str, Any]) -> SymbolicState:
    """Sets the flight status to 'booked'."""
    if state.get("flight_status") != "booked":
        state.set("flight_status", "booked")
        print("EFFECT: Flight booked.")
    return state


def check_not_blocked_card(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """
    This is the "learned" precondition that FDKA adds dynamically.
    It implements the logic for: Not(BlockedCard(payment, dates)).
    """
    card_type = params.get("payment", state.get("payment_method", ""))
    policy = state.get("corporate_card_policy")
    blackout_dates: List[str] = state.get("blackout_dates", []) or []
    travel_dates_str: str = state.get("travel_dates", "")

    is_corporate = isinstance(card_type, str) and ("CorporateCard" in card_type)
    is_blackout = any(date_str in str(travel_dates_str) for date_str in blackout_dates)

    if is_corporate and policy == "blocked_on_blackout_dates" and is_blackout:
        print("PRECONDITION CHECK: ❌ BlockedCard is TRUE. Check fails.")
        return False

    print("PRECONDITION CHECK: ✅ Not(BlockedCard) is TRUE. Check passes.")
    return True


# ========================================================================
# DYNAMIC PREDICATE & EFFECT FACTORIES (The Core of Symbolic Learning)
# ========================================================================

class PredicateFactory:
    """Dynamically creates predicate functions from patch details strings."""

    @staticmethod
    def create_precondition(details: str, operator_name: str) -> Optional[Callable]:
        details_clean = (details or "").strip()
        if "Not(BlockedCard" in details_clean: return check_not_blocked_card
        if "NetworkAvailable" in details_clean:
            def check_network_available(state: SymbolicState, params: Dict) -> bool:
                network_ok = state.get("network_available", True)
                print(f"PRECONDITION CHECK: {'✅' if network_ok else '❌'} NetworkAvailable")
                return network_ok

            check_network_available.__name__ = "check_network_available"
            return check_network_available

        # **FIX**: Handle simple predicate names without arguments, like 'is_flight_available'.
        if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', details_clean):
            # Check if it matches a globally known function in this module
            if details_clean in globals() and callable(globals()[details_clean]):
                print(f"  ✅ PREDICATE FACTORY: Matched existing predicate '{details_clean}'.")
                return globals()[details_clean]
            # Fallback to a generic state check for unknown simple predicates
            predicate_name = details_clean

            def generic_simple_check(state: SymbolicState, params: Dict) -> bool:
                result = state.get(predicate_name, True)
                print(f"PRECONDITION CHECK: {'✅' if result else '❌'} {predicate_name} (generic)")
                return result

            generic_simple_check.__name__ = predicate_name
            return generic_simple_check

        match = re.search(r'([A-Z][a-zA-Z]+)\s*\(', details_clean)
        if match:
            predicate_name = match.group(1)

            def generic_check(state: SymbolicState, params: Dict) -> bool:
                result = state.get(f"{predicate_name.lower()}_ok", True)
                print(f"PRECONDITION CHECK: {'✅' if result else '❌'} {predicate_name}")
                return result

            generic_check.__name__ = f"check_{predicate_name.lower()}"
            return generic_check

        print(f"  ⚠️ PREDICATE FACTORY: Could not parse pattern '{details_clean}'")
        return None


class EffectFactory:
    """Dynamically creates effect functions from patch details strings."""

    @staticmethod
    def create_effect(details: str, operator_name: str) -> Optional[Callable]:
        details_clean = (details or "").strip()
        if "IfThen" in details_clean and "NetworkAvailable" in details_clean:
            def conditional_booking_effect(state: SymbolicState, params: Dict) -> SymbolicState:
                if not state.get("network_available", True):
                    print("EFFECT: Skipped conditional effect (network unavailable).")
                    return state
                if "Hotel" in operator_name:
                    return book_hotel_effect(state, params)
                elif "Flight" in operator_name:
                    return book_flight_effect(state, params)
                return state

            conditional_booking_effect.__name__ = "conditional_effect_network_booking"
            return conditional_booking_effect
        print(f"  ⚠️ EFFECT FACTORY: Could not parse pattern '{details_clean}'")
        return None


# ========================================================================
# RULE POOL (Main Operator Registry)
# ========================================================================

class RulePool:
    """A repository for storing, retrieving, verifying, and updating symbolic operators."""

    def __init__(self, rule_pool_path: str):
        self.operators: Dict[str, Operator] = {}
        self.path = rule_pool_path
        self.operator_history: Dict[str, List[Dict[str, Any]]] = {}
        self.learned_predicates: Set[str] = set()
        self.learned_effects: Set[str] = set()
        self.predicate_factory = PredicateFactory()
        self.effect_factory = EffectFactory()
        self.stats = {'total_patches_applied': 0, 'patches_by_type': {}, 'operators_modified': set()}
        self.load_operators()

    def load_operators(self):
        book_hotel_op = Operator("BookHotel", ["location", "dates", "payment"], [is_card_valid, is_hotel_available],
                                 [book_hotel_effect])
        book_hotel_op.metadata = {"version": "1.0"}
        self.operators[book_hotel_op.name] = book_hotel_op
        self._snapshot_operator(book_hotel_op.name)
        book_flight_op = Operator("BookFlight", ["origin", "destination", "date"], [is_card_valid, is_flight_available],
                                  [book_flight_effect])
        book_flight_op.metadata = {"version": "1.0"}
        self.operators[book_flight_op.name] = book_flight_op
        self._snapshot_operator(book_flight_op.name)
        print(f"RULE_POOL: Loaded {len(self.operators)} initial operators: {list(self.operators.keys())}")

    def snapshot(self) -> Dict[str, Any]:
        return {"operators": copy.deepcopy(self.operators), "operator_history": copy.deepcopy(self.operator_history),
                "stats": copy.deepcopy(self.stats)}

    def restore(self, snap: Dict[str, Any]):
        self.operators = snap["operators"]
        self.operator_history = snap.get("operator_history", {})
        self.stats = snap.get("stats", self.stats)

    def get_operator(self, name: str) -> Optional[Operator]:
        return self.operators.get(name)

    # ========================================================================
    # ** NEWLY ADDED METHOD **
    # ========================================================================
    def list_preconditions(self, name: str) -> List[str]:
        """
        Returns the names of preconditions for a given operator.
        This is required by the FDKA serialization stage.
        """
        op = self.get_operator(name)
        if not op:
            return []
        return [getattr(fn, "__name__", "anonymous_predicate") for fn in op.preconditions]

    def evaluate_preconditions(self, op_name: str, params: Dict, state: SymbolicState) -> Tuple[bool, Optional[str]]:
        op = self.get_operator(op_name)
        if not op: return False, "UnknownOperator"
        for pred in op.preconditions:
            try:
                if not bool(pred(state, params)): return False, getattr(pred, "__name__", "anonymous")
            except Exception as e:
                return False, getattr(pred, "__name__", "anonymous")
        return True, None

    def update_operator(self, patch: Dict[str, Any]) -> bool:
        op_name, action, details, just, patch_id = patch.get("operator"), patch.get("action"), patch.get("details",
                                                                                                         ""), patch.get(
            "justification", ""), patch.get("id", "unknown")
        op = self.operators.get(op_name)
        if not op: return False
        self._snapshot_operator(op_name)
        print(f"\n  🔧 RULE_POOL: Applying patch {patch_id} to {op_name}")
        print(f"     Action: {action}, Details: {details}")
        success = False
        if action == "ADD_PRECONDITION":
            success = self._add_precondition(op, details, just)
        elif action == "REFINE_EFFECT":
            success = self._refine_effect(op, details, just)
        elif action == "UPDATE_TOOL_SCHEMA":
            success = self._update_schema(op, details, just)
        if success:
            # Only update metadata if a change was actually made or confirmed
            if not getattr(success, "__skipped__", False):
                self._update_metadata(op, patch_id)
                self.stats['total_patches_applied'] += 1
                self.stats['patches_by_type'][action] = self.stats['patches_by_type'].get(action, 0) + 1
                self.stats['operators_modified'].add(op_name)
            print(f"  ✅ RULE_POOL: Patch {patch_id} successfully applied (or was already present).")
            return True
        else:
            print(f"  ❌ RULE_POOL: Patch {patch_id} failed to apply.")
            return False

    def _add_precondition(self, op: Operator, details: str, just: str) -> bool:
        new_pre = self.predicate_factory.create_precondition(details, op.name)
        if not new_pre: return False
        pred_name = getattr(new_pre, "__name__", "anon")
        if pred_name in {getattr(fn, "__name__", "") for fn in op.preconditions}:
            print(f"  ℹ️ Precondition '{pred_name}' already exists. Skipping duplicate.")
            success = True
            setattr(success, "__skipped__", True)
            return success
        op.preconditions.append(new_pre)
        self.learned_predicates.add(pred_name)
        print(f"  ✅ Added precondition '{pred_name}' to {op.name}.")
        return True

    def _refine_effect(self, op: Operator, details: str, just: str) -> bool:
        new_eff = self.effect_factory.create_effect(details, op.name)
        if not new_eff:
            print(f"  ⚠️ Failed to create effect from: {details[:60]}")
            return False

        effect_name = getattr(new_eff, "__name__", "anon_eff")

        # **FIX**: Check BOTH function name AND source code for true duplicates
        existing_names = {getattr(fn, "__name__", "") for fn in op.effects}
        if effect_name in existing_names:
            # Double-check if it's truly identical by comparing details
            existing_details = [getattr(fn, "__details__", "") for fn in op.effects]
            if any(details in ed or ed in details for ed in existing_details if ed):
                print(f"  ℹ️ Effect '{effect_name}' with similar details already exists. Skipping duplicate.")
                # Return True to signal success for idempotency, avoiding canary failure
                success = True
                setattr(success, "__skipped__", True)  # Mark that no real change was made
                return success

        # Attach details for future comparison
        setattr(new_eff, "__details__", details)

        op.effects.append(new_eff)
        self.learned_effects.add(effect_name)
        print(f"  ✅ Added effect '{effect_name}' to {op.name}.")
        return True

    def _update_schema(self, op: Operator, details: str, just: str) -> bool:
        op.metadata['schema_update'] = details
        return True

    def _update_metadata(self, op: Operator, patch_id: str):
        try:
            op.metadata["version"] = f"{float(op.metadata.get('version', '1.0')) + 0.1:.1f}"
        except:
            op.metadata["version"] = "1.1"
        if "patch_history" not in op.metadata: op.metadata["patch_history"] = []
        op.metadata["patch_history"].append(patch_id)

    def _snapshot_operator(self, op_name: str):
        op = self.operators.get(op_name)
        if not op: return
        snapshot = {'version': op.metadata.get('version', '1.0'), 'preconditions': list(op.preconditions),
                    'effects': list(op.effects), 'metadata': copy.deepcopy(op.metadata),
                    'params': list(getattr(op, 'params', []))}
        if op_name not in self.operator_history: self.operator_history[op_name] = []
        self.operator_history[op_name].append(snapshot)