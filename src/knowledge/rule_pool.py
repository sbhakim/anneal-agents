# src/knowledge/rule_pool.py
"""
Manages the collection of all available operators (R_rules).
UPDATED:
- Added the missing `list_preconditions` method for FDKA serialization.
- Fixed idempotency bug: Effects are now checked by function name.
- Added ValidPayment predicate handler to support payment validation patches.
- CRITICAL FIX: Resolved bug where attaching an attribute to a boolean return value
  caused an AttributeError. Methods now return a (success, skipped) tuple.
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


def check_valid_payment(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """
    NEW: Validates that payment method is active and supported.
    This is the learned precondition for: ValidPayment(payment).

    Checks:
    - Payment method exists and is not None
    - Payment is not marked as expired, invalid, or blocked
    - Payment follows expected format (contains "Card")

    In production, this would integrate with a payment validation API.
    """
    payment = params.get("payment", state.get("payment_method", ""))

    # Check 1: Payment method must exist
    if not payment or payment == "None" or payment == "":
        print("PRECONDITION CHECK: ❌ ValidPayment - No payment method provided")
        return False

    # Check 2: Convert to string for validation
    payment_str = str(payment)

    # Check 3: Must contain "Card" to be a valid payment method
    if "Card" not in payment_str:
        print(f"PRECONDITION CHECK: ❌ ValidPayment - Invalid format: {payment_str}")
        return False

    # Check 4: Must not contain invalid/expired markers
    invalid_markers = ["expired", "invalid", "blocked", "declined", "suspended"]
    if any(marker in payment_str.lower() for marker in invalid_markers):
        print(f"PRECONDITION CHECK: ❌ ValidPayment - Payment marked as invalid: {payment_str}")
        return False

    # Check 5: Optional - Verify payment method is in known valid formats
    # Format examples: "CorporateCard:CC-5512", "PersonalCard:PC-1134"
    valid_formats = ["CorporateCard:", "PersonalCard:", "DebitCard:", "CreditCard:"]
    has_valid_format = any(fmt in payment_str for fmt in valid_formats)

    if not has_valid_format:
        print(f"PRECONDITION CHECK: ⚠️  ValidPayment - Unknown format (allowing): {payment_str}")
        # Allow unknown formats to avoid false negatives

    print(f"PRECONDITION CHECK: ✅ ValidPayment({payment_str}) - Check passes.")
    return True


# ========================================================================
# DYNAMIC PREDICATE & EFFECT FACTORIES (The Core of Symbolic Learning)
# ========================================================================

class PredicateFactory:
    """Dynamically creates predicate functions from patch details strings."""

    @staticmethod
    def create_precondition(details: str, operator_name: str) -> Optional[Callable]:
        details_clean = (details or "").strip()

        # Handler 1: Not(BlockedCard(...)) - Blackout date validation
        if "Not(BlockedCard" in details_clean:
            return check_not_blocked_card

        # Handler 2: NetworkAvailable() - Network connectivity check
        if "NetworkAvailable" in details_clean:
            def check_network_available(state: SymbolicState, params: Dict) -> bool:
                network_ok = state.get("network_available", True)
                print(f"PRECONDITION CHECK: {'✅' if network_ok else '❌'} NetworkAvailable")
                return network_ok

            check_network_available.__name__ = "check_network_available"
            return check_network_available

        # Handler 3: ValidPayment(...) - Payment method validation (NEW)
        if "ValidPayment" in details_clean:
            # Return the base function directly - it's already defined
            return check_valid_payment

        # Handler 4: Simple predicate names without arguments (e.g., 'is_flight_available')
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

        # Handler 5: Generic predicate pattern Predicate(args)
        match = re.search(r'([A-Z][a-zA-Z]+)\s*\(', details_clean)
        if match:
            predicate_name = match.group(1)

            # Special case: If we recognize the predicate name but it wasn't caught above
            if predicate_name == "ValidPayment":
                return check_valid_payment
            if predicate_name == "BlockedCard":
                return check_not_blocked_card

            # Generic fallback for unknown predicates
            def generic_check(state: SymbolicState, params: Dict) -> bool:
                result = state.get(f"{predicate_name.lower()}_ok", True)
                print(f"PRECONDITION CHECK: {'✅' if result else '❌'} {predicate_name} (generic)")
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

        # Handler 1: Conditional effects with network check
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
        return {
            "operators": copy.deepcopy(self.operators),
            "operator_history": copy.deepcopy(self.operator_history),
            "stats": copy.deepcopy(self.stats)
        }

    def restore(self, snap: Dict[str, Any]):
        self.operators = snap["operators"]
        self.operator_history = snap.get("operator_history", {})
        self.stats = snap.get("stats", self.stats)

    def get_operator(self, name: str) -> Optional[Operator]:
        return self.operators.get(name)

    # ========================================================================
    # ** ADDED METHOD **
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
        if not op:
            return False, "UnknownOperator"

        for pred in op.preconditions:
            try:
                if not bool(pred(state, params)):
                    return False, getattr(pred, "__name__", "anonymous")
            except Exception as e:
                return False, getattr(pred, "__name__", "anonymous")

        return True, None

    def update_operator(self, patch: Dict[str, Any]) -> bool:
        op_name = patch.get("operator")
        action = patch.get("action")
        details = patch.get("details", "")
        just = patch.get("justification", "")
        patch_id = patch.get("id", "unknown")

        op = self.operators.get(op_name)
        if not op:
            return False

        self._snapshot_operator(op_name)
        print(f"\n  🔧 RULE_POOL: Applying patch {patch_id} to {op_name}")
        print(f"     Action: {action}, Details: {details}")

        success, skipped = False, False
        if action == "ADD_PRECONDITION":
            success, skipped = self._add_precondition(op, details, just)
        elif action == "REFINE_EFFECT":
            success, skipped = self._refine_effect(op, details, just)
        elif action == "UPDATE_TOOL_SCHEMA":
            # This action doesn't have a duplicate check, so it's never skipped
            success = self._update_schema(op, details, just)
            skipped = False

        if success:
            # Only update metadata if a change was actually made (not skipped)
            if not skipped:
                self._update_metadata(op, patch_id)
                self.stats['total_patches_applied'] += 1
                self.stats['patches_by_type'][action] = self.stats['patches_by_type'].get(action, 0) + 1
                self.stats['operators_modified'].add(op_name)
            print(f"  ✅ RULE_POOL: Patch {patch_id} successfully applied (or was already present).")
            return True
        else:
            print(f"  ❌ RULE_POOL: Patch {patch_id} failed to apply.")
            return False

    def _add_precondition(self, op: Operator, details: str, just: str) -> Tuple[bool, bool]:
        new_pre = self.predicate_factory.create_precondition(details, op.name)
        if not new_pre:
            return False, False

        pred_name = getattr(new_pre, "__name__", "anon")

        # Check for duplicates
        if pred_name in {getattr(fn, "__name__", "") for fn in op.preconditions}:
            print(f"  ℹ️ Precondition '{pred_name}' already exists. Skipping duplicate.")
            return True, True

        op.preconditions.append(new_pre)
        self.learned_predicates.add(pred_name)
        print(f"  ✅ Added precondition '{pred_name}' to {op.name}.")
        return True, False

    def _refine_effect(self, op: Operator, details: str, just: str) -> Tuple[bool, bool]:
        new_eff = self.effect_factory.create_effect(details, op.name)
        if not new_eff:
            print(f"  ⚠️ Failed to create effect from: {details[:60]}")
            return False, False

        effect_name = getattr(new_eff, "__name__", "anon_eff")

        # Check BOTH function name AND source code for true duplicates
        existing_names = {getattr(fn, "__name__", "") for fn in op.effects}
        if effect_name in existing_names:
            # Double-check if it's truly identical by comparing details
            existing_details = [getattr(fn, "__details__", "") for fn in op.effects]
            if any(details in ed or ed in details for ed in existing_details if ed):
                print(f"  ℹ️ Effect '{effect_name}' with similar details already exists. Skipping duplicate.")
                return True, True

        # Attach details for future comparison
        setattr(new_eff, "__details__", details)

        op.effects.append(new_eff)
        self.learned_effects.add(effect_name)
        print(f"  ✅ Added effect '{effect_name}' to {op.name}.")
        return True, False

    def _update_schema(self, op: Operator, details: str, just: str) -> bool:
        op.metadata['schema_update'] = details
        return True

    def _update_metadata(self, op: Operator, patch_id: str):
        try:
            op.metadata["version"] = f"{float(op.metadata.get('version', '1.0')) + 0.1:.1f}"
        except:
            op.metadata["version"] = "1.1"

        if "patch_history" not in op.metadata:
            op.metadata["patch_history"] = []
        op.metadata["patch_history"].append(patch_id)

    def _snapshot_operator(self, op_name: str):
        op = self.operators.get(op_name)
        if not op:
            return

        snapshot = {
            'version': op.metadata.get('version', '1.0'),
            'preconditions': list(op.preconditions),
            'effects': list(op.effects),
            'metadata': copy.deepcopy(op.metadata),
            'params': list(getattr(op, 'params', []))
        }

        if op_name not in self.operator_history:
            self.operator_history[op_name] = []
        self.operator_history[op_name].append(snapshot)


# ========================================================================
# STANDALONE TESTING
# ========================================================================

if __name__ == "__main__":
    """
    Test the enhanced RulePool with ValidPayment support.
    """
    print("=" * 70)
    print("Testing Enhanced RulePool with ValidPayment")
    print("=" * 70)

    # Initialize RulePool
    rule_pool = RulePool("dummy_path.json")

    # Test 1: ValidPayment predicate creation
    print("\n[Test 1: ValidPayment Predicate Creation]")
    predicate = rule_pool.predicate_factory.create_precondition(
        "ValidPayment(payment)",
        "BookHotel"
    )

    if predicate:
        print(f"✅ Predicate created: {predicate.__name__}")

        # Test with valid payment
        mock_state = SymbolicState()
        mock_state.set("payment_method", "CorporateCard:CC-5512")
        result = predicate(mock_state, {"payment": "CorporateCard:CC-5512"})
        print(f"   Valid payment test: {'✅ PASS' if result else '❌ FAIL'}")

        # Test with invalid payment
        result2 = predicate(mock_state, {"payment": "ExpiredCard:EC-9999"})
        print(f"   Invalid payment test: {'✅ PASS' if not result2 else '❌ FAIL'}")
    else:
        print("❌ Failed to create predicate")

    # Test 2: Patch application with ValidPayment
    print("\n[Test 2: Patch Application]")
    patch = {
        "id": "test-patch-001",
        "operator": "BookHotel",
        "action": "ADD_PRECONDITION",
        "details": "ValidPayment(payment)",
        "justification": "Validate payment before booking"
    }

    success = rule_pool.update_operator(patch)
    print(f"Patch application: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Verify precondition was added
    hotel_op = rule_pool.get_operator("BookHotel")
    precond_names = [getattr(p, "__name__", "unknown") for p in hotel_op.preconditions]
    print(f"Current preconditions: {precond_names}")

    if "check_valid_payment" in precond_names:
        print("✅ ValidPayment precondition successfully added")
    else:
        print("❌ ValidPayment precondition not found")

    print("\n" + "=" * 70)
    print("✅ Enhanced RulePool testing complete")
    print("=" * 70)