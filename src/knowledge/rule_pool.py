# src/knowledge/rule_pool.py
"""
Manages the collection of all available operators (R_rules).
UPDATED:
- Added the missing `list_preconditions` method for FDKA serialization.
- Fixed idempotency bug: Effects are now checked by function name.
- Added ValidPayment predicate handler to support payment validation patches.
- CRITICAL FIX: Resolved bug where attaching an attribute to a boolean return value
  caused an AttributeError. Methods now return a (success, skipped) tuple.
- RECOVERY FIX: Updated EffectFactory to handle complex LLM patterns like
  'IfThen(ApiTimeoutRetry(api), ExecuteTool())' to resolve evolution halts.

MINIMUM RELIABILITY UPDATE (2026-01):
- Do not treat duplicate/no-op patches as "applied".
  Previously, skipped duplicates returned True from update_operator(), which inflated
  acceptance/commit metrics. Now, duplicates return False (no-op), so the system
  does not count them as committed.
- Adds a lightweight patch signature cache (operator+action+normalized details)
  to de-duplicate semantically identical patches even if factories vary formatting.
"""
from typing import Dict, Any, Callable, List, Tuple, Optional, Set
import re
import copy
import hashlib
from ..core.operators import Operator
from ..core.state import SymbolicState


# ========================================================================
# BASE PREDICATE LIBRARY (Initial Knowledge)
# ========================================================================

def is_card_valid(state: SymbolicState, params: Dict[str, Any]) -> bool:
    """
    Minimal payment validity predicate used by baseline operators.

    Critical: must fail when the system/injector marks payment invalid, otherwise
    constraint injections (payment_invalid / "(invalid)") will never trigger a
    PreconditionUnmet and the agent will "succeed" incorrectly.
    """
    payment = params.get("payment", state.get("payment_method"))
    if payment is None or payment == "" or payment == "None":
        return False

    # Honor explicit state flags used by the executor/injector.
    if state.get("payment_invalid") is True:
        return False
    if state.get("payment_valid") is False:
        return False

    p = str(payment)
    if "(invalid)" in p.lower():
        return False

    # Lightweight safety: common invalid markers (kept minimal).
    invalid_markers = ("expired", "blocked", "declined", "suspended")
    if any(m in p.lower() for m in invalid_markers):
        return False

    return True


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
    Learned predicate for: ValidPayment(payment).
    """
    # Honor explicit state flags first (used by injector/executor).
    if state.get("payment_invalid") is True:
        print("PRECONDITION CHECK: ❌ ValidPayment - State marked payment_invalid=True")
        return False
    if state.get("payment_valid") is False:
        print("PRECONDITION CHECK: ❌ ValidPayment - State marked payment_valid=False")
        return False

    payment = params.get("payment", state.get("payment_method", ""))

    # Check 1: Payment method must exist
    if not payment or payment == "None" or payment == "":
        print("PRECONDITION CHECK: ❌ ValidPayment - No payment method provided")
        return False

    payment_str = str(payment)

    # Check 2: Must contain "Card" to be a valid payment method
    if "Card" not in payment_str:
        print(f"PRECONDITION CHECK: ❌ ValidPayment - Invalid format: {payment_str}")
        return False

    # Check 3: Must not contain invalid/expired markers
    invalid_markers = ["expired", "invalid", "blocked", "declined", "suspended"]
    if any(marker in payment_str.lower() for marker in invalid_markers):
        print(f"PRECONDITION CHECK: ❌ ValidPayment - Payment marked as invalid: {payment_str}")
        return False

    # Check 4: Optional - Verify payment method is in known valid formats
    valid_formats = ["CorporateCard:", "PersonalCard:", "DebitCard:", "CreditCard:"]
    has_valid_format = any(fmt in payment_str for fmt in valid_formats)

    if not has_valid_format:
        print(f"PRECONDITION CHECK: ⚠️  ValidPayment - Unknown format (allowing): {payment_str}")

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
        details_l = details_clean.lower()

        # Handler 1: Not(BlockedCard(...)) - Blackout date validation (case-insensitive)
        if "not(blockedcard" in details_l:
            return check_not_blocked_card

        # Handler 2: NetworkAvailable() - Network connectivity check (case-insensitive)
        if "networkavailable" in details_l:
            def check_network_available(state: SymbolicState, params: Dict) -> bool:
                network_ok = state.get("network_available", True)
                print(f"PRECONDITION CHECK: {'✅' if network_ok else '❌'} NetworkAvailable")
                return network_ok

            check_network_available.__name__ = "check_network_available"
            return check_network_available

        # Handler 3: ValidPayment(...) - Payment method validation (case-insensitive)
        if "validpayment" in details_l:
            return check_valid_payment

        # Handler 4: Simple predicate names without arguments (e.g., 'is_flight_available')
        if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', details_clean):
            if details_clean in globals() and callable(globals()[details_clean]):
                print(f"  ✅ PREDICATE FACTORY: Matched existing predicate '{details_clean}'.")
                return globals()[details_clean]
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

            if predicate_name == "ValidPayment":
                return check_valid_payment
            if predicate_name == "BlockedCard":
                return check_not_blocked_card

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
        details_l = details_clean.lower()

        # Handler 1: Conditional effects with network check (case-insensitive)
        if "ifthen" in details_l and "networkavailable" in details_l:
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

        # Handler 2: API Timeout Retry logic (FDKA evolved pattern) (case-insensitive)
        # Recognizes: IfThen(ApiTimeoutRetry(api), ExecuteTool())
        #
        # IMPORTANT: This is a recovery-only MARKER (no-op). The executor owns retry semantics.
        if "apitimeoutretry" in details_l or "timeoutretry" in details_l:
            def timeout_retry_effect(state: SymbolicState, params: Dict) -> SymbolicState:
                print(f"EFFECT: Triggering evolution logic -> {details_clean}")
                print("EFFECT: ApiTimeoutRetry marker (executor performs the retry).")
                return state

            timeout_retry_effect.__name__ = "timeout_retry_recovery_effect"
            setattr(timeout_retry_effect, "__recovery_only__", True)
            setattr(timeout_retry_effect, "__details__", details_clean)
            return timeout_retry_effect

        # Handler 3: Simple state updates (original regex)
        match = re.match(r"(\w+)\((\w+)\)", details_clean)
        if match:
            pred, target = match.groups()

            def simple_effect(state: SymbolicState, params: Dict) -> SymbolicState:
                state.set(f"{target}_status", pred)
                print(f"EFFECT: Updated {target}_status to {pred}")
                return state

            simple_effect.__name__ = f"set_{target}_{pred}"
            return simple_effect

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
        self._patch_signature_cache: Set[str] = set()
        self.load_operators()

    def load_operators(self):
        book_hotel_op = Operator(
            "BookHotel",
            ["location", "dates", "payment"],
            [is_card_valid, is_hotel_available],
            [book_hotel_effect]
        )
        book_hotel_op.metadata = {"version": "1.0"}
        self.operators[book_hotel_op.name] = book_hotel_op
        self._snapshot_operator(book_hotel_op.name)

        book_flight_op = Operator(
            "BookFlight",
            ["origin", "destination", "date"],
            [is_card_valid, is_flight_available],
            [book_flight_effect]
        )
        book_flight_op.metadata = {"version": "1.0"}
        self.operators[book_flight_op.name] = book_flight_op
        self._snapshot_operator(book_flight_op.name)

        print(f"RULE_POOL: Loaded {len(self.operators)} initial operators: {list(self.operators.keys())}")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "operators": copy.deepcopy(self.operators),
            "operator_history": copy.deepcopy(self.operator_history),
            "stats": copy.deepcopy(self.stats),
            "_patch_signature_cache": copy.deepcopy(self._patch_signature_cache),
        }

    def restore(self, snap: Dict[str, Any]):
        self.operators = snap["operators"]
        self.operator_history = snap.get("operator_history", {})
        self.stats = snap.get("stats", self.stats)
        self._patch_signature_cache = snap.get("_patch_signature_cache", self._patch_signature_cache)

    def get_operator(self, name: str) -> Optional[Operator]:
        return self.operators.get(name)

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
            except Exception:
                return False, getattr(pred, "__name__", "anonymous")

        return True, None

    def _normalize_details(self, details: str) -> str:
        """Normalize patch details for stable signature comparisons."""
        s = (details or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    def _patch_signature(self, op_name: str, action: str, details: str) -> str:
        """Create a short, stable signature for a patch."""
        base = f"{op_name}||{action}||{self._normalize_details(details)}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

    def update_operator(self, patch: Dict[str, Any]) -> bool:
        op_name = patch.get("operator")
        action = patch.get("action")
        details = patch.get("details", "")
        just = patch.get("justification", "")
        patch_id = patch.get("id", "unknown")

        op = self.operators.get(op_name)
        if not op:
            return False

        sig = self._patch_signature(op_name, action, details)
        if sig in self._patch_signature_cache:
            print(f"\n  ℹ️ RULE_POOL: Patch {patch_id} is a duplicate/no-op (sig={sig}). Not applying.")
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
            success = self._update_schema(op, details, just)
            skipped = False

        if success and skipped:
            print(f"  ℹ️ RULE_POOL: Patch {patch_id} made no changes (already present). Not counting as applied.")
            return False

        if success:
            self._update_metadata(op, patch_id)
            self.stats['total_patches_applied'] += 1
            self.stats['patches_by_type'][action] = self.stats['patches_by_type'].get(action, 0) + 1
            self.stats['operators_modified'].add(op_name)
            self._patch_signature_cache.add(sig)
            print(f"  ✅ RULE_POOL: Patch {patch_id} successfully applied.")
            return True

        print(f"  ❌ RULE_POOL: Patch {patch_id} failed to apply.")
        return False

    def _add_precondition(self, op: Operator, details: str, just: str) -> Tuple[bool, bool]:
        new_pre = self.predicate_factory.create_precondition(details, op.name)
        if not new_pre:
            return False, False

        pred_name = getattr(new_pre, "__name__", "anon")

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

        # Safety: ensure timeout retry effects are tagged recovery-only even if factories drift.
        if ("timeout" in effect_name.lower() or "retry" in effect_name.lower()) and getattr(new_eff, "__recovery_only__", None) is None:
            setattr(new_eff, "__recovery_only__", True)

        existing_names = {getattr(fn, "__name__", "") for fn in op.effects}
        if effect_name in existing_names:
            existing_details = [getattr(fn, "__details__", "") for fn in op.effects]
            if any(details in ed or ed in details for ed in existing_details if ed):
                print(f"  ℹ️ Effect '{effect_name}' with similar details already exists. Skipping duplicate.")
                return True, True
            print(f"  ℹ️ Effect '{effect_name}' already exists. Skipping duplicate.")
            return True, True

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
        except Exception:
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

    rule_pool = RulePool("dummy_path.json")

    print("\n[Test 1: ValidPayment Predicate Creation]")
    predicate = rule_pool.predicate_factory.create_precondition(
        "ValidPayment(payment)",
        "BookHotel"
    )

    if predicate:
        print(f"✅ Predicate created: {predicate.__name__}")

        mock_state = SymbolicState()
        mock_state.set("payment_method", "CorporateCard:CC-5512")
        result = predicate(mock_state, {"payment": "CorporateCard:CC-5512"})
        print(f"   Valid payment test: {'✅ PASS' if result else '❌ FAIL'}")

        result2 = predicate(mock_state, {"payment": "ExpiredCard:EC-9999"})
        print(f"   Invalid payment test: {'✅ PASS' if not result2 else '❌ FAIL'}")
    else:
        print("❌ Failed to create predicate")

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
