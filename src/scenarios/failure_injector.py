# src/scenarios/failure_injector.py
"""
Implements controlled failure injection for the evaluation scenario.
Corresponds to the dynamic environment simulation in Section XII-D.

UPDATED:
- Added `patched_operators` tracking to allow the system to "win" against faults.
- Modified `should_fail` to respect committed patches, enabling TTA metrics.
- Added `mark_operator_patched` to bridge the gap between RulePool and Environment.

MINIMUM RELIABILITY UPDATE (2026-01):
- Make payment-related failures conditional on the *specific* payment identity so local repair
  (switching away from a bad corporate card) can actually recover.
- Cache failure details per (task, operator, payment_key) to keep injections stable
  across retries/attempts within a run while allowing different cards to experience
  different injected failures.
"""
from typing import Dict, Any, List, Set, Optional, Tuple
import random
import uuid


class FailureInjector:
    """
    Injects failures into tasks at a controlled rate to simulate open-world dynamics.
    Supports policy flips and ensures minimum failures in a prefix for visible adaptation.
    """

    def __init__(
            self,
            failure_rate: float = 0.3,
            policy_flip_at: int = 25,
            horizon: int = 50,
            blackout_dates: List[str] = None,
            min_failures_in_prefix: int = 3,
            prefix_len: int = 10,
    ):
        self.failure_rate = failure_rate
        self.policy_flip_at = policy_flip_at
        self.horizon = horizon
        self.blackout_dates = blackout_dates or []
        self.min_failures_in_prefix = min_failures_in_prefix
        self.prefix_len = min(prefix_len, horizon)

        # Track which operators have been successfully patched by FDKA
        self.patched_operators: Set[str] = set()

        # Cache stable failure details for consistency across attempts.
        # Keyed by (task, operator, payment_key) where payment_key is specific when possible.
        self._failure_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

        # Precompute failing tasks
        self.failing_tasks = self._select_failing_tasks()
        print(f"FAILURE_INJECTOR: Initialized. {len(self.failing_tasks)} tasks will fail.")

    def _select_failing_tasks(self) -> List[int]:
        """
        Select tasks that will fail, ensuring min_failures in prefix.
        """
        failing = set()

        # Force min_failures in prefix
        prefix_tasks = list(range(self.prefix_len))
        random.shuffle(prefix_tasks)
        for task in prefix_tasks[:self.min_failures_in_prefix]:
            failing.add(task)

        # Add remaining failures randomly across horizon
        target = int(self.failure_rate * self.horizon)
        all_tasks = list(range(self.horizon))
        random.shuffle(all_tasks)
        for task in all_tasks:
            if len(failing) >= target:
                break
            if task not in failing:
                failing.add(task)

        return sorted(list(failing))

    def mark_operator_patched(self, op_name: str):
        """
        Registers an operator as evolved/patched.
        Subsequent calls to should_fail for this operator will return False.
        """
        if op_name not in self.patched_operators:
            print(f"FAILURE_INJECTOR: Operator '{op_name}' marked as PATCHED. Faults will be suppressed.")
            self.patched_operators.add(op_name)

    def _payment_string(self, params: Optional[Dict[str, Any]], state: Optional[Any]) -> str:
        payment = None
        if isinstance(params, dict):
            payment = params.get("payment")
        if payment is None and state is not None:
            try:
                payment = state.get("payment_method") if hasattr(state, "get") else None
            except Exception:
                payment = None
        return str(payment or "")

    def _payment_kind(self, payment_str: str) -> str:
        if "CorporateCard" in payment_str:
            return "corporate"
        if "PersonalCard" in payment_str:
            return "personal"
        return "other"

    def _payment_key(self, params: Optional[Dict[str, Any]], state: Optional[Any]) -> str:
        """
        Prefer a specific, stable payment identity when present so switching cards changes the cache key.
        Falls back to a coarse kind when unknown.
        """
        p = self._payment_string(params, state)

        # Extract the token after "XCard:" if present; e.g., "CorporateCard:CC-NEW"
        m = None
        try:
            m = p.split()[0]  # strip trailing markers like "(invalid)"
        except Exception:
            m = p

        if ":" in m:
            head = m.split(":", 1)[0]
            if head.endswith("Card"):
                return m  # specific identity: "CorporateCard:CC-5512", "CorporateCard:CC-NEW", ...

        return self._payment_kind(p)

    def _cached_failure(self, task_id: int, op_name: str, payment_key: str) -> Dict[str, Any]:
        key = (task_id, op_name, payment_key)
        if key in self._failure_cache:
            return self._failure_cache[key]

        # Decide which failures are allowed for this payment identity.
        # Critical: invalid_payment (PAY-401) should ONLY target the baseline corporate card.
        allow_invalid_payment = (payment_key == "CorporateCard:CC-5512")

        r = random.random()
        if r < 0.50:
            error_type = "PreconditionUnmet"
            message = "Corporate cards blocked for reservations during blackout dates"
            policy_ref = "H-23"
            category = "blackout_blocked_card"
        elif allow_invalid_payment and r < 0.75:
            error_type = "PreconditionUnmet"
            message = "Payment method is invalid or expired"
            policy_ref = "PAY-401"
            category = "invalid_payment"
        else:
            error_type = "ToolError"
            message = "Booking API timeout"
            policy_ref = "API-503"
            category = "api_timeout"

        # Apply policy flip: expand blackout dates post-flip (only meaningful for blackout failures).
        if task_id >= self.policy_flip_at and category == "blackout_blocked_card":
            if "June" not in [d.split()[0] for d in self.blackout_dates]:
                self.blackout_dates.extend(["June 1", "June 2"])
            message += " (Policy updated: Extended blackout periods)"

        details = {
            "error": error_type,
            "operator": op_name,
            "message": message,
            "policy_ref": policy_ref,
            "category": category,
            "task_id": task_id,
            "trace_id": f"trace-{uuid.uuid4().hex[:8]}",
        }
        self._failure_cache[key] = details
        return details

    def should_fail(self, task_id: int, op_name: str, params: Dict[str, Any] = None, state: Any = None) -> bool:
        """
        Determines if the current task and operator should fail.

        UPDATED:
        - Backward compatible (params/state optional).
        - Conditional: failures depend on payment identity so local repair can recover.
        """
        if op_name in self.patched_operators:
            return False
        if task_id not in self.failing_tasks:
            return False

        p_str = self._payment_string(params, state)
        p_key = self._payment_key(params, state)
        info = self._cached_failure(task_id, op_name, p_key)

        # Blackout-blocked corporate cards should not fail once payment is non-corporate.
        if info.get("category") == "blackout_blocked_card":
            if self._payment_kind(p_str) != "corporate":
                return False

        # Invalid-payment failures should ONLY target the baseline corporate card.
        if info.get("category") == "invalid_payment":
            if p_key != "CorporateCard:CC-5512":
                return False

        return True

    def get_failure_details(self, task_id: int, op_name: str, params: Dict[str, Any] = None, state: Any = None) -> Dict[str, Any]:
        """
        Returns stable failure details for this (task, operator, payment_key).
        """
        p_key = self._payment_key(params, state)
        return dict(self._cached_failure(task_id, op_name, p_key))
