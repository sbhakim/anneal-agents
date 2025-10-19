# src/core/operators.py
"""
Defines the structure for symbolic operators used by the Planner and Executor.

Design goals for reliability:
- Preconditions must fail *gracefully* (return False, never raise) so the
  Executor can treat them as learnable failures instead of crashing.
- Operators can optionally declare `required_params`; these are validated
  before any user-defined preconditions run, preventing S1 fast-path on
  underspecified actions (e.g., BookFlight without date/payment).
- Optional `validators` allow richer checks (schema/range/domain) without
  conflating them with symbolic preconditions.
"""

from typing import List, Dict, Any, Callable, Optional


class Operator:
    """Represents a single action with preconditions and effects."""

    def __init__(
        self,
        name: str,
        params: List[str],
        preconditions: List[Callable],
        effects: List[Callable],
        required_params: Optional[List[str]] = None,
        validators: Optional[List[Callable]] = None,
    ):
        """
        Args:
            name: Operator identifier (e.g., 'BookFlight').
            params: Ordered parameter names the planner/executor will bind.
            preconditions: Functions (state, params) -> bool
            effects: Functions (state, params) -> state
            required_params: Hard requirements that must be present and non-empty
                             in `params` for the operator to be considered.
            validators: Functions (state, params) -> bool for schema/domain checks.
        """
        self.name = name
        self.params = params
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.required_params = required_params or []
        self.validators = validators or []
        self.metadata = {"version": "1.0", "trust_score": 1.0}

    def _check_required_params(self, params: Dict[str, Any]) -> bool:
        """Ensure all required params exist and are non-empty/non-None."""
        if not self.required_params:
            return True
        try:
            for key in self.required_params:
                if key not in params:
                    print(f"[Operator:{self.name}] Missing required param: '{key}'")
                    return False
                val = params.get(key)
                if val is None:
                    print(f"[Operator:{self.name}] Required param '{key}' is None")
                    return False
                if isinstance(val, str) and val.strip() == "":
                    print(f"[Operator:{self.name}] Required param '{key}' is empty")
                    return False
            return True
        except Exception as e:
            print(f"[Operator:{self.name}] Error checking required params: {e}")
            return False

    def _run_validators(self, state: "SymbolicState", params: Dict[str, Any]) -> bool:
        """Run optional schema/range/domain validators."""
        try:
            for validate in self.validators:
                if not validate(state, params):
                    return False
            return True
        except Exception as e:
            print(f"[Operator:{self.name}] Validator error: {e}")
            return False

    def check_preconditions(self, state: "SymbolicState", params: Dict[str, Any]) -> bool:
        """
        Checks preconditions against the state, now with grounded parameters.
        Returns:
            bool: True if all checks pass; False otherwise.
        Behavior:
            - Fails safely on any exception (returns False).
            - Enforces `required_params` prior to user preconditions.
            - Executes optional `validators` before symbolic preconditions.
        """
        # 1) Hard parameter grounding
        if not self._check_required_params(params):
            return False

        # 2) Optional validators (schema/domain)
        if self.validators and not self._run_validators(state, params):
            return False

        # 3) Symbolic preconditions
        try:
            # **FIX**: Do not pass 'self' to the individual precondition functions.
            # They are simple functions that only expect 'state' and 'params'.
            return all(bool(cond(state, params)) for cond in self.preconditions)
        except Exception as e:
            print(f"[Operator:{self.name}] Error checking preconditions: {e}")
            return False

    def apply_effects(self, state: "SymbolicState", params: Dict[str, Any]) -> "SymbolicState":
        """
        Applies the operator's effects to the state.
        Effects must be total and exception-safe from the caller's perspective:
        any raised exception is caught, logged, and the current state is returned.
        """
        new_state = state
        for effect in self.effects:
            try:
                new_state = effect(new_state, params)
            except Exception as e:
                print(f"[Operator:{self.name}] Error applying effect: {e}")
                # Fail-closed: do not mutate further on effect failure
                return new_state
        return new_state

    def __repr__(self):
        return f"Operator(name='{self.name}', required={self.required_params})"
