# src/core/operators.py
"""
Defines the structure for symbolic operators used by the Planner and Executor.
"""
from typing import List, Dict, Any, Callable

class Operator:
    """Represents a single action with preconditions and effects."""
    def __init__(self, name: str, params: List[str],
                 preconditions: List[Callable], effects: List[Callable]):
        self.name = name
        self.params = params
        self.preconditions = preconditions
        self.effects = effects
        self.metadata = {"version": "1.0", "trust_score": 1.0}

    def check_preconditions(self, state: "SymbolicState", params: Dict[str, Any]) -> bool:
        """
        Checks preconditions against the state, now with grounded parameters.
        """
        try:
            # **FIX**: Do not pass 'self' to the individual precondition functions.
            # They are simple functions that only expect 'state' and 'params'.
            return all(cond(state, params) for cond in self.preconditions)
        except Exception as e:
            print(f"Error checking precondition for {self.name}: {e}")
            return False

    def apply_effects(self, state: "SymbolicState", params: Dict[str, Any]) -> "SymbolicState":
        """Applies the operator's effects to the state."""
        for effect in self.effects:
            state = effect(state, params)
        return state

    def __repr__(self):
        return f"Operator(name='{self.name}')"