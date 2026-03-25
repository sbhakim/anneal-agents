# src/core/state.py
"""
Implements the typed symbolic state (Σ) for the ANNEAL architecture.
"""
from typing import Dict, Any
import copy
import re


class SymbolicState:
    """
    Represents the explicit symbolic state as a typed dictionary of predicates.
    """

    def __init__(self):
        """
        Initializes the state with a default configuration for the travel domain.
        """
        self.predicates = self._initialize_travel_domain()
        print(f"STATE: Initialized with {len(self.predicates)} predicates.")

    def _initialize_travel_domain(self) -> Dict[str, Any]:
        """Creates the default base state for the travel planning scenario."""
        return {
            # Payment information
            "payment_method": "CorporateCard:CC-5512",
            "payment_valid": True,
            "payment_invalid": False,  # ✅ added for compatibility with injected constraint state
            "card_type": "corporate",

            # Booking status
            "hotel_status": "available",
            "flight_status": "available",

            # **FIX**: Policy and blackout dates are part of the core state knowledge
            "corporate_card_policy": "blocked_on_blackout_dates",
            "blackout_dates": ["Apr 10", "Apr 11", "Apr 12", "Apr 13", "Apr 14", "Apr 15"],

            # Task-specific details (will be updated per task)
            "travel_dates": None,
            "destination": None,
        }

    def get(self, predicate: str, default: Any = None) -> Any:
        """
        Retrieve the value of a predicate from the state.
        """
        return self.predicates.get(predicate, default)

    def set(self, predicate: str, value: Any):
        """
        Set or update a predicate in the state.
        """
        self.predicates[predicate] = value
        # Using a direct print instead of a logger for simplicity in this file
        print(f"STATE: Updated '{predicate}' = {value}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to a plain dictionary for serialization or passing to other components.
        """
        return copy.deepcopy(self.predicates)

    def update_from_instruction(self, instruction: str):
        """
        **FIX**: Simulates a parser by updating the state from the instruction text.
        A real system would use a sophisticated NLP parser here. This makes the
        state dynamic and relevant to the current task.

        **MINIMUM RELIABILITY FIX (2026-01)**:
        Reset task-transient payment validity markers at the start of each task to
        prevent cross-task leakage like "(invalid) (invalid) (invalid)".
        """
        # --- Reset task-transient payment markers to avoid cross-task state pollution ---
        base_payment = "CorporateCard:CC-5512"

        # Normalize any previously-tagged payment method back to the base for each new task.
        # (If later you parse payment from instruction, you can override after this reset.)
        self.set("payment_method", base_payment)
        self.set("payment_valid", True)
        self.set("payment_invalid", False)

        # Simple regex to find dates like "Apr 10-12" or "June 20-22"
        date_match = re.search(r'(\w+\s\d+)-(\d+)', instruction)
        if date_match:
            month_day_start = date_match.group(1)
            day_end = date_match.group(2)
            full_date_range = f"{month_day_start}-{day_end}"
            self.set("travel_dates", full_date_range)
            print(f"STATE: Parsed travel dates: {self.get('travel_dates')}")
        else:
            # Handle single dates like "on June 13"
            single_date_match = re.search(r'on\s(\w+\s\d+)', instruction)
            if single_date_match:
                single_date = single_date_match.group(1)
                self.set("travel_dates", single_date)
                print(f"STATE: Parsed travel date: {self.get('travel_dates')}")
