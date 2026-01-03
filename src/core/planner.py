# src/core/planner.py
"""
Compiles a high-level instruction into a symbolic plan.
Corresponds to the Planner (Π) in the architecture.
"""
from typing import List, Dict, Any, Optional
import re
import copy

# Use relative import as requested
from ..knowledge.rule_pool import RulePool


class Planner:
    """
    A planner that creates and repairs sequences of operators to fulfill an instruction.
    """

    def __init__(self, config: Dict[str, Any], rule_pool: RulePool):
        """
        Initializes the Planner.

        Args:
            config: Planner-specific configuration.
            rule_pool: The shared, persistent RulePool instance.
        """
        self.config = config
        self.rule_pool = rule_pool

    def compile(self, instruction: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes a natural language instruction and returns an initial plan with grounded operators.
        """
        print(f"PLANNER: Compiling instruction -> '{instruction}'")
        plan: List[Dict[str, Any]] = []
        instruction_lower = instruction.lower()

        # Intent detection (minimal but fixes "cheapest travel option" PlanningFailed).
        wants_hotel = any(k in instruction_lower for k in ("hotel", "accommodation", "stay", "room"))
        wants_flight = any(k in instruction_lower for k in ("flight", "fly", "airfare", "plane", "ticket"))
        wants_cheapest = any(k in instruction_lower for k in ("cheapest", "lowest cost", "budget")) and any(
            k in instruction_lower for k in ("travel", "option", "itinerary")
        )

        # If user asks for cheapest travel option but doesn't explicitly say "flight/hotel",
        # default to booking a flight (PoC default).
        if wants_cheapest and not (wants_hotel or wants_flight):
            wants_flight = True

        # Plan flight bookings
        if wants_flight:
            op = self.rule_pool.get_operator("BookFlight")
            if op:
                params = self._ground_flight_params(instruction, state)
                if params:
                    grounded_op = {"operator": op, "params": params}
                    plan.append(grounded_op)
                    print(f"PLANNER: Added 'BookFlight' to plan with params: {params}")
            else:
                print("PLANNER: Warning - 'BookFlight' operator not found in Rule Pool.")

        # Plan hotel bookings
        if wants_hotel:
            op = self.rule_pool.get_operator("BookHotel")
            if op:
                params = self._ground_hotel_params(instruction, state)
                if params:
                    grounded_op = {"operator": op, "params": params}
                    plan.append(grounded_op)
                    print(f"PLANNER: Added 'BookHotel' to plan with params: {params}")
            else:
                print("PLANNER: Warning - Could not find 'BookHotel' operator in Rule Pool.")

        if plan:
            operator_names = [p["operator"].name for p in plan]
            print(f"PLANNER: Plan created with {len(plan)} step(s) -> {operator_names}")
        else:
            print("PLANNER: ❌ No plan generated (no matching operators or failed grounding).")

        return plan

    def replan(self, instruction: str, state: Dict[str, Any], failed_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attempts to generate an alternative plan when a precondition fails.
        This is the "Local Repair" mechanism discussed.
        """
        print("PLANNER: Attempting to replan after precondition failure...")

        for i, step in enumerate(failed_plan):
            if step["operator"].name == "BookHotel":
                if "CorporateCard" in step["params"].get("payment", ""):
                    print("PLANNER: Detected failure with Corporate Card. Attempting to switch payment method.")

                    new_plan = copy.deepcopy(failed_plan)

                    # Prefer "new corporate" if the instruction implies it; otherwise fallback to personal.
                    if "new corporate" in instruction.lower():
                        alternative_payment = "CorporateCard:CC-NEW"
                    else:
                        alternative_payment = "PersonalCard:PC-1134"

                    new_plan[i]["params"]["payment"] = alternative_payment
                    print(f"PLANNER: New plan generated with payment method: {alternative_payment}")
                    return new_plan

        print("PLANNER: ❌ No local repair strategy found for this failure.")
        return failed_plan

    def _ground_payment_method(self, instruction: str, state: Dict[str, Any]) -> str:
        # If the user explicitly asks for a new corporate account/card, represent it as a distinct identifier.
        il = instruction.lower()
        if "new corporate account" in il or "new corporate card" in il or "new corporate" in il:
            return "CorporateCard:CC-NEW"
        return state.get("payment_method", "CorporateCard:CC-5512")

    def _extract_date(self, instruction: str, state: Dict[str, Any], *, field_name: str) -> Optional[str]:
        """
        Minimal date extractor to reduce PlanningFailed:
        supports 'on June 7', 'around June 7', and ranges like 'June 1-3'.
        """
        # Range like "June 1-3"
        date_match = re.search(r"(\b[A-Za-z]+\s\d{1,2}-\d{1,2}\b)", instruction)
        if date_match:
            return date_match.group(1).strip()

        # "on June 7"
        on_match = re.search(r"\bon\s([A-Za-z]+\s\d{1,2})\b", instruction, re.IGNORECASE)
        if on_match:
            return on_match.group(1).strip()

        # "around June 7"
        around_match = re.search(r"\baround\s([A-Za-z]+\s\d{1,2})\b", instruction, re.IGNORECASE)
        if around_match:
            return around_match.group(1).strip()

        # Fall back to state travel_dates if present
        td = state.get("travel_dates")
        if isinstance(td, str) and td.strip():
            if "-" in td:
                return td.split("-")[0].strip()
            return td.strip()

        return None

    def _ground_hotel_params(self, instruction: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simulates a Neural Parser for grounding hotel booking details.
        """
        params: Dict[str, Any] = {}

        loc_match = re.search(r"in\s([\w\s]+?)(?=\sfor|\son|\saround|$)", instruction, re.IGNORECASE)
        if loc_match:
            params["location"] = loc_match.group(1).strip()

        dates = self._extract_date(instruction, state, field_name="dates")
        if dates:
            params["dates"] = dates

        params["payment"] = self._ground_payment_method(instruction, state)

        return params if "location" in params and "dates" in params else None

    def _ground_flight_params(self, instruction: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Simulates a Neural Parser for grounding flight booking details.

        Minimal correctness tightening:
        - require destination
        - require date (from instruction or state.travel_dates)
        - infer origin from state.flight_prices (cheapest) when missing
        """
        params: Dict[str, Any] = {}

        origin_match = re.search(r"from\s([\w\s]+?)(?=\sto|\son|\saround|$)", instruction, re.IGNORECASE)
        if origin_match:
            params["origin"] = origin_match.group(1).strip()

        # Primary destination pattern (FIX: stop before 'from ...' so we don't capture "X from Y")
        dest_match = re.search(
            r"to\s([\w\s]+?)(?=\sfrom|\sfor|\son|\saround|$)",
            instruction,
            re.IGNORECASE
        )
        if dest_match:
            params["destination"] = dest_match.group(1).strip()
        else:
            # Fallback: travel to/in <place> (helps "cheapest travel option" prompts)
            fallback_dest = re.search(
                r"(?:travel|trip)\s(?:to|in)\s([\w\s]+?)(?=\sfrom|\sfor|\son|\saround|$)",
                instruction,
                re.IGNORECASE
            )
            if fallback_dest:
                params["destination"] = fallback_dest.group(1).strip()

        date = self._extract_date(instruction, state, field_name="date")
        if date:
            params["date"] = date

        if "origin" not in params:
            fp = state.get("flight_prices")
            if isinstance(fp, dict) and fp:
                try:
                    params["origin"] = min(fp.keys(), key=lambda k: fp.get(k, float("inf")))
                except Exception:
                    pass

        # Include payment so preconditions can see the same identity consistently.
        params["payment"] = self._ground_payment_method(instruction, state)

        required = ("origin", "destination", "date")
        return params if all(k in params and params[k] for k in required) else None
