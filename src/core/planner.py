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

        wants_hotel = any(k in instruction_lower for k in ("hotel", "accommodation", "stay", "room"))
        wants_flight = any(k in instruction_lower for k in ("flight", "fly", "airfare", "plane", "ticket"))
        wants_cheapest = any(k in instruction_lower for k in ("cheapest", "lowest cost", "budget")) and any(
            k in instruction_lower for k in ("travel", "option", "itinerary")
        )

        if wants_cheapest and not (wants_hotel or wants_flight):
            wants_flight = True

        hotel_step = None
        flight_step = None

        # Ground hotel first when both are requested so flight can inherit location/dates if needed.
        if wants_hotel:
            op = self.rule_pool.get_operator("BookHotel")
            if op:
                params, missing = self._ground_hotel_params(instruction, state)
                hotel_step = {"operator": op, "params": params}
                if missing:
                    hotel_step["meta"] = {"needs_regrounding": missing}
                plan.append(hotel_step)
                print(f"PLANNER: Added 'BookHotel' to plan with params: {params}")
            else:
                print("PLANNER: Warning - Could not find 'BookHotel' operator in Rule Pool.")

        if wants_flight:
            op = self.rule_pool.get_operator("BookFlight")
            if op:
                params, missing = self._ground_flight_params(instruction, state)
                flight_step = {"operator": op, "params": params}
                if missing:
                    flight_step["meta"] = {"needs_regrounding": missing}
                plan.append(flight_step)
                print(f"PLANNER: Added 'BookFlight' to plan with params: {params}")
            else:
                print("PLANNER: Warning - 'BookFlight' operator not found in Rule Pool.")

        # Minimal cross-intent fill (hotel ↔ flight).
        if hotel_step and flight_step:
            hp = hotel_step.get("params", {}) or {}
            fp = flight_step.get("params", {}) or {}

            if not fp.get("destination") and hp.get("location"):
                fp["destination"] = hp["location"]
            if not hp.get("location") and fp.get("destination"):
                hp["location"] = fp["destination"]

            if not fp.get("date") and hp.get("dates"):
                fp["date"] = str(hp["dates"]).split("-")[0].strip()
            if not hp.get("dates") and fp.get("date"):
                hp["dates"] = fp["date"]

            hotel_step["params"] = hp
            flight_step["params"] = fp

            # Update missing markers (best-effort; executor can still repair).
            hotel_missing = [k for k in ("location", "dates") if not hp.get(k)]
            flight_missing = [k for k in ("origin", "destination", "date") if not fp.get(k)]
            if hotel_missing:
                hotel_step["meta"] = {"needs_regrounding": hotel_missing}
            else:
                hotel_step.pop("meta", None)
            if flight_missing:
                flight_step["meta"] = {"needs_regrounding": flight_missing}
            else:
                flight_step.pop("meta", None)

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
                current_payment = str((step.get("params") or {}).get("payment", "") or "")
                if "CorporateCard" in current_payment:
                    print("PLANNER: Detected failure with Corporate Card. Attempting to switch payment method.")

                    new_plan = copy.deepcopy(failed_plan)

                    if "new corporate" in instruction.lower():
                        alternative_payment = "CorporateCard:CC-NEW"
                        if "CC-NEW" in current_payment:
                            alternative_payment = "PersonalCard:PC-1134"
                    else:
                        alternative_payment = "PersonalCard:PC-1134"
                        if "PersonalCard" in current_payment:
                            alternative_payment = "CorporateCard:CC-NEW"

                    new_plan[i]["params"]["payment"] = alternative_payment
                    print(f"PLANNER: New plan generated with payment method: {alternative_payment}")
                    return new_plan

        print("PLANNER: ❌ No local repair strategy found for this failure.")
        return failed_plan

    def _ground_payment_method(self, instruction: str, state: Dict[str, Any]) -> str:
        il = instruction.lower()
        if "new corporate account" in il or "new corporate card" in il or "new corporate" in il:
            return "CorporateCard:CC-NEW"
        return state.get("payment_method", "CorporateCard:CC-5512")

    def _extract_date(self, instruction: str, state: Dict[str, Any], *, field_name: str) -> Optional[str]:
        """
        Minimal date extractor to reduce PlanningFailed:
        supports 'on June 7', 'around June 7', and ranges like 'June 1-3'.
        """
        date_match = re.search(r"(\b[A-Za-z]+\s\d{1,2}-\d{1,2}\b)", instruction)
        if date_match:
            return date_match.group(1).strip()

        on_match = re.search(r"\bon\s([A-Za-z]+\s\d{1,2})\b", instruction, re.IGNORECASE)
        if on_match:
            return on_match.group(1).strip()

        around_match = re.search(r"\baround\s([A-Za-z]+\s\d{1,2})\b", instruction, re.IGNORECASE)
        if around_match:
            return around_match.group(1).strip()

        td = state.get("travel_dates")
        if isinstance(td, str) and td.strip():
            if "-" in td:
                return td.split("-")[0].strip()
            return td.strip()

        return None

    def _ground_hotel_params(self, instruction: str, state: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Simulates a Neural Parser for grounding hotel booking details.
        Returns (params, missing_required).
        """
        params: Dict[str, Any] = {}
        missing: List[str] = []

        loc_match = re.search(r"in\s([\w\s]+?)(?=\sfor|\son|\saround|$)", instruction, re.IGNORECASE)
        if loc_match:
            params["location"] = loc_match.group(1).strip()
        else:
            for k in ("travel_location", "destination", "travel_destination", "city", "location"):
                v = state.get(k)
                if isinstance(v, str) and v.strip():
                    params["location"] = v.strip()
                    break

        dates = self._extract_date(instruction, state, field_name="dates")
        if dates:
            params["dates"] = dates
        else:
            td = state.get("travel_dates")
            if isinstance(td, str) and td.strip():
                params["dates"] = td.strip()

        params["payment"] = self._ground_payment_method(instruction, state)

        for k in ("location", "dates"):
            if not params.get(k):
                missing.append(k)

        return params, missing

    def _ground_flight_params(self, instruction: str, state: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Simulates a Neural Parser for grounding flight booking details.
        Returns (params, missing_required).
        """
        params: Dict[str, Any] = {}
        missing: List[str] = []

        origin_match = re.search(r"from\s([\w\s]+?)(?=\sto|\son|\saround|$)", instruction, re.IGNORECASE)
        if origin_match:
            params["origin"] = origin_match.group(1).strip()
        else:
            v = state.get("travel_origin")
            if isinstance(v, str) and v.strip():
                params["origin"] = v.strip()

        dest_match = re.search(
            r"to\s([\w\s]+?)(?=\sfrom|\sfor|\son|\saround|$)",
            instruction,
            re.IGNORECASE
        )
        if dest_match:
            params["destination"] = dest_match.group(1).strip()
        else:
            fallback_dest = re.search(
                r"(?:travel|trip)\s(?:to|in)\s([\w\s]+?)(?=\sfrom|\sfor|\son|\saround|$)",
                instruction,
                re.IGNORECASE
            )
            if fallback_dest:
                params["destination"] = fallback_dest.group(1).strip()
            else:
                for k in ("travel_destination", "travel_location", "destination", "location", "city"):
                    v = state.get(k)
                    if isinstance(v, str) and v.strip():
                        params["destination"] = v.strip()
                        break

        date = self._extract_date(instruction, state, field_name="date")
        if date:
            params["date"] = date
        else:
            td = state.get("travel_dates")
            if isinstance(td, str) and td.strip():
                params["date"] = td.split("-")[0].strip() if "-" in td else td.strip()

        if "origin" not in params:
            fp = state.get("flight_prices")
            if isinstance(fp, dict) and fp:
                try:
                    params["origin"] = min(fp.keys(), key=lambda k: fp.get(k, float("inf")))
                except Exception:
                    pass

        params["payment"] = self._ground_payment_method(instruction, state)

        for k in ("origin", "destination", "date"):
            if not params.get(k):
                missing.append(k)

        return params, missing
