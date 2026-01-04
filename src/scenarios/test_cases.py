# src/scenarios/test_cases.py
"""
Deterministic micro-suite for Canary testing.

Format:
- Each entry mirrors the ExperiencePool trace_record shape (minimally).
- Canary simulation only requires: entry["metadata"]["instruction"].

Notes:
- Cases are intentionally stable so before/after comparisons are fair.
- Operator filtering supports both single operators ("BookFlight") and composite labels
  like "BookFlight+BookHotel" when canary is invoked for a specific operator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _case(
    instruction: str,
    *,
    case_id: str,
    operator: str = "CANARY",
    success: bool = True,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canary test case compatible with the canary simulator."""
    return {
        "trace_id": -1,
        "trace": [],  # canary simulator reads only metadata.instruction
        "success": success,
        "operator": operator,
        "error_type": error_type,
        "metadata": {
            "instruction": instruction,
            "task_id": case_id,
        },
    }


# Deterministic suite (8 cases). Keep stable across experiments.
CANARY_SUITE: List[Dict[str, Any]] = [
    _case(
        "Book a flight from Newark to Seattle on June 13.",
        case_id="CANARY-01",
        operator="BookFlight",
    ),
    _case(
        "Book a hotel in Seattle for June 13-15.",
        case_id="CANARY-02",
        operator="BookHotel",
    ),
    _case(
        "Book a flight from Baltimore to Chicago on April 10.",
        case_id="CANARY-03",
        operator="BookFlight",
    ),
    _case(
        "Book a hotel in Chicago for April 10-12.",
        case_id="CANARY-04",
        operator="BookHotel",
    ),
    _case(
        "Book a flight from Philadelphia to Boston on May 20 and book a hotel in Boston for May 20-22.",
        case_id="CANARY-05",
        operator="BookFlight+BookHotel",
    ),
    _case(
        "Book a flight from Washington DC to San Francisco on July 2.",
        case_id="CANARY-06",
        operator="BookFlight",
    ),
    _case(
        "Book a hotel in San Francisco for July 2-5.",
        case_id="CANARY-07",
        operator="BookHotel",
    ),
    _case(
        "Book a flight from Newark to New York on March 1.",
        case_id="CANARY-08",
        operator="BookFlight",
    ),
]


def get_canary_suite() -> List[Dict[str, Any]]:
    """Return a copy of the deterministic canary micro-suite."""
    return list(CANARY_SUITE)


def get_canary_suite_for_operator(operator_name: str) -> List[Dict[str, Any]]:
    """
    Operator-specific suite selection.

    - If operator_name is empty/None -> return full suite.
    - If operator_name is "BookFlight" -> include "BookFlight" and composite cases containing it.
    """
    op = (operator_name or "").strip()
    if not op:
        return get_canary_suite()

    out: List[Dict[str, Any]] = []
    for ex in CANARY_SUITE:
        ex_op = str(ex.get("operator", ""))
        parts = [p.strip() for p in ex_op.split("+") if p.strip()]
        if op == ex_op or op in parts:
            out.append(ex)
    return out
