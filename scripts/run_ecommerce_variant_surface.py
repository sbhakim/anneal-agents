#!/usr/bin/env python3
"""Variant-surface e-commerce recurrence stress test.

This reuses the existing e-commerce stress runner but changes the holdout
instructions to a lexically different surface while keeping the same
PlaceOrder:tool_schema_drift root cause.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_ecommerce_stress as stress


HOLDOUT_START = 8

VARIANT_SURFACE_TASKS = [
    # Prefix: original direct order phrasing.
    "Place an order for 1 laptop for standard customer",
    "Place an order for 2 phones for premium_member customer",
    "Validate order for tablet with payment method credit_card",
    "Place an order for 1 headphones for standard customer",
    "Validate order for laptop with payment method paypal",
    "Place an order for 2 monitors for bulk_buyer customer",
    "Place an order for 1 phone for standard customer",
    "Validate order for keyboard with payment method credit_card",
    # Holdout: paraphrased purchase / checkout phrasing, same PlaceOrder operator.
    "Process order: keyboard quantity 1 standard tier",
    "Process order: tablets quantity 5 premium_member tier",
    "Process order: phone paid by credit_card",
    "Process order: laptop quantity 1 employee tier",
    "Process order: headphones paid by paypal",
    "Process order: monitors quantity 3 standard tier",
]


def _tokens(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def print_surface_shift_summary(tasks: List[str]) -> None:
    prefix = tasks[:HOLDOUT_START]
    holdout = tasks[HOLDOUT_START:]
    max_overlap = max(_jaccard(a, b) for a in prefix for b in holdout)
    mean_overlap = (
        sum(_jaccard(a, b) for a in prefix for b in holdout)
        / max(1, len(prefix) * len(holdout))
    )
    print("VARIANT SURFACE SUMMARY")
    print(f"  prefix tasks: {len(prefix)}")
    print(f"  holdout tasks: {len(holdout)}")
    print(f"  max prefix/holdout token Jaccard: {max_overlap:.3f}")
    print(f"  mean prefix/holdout token Jaccard: {mean_overlap:.3f}")


def main() -> None:
    stress.ECOMMERCE_STRESS_TASKS = list(VARIANT_SURFACE_TASKS)
    stress.HOLDOUT_START = HOLDOUT_START
    print_surface_shift_summary(VARIANT_SURFACE_TASKS)
    stress.main()


if __name__ == "__main__":
    main()
