# src/scenarios/failure_injector.py
"""
Implements controlled failure injection for the evaluation scenario.
Corresponds to the dynamic environment simulation in Section XII-D.

UPDATED:
- Added `patched_operators` tracking to allow the system to "win" against faults.
- Modified `should_fail` to respect committed patches, enabling TTA metrics.
- Added `mark_operator_patched` to bridge the gap between RulePool and Environment.
"""
from typing import Dict, Any, List, Set
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
        remaining_needed = int(self.failure_rate * self.horizon) - self.min_failures_in_prefix
        all_tasks = list(range(self.horizon))
        random.shuffle(all_tasks)
        for task in all_tasks:
            if len(failing) >= int(self.failure_rate * self.horizon):
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

    def should_fail(self, task_id: int, op_name: str) -> bool:
        """
        Determines if the current task and operator should fail.
        UPDATED: Returns False if the operator has been patched, allowing success.
        """
        # If the agent has evolved to handle this operator, suppress the injected failure
        if op_name in self.patched_operators:
            return False

        return task_id in self.failing_tasks

    def get_failure_details(self, task_id: int, op_name: str) -> Dict[str, Any]:
        """
        Returns failure details, varying error type (70% PreconditionUnmet, 30% ToolError).
        Applies policy flip if task_id >= policy_flip_at (e.g., expand blackout dates).
        """
        # Vary error type
        if random.random() < 0.7:
            error_type = "PreconditionUnmet"
            message = "Corporate cards blocked for reservations during blackout dates"
            policy_ref = f"Blackout dates: {', '.join(self.blackout_dates)}"
        else:
            error_type = "ToolError"
            message = "Booking API timeout"
            policy_ref = None

        # Apply policy flip: e.g., add more blackout dates post-flip
        if task_id >= self.policy_flip_at:
            # Example flip: Add June dates to blackout
            if "June" not in [d.split()[0] for d in self.blackout_dates]:
                self.blackout_dates.extend(["June 1", "June 2"])
            message += " (Policy updated: Extended blackout periods)"
            policy_ref = f"Updated blackout dates: {', '.join(self.blackout_dates)}"

        return {
            "error": error_type,
            "operator": op_name,
            "message": message,
            "policy_ref": policy_ref,
            "task_id": task_id,
            "trace_id": f"trace-{uuid.uuid4().hex[:8]}",
        }