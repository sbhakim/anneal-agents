# src/scenarios/travel_planning.py
"""
Travel planning scenario for SELFEVOLVE evaluation.
Generates tasks and handles failure injection for the demo.
Based on Section X walkthrough and Section XII evaluation protocol.
"""
from typing import List, Dict, Any, Optional, Set
import random


class TravelPlanningScenario:
    """
    Generates travel planning tasks and manages failure injection.
    Corresponds to the evaluation scenario from Section XII-D.

    UPDATED: Now supports configurable seeds for reproducible experiments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.num_tasks = config.get('num_tasks', 50)
        self.failure_rate = config.get('failure_rate', 0.3)
        self.policy_flip_at = config.get('policy_flip_at', 25)
        self.min_failures_in_prefix = config.get('min_failures_in_prefix', 3)
        self.prefix_len = config.get('prefix_len', min(10, self.num_tasks))
        self.blackout_dates: List[str] = config.get(
            'blackout_dates',
            ["April 21", "April 22", "April 23", "April 24", "April 25",
             "May 10", "May 11", "May 12", "May 13", "May 14"]
        )
        self.corporate_card_policy: str = config.get('corporate_card_policy', "blocked_on_blackout_dates")

        # **NEW**: Seed configuration for reproducibility
        self.task_seed = config.get('task_generation_seed', 7)
        self.failure_seed = config.get('failure_injector_seed', 42)

        # Generate tasks with specified seed
        self.tasks = self._generate_tasks(seed=self.task_seed)

        # Failure injector with specified seed
        self.failure_injector = FailureInjector(
            failure_rate=self.failure_rate,
            policy_flip_at=self.policy_flip_at,
            horizon=self.num_tasks,
            blackout_dates=self.blackout_dates,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed  # Pass seed to injector
        )
        print(f"SCENARIO: Generated {len(self.tasks)} travel planning tasks (seed={self.task_seed}).")
        print(f"SCENARIO: Failure rate = {self.failure_rate * 100}%, policy flip at task {self.policy_flip_at}")
        print(f"SCENARIO: Failure injector seed = {self.failure_seed}")

    def world_defaults(self) -> Dict[str, Any]:
        """Provide default world-state settings for each task."""
        return {
            "hotel_status": "available", "flight_status": "available",
            "corporate_card_policy": self.corporate_card_policy,
            "blackout_dates": list(self.blackout_dates),
        }

    def _generate_tasks(self, seed: int = 7) -> List[str]:
        """
        Generate a list of travel planning instructions.
        UPDATED: Now accepts seed parameter for reproducibility.
        """
        tasks = []
        destinations = ["San Francisco", "New York", "Chicago", "Seattle", "Boston"]
        origins = ["Newark", "Baltimore", "Philadelphia", "Washington DC"]
        # **FIXED**: Use provided seed instead of hardcoded 7
        random.seed(seed)
        for i in range(self.num_tasks):
            task_type = random.choice(["hotel_only", "flight_only", "combined"])
            dest, origin = random.choice(destinations), random.choice(origins)
            month, start_day = random.choice(["April", "May", "June"]), random.randint(1, 25)
            end_day = start_day + random.randint(1, 5)
            if task_type == "hotel_only":
                task = f"Book a hotel in {dest} for {month} {start_day}-{end_day}"
            elif task_type == "flight_only":
                task = f"Book a flight from {origin} to {dest} on {month} {start_day}"
            else:
                task = f"Book a hotel in {dest} for {month} {start_day}-{end_day} and a flight from {origin} on {month} {start_day}"
            tasks.append(task)
        return tasks

    def get_task(self, task_id: int) -> Optional[str]:
        if 0 <= task_id < len(self.tasks):
            return self.tasks[task_id]
        return None

    def should_inject_failure(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                              state: Optional[Any] = None) -> bool:
        return self.failure_injector.should_fail(task_id, op_name, params=params, state=state)

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                            state: Optional[Any] = None) -> Optional[Dict]:
        return self.failure_injector.get_failure_details(task_id, op_name, params=params, state=state)

    def mark_operator_patched(self, operator_name: str) -> None:
        self.failure_injector.mark_operator_patched(operator_name)


class FailureInjector:
    """
    Injects controlled failures to test FDKA adaptation.
    UPDATED: Now supports configurable seed for reproducible failure patterns.
    """

    def __init__(self, failure_rate: float, policy_flip_at: int, horizon: int,
                 blackout_dates: Optional[List[str]] = None, min_failures_in_prefix: int = 0, prefix_len: int = 0,
                 seed: int = 42):
        self.failure_rate = failure_rate
        self.policy_flip_at = policy_flip_at
        self.blackout_dates = set(blackout_dates or [])
        self.horizon = max(1, int(horizon))
        self.prefix_len = max(0, min(int(prefix_len), self.horizon))
        self.min_failures_in_prefix = max(0, int(min_failures_in_prefix))
        self.seed = seed
        self.patched_operators: Set[str] = set()
        self._patch_successes: Dict[str, int] = {}

        # **FIXED**: Use provided seed for deterministic failure selection
        random.seed(self.seed)
        total_fail = max(0, int(round(self.horizon * self.failure_rate)))
        base_pool = list(range(self.horizon))
        failing = set(random.sample(base_pool, total_fail)) if total_fail > 0 else set()

        if self.prefix_len > 0 and self.min_failures_in_prefix > 0:
            num_in_prefix = len(failing.intersection(range(self.prefix_len)))
            deficit = max(0, self.min_failures_in_prefix - num_in_prefix)
            if deficit > 0:
                prefix_indices = list(range(self.prefix_len))
                rng = random.Random(self.seed + 1000)  # Offset from main seed
                rng.shuffle(prefix_indices)
                extra = [idx for idx in prefix_indices if idx not in failing][:deficit]
                failing.update(extra)
        self.failing_tasks = failing

        self.failure_classes = {
            "blocked_card": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                             "message": "Corporate card blocked", "policy_ref": "H-23"},
            "api_timeout": {"operator": "BookFlight", "error_type": "ToolError", "message": "Booking API timeout",
                            "policy_ref": "API-503"},
            "invalid_payment": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                                "message": "Payment method invalid", "policy_ref": "PAY-401"},
        }
        print(f"FAILURE_INJECTOR: Initialized with seed={self.seed}. {len(self.failing_tasks)} tasks will fail.")

    def _is_corporate_card(self, params: Optional[Dict], state: Optional[Any]) -> bool:
        payment = params.get("payment") if isinstance(params, dict) else None
        if payment is None and state and hasattr(state, "get"): payment = state.get("payment_method")
        return isinstance(payment, str) and "CorporateCard" in payment

    def _in_blackout(self, state: Optional[Any]) -> bool:
        if not (state and hasattr(state, "get")): return False
        return any(b in str(state.get("travel_dates", "")) for b in self.blackout_dates)

    def should_fail(self, task_id: int, operator_name: str, params: Optional[Dict] = None,
                    state: Optional[Any] = None) -> bool:
        if operator_name in self.patched_operators:
            # If an operator is marked as permanently patched, it no longer fails.
            self.mark_operator_patched(operator_name)  # This will just print the success count
            return False

        if task_id not in self.failing_tasks:
            # If a task wasn't selected to fail, but the operator has a patch in progress,
            # we count this as a success towards permanent patching.
            if operator_name in self._patch_successes:
                self.mark_operator_patched(operator_name)
            return False

        if operator_name == "BookHotel":
            if task_id < self.policy_flip_at:
                return self._is_corporate_card(params, state) and self._in_blackout(state)
            return True

        if operator_name == "BookFlight":
            return (task_id % 5 == 0)

        return False

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                            state: Optional[Any] = None) -> Dict:
        if op_name == "BookHotel":
            failure_class = "blocked_card" if task_id < self.policy_flip_at else "invalid_payment"
        elif op_name == "BookFlight":
            failure_class = "api_timeout"
        else:
            failure_class = "blocked_card"
        failure_info = self.failure_classes[failure_class].copy()
        return {**failure_info, "error": failure_info["error_type"], "operator": op_name, "task_id": task_id,
                "trace_id": f"T-{task_id:04d}"}

    def mark_operator_patched(self, operator_name: str) -> None:
        if operator_name not in self.patched_operators:
            self._patch_successes[operator_name] = self._patch_successes.get(operator_name, 0) + 1
            if self._patch_successes[operator_name] >= 3:
                self.patched_operators.add(operator_name)
                print(
                    f"INJECTOR: 🎓 {operator_name} marked as permanently patched after {self._patch_successes[operator_name]} successes.")
            else:
                print(
                    f"INJECTOR: 📊 {operator_name} success count: {self._patch_successes[operator_name]}/3 before permanent patching.")