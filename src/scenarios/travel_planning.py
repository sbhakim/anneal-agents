# src/scenarios/travel_planning.py
"""
Travel planning scenario for SELFEVOLVE evaluation.
Generates tasks and handles failure injection for the demo.
Based on Section X walkthrough and Section XII evaluation protocol.
"""
from typing import List, Dict, Any, Optional, Set
import random

# Difficulty level configurations for Component Stress Test
DIFFICULTY_CONFIGS = {
    'easy': {
        'failure_rate': 0.2,
        'policy_flips': 0,
        'schema_changes': 0,
        'description': 'Simple failures with clear root causes'
    },
    'normal': {
        'failure_rate': 0.3,
        'policy_flips': 1,
        'schema_changes': 0,
        'description': 'Standard difficulty with a single policy flip'
    },
    'hard': {
        'failure_rate': 0.5,
        'policy_flips': 3,
        'schema_changes': 2,
        'description': 'Complex failures requiring governance and robust adaptation'
    },
    'adversarial': {
        'failure_rate': 0.7,
        'policy_flips': 5,  # Frequent policy oscillation
        'schema_changes': 3,  # Multiple breaking API changes
        'description': 'Stress test with rapid, conflicting environmental changes'
    }
}


class TravelPlanningScenario:
    """
    Generates travel planning tasks and manages failure injection.
    Corresponds to the evaluation scenario from Section XII-D.
    UPDATED: Now supports configurable seeds and multiple difficulty levels.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the scenario, including difficulty settings and failure injection.
        """
        self.difficulty = config.get('difficulty', 'normal')
        self.difficulty_config = DIFFICULTY_CONFIGS.get(self.difficulty, DIFFICULTY_CONFIGS['normal'])

        self.num_tasks = config.get('num_tasks', 50)
        self.failure_rate = self.difficulty_config['failure_rate']

        self.min_failures_in_prefix = config.get('min_failures_in_prefix', 3)
        self.prefix_len = config.get('prefix_len', min(10, self.num_tasks))
        self.blackout_dates: List[str] = config.get(
            'blackout_dates',
            ["April 21", "April 22", "April 23", "April 24", "April 25",
             "May 10", "May 11", "May 12", "May 13", "May 14"]
        )
        self.corporate_card_policy: str = config.get('corporate_card_policy', "blocked_on_blackout_dates")

        self.task_seed = config.get('task_generation_seed', 7)
        self.failure_seed = config.get('failure_injector_seed', 42)

        self.tasks = self._generate_tasks(seed=self.task_seed)

        self.failure_injector = FailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            blackout_dates=self.blackout_dates,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config
        )

        print(f"SCENARIO: Initialized with difficulty '{self.difficulty}'.")
        print(f"SCENARIO: Generated {len(self.tasks)} tasks (seed={self.task_seed}).")
        print(
            f"SCENARIO: Failure rate = {self.failure_rate * 100:.1f}%, Policy Flips = {self.difficulty_config['policy_flips']}, Schema Changes = {self.difficulty_config['schema_changes']}")
        print(f"SCENARIO: Failure injector seed = {self.failure_seed}")

    def world_defaults(self) -> Dict[str, Any]:
        """Provide default world-state settings for each task."""
        return {
            "hotel_status": "available", "flight_status": "available",
            "corporate_card_policy": self.corporate_card_policy,
            "blackout_dates": list(self.blackout_dates),
            "api_version": "v1"
        }

    def _generate_tasks(self, seed: int = 7) -> List[str]:
        """
        Generate a list of travel planning instructions.
        """
        tasks = []
        destinations = ["San Francisco", "New York", "Chicago", "Seattle", "Boston"]
        origins = ["Newark", "Baltimore", "Philadelphia", "Washington DC"]
        random.seed(seed)
        for i in range(self.num_tasks):
            task_type = random.choice(["hotel_only", "flight_only", "combined"])
            dest, origin = random.choice(destinations), random.choice(origins)
            month, start_day = random.choice(["April", "May", "June"]), random.randint(1, 25)
            end_day = start_day + random.randint(1, 5)

            if self.difficulty in ['hard', 'adversarial'] and i % 5 == 0:
                task_templates = [
                    f"Book a hotel in {dest} but avoid corporate card if dates {month} {start_day}-{end_day} are blocked.",
                    f"Find a flight from {origin} to {dest}, but if the API (v1) fails, try the new endpoint (v2)."
                ]
                task = random.choice(task_templates)
            else:
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
    UPDATED: Now supports difficulty levels and more complex failure types.
    """

    def __init__(self, failure_rate: float, horizon: int,
                 blackout_dates: Optional[List[str]] = None, min_failures_in_prefix: int = 0, prefix_len: int = 0,
                 seed: int = 42, difficulty_config: Optional[Dict] = None):
        """
        Initializes the FailureInjector.
        """
        self.failure_rate = failure_rate
        self.horizon = max(1, int(horizon))
        self.blackout_dates = set(blackout_dates or [])
        self.prefix_len = max(0, min(int(prefix_len), self.horizon))
        self.min_failures_in_prefix = max(0, int(min_failures_in_prefix))
        self.seed = seed
        self.difficulty_config = difficulty_config or DIFFICULTY_CONFIGS['normal']
        self.difficulty = next((key for key, value in DIFFICULTY_CONFIGS.items() if value == self.difficulty_config), 'normal')
        self.patched_operators: Set[str] = set()
        self._patch_successes: Dict[str, int] = {}

        self.policy_flip_points = self._get_event_points('policy_flips')
        self.schema_change_points = self._get_event_points('schema_changes')

        random.seed(self.seed)
        total_fail = max(0, int(round(self.horizon * self.failure_rate)))
        base_pool = list(range(self.horizon))
        failing = set(random.sample(base_pool, total_fail)) if total_fail > 0 else set()

        if self.prefix_len > 0 and self.min_failures_in_prefix > 0:
            num_in_prefix = len(failing.intersection(range(self.prefix_len)))
            deficit = max(0, self.min_failures_in_prefix - num_in_prefix)
            if deficit > 0:
                prefix_indices = list(range(self.prefix_len))
                rng = random.Random(self.seed + 1000)
                rng.shuffle(prefix_indices)
                extra = [idx for idx in prefix_indices if idx not in failing][:deficit]
                failing.update(extra)
        self.failing_tasks = failing

        self.failure_classes = {
            "blocked_card": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                             "message": "Corporate card blocked for reservations during blackout dates",
                             "policy_ref": "H-23", "category": "constraint"},
            "api_timeout": {"operator": "BookFlight", "error_type": "ToolError", "message": "Booking API timeout",
                            "policy_ref": "API-503", "category": "environmental"},
            "invalid_payment": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                                "message": "Payment method invalid", "policy_ref": "PAY-401", "category": "logical"},
            "api_schema_change": {"operator": "BookFlight", "error_type": "ToolError",
                                  "message": "API endpoint /v1/book is deprecated, use /v2/book",
                                  "policy_ref": "API-V2", "category": "environmental"},
        }
        print(f"FAILURE_INJECTOR: Initialized with seed={self.seed}. {len(self.failing_tasks)} tasks will fail.")

    def _get_event_points(self, event_type: str) -> List[int]:
        """Distributes events like policy flips or schema changes throughout the run."""
        count = self.difficulty_config.get(event_type, 0)
        if count == 0:
            return []
        return [int(i * self.horizon / (count + 1)) for i in range(1, count + 1)]

    def _is_corporate_card(self, params: Optional[Dict], state: Optional[Any]) -> bool:
        payment = params.get("payment") if isinstance(params, dict) else None
        if payment is None and state and hasattr(state, "get"): payment = state.get("payment_method")
        return isinstance(payment, str) and "CorporateCard" in payment

    def _in_blackout(self, state: Optional[Any]) -> bool:
        if not (state and hasattr(state, "get")): return False
        return any(b in str(state.get("travel_dates", "")) for b in self.blackout_dates)

    def should_fail(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                    state: Optional[Any] = None) -> bool:
        """
        ✅ CRITICAL FIX: This logic is updated to be consistent and reliable.
        It now directly checks if the current operator is the one designated to fail for this task.
        """
        # 1. Never fail an operator that has already been successfully patched.
        if op_name in self.patched_operators:
            return False

        # 2. If the task is not in the pre-selected list of failing tasks, it must succeed.
        if task_id not in self.failing_tasks:
            return False

        # 3. Determine which failure *should* happen on this task and who is responsible.
        # This makes `get_failure_details` the single source of truth.
        intended_failure = self.get_failure_details(task_id, op_name, params, state)
        failing_operator = intended_failure.get("operator")

        # 4. Only inject a failure if the current operator is the one designated to fail.
        if op_name == failing_operator:
            # For `blocked_card`, we add a final check to ensure the conditions are met,
            # making the failure more realistic.
            if intended_failure.get("policy_ref") == "H-23":
                return self._is_corporate_card(params, state) and self._in_blackout(state)
            return True

        return False

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                            state: Optional[Any] = None) -> Dict:
        """
        Determines the precise failure details for a given task, acting as the source of truth.
        """
        # First, check for special, difficulty-based events
        if task_id in self.schema_change_points and op_name == "BookFlight":
            failure_class = "api_schema_change"
        elif task_id in self.policy_flip_points and op_name == "BookHotel":
            failure_class = "blocked_card"
        # Then, assign a generic failure based on the operator for other failing tasks
        elif op_name == "BookHotel":
            failure_class = "invalid_payment"
        elif op_name == "BookFlight":
            failure_class = "api_timeout"
        else:
            # Fallback for any other operator, ensuring a failure can always be assigned
            failure_class = "blocked_card"

        failure_info = self.failure_classes[failure_class].copy()
        # Ensure the operator name in the details matches the one being executed
        failure_info["operator"] = op_name
        return {**failure_info, "error": failure_info["error_type"], "task_id": task_id,
                "trace_id": f"T-{task_id:04d}"}

    def mark_operator_patched(self, operator_name: str) -> None:
        """After 3 successful uses of a patch, mark the operator as fixed to stop injecting failures."""
        if operator_name not in self.patched_operators:
            self._patch_successes[operator_name] = self._patch_successes.get(operator_name, 0) + 1
            if self._patch_successes[operator_name] >= 3:
                self.patched_operators.add(operator_name)
                print(
                    f"INJECTOR: 🎓 {operator_name} marked as permanently patched after {self._patch_successes[operator_name]} successes.")
                print(f"         ✓ Future tasks will not inject failures for this operator.")
            else:
                print(
                    f"INJECTOR: 📊 {operator_name} success count: {self._patch_successes[operator_name]}/3 before permanent patching.")