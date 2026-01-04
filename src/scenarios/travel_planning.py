# src/scenarios/travel_planning.py
"""
Travel planning scenario for SELFEVOLVE evaluation.
Generates tasks and handles failure injection for the demo.
Based on Section X walkthrough and Section XII evaluation protocol.
"""
from typing import List, Dict, Any, Optional, Set, Tuple
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
            "api_version": "v1",
            "budget": 2000,
            "flight_prices": {"Newark": 350, "Baltimore": 250, "Philadelphia": 300, "Washington DC": 320},
            "hotel_prices": {"Chicago": 150, "New York": 250, "San Francisco": 220, "Seattle": 180, "Boston": 200}
        }

    def _generate_tasks(self, seed: int = 7) -> List[str]:
        """
        Generate a list of travel planning instructions, now with more complex adversarial tasks.
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

            if self.difficulty in ['hard', 'adversarial'] and i % 4 == 0:
                budget = random.randint(400, 700)
                adversarial_templates = [
                    f"URGENT: Book a hotel in {dest} for {month} {start_day}-{end_day} using a new corporate account.",
                    f"Find the cheapest travel option to {dest} from {origin} around {month} {start_day}. Budget is ${budget}.",
                    f"Book a non-refundable flight from {origin} to {dest} on {month} {start_day}, and also book a hotel that must be fully refundable."
                ]
                task = random.choice(adversarial_templates)
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
    UPDATED: Now supports more complex, feature-rich failure types.

    MINIMUM RELIABILITY UPDATES (2026-01):
    1) Cache intended failure so should_fail() and get_failure_details() are consistent.
    2) Make ToolError timeouts recoverable within a task by injecting at most once per
       (task_id, operator, payment_key) (lets retry logic demonstrate value).
    """

    def __init__(self, failure_rate: float, horizon: int,
                 blackout_dates: Optional[List[str]] = None, min_failures_in_prefix: int = 0, prefix_len: int = 0,
                 seed: int = 42, difficulty_config: Optional[Dict] = None):
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

        # Cache key includes payment identity so switching cards can change injected outcomes.
        self._intended_failure_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        self._task_op_injection_counts: Dict[Tuple[int, str, str], int] = {}

        self.policy_flip_points = self._get_event_points('policy_flips')
        self.schema_change_points = self._get_event_points('schema_changes')

        random.seed(self.seed)
        total_fail = max(0, int(round(self.horizon * self.failure_rate)))
        base_pool = list(range(self.horizon))
        self.failing_tasks = set(random.sample(base_pool, total_fail)) if total_fail > 0 else set()

        self.failure_classes = {
            "blocked_card": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                             "message": "Corporate card blocked for reservations during blackout dates",
                             "policy_ref": "H-23", "category": "constraint"},
            "api_timeout": {"operator": "BookFlight", "error_type": "ToolError", "message": "Booking API timeout",
                            "policy_ref": "API-503", "category": "environmental"},
            "api_schema_change": {"operator": "BookFlight", "error_type": "ToolError",
                                  "message": "API endpoint /v1/book is deprecated, use /v2/book",
                                  "policy_ref": "API-V2", "category": "environmental"},
            "booking_cascade_failure": {
                "operator": "BookHotel", "error_type": "ToolError",
                "message": "Downstream service failed. Cause is ambiguous.", "category": "ambiguous"
            },
            "policy_conflict_error": {
                "operator": "ConfirmBooking", "error_type": "PreconditionUnmet",
                "message": "Cannot confirm booking with conflicting refund policies.", "category": "constraint"
            },
            "invalid_payment": {"operator": "BookHotel", "error_type": "PreconditionUnmet",
                                "message": "Payment method is invalid or expired.",
                                "policy_ref": "PAY-401", "category": "logical"}
        }
        print(f"FAILURE_INJECTOR: Initialized with seed={self.seed}. {len(self.failing_tasks)} tasks will fail.")

    def _get_event_points(self, event_type: str) -> List[int]:
        count = self.difficulty_config.get(event_type, 0)
        if count == 0:
            return []
        return [int(i * self.horizon / (count + 1)) for i in range(1, count + 1)]

    def _payment_key(self, params: Optional[Dict], state: Optional[Any]) -> str:
        payment = params.get("payment") if isinstance(params, dict) else None
        if payment is None and state and hasattr(state, "get"):
            payment = state.get("payment_method")
        return str(payment) if payment is not None else ""

    def _is_corporate_card(self, params: Optional[Dict], state: Optional[Any]) -> bool:
        payment = params.get("payment") if isinstance(params, dict) else None
        if payment is None and state and hasattr(state, "get"):
            payment = state.get("payment_method")
        return isinstance(payment, str) and "CorporateCard" in payment

    def _in_blackout(self, state: Optional[Any]) -> bool:
        if not (state and hasattr(state, "get")):
            return False
        return any(b in str(state.get("travel_dates", "")) for b in self.blackout_dates)

    def _is_recoverable_timeout(self, failure_info: Dict[str, Any]) -> bool:
        """True if this failure is a ToolError timeout suitable for one-shot injection."""
        if not isinstance(failure_info, dict):
            return False
        if failure_info.get("error_type") != "ToolError":
            return False
        msg = str(failure_info.get("message", "")).lower()
        pref = str(failure_info.get("policy_ref", "")).upper()
        return ("timeout" in msg) or (pref == "API-503")

    def should_fail(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                    state: Optional[Any] = None) -> bool:
        # Permanently patched operators: never inject
        if op_name in self.patched_operators:
            return False

        # Non-failing task: never inject
        if task_id not in self.failing_tasks:
            return False

        intended_failure = self.get_failure_details(task_id, op_name, params, state) or {}
        failing_operator = intended_failure.get("operator")

        if op_name != failing_operator:
            return False

        # Policy-conditioned failure: only when corporate card + blackout overlap
        if intended_failure.get("policy_ref") == "H-23":
            return self._is_corporate_card(params, state) and self._in_blackout(state)

        # Payment invalidation should only target corporate cards so switching payment can recover.
        if intended_failure.get("policy_ref") == "PAY-401":
            return self._is_corporate_card(params, state)

        # For ToolError timeouts, inject at most once per task/operator/payment to allow recovery/retry
        if self._is_recoverable_timeout(intended_failure):
            key = (int(task_id), str(op_name), self._payment_key(params, state))
            count = self._task_op_injection_counts.get(key, 0)
            if count >= 1:
                return False
            self._task_op_injection_counts[key] = count + 1
            return True

        return True

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                            state: Optional[Any] = None) -> Dict:
        payment_key = self._payment_key(params, state)
        cache_key = (int(task_id), str(op_name), payment_key)
        if cache_key in self._intended_failure_cache:
            cached = self._intended_failure_cache[cache_key].copy()
            cached["operator"] = op_name
            return cached

        instruction = ""
        if state and hasattr(state, 'get'):
            instruction = state.get('instruction', '')

        if self.difficulty == 'adversarial':
            if "URGENT" in instruction:
                failure_class = "booking_cascade_failure"
            elif "non-refundable" in instruction and "refundable" in instruction:
                failure_class = "policy_conflict_error"
            elif task_id in self.schema_change_points and op_name == "BookFlight":
                failure_class = "api_schema_change"
            elif task_id in self.policy_flip_points and op_name == "BookHotel":
                failure_class = "blocked_card"
            else:
                failure_class = "api_timeout" if random.random() < 0.5 else "invalid_payment"
        elif task_id in self.schema_change_points and op_name == "BookFlight":
            failure_class = "api_schema_change"
        elif task_id in self.policy_flip_points and op_name == "BookHotel":
            failure_class = "blocked_card"
        elif op_name == "BookHotel":
            failure_class = "invalid_payment"
        else:
            failure_class = "api_timeout"

        failure_info = self.failure_classes[failure_class].copy()
        failure_info["operator"] = op_name

        out = {
            **failure_info,
            "error": failure_info["error_type"],
            "task_id": task_id,
            "trace_id": f"T-{task_id:04d}"
        }

        self._intended_failure_cache[cache_key] = out.copy()
        return out

    def mark_operator_patched(self, operator_name: str) -> None:
        if operator_name not in self.patched_operators:
            self._patch_successes[operator_name] = self._patch_successes.get(operator_name, 0) + 1
            if self._patch_successes[operator_name] >= 3:
                self.patched_operators.add(operator_name)
                print(f"INJECTOR: 🎓 {operator_name} marked as permanently patched...")
                print(f"         ✓ Future tasks will not inject failures for this operator.")
            else:
                print(f"INJECTOR: 📊 {operator_name} success count: {self._patch_successes[operator_name]}/3 before permanent patching.")
