# src/scenarios/travel_planning.py
"""
Travel planning scenario for ANNEAL evaluation.
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
        self.task_overrides = [str(task) for task in (config.get('task_overrides') or []) if str(task).strip()]

        # Respect an explicit scenario failure_rate override when provided; otherwise fall back to difficulty defaults.
        cfg_failure_rate = config.get('failure_rate', None)
        self.failure_rate = float(cfg_failure_rate) if cfg_failure_rate is not None else self.difficulty_config['failure_rate']

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

        if self.task_overrides:
            self.tasks = list(self.task_overrides)
            self.num_tasks = len(self.tasks)
        else:
            self.tasks = self._generate_tasks(seed=self.task_seed)

        self.failure_injector = FailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            blackout_dates=self.blackout_dates,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config,
            forced_failures=config.get('forced_failures') or [],
            patch_successes_to_suppress=config.get('patch_successes_to_suppress', 3),
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

        # Use a local RNG for reproducible task generation without mutating global random state.
        rng = random.Random(seed)

        for i in range(self.num_tasks):
            task_type = rng.choice(["hotel_only", "flight_only", "combined"])
            dest, origin = rng.choice(destinations), rng.choice(origins)
            month, start_day = rng.choice(["April", "May", "June"]), rng.randint(1, 25)
            end_day = start_day + rng.randint(1, 5)

            if self.difficulty in ['hard', 'adversarial'] and i % 4 == 0:
                budget = rng.randint(400, 700)
                adversarial_templates = [
                    f"URGENT: Book a hotel in {dest} for {month} {start_day}-{end_day} using a new corporate account.",
                    f"Find the cheapest travel option to {dest} from {origin} around {month} {start_day}. Budget is ${budget}.",
                    f"Book a non-refundable flight from {origin} to {dest} on {month} {start_day}, and also book a hotel that must be fully refundable."
                ]
                task = rng.choice(adversarial_templates)
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
       (task_id, operator, payment_key, policy_ref) (lets retry logic demonstrate value).
    3) Treat API schema drift (API-V2) as one-shot recoverable as well.
    4) Use a local RNG to avoid global seeding side effects and improve run reproducibility.
    """

    def __init__(self, failure_rate: float, horizon: int,
                 blackout_dates: Optional[List[str]] = None, min_failures_in_prefix: int = 0, prefix_len: int = 0,
                 seed: int = 42, difficulty_config: Optional[Dict] = None,
                 forced_failures: Optional[List[Dict[str, Any]]] = None,
                 patch_successes_to_suppress: int = 3):
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
        self.patch_successes_to_suppress = max(1, int(patch_successes_to_suppress))

        # Local RNG: deterministic without mutating global random state.
        self.rng = random.Random(self.seed)

        # Cache key includes payment identity so switching cards can change injected outcomes.
        self._intended_failure_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        self._task_op_injection_counts: Dict[Tuple[int, str, str, str], int] = {}

        self.policy_flip_points = self._get_event_points('policy_flips')
        self.schema_change_points = self._get_event_points('schema_changes')

        self.failing_tasks = set(self._select_failing_tasks())

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
        self.forced_failures = self._normalize_forced_failures(forced_failures or [])
        self.failing_tasks.update(ff["task_id"] for ff in self.forced_failures)
        print(f"FAILURE_INJECTOR: Initialized with seed={self.seed}. {len(self.failing_tasks)} tasks will fail.")

    def _normalize_forced_failures(self, forced_failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in forced_failures or []:
            if not isinstance(item, dict):
                continue
            try:
                task_id = int(item.get("task_id"))
            except Exception:
                continue
            operator = str(item.get("operator") or "").strip()
            if not operator:
                continue

            failure_class = str(item.get("failure_class") or "").strip()
            base: Dict[str, Any] = {}
            if failure_class:
                base = dict(self.failure_classes.get(failure_class, {}))

            error_type = str(item.get("error_type") or base.get("error_type") or item.get("error") or "ToolError")
            policy_ref = str(item.get("policy_ref") or base.get("policy_ref") or "")
            message = str(item.get("message") or base.get("message") or error_type)
            category = str(item.get("category") or base.get("category") or "forced")

            normalized.append({
                "task_id": task_id,
                "operator": operator,
                "error_type": error_type,
                "error": error_type,
                "message": message,
                "policy_ref": policy_ref,
                "category": category,
                "trace_id": f"T-{task_id:04d}",
            })
        return normalized

    def _get_event_points(self, event_type: str) -> List[int]:
        count = self.difficulty_config.get(event_type, 0)
        if count == 0:
            return []
        return [int(i * self.horizon / (count + 1)) for i in range(1, count + 1)]

    def _select_failing_tasks(self) -> List[int]:
        """
        Select failing tasks with a guaranteed minimum in the prefix window.
        This keeps early adaptation pressure consistent across horizons.
        """
        failing: Set[int] = set()

        if self.prefix_len > 0 and self.min_failures_in_prefix > 0:
            prefix_tasks = list(range(self.prefix_len))
            self.rng.shuffle(prefix_tasks)
            for task in prefix_tasks[:min(self.min_failures_in_prefix, len(prefix_tasks))]:
                failing.add(task)

        target = max(0, int(round(self.horizon * self.failure_rate)))
        all_tasks = list(range(self.horizon))
        self.rng.shuffle(all_tasks)
        for task in all_tasks:
            if len(failing) >= target:
                break
            failing.add(task)

        return sorted(failing)

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
        if not isinstance(failure_info, dict):
            return False
        if failure_info.get("error_type") != "ToolError":
            return False
        msg = str(failure_info.get("message", "")).lower()
        pref = str(failure_info.get("policy_ref", "")).upper()
        return ("timeout" in msg) or (pref == "API-503")

    def _is_recoverable_schema_change(self, failure_info: Dict[str, Any]) -> bool:
        if not isinstance(failure_info, dict):
            return False
        if failure_info.get("error_type") != "ToolError":
            return False
        msg = str(failure_info.get("message", "")).lower()
        pref = str(failure_info.get("policy_ref", "")).upper()
        return (pref == "API-V2") or ("/v1/book" in msg) or ("deprecated" in msg) or ("use /v2" in msg) or ("use v2" in msg)

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

        # If the agent has migrated to v2, don't keep injecting v1 deprecation.
        if self._is_recoverable_schema_change(intended_failure):
            if state and hasattr(state, "get") and str(state.get("api_version", "")).lower() == "v2":
                return False

        # For retryable ToolError classes, inject at most once per task/operator/payment/policy
        if self._is_recoverable_timeout(intended_failure) or self._is_recoverable_schema_change(intended_failure):
            pref = str(intended_failure.get("policy_ref", "") or "")
            key = (int(task_id), str(op_name), self._payment_key(params, state), pref)
            count = self._task_op_injection_counts.get(key, 0)
            if count >= 1:
                return False
            self._task_op_injection_counts[key] = count + 1
            return True

        return True

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict] = None,
                            state: Optional[Any] = None) -> Dict:
        for forced in self.forced_failures:
            if forced.get("task_id") == int(task_id) and forced.get("operator") == str(op_name):
                out = forced.copy()
                out["operator"] = op_name
                return out

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
                failure_class = "api_timeout" if self.rng.random() < 0.5 else "invalid_payment"
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
            if self._patch_successes[operator_name] >= self.patch_successes_to_suppress:
                self.patched_operators.add(operator_name)
                print(f"INJECTOR: 🎓 {operator_name} marked as permanently patched...")
                print(f"         ✓ Future tasks will not inject failures for this operator.")
            else:
                print(
                    f"INJECTOR: 📊 {operator_name} success count: "
                    f"{self._patch_successes[operator_name]}/{self.patch_successes_to_suppress} "
                    f"before permanent patching."
                )
