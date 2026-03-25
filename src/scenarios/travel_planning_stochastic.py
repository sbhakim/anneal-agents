from typing import Dict, Any, Optional, List, Tuple
import random
from .travel_planning import TravelPlanningScenario, FailureInjector


class StochasticTravelScenario(TravelPlanningScenario):
    """
    Stochastic variant of travel planning with noise injection.
    Validates metacognitive arbitration and verify-before-act components.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        noise_config = config.get('noise', {})
        self.transient_failure_rate = noise_config.get('transient_failure_rate', 0.15)
        self.uncertainty_injection_rate = noise_config.get('uncertainty_injection', 0.20)
        self.dynamic_policy_shift_rate = noise_config.get('dynamic_policy_shifts', 0.10)
        self.resource_contention_rate = noise_config.get('resource_contention', 0.12)

        self.noise_seed = noise_config.get('seed', 99)
        self.noise_rng = random.Random(self.noise_seed)

        self.transient_failure_types = ['timeout', 'rate_limit', 'service_unavailable', 'network_error']
        self.policy_shift_history: List[Dict[str, Any]] = []
        self.resource_locks: Dict[str, int] = {}  # Track contended resources

        # Override failure injector with stochastic-aware version
        self.failure_injector = StochasticFailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            blackout_dates=self.blackout_dates,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config,
            stochastic_config={
                "enabled": True,
                "seed": self.noise_seed,
                "transient_failure_rate": self.transient_failure_rate,
            },
        )

        print(f"STOCHASTIC: Transient failures={self.transient_failure_rate*100:.1f}%")
        print(f"STOCHASTIC: Uncertainty injection={self.uncertainty_injection_rate*100:.1f}%")
        print(f"STOCHASTIC: Dynamic policy shifts={self.dynamic_policy_shift_rate*100:.1f}%")

    def inject_transient_failure(self, operator_name: str, params: Dict[str, Any],
                                task_id: int) -> Optional[Dict[str, Any]]:
        """Inject transient API failures requiring retry or S2 pathway."""

        if self.noise_rng.random() > self.transient_failure_rate:
            return None

        failure_type = self.noise_rng.choice(self.transient_failure_types)

        failures = {
            'timeout': {
                'error': 'ToolError',
                'error_type': 'ToolError',
                'message': 'API request timed out after 30s',
                'retry_after': self.noise_rng.uniform(1, 5),
                'transient': True,
            },
            'rate_limit': {
                'error': 'ToolError',
                'error_type': 'ToolError',
                'message': 'Rate limit exceeded: 100 requests/min',
                'retry_after': self.noise_rng.uniform(10, 60),
                'transient': True,
            },
            'service_unavailable': {
                'error': 'ToolError',
                'error_type': 'ToolError',
                'message': '503 Service temporarily unavailable',
                'retry_after': self.noise_rng.uniform(5, 30),
                'transient': True,
            },
            'network_error': {
                'error': 'ToolError',
                'error_type': 'ToolError',
                'message': 'Connection reset by peer',
                'retry_after': self.noise_rng.uniform(2, 10),
                'transient': True,
            },
        }

        return failures[failure_type]

    def inject_uncertainty(self, operator_name: str, params: Dict[str, Any],
                          state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Inject low-confidence responses requiring verification.
        Triggers VERIFY pathway when confidence < tau_u.
        """

        if self.noise_rng.random() > self.uncertainty_injection_rate:
            return None

        confidence = self.noise_rng.uniform(0.4, 0.7)  # Below typical tau_u=0.7

        reasons = [
            'Multiple conflicting data sources',
            'Incomplete API response',
            'Ambiguous date parsing',
            'Price volatility detected',
            'Location geocoding uncertainty',
        ]

        return {
            'uncertainty_detected': True,
            'confidence': confidence,
            'reason': self.noise_rng.choice(reasons),
            'requires_verification': confidence < 0.7,
            'verification_method': 'crosscheck_alternative_api' if confidence < 0.5 else 'sanity_check',
        }

    def inject_dynamic_policy_shift(self, task_id: int, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Inject mid-execution policy changes.
        Tests adaptation to evolving constraints.

        ~30% of shifts are compound (affect multiple dimensions simultaneously),
        creating multi-step precondition violations that require deeper
        deliberation (S2 pathway) to resolve preemptively.
        """

        if self.noise_rng.random() > self.dynamic_policy_shift_rate:
            return None

        # ~40% chance of compound shift (card policy + resource availability)
        if self.noise_rng.random() < 0.40:
            return self._shift_compound(state, task_id)

        shift_types = [
            ('blackout_dates', lambda: self._shift_blackout_dates(state)),
            ('corporate_card_policy', lambda: self._shift_card_policy(state)),
            ('budget_limit', lambda: self._shift_budget(state)),
        ]

        shift_type, shift_fn = self.noise_rng.choice(shift_types)
        shift_result = shift_fn()

        self.policy_shift_history.append({
            'task_id': task_id,
            'shift_type': shift_type,
            'old_value': shift_result['old'],
            'new_value': shift_result['new'],
        })

        return shift_result

    def _shift_compound(self, state: Any, task_id: int) -> Dict[str, Any]:
        """
        Compound policy shift: block corporate card AND mark resources as
        temporarily unavailable.  Creates 3+ precondition violations on
        booking operators that exceed a 2-hop repair budget (S1 limit) but
        are resolvable with S2's extended repair budget + preemptive look-ahead.
        """
        # 1. Block corporate card
        old_policy = self._state_get(state, 'corporate_card_policy', 'blocked_on_blackout_dates')
        self._state_set(state, 'corporate_card_policy', 'blocked_on_blackout_dates')

        # Taint payment as invalid (forces clear_invalid + switch_personal = 2 hops)
        # Lock prevents _sync_payment_from_params from clearing the taint.
        pm = self._state_get(state, 'payment_method', None)
        if pm and isinstance(pm, str) and "(invalid)" not in pm.lower():
            self._state_set(state, 'payment_method', f"{pm} (invalid)")
            self._state_set(state, 'payment_invalid', True)
            self._state_set(state, 'payment_valid', False)
            self._state_set(state, 'payment_shift_locked', True)

        # Ensure travel_dates overlaps a blackout date so check_not_blocked_card fires.
        travel_dates = str(self._state_get(state, 'travel_dates', '') or '')
        blackout_dates = self._state_get(state, 'blackout_dates', []) or []
        overlap = any(bd in travel_dates for bd in blackout_dates)
        if not overlap and blackout_dates:
            # Pick a blackout date and inject it into travel_dates
            bd = self.noise_rng.choice(blackout_dates)
            self._state_set(state, 'travel_dates', bd)

        # 2. Temporarily mark hotel as unavailable (requires backoff_refresh = +1 hop)
        old_hotel = self._state_get(state, 'hotel_status', 'available')
        self._state_set(state, 'hotel_status', 'unavailable')

        self.policy_shift_history.append({
            'task_id': task_id,
            'shift_type': 'compound',
            'old_value': {'policy': old_policy, 'hotel': old_hotel},
            'new_value': {'policy': 'blocked_on_blackout_dates', 'hotel': 'unavailable'},
        })

        return {
            'old': {'policy': old_policy, 'hotel': old_hotel},
            'new': {'policy': 'blocked_on_blackout_dates', 'hotel': 'unavailable'},
            'action': 'compound_shift',
            'message': 'Compound policy shift: card blocked + hotel temporarily unavailable',
        }

    def inject_resource_contention(self, operator_name: str, params: Dict[str, Any],
                                   task_id: int) -> Optional[Dict[str, Any]]:
        """
        Inject resource contention (race conditions).
        Tests concurrency handling and retry logic.
        """

        if self.noise_rng.random() > self.resource_contention_rate:
            return None

        resource_key = f"{operator_name}:{params.get('location', 'default')}"

        # Simulate lock held by another process
        if resource_key not in self.resource_locks:
            self.resource_locks[resource_key] = task_id
            return None  # We got the lock

        # Resource locked
        return {
            'error': 'ToolError',
            'error_type': 'ToolError',
            'message': f'Resource {resource_key} locked by task {self.resource_locks[resource_key]}',
            'retry_after': self.noise_rng.uniform(0.5, 3),
            'contention': True,
        }

    def release_resource_lock(self, operator_name: str, params: Dict[str, Any]):
        """Release resource lock after operation completes."""
        resource_key = f"{operator_name}:{params.get('location', 'default')}"
        if resource_key in self.resource_locks:
            del self.resource_locks[resource_key]

    def _state_get(self, state: Any, key: str, default: Any = None) -> Any:
        if hasattr(state, "get"):
            return state.get(key, default)
        if isinstance(state, dict):
            return state.get(key, default)
        return default

    def _state_set(self, state: Any, key: str, value: Any) -> None:
        if hasattr(state, "set"):
            state.set(key, value)
        elif isinstance(state, dict):
            state[key] = value

    def _shift_blackout_dates(self, state: Any) -> Dict[str, Any]:
        """Simulate blackout date policy change."""
        old_dates = self._state_get(state, 'blackout_dates', [])

        # Add or remove dates
        if self.noise_rng.random() < 0.5:
            new_date = f"April {self.noise_rng.randint(1, 30)}"
            new_dates = old_dates + [new_date]
            action = 'added'
        else:
            new_dates = old_dates[:-1] if len(old_dates) > 1 else old_dates
            action = 'removed'

        self._state_set(state, 'blackout_dates', new_dates)

        return {
            'old': old_dates,
            'new': new_dates,
            'action': action,
            'message': f'Corporate blackout dates {action}',
        }

    def _shift_card_policy(self, state: Any) -> Dict[str, Any]:
        """Simulate corporate card policy change."""
        policies = ['blocked_on_blackout_dates', 'always_allowed', 'manager_approval_required']
        old_policy = self._state_get(state, 'corporate_card_policy', 'blocked_on_blackout_dates')
        new_policy = self.noise_rng.choice([p for p in policies if p != old_policy])

        self._state_set(state, 'corporate_card_policy', new_policy)

        return {
            'old': old_policy,
            'new': new_policy,
            'message': f'Corporate card policy changed to {new_policy}',
        }

    def _shift_budget(self, state: Any) -> Dict[str, Any]:
        """Simulate budget limit adjustment."""
        old_budget = self._state_get(state, 'budget', 2000)
        new_budget = self.noise_rng.choice([1500, 2500, 3000])

        self._state_set(state, 'budget', new_budget)

        return {
            'old': old_budget,
            'new': new_budget,
            'message': f'Budget limit adjusted to ${new_budget}',
        }

    def get_noise_statistics(self) -> Dict[str, Any]:
        """Return statistics on injected noise."""
        return {
            'transient_failure_rate': self.transient_failure_rate,
            'uncertainty_injection_rate': self.uncertainty_injection_rate,
            'policy_shifts': len(self.policy_shift_history),
            'policy_shift_history': self.policy_shift_history,
            'active_resource_locks': len(self.resource_locks),
        }


class StochasticFailureInjector(FailureInjector):
    """
    Extended failure injector with stochastic noise.
    Combines deterministic failure schedule with probabilistic noise.
    """

    def __init__(self, *args, stochastic_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.stochastic_config = stochastic_config or {}
        self.noise_enabled = self.stochastic_config.get('enabled', True)
        self.noise_seed = self.stochastic_config.get('seed', 101)
        self.noise_rng = random.Random(self.noise_seed)
        self.transient_failure_rate = float(self.stochastic_config.get('transient_failure_rate', 0.05))
        self.transient_failure_types = ['timeout', 'rate_limit', 'service_unavailable', 'network_error']
        self._stochastic_failure_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    def should_fail(self, task_id: int, operator_name: str,
                    params: Optional[Dict[str, Any]] = None,
                    state: Optional[Dict[str, Any]] = None) -> bool:
        if super().should_fail(task_id, operator_name, params=params, state=state):
            return True

        if not self.noise_enabled:
            return False

        if self.noise_rng.random() < self.transient_failure_rate:
            key = (int(task_id), str(operator_name), self._payment_key(params, state))
            self._stochastic_failure_cache[key] = self._build_transient_failure(task_id, operator_name)
            return True

        return False

    def get_failure_details(self, task_id: int, op_name: str, params: Optional[Dict[str, Any]] = None,
                            state: Optional[Any] = None) -> Dict[str, Any]:
        key = (int(task_id), str(op_name), self._payment_key(params, state))
        if key in self._stochastic_failure_cache:
            return dict(self._stochastic_failure_cache[key])
        return super().get_failure_details(task_id, op_name, params=params, state=state)

    def _build_transient_failure(self, task_id: int, operator_name: str) -> Dict[str, Any]:
        failure_type = self.noise_rng.choice(self.transient_failure_types)
        message_map = {
            'timeout': 'API request timed out after 30s',
            'rate_limit': 'Rate limit exceeded: 100 requests/min',
            'service_unavailable': '503 Service temporarily unavailable',
            'network_error': 'Connection reset by peer',
        }
        return {
            'error': 'ToolError',
            'error_type': 'ToolError',
            'message': message_map.get(failure_type, 'Transient tool error'),
            'retry_after': self.noise_rng.uniform(1, 30),
            'transient': True,
            'operator': operator_name,
            'task_id': task_id,
            'trace_id': f"T-{task_id:04d}-N",
        }
