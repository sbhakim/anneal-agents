from typing import List, Dict, Any, Optional, Set, Tuple
import random
import json
from datetime import datetime, timedelta


DIFFICULTY_CONFIGS = {
    'easy': {
        'failure_rate': 0.2,
        'inventory_volatility': 0.1,
        'promo_conflicts': 0,
        'policy_changes': 0,
    },
    'normal': {
        'failure_rate': 0.3,
        'inventory_volatility': 0.2,
        'promo_conflicts': 1,
        'policy_changes': 1,
    },
    'hard': {
        'failure_rate': 0.5,
        'inventory_volatility': 0.4,
        'promo_conflicts': 3,
        'policy_changes': 2,
    },
}


class EcommerceScenario:
    """E-commerce order management domain for cross-domain validation."""

    def __init__(self, config: Dict[str, Any]):
        self.difficulty = config.get('difficulty', 'normal')
        self.difficulty_config = DIFFICULTY_CONFIGS.get(self.difficulty, DIFFICULTY_CONFIGS['normal'])

        self.num_tasks = config.get('num_tasks', 30)
        self.failure_rate = config.get('failure_rate', self.difficulty_config['failure_rate'])
        self.min_failures_in_prefix = config.get('min_failures_in_prefix', 3)
        self.prefix_len = config.get('prefix_len', min(10, self.num_tasks))

        self.task_seed = config.get('task_generation_seed', 13)
        self.failure_seed = config.get('failure_injector_seed', 17)

        # High inventory limits to focus on patchable failures, not constraints
        self.inventory_limits = config.get('inventory_limits', {
            'laptop': 1000, 'phone': 1000, 'tablet': 1000,
            'headphones': 1000, 'monitor': 1000, 'keyboard': 1000
        })

        # Relaxed promo policies to reduce constraint failures
        self.promo_policies = config.get('promo_policies', {
            'max_discount_percent': 50,
            'stackable_promos': True,  # Allow stacking to avoid conflicts
            'employee_discount_requires_verification': False,
            'bulk_discount_min_quantity': 5
        })

        # Minimal shipping restrictions to focus on patchable bugs
        self.shipping_restrictions = config.get('shipping_restrictions', [])

        # Support stress-test task overrides (plain instruction strings).
        # When provided, replaces generated tasks so the stress script can
        # inject PlaceOrder-dominant holdout tasks without changing core logic.
        task_overrides = [str(t) for t in (config.get('task_overrides') or []) if str(t).strip()]
        if task_overrides:
            self.tasks = [
                {'task_id': i, 'instruction': instr, 'product': 'laptop', 'quantity': 1,
                 'customer_type': 'standard', 'promo_code': None,
                 'location': 'standard', 'payment_method': 'credit_card'}
                for i, instr in enumerate(task_overrides)
            ]
            self.num_tasks = len(self.tasks)
        else:
            self.tasks = self._generate_tasks(seed=self.task_seed)

        # placeorder_force_mode: when set, forces PlaceOrder failures to a single
        # mode for controlled stress-test experiments (e.g. 'tool_schema_drift').
        placeorder_force_mode = config.get('placeorder_force_mode', None)

        self.failure_injector = EcommerceFailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            inventory_limits=self.inventory_limits,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config,
            placeorder_force_mode=placeorder_force_mode,
        )

        print(f"ECOMMERCE: Initialized with difficulty '{self.difficulty}'")
        print(f"ECOMMERCE: Generated {len(self.tasks)} tasks (seed={self.task_seed})")
        print(f"ECOMMERCE: Failure rate = {self.failure_rate * 100:.1f}%")

    def world_defaults(self) -> Dict[str, Any]:
        """Default world state for each task."""
        return {
            'inventory': dict(self.inventory_limits),
            'promo_policies': dict(self.promo_policies),
            'shipping_restrictions': list(self.shipping_restrictions),
            'payment_methods': ['credit_card', 'debit_card', 'paypal', 'employee_account'],
            'payment_method': 'credit_card',
            'tax_rate': 0.08,
            'free_shipping_threshold': 50,
            'return_window_days': 30,
            'active_promos': ['SAVE10', 'BULK15', 'EMPLOYEE20'],
            'restricted_items': [],
            'required_auth_mode': 'legacy_auth_token',
            'allowed_auth_modes': ['legacy_auth_token', 'signed_session_token'],
            'auth_schema_version': 'legacy_auth_token',
            # Explicit resets so injected flags don't persist across tasks
            'payment_invalid': False,
            'payment_valid': True,
            'policy_violation': False,
            'price_changed': False,
        }

    def _generate_tasks(self, seed: int) -> List[Dict[str, Any]]:
        """Generate diverse e-commerce task instances."""
        rng = random.Random(seed)
        tasks = []

        products = ['laptop', 'phone', 'tablet', 'headphones', 'monitor', 'keyboard']
        customers = ['standard', 'employee', 'bulk_buyer', 'premium_member']
        promos = ['SAVE10', 'BULK15', 'EMPLOYEE20', 'FREESHIP', None]

        templates = [
            "Place an order for {quantity} {product}(s) for {customer_type} customer",
            "Process order with {product} using promo code {promo}",
            "Calculate total for {quantity} {product}(s) with shipping to {location}",
            "Apply {promo} discount to order containing {product}",
            "Process refund for {product} purchased {days} days ago",
            "Update inventory after selling {quantity} {product}(s)",
            "Validate order for {product} with payment method {payment}",
            "Ship {quantity} {product}(s) to {location} address",
        ]

        for i in range(self.num_tasks):
            template = rng.choice(templates)
            product = rng.choice(products)
            quantity = rng.choice([1, 2, 5, 10, 15, 20])
            customer_type = rng.choice(customers)
            promo = rng.choice(promos)
            location = rng.choice(['standard', 'APO', 'FPO', 'international'])
            payment = rng.choice(['credit_card', 'employee_account', 'paypal'])
            days_ago = rng.randint(1, 60)

            instruction = template.format(
                quantity=quantity,
                product=product,
                customer_type=customer_type,
                promo=promo or 'none',
                location=location,
                payment=payment,
                days=days_ago
            )

            tasks.append({
                'task_id': i,
                'instruction': instruction,
                'product': product,
                'quantity': quantity,
                'customer_type': customer_type,
                'promo_code': promo,
                'location': location,
                'payment_method': payment,
            })

        return tasks

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve task instruction by ID (matches TravelPlanningScenario interface)."""
        if 0 <= task_id < len(self.tasks):
            return self.tasks[task_id].get('instruction')
        return None

    def get_task_payload(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full task payload by ID (for domain-specific analyses)."""
        if 0 <= task_id < len(self.tasks):
            return dict(self.tasks[task_id])
        return None

    def should_fail(self, task_id: int, operator_name: str,
                    params: Optional[Dict[str, Any]] = None,
                    state: Optional[Any] = None) -> bool:
        """Delegate to failure injector (used by baselines via _scenario_should_fail)."""
        return self.failure_injector.should_fail(task_id, operator_name, params=params, state=state)

    def get_failure_details(self, task_id: int, operator_name: str,
                            params: Optional[Dict[str, Any]] = None,
                            state: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Delegate to failure injector (used by baselines via _scenario_failure_details)."""
        return self.failure_injector.get_failure_details(task_id, operator_name, params=params, state=state)

    def mark_operator_patched(self, operator_name: str) -> None:
        """Delegate to the failure injector (matches TravelPlanningScenario interface)."""
        self.failure_injector.mark_operator_patched(operator_name)


class EcommerceFailureInjector:
    """Injects failures into e-commerce operations."""

    FAILURE_MODES = [
        'inventory_insufficient',
        'promo_expired',
        'promo_conflict',
        'shipping_restricted',
        'payment_declined',
        'policy_violation',
        'price_changed',
        'return_window_exceeded',
        'tool_schema_drift',   # Patchable: API v1→v2 field rename; triggers UPDATE_TOOL_SCHEMA
        'auth_schema_drift',   # Patchable but governance-sensitive: auth_token -> signed_session_token
        'missing_field_validation',  # Patchable: missing required field; triggers ADD_PRECONDITION
        'api_timeout',               # Patchable: service timeout; triggers REFINE_EFFECT
    ]

    def __init__(self, failure_rate: float, horizon: int, inventory_limits: Dict[str, int],
                 min_failures_in_prefix: int, prefix_len: int, seed: int,
                 difficulty_config: Dict[str, Any],
                 placeorder_force_mode: Optional[str] = None):
        self.failure_rate = failure_rate
        self.horizon = horizon
        self.inventory_limits = inventory_limits
        self.min_failures_in_prefix = min_failures_in_prefix
        self.prefix_len = prefix_len
        self.difficulty_config = difficulty_config
        # When set, forces PlaceOrder to always use this failure mode (controlled experiments).
        self.placeorder_force_mode = placeorder_force_mode
        self.rng = random.Random(seed)

        self.failure_schedule = self._generate_failure_schedule()
        self.patched_operators: Set[str] = set()
        self.failure_counts = {mode: 0 for mode in self.FAILURE_MODES}
        self._intended_failure_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    def _generate_failure_schedule(self) -> List[bool]:
        """Pre-generate which tasks will fail."""
        schedule = [False] * self.horizon

        # Ensure minimum failures in prefix
        prefix_failures = 0
        for i in range(min(self.prefix_len, self.horizon)):
            if self.rng.random() < self.failure_rate:
                schedule[i] = True
                prefix_failures += 1

        while prefix_failures < self.min_failures_in_prefix and self.prefix_len > 0:
            idx = self.rng.randint(0, min(self.prefix_len - 1, self.horizon - 1))
            if not schedule[idx]:
                schedule[idx] = True
                prefix_failures += 1

        # Rest of schedule
        for i in range(self.prefix_len, self.horizon):
            schedule[i] = self.rng.random() < self.failure_rate

        return schedule

    def _context_key(self, params: Optional[Dict[str, Any]]) -> str:
        if not isinstance(params, dict):
            return ""
        parts = [
            str(params.get("product", "")),
            str(params.get("promo_code", "")),
            str(params.get("location", "")),
            str(params.get("payment_method", "")),
        ]
        return "|".join(parts)

    def should_fail(self, task_id: int, operator_name: str, params: Optional[Dict[str, Any]] = None,
                    state: Optional[Dict[str, Any]] = None) -> bool:
        """Return True if operation should fail (Executor-compatible)."""

        # Don't fail patched operators (learned)
        if operator_name in self.patched_operators:
            return False

        # Negative task IDs are used by canary simulation (task_id=-1);
        # don't inject failures during canary testing.
        if task_id < 0 or task_id >= len(self.failure_schedule) or not self.failure_schedule[task_id]:
            return False

        cache_key = (task_id, operator_name, self._context_key(params))
        if cache_key in self._intended_failure_cache:
            return True

        # Select failure mode based on operator and params
        failure_mode = self._select_failure_mode(operator_name, params or {}, state or {})

        if failure_mode:
            self.failure_counts[failure_mode] += 1
            self._intended_failure_cache[cache_key] = self._generate_failure(
                failure_mode, operator_name, params or {}, state or {}
            )
            return True

        return False

    def get_failure_details(self, task_id: int, operator_name: str, params: Optional[Dict[str, Any]] = None,
                            state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Return cached failure details for a scheduled failure."""
        if task_id >= len(self.failure_schedule) or not self.failure_schedule[task_id]:
            return None
        if operator_name in self.patched_operators:
            return None

        cache_key = (task_id, operator_name, self._context_key(params))
        if cache_key in self._intended_failure_cache:
            return dict(self._intended_failure_cache[cache_key])

        if not self.should_fail(task_id, operator_name, params=params, state=state):
            return None

        return dict(self._intended_failure_cache.get(cache_key, {})) or None

    def _select_failure_mode(self, operator_name: str, params: Dict[str, Any],
                            state: Dict[str, Any]) -> Optional[str]:
        """Choose appropriate failure mode weighted toward patchable bugs."""

        if operator_name == 'PlaceOrder':
            # Controlled stress-test override: always inject the specified mode.
            # Used by run_ecommerce_stress.py to guarantee tool_schema_drift fires,
            # mirroring how the travel stress test controls BookFlight:API_drift.
            if self.placeorder_force_mode is not None:
                return self.placeorder_force_mode

            # PlaceOrder always gets tool_schema_drift (UPDATE_TOOL_SCHEMA).
            # Other patchable ToolError types (ADD_PRECONDITION, REFINE_EFFECT) are
            # assigned to ApplyPromoCode and CalculateShipping respectively, ensuring
            # patch diversity across operators.
            rand = self.rng.random()

            if rand < 0.45:  # 45% - Tool schema drift (patchable, triggers FDKA)
                return 'tool_schema_drift'
            elif rand < 0.65:  # 20% - Payment declined (patchable, in-episode repair)
                return 'payment_declined'
            elif rand < 0.78:  # 13% - Promo issues (constraint)
                if params.get('promo_code') and self.rng.random() < 0.5:
                    return 'promo_expired'  # Constraint
                return 'payment_declined'  # Patchable fallback
            elif rand < 0.90:  # 12% - Price/policy (constraints, for realism)
                if self.rng.random() < 0.5:
                    return 'price_changed'
                return 'policy_violation'
            else:  # 10% - Inventory (constraint, for realism)
                product = params.get('product', '')
                inventory = state.get('inventory', {})
                try:
                    quantity = int(params.get('quantity', 1) or 1)
                except (TypeError, ValueError):
                    quantity = 1
                if inventory.get(product, 9999) < quantity:
                    return 'inventory_insufficient'
                return 'tool_schema_drift'  # Fallback to patchable FDKA target

        elif operator_name == 'CalculateShipping':
            # Patchable: service timeout triggers REFINE_EFFECT via FDKA.
            # High probability ensures this operator contributes a distinct patch type.
            if self.rng.random() < 0.75:
                return 'api_timeout'
            return 'payment_declined'  # Fallback (in-episode repair)

        elif operator_name == 'ApplyPromoCode':
            # Patchable: missing field validation triggers ADD_PRECONDITION via FDKA.
            # High probability ensures this operator contributes a distinct patch type.
            rand = self.rng.random()
            if rand < 0.70:
                return 'missing_field_validation'
            elif rand < 0.90:
                return 'payment_declined'  # Validation failure (patchable, in-episode)
            return 'promo_expired'  # Constraint

        elif operator_name == 'ProcessRefund':
            # Avoid time-window constraints
            return 'payment_declined'  # Validation failure (patchable)

        elif operator_name == 'UpdateInventory':
            return 'payment_declined'  # Validation failure (patchable)

        # Default to patchable failure
        return 'payment_declined'

    def _generate_failure(self, failure_mode: str, operator_name: str,
                         params: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate failure details for given mode."""

        failures = {
            'inventory_insufficient': {
                'error': 'InsufficientInventory',
                'message': f"Only {state.get('inventory', {}).get(params.get('product', ''), 0)} units available",
                'recoverable': True,
            },
            'promo_expired': {
                'error': 'PromoCodeExpired',
                'message': f"Promo code '{params.get('promo_code', '')}' expired on 2024-01-15",
                'recoverable': True,
            },
            'promo_conflict': {
                'error': 'PromoConflict',
                'message': 'Cannot stack multiple promotional discounts',
                'recoverable': True,
            },
            'shipping_restricted': {
                'error': 'ShippingRestricted',
                'message': f"Cannot ship to {params.get('location', '')} addresses",
                'recoverable': True,
            },
            'payment_declined': {
                'error': 'PaymentDeclined',
                'message': 'Payment method declined by processor',
                'recoverable': True,
            },
            'policy_violation': {
                'error': 'PolicyViolation',
                'message': 'Order violates company purchase policy',
                'recoverable': True,
            },
            'price_changed': {
                'error': 'PriceChanged',
                'message': 'Item price changed since cart add',
                'recoverable': True,
            },
            'return_window_exceeded': {
                'error': 'ReturnWindowExceeded',
                'message': f"Return window of {state.get('return_window_days', 30)} days exceeded",
                'recoverable': False,
            },
            'tool_schema_drift': {
                'error': 'ToolError',
                'api_error_type': 'ApiSchemaDeprecated',
                'message': "PlaceOrder API v1 deprecated: endpoint /api/v1/orders replaced by /api/v2/orders",
                'recoverable': False,
            },
            'auth_schema_drift': {
                'error': 'ToolError',
                'api_error_type': 'AuthSchemaDeprecated',
                'message': (
                    'PlaceOrder authentication schema deprecated: legacy auth_token bearer flow removed; '
                    'signed_session_token is now required for checkout requests.'
                ),
                'recoverable': False,
            },
            'missing_field_validation': {
                # ToolError (bypasses verify-before-act) without drift markers →
                # LLM/mock proposes ADD_PRECONDITION.  Message uses a field the
                # operator actually owns so the resulting precondition passes canary.
                'error': 'ToolError',
                'api_error_type': 'MissingFieldValidation',
                'message': f"{operator_name} rejected: payment validation failed for {params.get('product', 'item')} — invalid or unsupported payment method",
                'recoverable': False,
            },
            'api_timeout': {
                # ToolError (bypasses verify-before-act) without drift markers →
                # LLM/mock proposes REFINE_EFFECT with network/timeout guard.
                'error': 'ToolError',
                'api_error_type': 'ApiTimeout',
                'message': f"{operator_name} API timeout: service unavailable after 30s while processing order for {params.get('product', 'item')}",
                'recoverable': False,
            },
        }

        return failures.get(failure_mode, {
            'error': 'UnknownError',
            'message': 'Operation failed',
            'recoverable': True,
        })

    def mark_operator_patched(self, operator_name: str):
        """Mark operator as patched (won't fail again)."""
        self.patched_operators.add(operator_name)
        print(f"ECOMMERCE_INJECTOR: Marked '{operator_name}' as patched")

    def get_statistics(self) -> Dict[str, Any]:
        """Return injector statistics."""
        total_failures = sum(self.failure_counts.values())
        return {
            'total_scheduled_failures': sum(self.failure_schedule),
            'total_injected_failures': total_failures,
            'failure_breakdown': dict(self.failure_counts),
            'patched_operators': list(self.patched_operators),
        }
