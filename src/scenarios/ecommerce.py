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

        self.tasks = self._generate_tasks(seed=self.task_seed)
        self.failure_injector = EcommerceFailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            inventory_limits=self.inventory_limits,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config
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
            'restricted_items': [],  # Can be updated mid-run for policy changes
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
    ]

    def __init__(self, failure_rate: float, horizon: int, inventory_limits: Dict[str, int],
                 min_failures_in_prefix: int, prefix_len: int, seed: int,
                 difficulty_config: Dict[str, Any]):
        self.failure_rate = failure_rate
        self.horizon = horizon
        self.inventory_limits = inventory_limits
        self.min_failures_in_prefix = min_failures_in_prefix
        self.prefix_len = prefix_len
        self.difficulty_config = difficulty_config
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

        if task_id >= len(self.failure_schedule) or not self.failure_schedule[task_id]:
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
            # Weighted toward patchable failures (80% patchable)
            rand = self.rng.random()

            if rand < 0.30:  # 30% - Payment declined (patchable)
                return 'payment_declined'
            elif rand < 0.50:  # 20% - Payment validation (patchable)
                return 'payment_declined'
            elif rand < 0.70:  # 20% - Promo issues (10% constraint, 10% patchable)
                if params.get('promo_code') and self.rng.random() < 0.5:
                    return 'promo_expired'  # Constraint
                return 'payment_declined'  # Patchable fallback
            elif rand < 0.85:  # 15% - Price/policy (constraints, for realism)
                if self.rng.random() < 0.5:
                    return 'price_changed'
                return 'policy_violation'
            else:  # 15% - Inventory (constraint, for realism)
                product = params.get('product', '')
                inventory = state.get('inventory', {})
                if inventory.get(product, 9999) < params.get('quantity', 1):
                    return 'inventory_insufficient'
                return 'payment_declined'  # Fallback to patchable

        elif operator_name == 'CalculateShipping':
            # Mostly patchable (avoid shipping restrictions)
            return 'payment_declined'  # Reuse payment validation

        elif operator_name == 'ApplyPromoCode':
            # Mix of validation and constraints
            if self.rng.random() < 0.7:
                return 'payment_declined'  # Validation failure (patchable)
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
