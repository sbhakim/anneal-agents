# src/scenarios/itsm.py
"""
IT Service Management (ITSM) domain scenario for ANNEAL cross-domain evaluation.

This scenario validates that FDKA generalises beyond travel planning and e-commerce
to enterprise IT operations — a third structurally distinct domain.

Operators
---------
  ProvisionAccess[user=..., resource=..., role=..., requester=...]
  DeployPatch[system=..., patch_id=..., environment=..., approved_by=...]
  ResetCredentials[user=..., system=..., method=...]
  CreateTicket[category=..., priority=..., description=...]
  Finish[success | failure: <reason>]

Failure taxonomy
----------------
PATCHABLE — structural bugs that FDKA should detect and repair:

  approval_required           ProvisionAccess : PreconditionUnmet
      Admin-role provisioning invoked without a manager approval code.
      Patch: ADD_PRECONDITION — require approval_code when role ∈ {admin, privileged}.

  change_window_violation     DeployPatch : PreconditionUnmet
      Production deployment attempted outside an approved maintenance window.
      Patch: ADD_PRECONDITION — verify change_window_slot before env=prod deploy.

  mfa_verification_required   ResetCredentials : PreconditionUnmet
      Credential reset called without prior MFA completion.
      Patch: ADD_PRECONDITION — complete MFA before ResetCredentials.

  api_schema_deprecated       DeployPatch : ToolError
      Patch-management API v1 endpoint deprecated; v2 required.
      Patch: MODIFY_PARAMETERS — update endpoint to /api/v2/deployments.

HARD CONSTRAINT — organisational access-control rule; FDKA trigger is expected
but governance should score the proposed patch below acceptance threshold θ:

  priority_escalation_denied  CreateTicket : AccessDenied
      Critical-priority ticket creation requires team-lead role.
      The standard_user role cannot be patched around an RBAC constraint.
      Injector: operator is NEVER marked as permanently patched.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Difficulty presets
# =============================================================================

DIFFICULTY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "failure_rate": 0.25,
        "schema_changes": 0,
        "policy_changes": 0,
        "description": "Single patchable failure class, no environmental drift",
    },
    "normal": {
        "failure_rate": 0.40,
        "schema_changes": 0,
        "policy_changes": 1,
        "description": "Mix of patchable failures with one policy-change point",
    },
    "hard": {
        "failure_rate": 0.60,
        "schema_changes": 1,
        "policy_changes": 2,
        "description": "Multiple patchable classes + schema drift + constraint failure",
    },
}


# =============================================================================
# ITSMScenario
# =============================================================================

class ITSMScenario:
    """
    IT Service Management scenario for ANNEAL.

    Generates realistic ITSM task instructions and manages failure injection
    across four operators.  Failure class assignment is coordinated with task
    generation so that every failing task's natural-language instruction calls
    the operator that will fail — ensuring the LLM-based baselines are not
    penalised by an impossible instruction mismatch.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.difficulty = config.get("difficulty", "normal")
        self.difficulty_config = DIFFICULTY_CONFIGS.get(
            self.difficulty, DIFFICULTY_CONFIGS["normal"]
        )

        self.num_tasks = int(config.get("num_tasks", 25))
        self.failure_rate = float(
            config.get("failure_rate", self.difficulty_config["failure_rate"])
        )
        self.min_failures_in_prefix = int(config.get("min_failures_in_prefix", 4))
        self.prefix_len = int(config.get("prefix_len", min(10, self.num_tasks)))

        self.task_seed = int(config.get("task_generation_seed", 31))
        self.failure_seed = int(config.get("failure_injector_seed", 53))

        # Failure injector is created first so task generation can be coordinated.
        self.failure_injector = ITSMFailureInjector(
            failure_rate=self.failure_rate,
            horizon=self.num_tasks,
            min_failures_in_prefix=self.min_failures_in_prefix,
            prefix_len=self.prefix_len,
            seed=self.failure_seed,
            difficulty_config=self.difficulty_config,
        )

        # Tasks are generated after the injector so instructions match assignments.
        self.tasks: List[str] = self._generate_tasks(seed=self.task_seed)

        print(f"ITSM: Initialised difficulty='{self.difficulty}'")
        print(f"ITSM: {self.num_tasks} tasks generated (task_seed={self.task_seed})")
        print(f"ITSM: failure_rate={self.failure_rate:.0%}, "
              f"failing_tasks={len(self.failure_injector.failing_tasks)}")
        patchable = sum(
            1 for a in self.failure_injector.task_failure_assignments.values()
            if a["patchable"]
        )
        unpatchable = len(self.failure_injector.task_failure_assignments) - patchable
        print(f"ITSM: patchable failures={patchable}, "
              f"hard-constraint (unpatchable)={unpatchable}")

    # ------------------------------------------------------------------
    # Scenario interface (mirrors TravelPlanningScenario / EcommerceScenario)
    # ------------------------------------------------------------------

    def world_defaults(self) -> Dict[str, Any]:
        """Initial world state for each task."""
        return {
            "change_windows": ["Tuesday 22:00-02:00", "Saturday 00:00-06:00"],
            "current_time_slot": "weekday_business_hours",   # outside window
            "api_version": "v1",
            "roles_requiring_approval": ["admin", "privileged", "superuser"],
            "mfa_verified_users": set(),
            "approval_codes_issued": {},
            "patched_operators": set(),
        }

    def get_task(self, task_id: int) -> Optional[str]:
        if 0 <= task_id < len(self.tasks):
            return self.tasks[task_id]
        return None

    def should_inject_failure(
        self,
        task_id: int,
        op_name: str,
        params: Optional[Dict] = None,
        state: Optional[Any] = None,
    ) -> bool:
        return self.failure_injector.should_fail(
            task_id, op_name, params=params, state=state
        )

    def get_failure_details(
        self,
        task_id: int,
        op_name: str,
        params: Optional[Dict] = None,
        state: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.failure_injector.get_failure_details(
            task_id, op_name, params=params, state=state
        )

    def mark_operator_patched(self, operator_name: str) -> None:
        self.failure_injector.mark_operator_patched(operator_name)

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def _generate_tasks(self, seed: int) -> List[str]:
        rng = random.Random(seed)

        users = [
            "alice.johnson", "bob.smith", "charlie.wang",
            "diana.patel", "eve.rodriguez", "frank.kim",
            "grace.okonkwo", "hector.lima",
        ]
        departments = ["Finance", "Engineering", "HR", "Sales", "Legal", "Operations"]
        resources = [
            "billing-system", "employee-records", "source-code-repo",
            "financial-reports", "audit-logs", "customer-database",
            "monitoring-dashboard", "payroll-portal",
        ]
        systems = [
            "VPN", "payroll-system", "billing-portal",
            "code-repository", "monitoring-dashboard", "identity-provider",
            "data-warehouse", "ticketing-system",
        ]
        patch_ids = [
            "KB-2024-0891", "SEC-2024-0112", "HOTFIX-2024-0445",
            "PATCH-2024-0773", "CVE-2024-3141", "CRITICAL-2024-0099",
        ]
        environments = ["prod", "staging"]
        methods = ["email", "sms", "security_key"]
        priorities = ["low", "normal", "high", "critical"]
        categories = [
            "network-outage", "authentication-failure", "performance-degradation",
            "data-corruption", "access-violation", "service-unavailable",
        ]
        services = [
            "checkout-service", "payment-gateway", "user-auth",
            "reporting-pipeline", "data-ingestion", "notification-service",
        ]
        requesters = ["manager.taylor", "lead.chen", "admin.osei"]

        tasks: List[str] = []
        assignments = self.failure_injector.task_failure_assignments

        for i in range(self.num_tasks):
            u = rng.choice(users)
            dept = rng.choice(departments)
            res = rng.choice(resources)
            sys_ = rng.choice(systems)
            pid = rng.choice(patch_ids)
            env = rng.choice(environments)
            meth = rng.choice(methods)
            pri = rng.choice(priorities)
            cat = rng.choice(categories)
            svc = rng.choice(services)
            req = rng.choice(requesters)

            if i in assignments:
                # Generate instruction that naturally exercises the assigned operator.
                fc = assignments[i]["class"]
                op = assignments[i]["operator"]
                task = self._task_for_assignment(
                    fc, op,
                    user=u, department=dept, resource=res, system=sys_,
                    patch_id=pid, environment=env, method=meth,
                    category=cat, service=svc, requester=req,
                    rng=rng,
                )
            else:
                task = self._random_task(
                    user=u, department=dept, resource=res, system=sys_,
                    patch_id=pid, environment=env, method=meth,
                    priority=pri, category=cat, service=svc, requester=req,
                    rng=rng,
                )

            tasks.append(task)

        return tasks

    def _task_for_assignment(
        self, failure_class: str, operator: str, **kw
    ) -> str:
        """Return a task instruction that calls 'operator' and will trigger 'failure_class'."""
        rng: random.Random = kw["rng"]

        if operator == "ProvisionAccess":
            # approval_required fires on role=admin
            role = "admin"
            templates = [
                f"Provision {role} access for {kw['user']} in the "
                f"{kw['department']} department to the {kw['resource']} resource",
                f"Grant {role} privileges to {kw['user']} for {kw['resource']} "
                f"as requested by {kw['requester']}",
                f"Set up {role} account for new hire {kw['user']} on "
                f"{kw['resource']} (approved by {kw['requester']})",
            ]
            return rng.choice(templates)

        if operator == "DeployPatch":
            if failure_class == "api_schema_deprecated":
                env = "prod"
            else:
                # change_window_violation fires on env=prod
                env = "prod"
            templates = [
                f"Deploy security patch {kw['patch_id']} to {env} "
                f"{kw['system']} servers",
                f"Apply hotfix {kw['patch_id']} to {env} environment on "
                f"{kw['system']}",
                f"Roll out patch {kw['patch_id']} to {env} {kw['system']} "
                f"as approved by {kw['requester']}",
            ]
            return rng.choice(templates)

        if operator == "ResetCredentials":
            # mfa_verification_required fires always
            templates = [
                f"Reset credentials for {kw['user']} on the {kw['system']} "
                f"system using {kw['method']} verification",
                f"Unlock and reset password for {kw['user']} locked out of "
                f"{kw['system']}",
                f"Perform credential rotation for {kw['user']} on "
                f"{kw['system']}",
            ]
            return rng.choice(templates)

        if operator == "CreateTicket":
            # priority_escalation_denied fires on priority=critical
            priority = "critical"
            templates = [
                f"Create a {priority}-priority incident ticket for "
                f"{kw['category']} affecting the {kw['service']}",
                f"Open a {priority} incident for {kw['service']} "
                f"{kw['category']} issue (immediate action required)",
                f"File a {priority} severity ticket: {kw['service']} "
                f"experiencing {kw['category']}",
            ]
            return rng.choice(templates)

        # Fallback
        return f"Complete IT service task for {kw['user']}"

    def _random_task(self, **kw) -> str:
        """Random healthy-path ITSM task (no failure expected)."""
        rng: random.Random = kw["rng"]
        templates = [
            f"Provision readonly access for {kw['user']} to the "
            f"{kw['resource']} dashboard",
            f"Deploy patch {kw['patch_id']} to {kw['system']} staging "
            f"environment for testing",
            f"Reset credentials for {kw['user']} on {kw['system']} "
            f"using {kw['method']} verification",
            f"Create a {kw['priority']}-priority ticket for "
            f"{kw['category']} in {kw['service']}",
            f"Provision contributor access for {kw['user']} in "
            f"{kw['department']} to {kw['resource']}",
            f"Apply configuration update to {kw['system']} in staging",
        ]
        return rng.choice(templates)


# =============================================================================
# ITSMFailureInjector
# =============================================================================

class ITSMFailureInjector:
    """
    Injects failures into ITSM operator executions.

    Failure class assignment is done at init time so that task generation
    can produce matching instructions.  Each failing task is assigned exactly
    one failure class targeting one operator.

    Hard-constraint failures (priority_escalation_denied) are never suppressed
    by mark_operator_patched — the RBAC constraint cannot be removed by an
    operator-level patch.
    """

    # Failure classes that are NOT patchable (organisational hard constraints).
    HARD_CONSTRAINT_CLASSES: Set[str] = {"priority_escalation_denied"}

    FAILURE_CLASSES: Dict[str, Dict[str, Any]] = {
        # ---- Patchable -------------------------------------------------------
        "approval_required": {
            "operator": "ProvisionAccess",
            "error_type": "PreconditionUnmet",
            "error": "PreconditionUnmet",
            "message": (
                "Admin role provisioning requires a manager approval code. "
                "Parameter 'approval_code' is missing. "
                "Obtain approval from your manager before calling ProvisionAccess with role=admin."
            ),
            "policy_ref": "IAM-401",
            "category": "structural",
            "patchable": True,
        },
        "change_window_violation": {
            "operator": "DeployPatch",
            "error_type": "PreconditionUnmet",
            "error": "PreconditionUnmet",
            "message": (
                "Production deployments require an approved change-window slot. "
                "Current time is outside all approved maintenance windows "
                "(Tuesday 22:00-02:00, Saturday 00:00-06:00). "
                "Book a change window or deploy to staging first."
            ),
            "policy_ref": "CHG-301",
            "category": "structural",
            "patchable": True,
        },
        "mfa_verification_required": {
            "operator": "ResetCredentials",
            "error_type": "PreconditionUnmet",
            "error": "PreconditionUnmet",
            "message": (
                "Credential reset requires MFA verification to be completed first. "
                "The user has not completed MFA for this session. "
                "Call InitiateMFA or use method=security_key with a registered device."
            ),
            "policy_ref": "SEC-202",
            "category": "structural",
            "patchable": True,
        },
        "api_schema_deprecated": {
            "operator": "DeployPatch",
            "error_type": "ToolError",
            "error": "ToolError",
            "message": (
                "Patch management API v1 is deprecated and has been shut down. "
                "Endpoint /api/v1/patches no longer accepts requests. "
                "Migrate to /api/v2/deployments — update the 'api_version' parameter."
            ),
            "policy_ref": "API-V2",
            "category": "environmental",
            "patchable": True,
        },
        # ---- Hard constraint (not patchable) ---------------------------------
        "priority_escalation_denied": {
            "operator": "CreateTicket",
            "error_type": "AccessDenied",
            "error": "AccessDenied",
            "message": (
                "Critical-priority ticket creation requires team-lead role or above. "
                "Current role: standard_user. Access denied. "
                "Use priority=high or escalate via your team lead."
            ),
            "policy_ref": "RBAC-101",
            "category": "constraint",
            "patchable": False,
        },
    }

    def __init__(
        self,
        failure_rate: float,
        horizon: int,
        min_failures_in_prefix: int,
        prefix_len: int,
        seed: int,
        difficulty_config: Dict[str, Any],
    ) -> None:
        self.failure_rate = failure_rate
        self.horizon = horizon
        self.min_failures_in_prefix = min_failures_in_prefix
        self.prefix_len = prefix_len
        self.difficulty_config = difficulty_config
        self.rng = random.Random(seed)

        # Schema-change and policy-change event task indices.
        self.schema_change_points: Set[int] = set(
            self._get_event_points("schema_changes")
        )
        self.policy_change_points: Set[int] = set(
            self._get_event_points("policy_changes")
        )

        # Which tasks will fail (deterministic, seeded).
        self.failing_tasks: Set[int] = self._select_failing_tasks()

        # Assign one failure class + operator to each failing task.
        self.task_failure_assignments: Dict[int, Dict[str, Any]] = (
            self._assign_failure_classes()
        )

        # Runtime state.
        self.patched_operators: Set[str] = set()
        self._patch_successes: Dict[str, int] = {}
        self._failure_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

        print(
            f"ITSM_INJECTOR: seed={seed}, "
            f"failing_tasks={sorted(self.failing_tasks)}"
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _get_event_points(self, event_type: str) -> List[int]:
        count = int(self.difficulty_config.get(event_type, 0))
        if count == 0:
            return []
        return [int(i * self.horizon / (count + 1)) for i in range(1, count + 1)]

    def _select_failing_tasks(self) -> Set[int]:
        """Select which task IDs will fail, with minimum guaranteed in prefix."""
        failing: Set[int] = set()
        prefix_tasks = list(range(self.prefix_len))
        self.rng.shuffle(prefix_tasks)
        for t in prefix_tasks[: min(self.min_failures_in_prefix, self.prefix_len)]:
            failing.add(t)

        target = max(0, round(self.horizon * self.failure_rate))
        all_tasks = list(range(self.horizon))
        self.rng.shuffle(all_tasks)
        for t in all_tasks:
            if len(failing) >= target:
                break
            failing.add(t)
        return failing

    def _assign_failure_classes(self) -> Dict[int, Dict[str, Any]]:
        """
        Assign one failure class to each failing task.

        Distribution (approximate):
          35% approval_required       (ProvisionAccess — patchable)
          25% change_window_violation (DeployPatch    — patchable)
          20% mfa_verification_required (ResetCredentials — patchable)
          10% api_schema_deprecated   (DeployPatch    — patchable, schema-change tasks)
          10% priority_escalation_denied (CreateTicket — hard constraint)
        """
        rotation = [
            "approval_required",
            "change_window_violation",
            "approval_required",
            "mfa_verification_required",
            "change_window_violation",
            "approval_required",
            "mfa_verification_required",
            "priority_escalation_denied",
            "change_window_violation",
            "approval_required",
        ]
        assignments: Dict[int, Dict[str, Any]] = {}
        rot_idx = 0

        for task_id in sorted(self.failing_tasks):
            # Schema-change points always get api_schema_deprecated.
            if task_id in self.schema_change_points:
                fc_name = "api_schema_deprecated"
            else:
                fc_name = rotation[rot_idx % len(rotation)]
                rot_idx += 1

            fc = self.FAILURE_CLASSES[fc_name]
            assignments[task_id] = {
                "class": fc_name,
                "operator": fc["operator"],
                "patchable": fc["patchable"],
            }

        return assignments

    # ------------------------------------------------------------------
    # Runtime failure injection interface
    # ------------------------------------------------------------------

    def should_fail(
        self,
        task_id: int,
        operator_name: str,
        params: Optional[Dict] = None,
        state: Optional[Any] = None,
    ) -> bool:
        if task_id not in self.failing_tasks:
            return False

        assignment = self.task_failure_assignments.get(task_id)
        if assignment is None:
            return False

        # The failure only fires when the ASSIGNED operator is called.
        if operator_name != assignment["operator"]:
            return False

        fc_name = assignment["class"]

        # Hard-constraint failures are NEVER suppressed by patching.
        if fc_name in self.HARD_CONSTRAINT_CLASSES:
            return True

        # Patchable failures stop firing once the operator is patched.
        if operator_name in self.patched_operators:
            return False

        # api_schema_deprecated: one-shot within a task (retry with v2 recovers).
        if fc_name == "api_schema_deprecated":
            cache_key = (task_id, operator_name)
            if cache_key in self._failure_cache:
                # Already fired once this task; allow retry to succeed.
                return False

        return True

    def get_failure_details(
        self,
        task_id: int,
        operator_name: str,
        params: Optional[Dict] = None,
        state: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_fail(task_id, operator_name, params, state):
            return None

        assignment = self.task_failure_assignments[task_id]
        fc_name = assignment["class"]
        fc = dict(self.FAILURE_CLASSES[fc_name])

        details: Dict[str, Any] = {
            **fc,
            "operator": operator_name,
            "task_id": task_id,
            "trace_id": f"ITSM-{task_id:04d}",
            "failure_class": fc_name,
        }

        # Cache api_schema_deprecated to enforce one-shot behaviour.
        if fc_name == "api_schema_deprecated":
            cache_key = (task_id, operator_name)
            self._failure_cache[cache_key] = details

        return details

    # ------------------------------------------------------------------
    # Patch management
    # ------------------------------------------------------------------

    def mark_operator_patched(self, operator_name: str) -> None:
        """
        Record a successful execution of a previously-failing operator.
        After 3 successes the operator is permanently exempted from injection.

        Hard-constraint operators are ignored — their failure cannot be
        resolved by an operator-level patch.
        """
        # Identify the failure class assigned to this operator across tasks.
        fc_for_op = next(
            (
                fc_name
                for fc_name, fc in self.FAILURE_CLASSES.items()
                if fc["operator"] == operator_name
            ),
            None,
        )

        if fc_for_op and fc_for_op in self.HARD_CONSTRAINT_CLASSES:
            print(
                f"ITSM_INJECTOR: ⚠  Cannot patch {operator_name} — "
                f"hard RBAC constraint ('{fc_for_op}'). "
                f"Failure will persist regardless of proposed patch."
            )
            return

        self._patch_successes[operator_name] = (
            self._patch_successes.get(operator_name, 0) + 1
        )
        threshold = 3
        if self._patch_successes[operator_name] >= threshold:
            self.patched_operators.add(operator_name)
            print(
                f"ITSM_INJECTOR: ✓ {operator_name} permanently patched "
                f"({threshold} successes confirmed)."
            )
        else:
            print(
                f"ITSM_INJECTOR: Patch progress {operator_name}: "
                f"{self._patch_successes[operator_name]}/{threshold}"
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        class_counts: Dict[str, int] = {}
        patchable = 0
        unpatchable = 0
        for a in self.task_failure_assignments.values():
            class_counts[a["class"]] = class_counts.get(a["class"], 0) + 1
            if a["patchable"]:
                patchable += 1
            else:
                unpatchable += 1
        return {
            "total_failing_tasks": len(self.failing_tasks),
            "patchable": patchable,
            "unpatchable_hard_constraint": unpatchable,
            "failure_class_distribution": class_counts,
            "patched_operators": list(self.patched_operators),
            "schema_change_points": sorted(self.schema_change_points),
            "policy_change_points": sorted(self.policy_change_points),
        }
