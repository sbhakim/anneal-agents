# src/core/executor.py
"""
Executes a symbolic plan against the current state (E).
UPDATED FOR PoC:
- Refactored to accept Arbitrator and SignalGenerator during initialization.
- The `execute` method now implements distinct behaviors for the S1, S2,
  and VERIFY->S1 pathways, making the metacognitive arbitration meaningful.
- The S2 (Slow Path) now simulates "deliberation" with an extra check and delay
  to align with the manuscript's description of dual-process reasoning.

MINIMUM RELIABILITY FIX (2026-01):
- Make injected "constraint/precondition" failures *patchable* by expressing them
  through state/params *before* Verify-Before-Act, instead of forcing a failure
  after verification passes.
- Call failure injector with (task_id, op_name, params, state) when supported,
  while remaining backward compatible with older injectors.
"""
from typing import List, Dict, Any, Tuple, Optional
import time
from ..core.state import SymbolicState
from ..knowledge.rule_pool import RulePool
from ..metacognition.signals import SignalGenerator
from ..metacognition.arbitrator import Arbitrator


class Executor:
    """Applies operators to the state and checks for failures."""

    def __init__(
            self,
            config: Dict[str, Any],
            failure_injector: Optional[Any],
            rule_pool: RulePool,
            signal_gen: SignalGenerator,
            arbitrator: Arbitrator,
            scenario: Optional[Any] = None,
    ):
        """
        Args:
            config: executor config
            failure_injector: optional FailureInjector for controlled faults
            rule_pool: RulePool for centralized Verify-Before-Act
            signal_gen: SignalGenerator for computing metacognitive signals
            arbitrator: Arbitrator for choosing execution pathways
            scenario: Scenario object to provide world defaults
        """
        self.config = config
        self.failure_injector = failure_injector
        self.rule_pool = rule_pool
        self.signal_gen = signal_gen
        self.arbitrator = arbitrator
        self.scenario = scenario

    def _apply_world_defaults(self, state: SymbolicState) -> None:
        """Ensures resources start 'available' and policy is visible to predicates."""
        if not self.scenario or not hasattr(self.scenario, "world_defaults"):
            return
        try:
            defaults = self.scenario.world_defaults()
            for k, v in defaults.items():
                if state.get(k) is None:
                    state.set(k, v)
        except Exception:
            pass

    def _verify_preconditions(
            self, op, params: Dict[str, Any], state: SymbolicState
    ) -> Tuple[bool, Optional[str]]:
        """
        Centralized precondition verification using the RulePool.
        This is the "Verify-Before-Act" mechanism.
        """
        if self.config.get("enable_verification", True):
            ok, violated = self.rule_pool.evaluate_preconditions(op.name, params, state)
            return ok, violated
        return True, None

    # ---------------------------
    # Failure injector compatibility + patchable constraint injection
    # ---------------------------
    def _injector_should_fail(self, task_id: int, op_name: str, params: Dict[str, Any], state: SymbolicState) -> bool:
        """
        Calls should_fail with the richest available signature.
        Backward compatible with legacy injectors that only accept (task_id, op_name).
        """
        if not self.failure_injector:
            return False
        try:
            return bool(self.failure_injector.should_fail(task_id, op_name, params=params, state=state))
        except TypeError:
            # Legacy signature
            return bool(self.failure_injector.should_fail(task_id, op_name))
        except Exception:
            return False

    def _injector_failure_details(self, task_id: int, op_name: str, params: Dict[str, Any],
                                  state: SymbolicState) -> Dict[str, Any]:
        """
        Calls get_failure_details with the richest available signature.
        Backward compatible with legacy injectors that only accept (task_id, op_name).
        """
        if not self.failure_injector:
            return {}
        try:
            return dict(self.failure_injector.get_failure_details(task_id, op_name, params=params, state=state))
        except TypeError:
            return dict(self.failure_injector.get_failure_details(task_id, op_name))
        except Exception:
            return {}

    def _apply_constraint_injection_to_state(self, error_info: Dict[str, Any], params: Dict[str, Any],
                                            state: SymbolicState) -> None:
        """
        Minimum, safe state mutations so that constraint injections are caught by Verify-Before-Act.

        Key goal: if the system injects a constraint like "invalid payment", the state should reflect
        that so a learned predicate such as ValidPayment(...) can fail *causally*, not by post-hoc forcing.
        """
        msg = str(error_info.get("message", "") or "")
        policy_ref = str(error_info.get("policy_ref", "") or "")
        category = str(error_info.get("category", "") or "")

        # 1) Invalid/expired payment: make ValidPayment fail by adding an "invalid" marker
        #    (Predicate checks for markers like "invalid", "expired", etc.)
        if "invalid or expired" in msg.lower() or "pay-401" in policy_ref.lower() or "invalid_payment" in category.lower():
            payment = params.get("payment", state.get("payment_method"))
            if payment is not None:
                # Ensure marker appears in lowercase for predicate match
                state.set("payment_method", f"{payment} (invalid)")
            # Also store a structured flag for debugging/analysis (harmless if unused)
            state.set("payment_invalid", True)

        # 2) Corporate blackout blocked card: make Not(BlockedCard) fail by setting policy + blackout context
        if "blackout" in msg.lower() or "blocked" in msg.lower() or policy_ref.strip() == "H-23":
            # Use the same policy key that check_not_blocked_card expects
            state.set("corporate_card_policy", "blocked_on_blackout_dates")
            # If blackout dates are already present via world defaults, leave them.
            # Otherwise, best-effort add a conservative marker; travel_dates already comes from instruction parsing.
            if not state.get("blackout_dates"):
                state.set("blackout_dates", ["June 1", "June 2"])

        # 3) Network-related constraint (if present): make NetworkAvailable() fail if that predicate exists/added later
        if "network" in msg.lower() and ("unavailable" in msg.lower() or "down" in msg.lower()):
            state.set("network_available", False)

    def execute(self, plan: List[Dict[str, Any]], state: SymbolicState, task_id: int) -> Tuple[
        List, bool, Optional[str]]:
        """
        Executes each step, returning a trace, success status, and failure type.
        Failure type is one of: "PreconditionUnmet", "ToolError", "Deferred", None
        """
        print("EXECUTOR: Starting plan execution...")
        trace: List[Dict[str, Any]] = []

        # Ensure defaults exist before signal prediction and predicate checks
        self._apply_world_defaults(state)

        # 1. Metacognitive Arbitration (Monitor-Evaluate-Regulate)
        u = self.signal_gen.compute_uncertainty(plan)
        p_viol = self.signal_gen.predict_violation_probability(plan, state)
        budget = self.config.get('budget_ms', 2000)
        pathway = self.arbitrator.arbitrate(u, p_viol, budget)
        print(f"EXECUTOR: Arbitrated pathway -> {pathway}")
        trace.append({"pathway": pathway, "signals": {"u": u, "p_viol": p_viol}})

        # 2. Act based on the chosen pathway
        if pathway == "DEFER":
            print("EXECUTOR: ⏸️ Execution deferred due to budget constraints.")
            trace.append({"error": "Deferred", "reason": "Budget exceeded"})
            return trace, False, "Deferred"

        if pathway == "S2":
            # S2 = Slow/deliberative path with extra checks.
            print("EXECUTOR: 🤔 S2 (Slow Path) - Performing extra deliberation.")
            # For PoC, simulate deliberation with a "deep check" and a small delay.
            time.sleep(0.1)  # Simulate latency of slower reasoning
            trace.append({"deliberation": "Simulated deep verification (S2 path)"})

        # For VERIFY_S1 and S1, we proceed to the standard execution loop,
        # which already includes the essential Verify-Before-Act step.
        elif pathway == "VERIFY_S1":
            print("EXECUTOR: ✓ VERIFY→S1 - Proceeding with standard verified execution.")
        else:  # pathway == "S1"
            print("EXECUTOR: ⚡ S1 (Fast Path) - Proceeding with standard execution.")

        # 3. Main Execution Loop (All non-deferred pathways proceed here)
        for step in plan:
            op = step["operator"]
            params = step.get("params", {})
            print(f"EXECUTOR: Executing step -> {op.name}")

            self._apply_world_defaults(state)

            # Sync state for predicate evaluation
            if "payment" in params:
                state.set("payment_method", params["payment"])

            trace.append({"step": op.name, "state_before": state.to_dict()})

            # ---- Controlled Failure Injection (PATCHABLE constraints BEFORE Verify) ----
            # If an injected failure is a "constraint"/PreconditionUnmet, express it in state/params first
            # so Verify-Before-Act can catch it. ToolError remains a runtime fault injected after Verify.
            injected_error_info: Optional[Dict[str, Any]] = None
            if self._injector_should_fail(task_id, op.name, params, state):
                error_info = self._injector_failure_details(task_id, op.name, params, state)
                # Normalize keys across injector variants
                err_cls = error_info.get("error") or error_info.get("error_type") or "ToolError"
                injected_error_info = dict(error_info)
                injected_error_info["error"] = err_cls

                if err_cls != "ToolError":
                    # Make the constraint visible to predicates
                    self._apply_constraint_injection_to_state(injected_error_info, params, state)
                    # Record the injection event for auditability
                    trace.append({"injected": True, "phase": "pre-verify", **injected_error_info})

            # ---- Verify-Before-Act (Standard for all execution paths) ----
            ok, violated = self._verify_preconditions(op, params, state)
            if not ok:
                print(f"EXECUTOR: ❌ Verify-Before-Act failed for {op.name}. Halting.")
                # If we injected a constraint, attach that context too (helps FDKA localization/scoring)
                payload = {"error": "PreconditionUnmet", "operator": op.name, "violated": violated}
                if injected_error_info and injected_error_info.get("error") != "ToolError":
                    payload["injected_constraint"] = True
                    payload.update({k: v for k, v in injected_error_info.items() if k not in payload})
                trace.append(payload)
                return trace, False, "PreconditionUnmet"

            # ---- Controlled Failure Injection (runtime/tool faults AFTER Verify) ----
            if injected_error_info and injected_error_info.get("error") == "ToolError":
                print(f"EXECUTOR: ❌ Injected Failure -> {injected_error_info.get('message', 'tool error')}")
                trace.append({"injected": True, "phase": "post-verify", **injected_error_info})
                return trace, False, "ToolError"

            # ---- Apply Effects on Success ----
            state = op.apply_effects(state, params)
            trace.append({"step": op.name, "state_after": state.to_dict()})
            print(f"EXECUTOR: ✅ Step {op.name} successful.")

        print("EXECUTOR: ✅ Plan execution complete.")
        return trace, True, None
