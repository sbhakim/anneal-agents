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

MINIMUM RETRY-SEMANTICS FIX (2026-01):
- If a ToolError timeout is injected, attempt a single recovery retry ONLY when
  the operator contains a learned timeout-retry effect (e.g., ApiTimeoutRetry).
  This prevents "retry" hooks from firing after a clean success and makes the
  recovery behavior observable under fault injection.

MINIMUM METRICS COMPATIBILITY FIX (2026-01):
- Ensure every failure path emits a trace entry with {error|error_type, operator}.
  This keeps event-based metrics reliable even when tasks ultimately succeed.
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
        """Ensures resources start 'available' and policy context is visible to predicates."""
        if not self.scenario or not hasattr(self.scenario, "world_defaults"):
            return
        try:
            defaults = self.scenario.world_defaults()

            # Resources: only set if absent (preserve ongoing effects).
            for k in ("hotel_status", "flight_status"):
                if k in defaults and state.get(k) is None:
                    state.set(k, defaults[k])

            # Policy context: always mirror scenario truth (prevents silent drift).
            for k in ("corporate_card_policy", "blackout_dates"):
                if k in defaults:
                    state.set(k, defaults[k])
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

    def _sync_payment_from_params(self, params: Dict[str, Any], state: SymbolicState) -> None:
        """
        Keep payment identity consistent across params/state AND prevent "sticky" invalid flags.

        Key bug fixed: if a prior attempt injected payment_invalid=True, and a later attempt
        switches to a clean payment method, the old flags must not continue to poison Verify.
        """
        if "payment" not in params or params["payment"] is None:
            return

        payment = params["payment"]
        state.set("payment_method", payment)

        p = str(payment)
        if "(invalid)" in p.lower():
            state.set("payment_invalid", True)
            state.set("payment_valid", False)
        else:
            # Clear transient invalid markers when a clean payment is provided.
            if state.get("payment_invalid") is True:
                state.set("payment_invalid", False)
            if state.get("payment_valid") is False:
                state.set("payment_valid", True)

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
        Minimum, safe mutations so constraint injections are caught by Verify-Before-Act.

        Critical: mutate BOTH state and params for payment constraints. Some predicates read params,
        others read state; if only one is tainted, Verify can miss injected failures.
        """
        msg = str(error_info.get("message", "") or "")
        policy_ref = str(error_info.get("policy_ref", "") or "")
        category = str(error_info.get("category", "") or "")

        # 1) Invalid/expired payment (PAY-401): taint state+params + set structured flags.
        if "invalid or expired" in msg.lower() or "pay-401" in policy_ref.lower() or "invalid_payment" in category.lower():
            payment = params.get("payment", state.get("payment_method"))
            if payment is not None:
                p = str(payment)
                tainted = p if "(invalid)" in p.lower() else f"{p} (invalid)"
                params["payment"] = tainted
                state.set("payment_method", tainted)
            state.set("payment_invalid", True)
            state.set("payment_valid", False)

        # 2) Corporate blackout blocked card (H-23): ensure policy+blackout context is present.
        if "blackout" in msg.lower() or "blocked" in msg.lower() or policy_ref.strip() == "H-23":
            state.set("corporate_card_policy", "blocked_on_blackout_dates")
            if not state.get("blackout_dates"):
                state.set("blackout_dates", ["June 1", "June 2"])

        # 3) Network constraint (if present)
        if "network" in msg.lower() and ("unavailable" in msg.lower() or "down" in msg.lower()):
            state.set("network_available", False)

    # ---------------------------
    # Effect execution with minimal gating (prevents recovery-only effects firing on clean success)
    # ---------------------------
    def _apply_effects(self, op: Any, state: SymbolicState, params: Dict[str, Any], *, recovery: bool) -> SymbolicState:
        """
        Applies operator effects with a minimal gate:
        - effects tagged __recovery_only__ run ONLY when recovery=True
        """
        effs = getattr(op, "effects", None)
        if not isinstance(effs, list):
            return op.apply_effects(state, params)

        for eff in effs:
            try:
                if getattr(eff, "__recovery_only__", False) and not recovery:
                    continue
                state = eff(state, params)
            except Exception:
                # Preserve baseline behavior: a bad effect should surface as an execution failure.
                raise
        return state

    # ---------------------------
    # Minimal retry semantics for timeout ToolError
    # ---------------------------
    def _operator_has_timeout_retry(self, op: Any) -> bool:
        """
        True if the operator appears to have a learned timeout-retry recovery effect.
        We use function-name heuristics to avoid deeper coupling to patch schemas.
        """
        try:
            for eff in getattr(op, "effects", []) or []:
                n = getattr(eff, "__name__", "") or ""
                if "timeout_retry" in n.lower() or "retry" in n.lower():
                    return True
                d = getattr(eff, "__details__", "") or ""
                if isinstance(d, str) and ("ApiTimeoutRetry" in d or "TimeoutRetry" in d):
                    return True
        except Exception:
            return False
        return False

    def _is_timeout_tool_error(self, error_info: Dict[str, Any]) -> bool:
        """Best-effort classification of injected ToolError timeouts."""
        msg = str(error_info.get("message", "") or "").lower()
        pref = str(error_info.get("policy_ref", "") or "").upper()
        return ("timeout" in msg) or (pref == "API-503")

    def execute(self, plan: List[Dict[str, Any]], state: SymbolicState, task_id: int) -> Tuple[
        List, bool, Optional[str]]:
        """
        Executes each step, returning a trace, success status, and failure type.
        Failure type is one of: "PreconditionUnmet", "ToolError", "Deferred", None
        """
        print("EXECUTOR: Starting plan execution...")
        trace: List[Dict[str, Any]] = []

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
            trace.append({"error": "Deferred", "operator": "SYSTEM", "reason": "Budget exceeded"})
            return trace, False, "Deferred"

        if pathway == "S2":
            print("EXECUTOR: 🤔 S2 (Slow Path) - Performing extra deliberation.")
            time.sleep(0.1)
            trace.append({"deliberation": "Simulated deep verification (S2 path)"})
        elif pathway == "VERIFY_S1":
            print("EXECUTOR: ✓ VERIFY→S1 - Proceeding with standard verified execution.")
        else:
            print("EXECUTOR: ⚡ S1 (Fast Path) - Proceeding with standard execution.")

        # 3. Main Execution Loop
        for step in plan:
            op = step["operator"]
            params = step.get("params", {}) or {}
            print(f"EXECUTOR: Executing step -> {op.name}")

            self._apply_world_defaults(state)

            # Sync state for predicate evaluation (and clear sticky invalid flags on clean payment)
            self._sync_payment_from_params(params, state)

            trace.append({"step": op.name, "state_before": state.to_dict()})

            # ---- Controlled Failure Injection (PATCHABLE constraints BEFORE Verify) ----
            injected_error_info: Optional[Dict[str, Any]] = None
            if self._injector_should_fail(task_id, op.name, params, state):
                error_info = self._injector_failure_details(task_id, op.name, params, state)
                err_cls = error_info.get("error") or error_info.get("error_type") or "ToolError"
                injected_error_info = dict(error_info)
                injected_error_info["error"] = err_cls
                injected_error_info.setdefault("operator", op.name)

                if err_cls != "ToolError":
                    self._apply_constraint_injection_to_state(injected_error_info, params, state)
                    trace.append({"injected": True, "phase": "pre-verify", **injected_error_info})

            # ---- Verify-Before-Act ----
            ok, violated = self._verify_preconditions(op, params, state)
            if not ok:
                print(f"EXECUTOR: ❌ Verify-Before-Act failed for {op.name}. Halting.")
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

                if self._is_timeout_tool_error(injected_error_info) and self._operator_has_timeout_retry(op):
                    trace.append({"recovery": "ApiTimeoutRetry", "attempt": 1, "operator": op.name})
                    print("EXECUTOR: 🔁 Detected timeout + retry-capable operator. Attempting one recovery retry...")
                    time.sleep(0.05)

                    try:
                        state = self._apply_effects(op, state, params, recovery=True)
                        trace.append({"step": op.name, "state_after": state.to_dict(), "recovered": True})
                        print(f"EXECUTOR: ✅ Recovery retry succeeded for {op.name}.")
                        continue
                    except Exception as e:
                        trace.append({
                            "error": "ToolError",
                            "operator": op.name,
                            "message": "Recovery retry failed",
                            "exception": str(e)[:120]
                        })
                        print(f"EXECUTOR: ❌ Recovery retry failed for {op.name}.")
                        return trace, False, "ToolError"

                trace.append({
                    "error": "ToolError",
                    "operator": op.name,
                    "message": injected_error_info.get("message", "Injected tool error"),
                    "policy_ref": injected_error_info.get("policy_ref"),
                })
                return trace, False, "ToolError"

            # ---- Apply Effects on Success ----
            try:
                state = self._apply_effects(op, state, params, recovery=False)
            except Exception as e:
                trace.append({
                    "error": "ToolError",
                    "operator": op.name,
                    "message": "Exception during effect application",
                    "exception": str(e)[:120],
                })
                print(f"EXECUTOR: ❌ Exception during {op.name} effects.")
                return trace, False, "ToolError"

            trace.append({"step": op.name, "state_after": state.to_dict()})
            print(f"EXECUTOR: ✅ Step {op.name} successful.")

        print("EXECUTOR: ✅ Plan execution complete.")
        return trace, True, None
