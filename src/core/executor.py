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

MINIMUM TRACE SEMANTICS FIX (2026-01):
- Add `event_type` for trace entries and avoid emitting `error`/`error_type` for
  non-failure annotations (e.g., pre-verify constraint injections that are later repaired).
  This prevents metrics from counting "failures" when the step ultimately succeeds.

MINIMUM OBSERVED-FAILURE SEMANTICS (2026-01):
- Emit a `verify_failed` event (with error/error_type) ONLY when verification actually fails.
- Mark `step_success` as recovered when a verify failure was repaired, so metrics can count
  constraint recoveries without introducing phantom failures.

MINIMUM DEFENSIBILITY UPDATE (2026-01):
- Emit a recovery summary + optional instability hint (no error/error_type) when a task succeeds
  but required repeated recoveries. This supports analysis and can drive later escalation.
"""
from typing import List, Dict, Any, Tuple, Optional
import time
from collections import defaultdict
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

            # Network is used by some patches/predicates; keep a sensible default when missing.
            if "network_available" in defaults and state.get("network_available") is None:
                state.set("network_available", defaults["network_available"])

            # API version baseline.
            if "api_version" in defaults and state.get("api_version") is None:
                state.set("api_version", defaults["api_version"])
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
            if state.get("payment_invalid") is True:
                state.set("payment_invalid", False)
            if state.get("payment_valid") is False:
                state.set("payment_valid", True)

    def _apply_tool_schema_update_hint(self, op: Any, params: Dict[str, Any], state: SymbolicState) -> None:
        """
        Reflect committed UPDATE_TOOL_SCHEMA metadata into runtime state/params.
        This is intentionally conservative: only version/endpoint drift hints are applied.
        """
        try:
            meta = getattr(op, "metadata", {}) or {}
            su = meta.get("schema_update") or ""
            if not isinstance(su, str) or not su.strip():
                return

            s = su.lower()
            if ("/v2/" in s) or ("use /v2" in s) or ("use v2" in s) or ("api-v2" in s) or ("v2/book" in s):
                if str(state.get("api_version") or "").lower() != "v2":
                    state.set("api_version", "v2")
                params.setdefault("api_version", "v2")

                if "/v2/book" in s or "v2/book" in s:
                    state.set("booking_endpoint", "/v2/book")
                    params.setdefault("endpoint", "/v2/book")
        except Exception:
            return

    # ---------------------------
    # Verify→Repair (h=1–2) templates
    # ---------------------------
    def _repair_clear_invalid_marker(self, params: Dict[str, Any], state: SymbolicState) -> bool:
        p = params.get("payment")
        if not p or not isinstance(p, str) or "(invalid)" not in p.lower():
            return False
        cleaned = p.replace("(invalid)", "").replace("(INVALID)", "").strip()
        if cleaned == p:
            return False
        params["payment"] = cleaned
        state.set("payment_method", cleaned)
        state.set("payment_invalid", False)
        state.set("payment_valid", True)
        return True

    def _repair_switch_to_personal_payment(self, params: Dict[str, Any], state: SymbolicState) -> bool:
        cur = params.get("payment", state.get("payment_method"))
        cur_s = str(cur or "")
        if not cur_s:
            return False

        candidates = []
        for k in ("personal_payment_method", "backup_payment_method", "alt_payment_method"):
            v = state.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

        candidates.append("PersonalCard:PC-1134")

        if "personalcard" in cur_s.lower():
            return False

        new_p = candidates[0]
        params["payment"] = new_p
        state.set("payment_method", new_p)
        state.set("payment_invalid", False)
        state.set("payment_valid", True)
        return True

    def _repair_reground_params_from_state(self, params: Dict[str, Any], state: SymbolicState) -> bool:
        changed = False

        def _fill(key: str, fallbacks: List[str]) -> None:
            nonlocal changed
            v = params.get(key)
            if v is not None and (not isinstance(v, str) or v.strip() != ""):
                return
            for sk in fallbacks:
                sv = state.get(sk)
                if sv is None:
                    continue
                if isinstance(sv, str) and sv.strip() == "":
                    continue
                params[key] = sv
                changed = True
                return

        _fill("dates", ["travel_dates", "date", "travel_date"])
        _fill("date", ["travel_date", "travel_dates", "dates"])
        _fill("location", ["travel_location", "destination", "travel_destination", "city"])
        _fill("destination", ["travel_destination", "travel_location", "location"])
        _fill("origin", ["travel_origin"])

        if params.get("payment") in (None, "", "None"):
            pm = state.get("payment_method")
            if pm not in (None, "", "None"):
                params["payment"] = pm
                changed = True

        return changed

    def _repair_backoff_and_refresh(self, params: Dict[str, Any], state: SymbolicState) -> bool:
        time.sleep(float(self.config.get("repair_backoff_s", 0.03)))
        self._apply_world_defaults(state)
        return True

    def _select_repair_action(
            self,
            violated: Optional[str],
            injected_error_info: Optional[Dict[str, Any]],
            params: Dict[str, Any],
            state: SymbolicState
    ) -> Optional[Tuple[str, Any]]:
        v = (violated or "").lower()
        msg = str((injected_error_info or {}).get("message", "") or "").lower()

        if "payment" in v or "card" in v or "blocked" in v:
            if self._repair_clear_invalid_marker(params, state):
                return "CLEAR_INVALID_PAYMENT_MARKER", lambda: True
            return "SWITCH_TO_PERSONAL_PAYMENT", lambda: self._repair_switch_to_personal_payment(params, state)

        if "network" in v or "timeout" in msg:
            return "RETRY_BACKOFF_REFRESH", lambda: self._repair_backoff_and_refresh(params, state)

        if violated in (None, "", "anonymous"):
            return "REGROUND_PARAMS_FROM_STATE", lambda: self._repair_reground_params_from_state(params, state)

        return "RETRY_BACKOFF_REFRESH", lambda: self._repair_backoff_and_refresh(params, state)

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
        """
        msg = str(error_info.get("message", "") or "")
        policy_ref = str(error_info.get("policy_ref", "") or "")
        category = str(error_info.get("category", "") or "")

        if "invalid or expired" in msg.lower() or "pay-401" in policy_ref.lower() or "invalid_payment" in category.lower():
            payment = params.get("payment", state.get("payment_method"))
            if payment is not None:
                p = str(payment)
                tainted = p if "(invalid)" in p.lower() else f"{p} (invalid)"
                params["payment"] = tainted
                state.set("payment_method", tainted)
            state.set("payment_invalid", True)
            state.set("payment_valid", False)

        if "blackout" in msg.lower() or "blocked" in msg.lower() or policy_ref.strip() == "H-23":
            state.set("corporate_card_policy", "blocked_on_blackout_dates")
            if not state.get("blackout_dates"):
                state.set("blackout_dates", ["June 1", "June 2"])

        if "network" in msg.lower() and ("unavailable" in msg.lower() or "down" in msg.lower()):
            state.set("network_available", False)

    # ---------------------------
    # Effect execution with minimal gating
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
                raise
        return state

    # ---------------------------
    # Minimal retry semantics for timeout ToolError
    # ---------------------------
    def _operator_has_timeout_retry(self, op: Any) -> bool:
        """
        True if the operator appears to have a learned timeout-retry recovery effect.
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

        # Track recoveries for analysis (does not affect success/failure semantics).
        recovered_counts: Dict[str, int] = defaultdict(int)

        self._apply_world_defaults(state)

        u = self.signal_gen.compute_uncertainty(plan)
        p_viol = self.signal_gen.predict_violation_probability(plan, state)
        budget = self.config.get('budget_ms', 2000)
        pathway = self.arbitrator.arbitrate(u, p_viol, budget)
        print(f"EXECUTOR: Arbitrated pathway -> {pathway}")
        trace.append({"event_type": "meta", "pathway": pathway, "signals": {"u": u, "p_viol": p_viol}})

        if pathway == "DEFER":
            print("EXECUTOR: ⏸️ Execution deferred due to budget constraints.")
            trace.append({
                "event_type": "step_failure",
                "error": "Deferred",
                "error_type": "Deferred",
                "operator": "SYSTEM",
                "reason": "Budget exceeded"
            })
            return trace, False, "Deferred"

        if pathway == "S2":
            print("EXECUTOR: 🤔 S2 (Slow Path) - Performing extra deliberation.")
            time.sleep(0.1)
            trace.append({"event_type": "deliberation", "detail": "Simulated deep verification (S2 path)"})
        elif pathway == "VERIFY_S1":
            print("EXECUTOR: ✓ VERIFY→S1 - Proceeding with standard verified execution.")
        else:
            print("EXECUTOR: ⚡ S1 (Fast Path) - Proceeding with standard execution.")

        max_repairs = int(self.config.get("repair_hops", 2))
        enable_repair = bool(self.config.get("enable_repair", True))

        for step_idx, step in enumerate(plan):
            op = step["operator"]
            params = step.get("params", {}) or {}
            print(f"EXECUTOR: Executing step -> {op.name}")

            self._apply_world_defaults(state)
            self._sync_payment_from_params(params, state)
            self._apply_tool_schema_update_hint(op, params, state)

            trace.append({
                "event_type": "step_start",
                "step": op.name,
                "step_idx": step_idx,
                "state_before": state.to_dict()
            })

            injected_error_info: Optional[Dict[str, Any]] = None
            if self._injector_should_fail(task_id, op.name, params, state):
                error_info = self._injector_failure_details(task_id, op.name, params, state)

                err_cls = error_info.get("error") or error_info.get("error_type") or "ToolError"
                injected_error_info = dict(error_info)
                injected_error_info["error"] = err_cls
                injected_error_info["error_type"] = err_cls
                injected_error_info.setdefault("operator", op.name)

                # For constraint-style injections, mutate state/params so Verify-Before-Act can catch them.
                # IMPORTANT: Do NOT emit `error`/`error_type` here; it may be repaired and should not be
                # counted as a failure event unless verification actually fails.
                if err_cls != "ToolError":
                    self._apply_constraint_injection_to_state(injected_error_info, params, state)
                    trace.append({
                        "event_type": "fault_injected",
                        "injected": True,
                        "phase": "pre-verify",
                        "operator": op.name,
                        "step_idx": step_idx,
                        "fault_class": err_cls,
                        "message": injected_error_info.get("message"),
                        "policy_ref": injected_error_info.get("policy_ref"),
                        "category": injected_error_info.get("category"),
                        "task_id": injected_error_info.get("task_id"),
                        "trace_id": injected_error_info.get("trace_id"),
                    })

            # Track whether this step had a real verify failure that was later repaired.
            had_verify_failure = False
            verify_failed_violated = None

            ok, violated = self._verify_preconditions(op, params, state)
            if not ok:
                had_verify_failure = True
                verify_failed_violated = violated
                # This is a real failure signal (verification failed). Metrics should count it even if repaired.
                payload = {
                    "event_type": "verify_failed",
                    "error": "PreconditionUnmet",
                    "error_type": "PreconditionUnmet",
                    "operator": op.name,
                    "step_idx": step_idx,
                    "violated": violated
                }
                if injected_error_info and injected_error_info.get("error") != "ToolError":
                    payload["injected_constraint"] = True
                    payload.update({k: v for k, v in injected_error_info.items() if k not in payload})
                trace.append(payload)

            if not ok and enable_repair:
                v = violated
                for attempt in range(1, max_repairs + 1):
                    sel = self._select_repair_action(v, injected_error_info, params, state)
                    if not sel:
                        break
                    repair_name, repair_fn = sel
                    trace.append({
                        "event_type": "repair_attempt",
                        "repair": repair_name,
                        "attempt": attempt,
                        "operator": op.name,
                        "step_idx": step_idx,
                        "violated": v
                    })
                    try:
                        applied = bool(repair_fn())
                    except Exception:
                        applied = False
                    if not applied:
                        break
                    self._sync_payment_from_params(params, state)
                    self._apply_tool_schema_update_hint(op, params, state)
                    ok, v = self._verify_preconditions(op, params, state)
                    if ok:
                        violated = None
                        break
                violated = v if not ok else None

            if not ok:
                print(f"EXECUTOR: ❌ Verify-Before-Act failed for {op.name}. Halting.")
                payload = {
                    "event_type": "step_failure",
                    "error": "PreconditionUnmet",
                    "error_type": "PreconditionUnmet",
                    "operator": op.name,
                    "step_idx": step_idx,
                    "violated": violated
                }
                if injected_error_info and injected_error_info.get("error") != "ToolError":
                    payload["injected_constraint"] = True
                    payload.update({k: v for k, v in injected_error_info.items() if k not in payload})
                trace.append(payload)
                return trace, False, "PreconditionUnmet"

            if injected_error_info and injected_error_info.get("error") == "ToolError":
                print(f"EXECUTOR: ❌ Injected Failure -> {injected_error_info.get('message', 'tool error')}")
                trace.append({
                    "event_type": "fault_injected",
                    "injected": True,
                    "phase": "post-verify",
                    "step_idx": step_idx,
                    **injected_error_info
                })

                if self._is_timeout_tool_error(injected_error_info) and self._operator_has_timeout_retry(op):
                    trace.append({
                        "event_type": "recovery_retry",
                        "recovery": "ApiTimeoutRetry",
                        "attempt": 1,
                        "operator": op.name,
                        "step_idx": step_idx
                    })
                    print("EXECUTOR: 🔁 Detected timeout + retry-capable operator. Attempting one recovery retry...")
                    time.sleep(0.05)

                    try:
                        state = self._apply_effects(op, state, params, recovery=True)
                        # NOTE: recovered marker without error/error_type avoids phantom failure events.
                        trace.append({
                            "event_type": "step_success",
                            "step": op.name,
                            "step_idx": step_idx,
                            "state_after": state.to_dict(),
                            "recovered": True,
                            "recovered_from": "ToolError",
                            "recovery": "ApiTimeoutRetry"
                        })
                        recovered_counts[f"{op.name}:ToolError"] += 1
                        print(f"EXECUTOR: ✅ Recovery retry succeeded for {op.name}.")
                        continue
                    except (RuntimeError, ValueError, AttributeError, TypeError, KeyError) as e:
                        trace.append({
                            "event_type": "step_failure",
                            "error": "ToolError",
                            "error_type": "ToolError",
                            "operator": op.name,
                            "step_idx": step_idx,
                            "message": "Recovery retry failed",
                            "exception": str(e)[:120]
                        })
                        print(f"EXECUTOR: ❌ Recovery retry failed for {op.name}.")
                        return trace, False, "ToolError"

                trace.append({
                    "event_type": "step_failure",
                    "error": "ToolError",
                    "error_type": "ToolError",
                    "operator": op.name,
                    "step_idx": step_idx,
                    "message": injected_error_info.get("message", "Injected tool error"),
                    "policy_ref": injected_error_info.get("policy_ref"),
                })
                return trace, False, "ToolError"

            try:
                state = self._apply_effects(op, state, params, recovery=False)
            except (RuntimeError, ValueError, AttributeError, TypeError, KeyError) as e:
                trace.append({
                    "event_type": "step_failure",
                    "error": "ToolError",
                    "error_type": "ToolError",
                    "operator": op.name,
                    "step_idx": step_idx,
                    "message": "Exception during effect application",
                    "exception": str(e)[:120],
                })
                print(f"EXECUTOR: ❌ Exception during {op.name} effects.")
                return trace, False, "ToolError"

            success_event = {"event_type": "step_success", "step": op.name, "step_idx": step_idx, "state_after": state.to_dict()}
            if had_verify_failure:
                success_event["recovered"] = True
                success_event["recovered_from"] = "PreconditionUnmet"
                success_event["violated_initial"] = verify_failed_violated
                recovered_counts[f"{op.name}:PreconditionUnmet"] += 1
            trace.append(success_event)
            print(f"EXECUTOR: ✅ Step {op.name} successful.")

        # Audit-only: summarize recoveries (does not change success semantics).
        if recovered_counts and bool(self.config.get("emit_recovery_summary", True)):
            trace.append({
                "event_type": "recovery_summary",
                "task_id": task_id,
                "recovered_counts": dict(recovered_counts)
            })

            # Optional analysis hook: repeated recoveries indicate instability (useful for later escalation).
            thr = int(self.config.get("recovered_failure_alert_threshold", 2))
            hot = sorted([k for k, c in recovered_counts.items() if c >= thr])
            if hot:
                trace.append({
                    "event_type": "instability_hint",
                    "task_id": task_id,
                    "threshold": thr,
                    "hot_failure_keys": hot,
                    "suggest_fdka": True
                })

        print("EXECUTOR: ✅ Plan execution complete.")
        return trace, True, None
