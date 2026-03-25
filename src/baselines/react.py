# src/baselines/react.py
"""
ReAct Baseline (Yao et al., ICLR 2023)
"ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.03629

Key properties of this implementation:
- Thought → Action → Observation loop (per-step LLM calls)
- NO cross-episode memory: each task starts completely fresh
- Real LLM calls via OpenAI (same model as ANNEAL: gpt-4o-mini)
- NO symbolic operators, NO verify-before-act, NO FDKA, NO governance
- Failure signal arrives as natural language Observation text
- max_steps guard prevents infinite loops (ReAct's primary failure mode)

Expected behaviour vs ANNEAL:
- SR ~40-55%: LLM can reason within an episode but cannot patch operators
- RFR_terminal high (>40%): same failure class repeats across tasks (no memory)
- TTA = infinity: no cross-episode structural learning by design
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..scenarios import SCENARIO_MAP
from ..scenarios.travel_planning import TravelPlanningScenario
from ..utils.metrics import MetricsCollector
from ..utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Few-shot prompt template (travel domain)
# One example showing a successful booking via Thought/Action/Observation.
# Kept short (1 example) to minimise token usage across 25-50 tasks.
# ---------------------------------------------------------------------------
_TRAVEL_FEW_SHOT = """
Example:
Task: Book a hotel in Chicago for May 3-5
Thought 1: I need to book a hotel in Chicago for May 3-5. Let me check the payment method and book.
Action 1: BookHotel[location=Chicago, dates=May 3-5, payment=CorporateCard:CC-5512]
Observation 1: Success. Hotel booked in Chicago for May 3-5.
Thought 2: The hotel is booked. Task is complete.
Action 2: Finish[success]
""".strip()

_ECOMMERCE_FEW_SHOT = """
Example:
Task: Place an order for 2 laptops using credit_card
Thought 1: I need to place an order for 2 laptops. Payment is credit_card.
Action 1: PlaceOrder[product=laptop, quantity=2, payment_method=credit_card]
Observation 1: Success. Order placed for 2 laptops.
Thought 2: Order is complete.
Action 2: Finish[success]
""".strip()

_ITSM_FEW_SHOT = """
Example:
Task: Provision readonly access for bob.smith to the reporting-dashboard resource
Thought 1: I need to provision readonly access for bob.smith to reporting-dashboard.
Action 1: ProvisionAccess[user=bob.smith, resource=reporting-dashboard, role=readonly, requester=manager.taylor]
Observation 1: Success. Readonly access provisioned for bob.smith.
Thought 2: Task is complete.
Action 2: Finish[success]
""".strip()

_SYSTEM_PROMPT = """You are a task execution agent. You operate in a Thought-Action-Observation loop.

Available actions depend on the domain:
Travel    : BookHotel[location=..., dates=..., payment=...],
            BookFlight[origin=..., destination=..., date=..., payment=...],
            Finish[success|failure]
E-commerce: PlaceOrder[product=..., quantity=..., payment_method=...],
            ApplyPromoCode[promo_code=...], CalculateShipping[location=...],
            ProcessRefund[order_id=..., days_since_purchase=...],
            UpdateInventory[product=..., delta=...], Finish[success|failure]
ITSM      : ProvisionAccess[user=..., resource=..., role=..., requester=...],
            DeployPatch[system=..., patch_id=..., environment=..., approved_by=...],
            ResetCredentials[user=..., system=..., method=...],
            CreateTicket[category=..., priority=..., description=...],
            Finish[success|failure]

Rules:
- Each Action must be on its own line starting with "Action N:" where N is the step number.
- Use Finish[success] when the task is complete.
- Use Finish[failure: <reason>] when the task cannot be completed.
- Do NOT repeat the same failed action without changing parameters.
- Max 8 steps per task.

{few_shot}
"""

# Regex to parse: Action N: OperatorName[key=val, key=val]
_ACTION_RE = re.compile(
    r"Action\s+\d+\s*:\s*(\w+)\[([^\]]*)\]",
    re.IGNORECASE
)
# Regex to parse: Action N: Finish[...]
_FINISH_RE = re.compile(
    r"Action\s+\d+\s*:\s*Finish\[([^\]]*)\]",
    re.IGNORECASE
)


def _parse_action(text: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Parse the FIRST action line (by step number) from LLM output.
    Returns (operator_name, params_dict) or (None, {}) if unparseable.
    Finish actions return ("Finish", {"result": "..."}).

    NOTE: We find the lowest-numbered Action N: line to prevent the LLM from
    accidentally jumping ahead to Finish when it outputs a full trajectory in
    one response. Operator actions are always preferred over Finish at the same
    step number.
    """
    # Collect all Action N: occurrences with their step numbers
    all_re = re.compile(
        r"Action\s+(\d+)\s*:\s*(\w+)\[([^\]]*)\]",
        re.IGNORECASE
    )
    finish_all_re = re.compile(
        r"Action\s+(\d+)\s*:\s*Finish\[([^\]]*)\]",
        re.IGNORECASE
    )

    candidates: List[Tuple[int, str, Dict[str, str]]] = []

    for m in all_re.finditer(text):
        step_num = int(m.group(1))
        operator = m.group(2).strip()
        raw_params = m.group(3).strip()
        params: Dict[str, str] = {}
        if raw_params:
            for part in raw_params.split(","):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    params[k.strip()] = v.strip()
                elif part:
                    params.setdefault("value", part)
        candidates.append((step_num, operator, params))

    for m in finish_all_re.finditer(text):
        step_num = int(m.group(1))
        candidates.append((step_num, "Finish", {"result": m.group(2).strip()}))

    if not candidates:
        return None, {}

    # Take the action with the lowest step number.
    # If tied, prefer non-Finish operators so we execute the action first.
    candidates.sort(key=lambda x: (x[0], 1 if x[1] == "Finish" else 0))
    _, operator, params = candidates[0]
    return operator, params


class ReActAgent:
    """
    ReAct agent baseline.

    Implements the canonical Thought→Action→Observation loop with:
    - Real LLM inference (OpenAI gpt-4o-mini, temperature=0.3)
    - No persistent memory across tasks
    - No symbolic operator patching
    - Same scenario/metrics infrastructure as ANNEAL for fair comparison
    """

    MAX_STEPS = 8  # match paper (6-10 typical range)
    _ALLOWED_OPS = {
        "travel_planning": {"BookHotel", "BookFlight"},
        "travel_stochastic": {"BookHotel", "BookFlight"},
        "ecommerce": {"PlaceOrder", "ApplyPromoCode", "CalculateShipping", "ProcessRefund", "UpdateInventory"},
        "itsm": {"ProvisionAccess", "DeployPatch", "ResetCredentials", "CreateTicket"},
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        log_cfg = dict(config.get("logging", {}))
        log_cfg["level"] = log_cfg.get("level", "WARNING")
        self.logger = setup_logger(log_cfg, name="react")

        self.logger.info("=" * 70)
        self.logger.info("Initializing ReAct Baseline (Yao et al., ICLR 2023)...")
        self.logger.info("=" * 70)

        # Scenario (same as ANNEAL — fair comparison)
        scenario_name = config.get("scenario", {}).get("name", "travel_planning")
        scenario_class = SCENARIO_MAP.get(scenario_name, TravelPlanningScenario)
        self.scenario = scenario_class(config["scenario"])
        self.scenario_name = scenario_name
        self.allowed_operators = set(self._ALLOWED_OPS.get(scenario_name, {"BookHotel", "BookFlight"}))

        # Metrics (identical collector to ANNEAL)
        self.metrics = MetricsCollector()
        self.metrics.set_run_metadata({
            "baseline": "react",
            "difficulty": config.get("scenario", {}).get("difficulty"),
            "seed": config.get("scenario", {}).get("task_generation_seed"),
            "llm_provider": self.llm_provider if hasattr(self, "llm_provider") else None,
            "model": self.model if hasattr(self, "model") else None,
            "fdka_enabled": False,
        })

        # LLM config — read from same fdka.propose_edit block for consistency
        pe_cfg = config.get("fdka", {}).get("propose_edit", {}) or {}
        self.llm_provider = pe_cfg.get("llm_provider", "mock")
        self.model = pe_cfg.get("model", "gpt-4o-mini")
        self.temperature = float(pe_cfg.get("temperature", 0.3))
        self.max_tokens = 512
        self.timeout = float(pe_cfg.get("timeout_sec", 60.0))
        self.metrics.set_run_metadata({
            "baseline": "react",
            "difficulty": config.get("scenario", {}).get("difficulty"),
            "seed": config.get("scenario", {}).get("task_generation_seed"),
            "llm_provider": self.llm_provider,
            "model": self.model,
            "fdka_enabled": False,
        })

        # Initialise OpenAI client if needed
        self._openai_client = None
        if self.llm_provider == "openai":
            self._init_openai()

        # Select few-shot example by domain
        if "ecommerce" in scenario_name:
            self._few_shot = _ECOMMERCE_FEW_SHOT
        elif "itsm" in scenario_name:
            self._few_shot = _ITSM_FEW_SHOT
        else:
            self._few_shot = _TRAVEL_FEW_SHOT
        self._system_prompt = _SYSTEM_PROMPT.format(few_shot=self._few_shot)

        self.logger.info(f"✅ ReAct baseline ready")
        self.logger.info(f"   LLM: {self.model} (provider: {self.llm_provider})")
        self.logger.info(f"   Max steps per task: {self.MAX_STEPS}")
        self.logger.info(f"   Cross-episode memory: DISABLED (by design)")
        self.logger.info("=" * 70)

    # ------------------------------------------------------------------
    # OpenAI init
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        try:
            import openai
            self._openai_client = openai.OpenAI()
            self.logger.info("REACT: OpenAI client initialised.")
        except Exception as e:
            self.logger.warning(f"REACT: OpenAI init failed ({e}). Falling back to mock.")
            self.llm_provider = "mock"

    # ------------------------------------------------------------------
    # Public API — matches ANNEALSystem interface used by main.py
    # ------------------------------------------------------------------

    def run_evaluation(self) -> MetricsCollector:
        print(f"\n{'=' * 70}")
        print("🔁 ReAct BASELINE EVALUATION (Yao et al., ICLR 2023)")
        print(f"   Provider: {self.llm_provider} | Model: {self.model}")
        print(f"   Memory: NONE (fresh context each task — pure ReAct)")
        print(f"{'=' * 70}")

        num_tasks = self.scenario.num_tasks
        for task_id in range(num_tasks):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self._run_react_episode(task_id, instruction)

        print(f"\n{'=' * 70}")
        print("✅ ReAct EVALUATION COMPLETE")
        print(f"{'=' * 70}")
        self.metrics.print_summary()
        output_dir = Path(self.config.get("output", {}).get("results_dir", "data/results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.save(output_dir / "metrics.json")
        return self.metrics

    def run_task(self, task_id: int, instruction: str) -> Dict[str, Any]:
        """Single-task interface (mirrors ANNEALSystem.run_task)."""
        self._run_react_episode(task_id, instruction)
        task_data = next(
            (t for t in self.metrics.tasks if t.get("task_id") == task_id), None
        )
        status = "success" if (task_data and task_data.get("success")) else "failure"
        return {
            "status": status,
            "trace": task_data.get("trace", []) if task_data else [],
            "metrics": self.metrics.get_summary(),
        }

    # ------------------------------------------------------------------
    # Core ReAct episode loop
    # ------------------------------------------------------------------

    def _run_react_episode(self, task_id: int, instruction: str) -> None:
        """
        Run one task as a ReAct episode.
        Context is built fresh — NO memory from previous tasks.
        """
        print(f"\n{'=' * 70}")
        print(f"Task {task_id + 1}: {instruction}")
        print(f"{'=' * 70}")

        # Fresh context per task (the defining property of ReAct vs Reflexion)
        context = (
            f"{self._system_prompt}\n\n"
            f"Now solve:\nTask: {instruction}\n"
        )

        trace: List[Dict] = []
        success = False
        last_failure: Optional[Dict] = None
        step = 0
        consecutive_same_action = 0
        last_action_sig = ""

        for step in range(1, self.MAX_STEPS + 1):
            # 1. Call LLM → get Thought + Action
            llm_output = self._call_llm(context, step)
            print(f"  [Step {step}] LLM → {llm_output[:120].strip()}...")

            # 2. Parse action
            operator, params = _parse_action(llm_output)

            # 3. Handle unparseable output (treat as failure)
            if operator is None:
                observation = "ERROR: Could not parse a valid Action from your response. Please format as 'Action N: OperatorName[key=val]'."
                print(f"  [Step {step}] ⚠️  Unparseable action.")
                trace.append({
                    "step": step, "event_type": "parse_error",
                    "llm_output": llm_output, "observation": observation
                })
                context += f"\n{llm_output}\nObservation {step}: {observation}\n"
                continue

            # 4. Finish action → episode ends
            if operator == "Finish":
                result = params.get("result", "")
                success = "failure" not in result.lower()
                print(f"  [Step {step}] {'✅' if success else '❌'} Finish[{result}]")
                trace.append({
                    "step": step, "event_type": "finish",
                    "result": result, "success": success
                })
                break

            if operator not in self.allowed_operators:
                observation = (
                    f"ERROR: Unsupported action '{operator}' for scenario '{self.scenario_name}'. "
                    f"Allowed actions: {', '.join(sorted(self.allowed_operators))}."
                )
                print(f"  [Step {step}] ⚠️  Unsupported action '{operator}'.")
                trace.append({
                    "step": step,
                    "event_type": "invalid_action",
                    "operator": operator,
                    "params": params,
                    "error": "InvalidAction",
                    "observation": observation,
                })
                context += f"\n{llm_output}\nObservation {step}: {observation}\n"
                last_failure = {
                    "operator": operator,
                    "error_type": "InvalidAction",
                    "message": observation,
                    "policy_ref": "",
                }
                continue

            # 5. Detect repetitive loop (ReAct's primary failure mode)
            action_sig = f"{operator}:{json.dumps(params, sort_keys=True)}"
            if action_sig == last_action_sig:
                consecutive_same_action += 1
                if consecutive_same_action >= 2:
                    observation = f"ERROR: You have repeated the same action '{operator}' with identical parameters. Try a different approach or call Finish[failure: stuck in loop]."
                    print(f"  [Step {step}] 🔄 Repetition detected — injecting loop warning.")
                    context += f"\n{llm_output}\nObservation {step}: {observation}\n"
                    trace.append({
                        "step": step, "event_type": "loop_warning",
                        "operator": operator, "params": params
                    })
                    continue
            else:
                consecutive_same_action = 0
            last_action_sig = action_sig

            # 6. Execute action → get observation
            observation, step_success, failure_info = self._execute_action(
                operator, params, task_id, step
            )
            print(f"  [Step {step}] {'✅' if step_success else '❌'} {operator} → {observation[:80]}")

            trace.append({
                "step": step, "event_type": "step_success" if step_success else "step_failure",
                "operator": operator, "params": params,
                "observation": observation, "success": step_success,
                "failure_info": failure_info,
            })
            if not step_success and failure_info:
                trace[-1]["error"] = failure_info.get("error_type", "Error")
                trace[-1]["error_type"] = failure_info.get("error_type", "Error")
                trace[-1]["policy_ref"] = failure_info.get("policy_ref", "")
            elif step_success and last_failure:
                trace[-1]["recovered"] = True
                trace[-1]["recovered_from"] = (
                    f"{last_failure.get('operator', 'unknown')}:"
                    f"{last_failure.get('error_type', 'unknown')}"
                )

            if failure_info:
                last_failure = failure_info

            context += f"\n{llm_output}\nObservation {step}: {observation}\n"

        else:
            # max_steps exhausted without Finish
            print(f"  ❌ Max steps ({self.MAX_STEPS}) reached without finishing.")
            trace.append({"step": step, "event_type": "max_steps_exceeded"})
            success = False

        # Record metrics — same interface as all other agents
        self.metrics.record_task(task_id, success, trace)

        if last_failure and not success:
            # Push failure info so MetricsCollector can track repeat failures
            failure_key = (
                f"{last_failure.get('operator', 'unknown')}:"
                f"{last_failure.get('error_type', 'unknown')}"
            )
            if hasattr(self.metrics, "record_failure_event"):
                self.metrics.record_failure_event(failure_key, task_id)

        print(f"  {'✅ Task Succeeded' if success else '❌ Task Failed'}")

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, context: str, step: int) -> str:
        """
        Call LLM to generate next Thought + Action.
        Falls back to rule-based mock when provider is 'mock' or client unavailable.
        """
        prompt = context + f"\nThought {step}:"

        if self.llm_provider == "openai" and self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    # Stop BEFORE the agent generates its own Observation token —
                    # we supply the real observation from the environment.
                    stop=[f"\nObservation {step}:", f"Observation {step}:"],
                )
                content = response.choices[0].message.content or ""
                # Prepend "Thought N:" prefix if LLM omitted it
                if not content.strip().lower().startswith("thought"):
                    content = f"Thought {step}: {content.strip()}"
                return content
            except Exception as e:
                self.logger.warning(f"REACT: LLM call failed at step {step}: {e}. Using mock.")

        # Mock fallback — deterministic rule-based (matches scenario patterns)
        return self._mock_llm(context, step)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self,
        operator: str,
        params: Dict[str, str],
        task_id: int,
        step: int,
    ) -> Tuple[str, bool, Optional[Dict]]:
        """
        Execute an operator action against the scenario.
        Returns (observation_str, step_success, failure_info_or_None).
        """
        # Normalise payment field (accept both 'payment' and 'payment_method')
        if "payment_method" in params and "payment" not in params:
            params = dict(params, payment=params["payment_method"])

        # Check failure injection
        should_fail = self._scenario_should_fail(task_id, operator, params)

        if should_fail:
            failure_info = self._scenario_failure_details(task_id, operator, params) or {}
            error_type = failure_info.get("error_type", "Error")
            message = failure_info.get("message", "Operation failed.")
            policy_ref = failure_info.get("policy_ref", "")
            obs = f"FAILED ({error_type}): {message}"
            if policy_ref:
                obs += f" [policy: {policy_ref}]"
            return obs, False, {
                "operator": operator,
                "error_type": error_type,
                "message": message,
                "policy_ref": policy_ref,
            }

        # Success path — build confirmation observation
        obs = self._success_observation(operator, params)
        return obs, True, None

    def _scenario_should_fail(self, task_id: int, operator: str, params: Dict[str, str]) -> bool:
        """Support both travel-style and ecommerce-style scenario failure APIs."""
        if hasattr(self.scenario, "should_inject_failure"):
            return bool(self.scenario.should_inject_failure(task_id, operator, params, state=None))
        if hasattr(self.scenario, "should_fail"):
            return bool(self.scenario.should_fail(task_id, operator, params=params, state=None))
        return False

    def _scenario_failure_details(self, task_id: int, operator: str, params: Dict[str, str]) -> Optional[Dict]:
        if hasattr(self.scenario, "get_failure_details"):
            return self.scenario.get_failure_details(task_id, operator, params=params, state=None)
        return None

    def _success_observation(self, operator: str, params: Dict[str, str]) -> str:
        obs_map = {
            "BookHotel": lambda p: f"Success. Hotel booked in {p.get('location', '?')} for {p.get('dates', '?')}.",
            "BookFlight": lambda p: f"Success. Flight booked from {p.get('origin', '?')} to {p.get('destination', '?')} on {p.get('date', '?')}.",
            "PlaceOrder": lambda p: f"Success. Order placed for {p.get('quantity', '?')} x {p.get('product', '?')}.",
            "ApplyPromoCode": lambda p: f"Success. Promo code {p.get('promo_code', '?')} applied.",
            "CalculateShipping": lambda p: f"Success. Shipping calculated for {p.get('location', '?')}.",
            "ProcessRefund": lambda p: f"Success. Refund processed for order {p.get('order_id', '?')}.",
            "UpdateInventory": lambda p: f"Success. Inventory updated for {p.get('product', '?')}.",
        }
        fn = obs_map.get(operator)
        return fn(params) if fn else f"Success. {operator} completed."

    # ------------------------------------------------------------------
    # Mock LLM (deterministic fallback — no real LLM, for testing only)
    # ------------------------------------------------------------------

    def _mock_llm(self, context: str, step: int) -> str:
        """
        Rule-based mock that mimics minimal ReAct behaviour.
        Used when llm_provider='mock' or when the real LLM call fails.
        NOTE: This does NOT use any cross-episode memory (preserving ReAct semantics).
        """
        ctx_lower = context.lower()

        # If a previous observation indicates failure, try alternate params
        if "failed" in ctx_lower and "corporate" in ctx_lower and "blocked" in ctx_lower:
            # ReAct can reason within-episode: switch to personal card
            if "hotel" in ctx_lower:
                return (
                    f"Thought {step}: The corporate card is blocked. I should try a personal card.\n"
                    f"Action {step}: BookHotel[location=Chicago, dates=April 10-12, payment=PersonalCard:PC-1134]"
                )

        if "failed" in ctx_lower and "timeout" in ctx_lower:
            # Retry once with same params
            if "bookflight" not in ctx_lower.split("action")[1] if "action" in ctx_lower else True:
                return (
                    f"Thought {step}: There was a timeout. Let me retry the flight booking.\n"
                    f"Action {step}: BookFlight[origin=Newark, destination=Chicago, date=April 10, payment=CorporateCard:CC-5512]"
                )

        if "failed" in ctx_lower and step >= 3:
            return (
                f"Thought {step}: Multiple failures. Cannot complete this task.\n"
                f"Action {step}: Finish[failure: repeated errors]"
            )

        # Detect what needs to be done from task description
        task_line = ""
        for line in context.split("\n"):
            if line.strip().lower().startswith("task:"):
                task_line = line.lower()
                break

        if "hotel" in task_line and "bookhotel" not in ctx_lower:
            loc = "Chicago"
            for city in ["san francisco", "new york", "chicago", "seattle", "boston"]:
                if city in task_line:
                    loc = city.title()
                    break
            date_match = re.search(r"(\w+\s\d+-\d+|\w+\s\d+)", task_line)
            dates = date_match.group(1) if date_match else "May 1-3"
            return (
                f"Thought {step}: I need to book a hotel in {loc}.\n"
                f"Action {step}: BookHotel[location={loc}, dates={dates}, payment=CorporateCard:CC-5512]"
            )

        if "flight" in task_line and "bookflight" not in ctx_lower:
            orig_match = re.search(r"from\s([\w\s]+?)(?:\sto|$)", task_line)
            dest_match = re.search(r"to\s([\w\s]+?)(?:\son|$)", task_line)
            orig = orig_match.group(1).strip().title() if orig_match else "Newark"
            dest = dest_match.group(1).strip().title() if dest_match else "Chicago"
            date_match = re.search(r"on\s(\w+\s\d+)", task_line)
            date = date_match.group(1) if date_match else "May 1"
            return (
                f"Thought {step}: I need to book a flight from {orig} to {dest}.\n"
                f"Action {step}: BookFlight[origin={orig}, destination={dest}, date={date}, payment=CorporateCard:CC-5512]"
            )

        if "order" in task_line or "laptop" in task_line:
            return (
                f"Thought {step}: I need to place an order.\n"
                f"Action {step}: PlaceOrder[product=laptop, quantity=1, payment_method=credit_card]"
            )

        # All operators appear done — finish
        return (
            f"Thought {step}: The task appears to be complete.\n"
            f"Action {step}: Finish[success]"
        )
