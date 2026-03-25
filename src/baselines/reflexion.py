# src/baselines/reflexion.py
"""
Reflexion Baseline (Shinn et al., NeurIPS 2023)
"Reflexion: Language Agents with Verbal Reinforcement Learning"
https://arxiv.org/abs/2303.11366

Key properties of this implementation:
- Same Thought → Action → Observation loop as ReAct (per-step LLM calls)
- After a FAILED episode: LLM generates a verbal post-mortem (reflection)
- Post-mortem is stored in a token-bounded episodic buffer
- Buffer is prepended to the NEXT episode's system prompt only
- Successful episodes do NOT generate reflections
- NO symbolic operators, NO verify-before-act, NO FDKA, NO governance

Architectural distinctions:
  vs. ReAct:        Reflexion adds cross-episode failure memory; ReAct has none.
  vs. LLM-Reflect:  Reflexion uses step-by-step T→A→O (not upfront planning);
                    reflections are failure-only verbal post-mortems (not mixed
                    general context); reflection fires at episode boundary only
                    (not at each within-episode attempt).
  vs. ANNEAL:   Reflexion stores unstructured text; ANNEAL stores typed
                    symbolic operator patches with formal semantics and governance.

Expected behaviour vs ANNEAL:
- SR ~80-92%: cross-episode verbal memory helps avoid known failure patterns
- RFR_term  : non-zero; textual reasoning cannot eliminate failure structurally
- TTA       : infinity; no operator-level knowledge is ever modified
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..scenarios import SCENARIO_MAP
from ..scenarios.travel_planning import TravelPlanningScenario
from ..utils.metrics import MetricsCollector
from ..utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Token budget for the episodic reflection buffer.
# Paper (Shinn et al., NeurIPS 2023) sets Ω=1-3; AlfWorld code uses [-3:].
# We use Ω=3 and a character cap (~200 chars × 3 entries ≈ 150 tokens) to
# stay faithful to the paper while keeping the buffer within context budget.
# ---------------------------------------------------------------------------
_BUFFER_MAX_CHARS = 600    # hard limit on total buffer text (~150 tokens)
_BUFFER_MAX_SIZE  = 3      # Ω=3 as per paper (Shinn et al., NeurIPS 2023)

# ---------------------------------------------------------------------------
# Few-shot examples — identical to ReAct so the loop structure is the same.
# ---------------------------------------------------------------------------
_TRAVEL_FEW_SHOT = """
Example:
Task: Book a hotel in Chicago for May 3-5
Thought 1: I need to book a hotel in Chicago for May 3-5.
Action 1: BookHotel[location=Chicago, dates=May 3-5, payment=CorporateCard:CC-5512]
Observation 1: Success. Hotel booked in Chicago for May 3-5.
Thought 2: Task complete.
Action 2: Finish[success]
""".strip()

_ECOMMERCE_FEW_SHOT = """
Example:
Task: Place an order for 2 laptops using credit_card
Thought 1: I need to place an order for 2 laptops with credit_card.
Action 1: PlaceOrder[product=laptop, quantity=2, payment_method=credit_card]
Observation 1: Success. Order placed.
Thought 2: Task complete.
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

_ACTION_BLOCKS = {
    "travel_planning": """
Available actions for the current domain (travel planning) only:
- BookHotel[location=..., dates=..., payment=...]
- BookFlight[origin=..., destination=..., date=..., payment=...]
- Finish[success|failure]
""".strip(),
    "travel_stochastic": """
Available actions for the current domain (travel planning) only:
- BookHotel[location=..., dates=..., payment=...]
- BookFlight[origin=..., destination=..., date=..., payment=...]
- Finish[success|failure]
""".strip(),
    "ecommerce": """
Available actions for the current domain (e-commerce) only:
- PlaceOrder[product=..., quantity=..., payment_method=...]
- ApplyPromoCode[promo_code=...]
- CalculateShipping[location=...]
- ProcessRefund[order_id=..., days_since_purchase=...]
- UpdateInventory[product=..., delta=...]
- Finish[success|failure]
""".strip(),
    "itsm": """
Available actions for the current domain (ITSM) only:
- ProvisionAccess[user=..., resource=..., role=..., requester=...]
- DeployPatch[system=..., patch_id=..., environment=..., approved_by=...]
- ResetCredentials[user=..., system=..., method=...]
- CreateTicket[category=..., priority=..., description=...]
- Finish[success|failure]
""".strip(),
}

_BASE_SYSTEM = """You are a task execution agent. You operate in a \
Thought-Action-Observation loop.

{action_block}

Rules:
- Use only the actions listed for the current domain.
- Each Action must start with "Action N:" where N is the current step number.
- Use Finish[success] when the task is complete.
- Use Finish[failure: <reason>] only when the task truly cannot be completed.
- Do NOT repeat a failed action with identical parameters.
- Max 8 steps per task.

{few_shot}
"""


# ---------------------------------------------------------------------------
# Action parser — identical logic to react.py (duplicated to keep baselines
# self-contained and avoid cross-file coupling).
# ---------------------------------------------------------------------------
def _parse_action(text: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Return (operator, params) for the LOWEST-numbered Action line in text.
    Operator actions are preferred over Finish at the same step number.
    Returns (None, {}) when no parseable action is found.
    """
    all_re = re.compile(
        r"Action\s+(\d+)\s*:\s*(\w+)\[([^\]]*)\]",
        re.IGNORECASE,
    )
    finish_re = re.compile(
        r"Action\s+(\d+)\s*:\s*Finish\[([^\]]*)\]",
        re.IGNORECASE,
    )

    candidates: List[Tuple[int, str, Dict[str, str]]] = []

    for m in all_re.finditer(text):
        step_num = int(m.group(1))
        operator = m.group(2).strip()
        raw_params = m.group(3).strip()
        params: Dict[str, str] = {}
        for part in raw_params.split(","):
            part = part.strip()
            if "=" in part:
                k, _, v = part.partition("=")
                params[k.strip()] = v.strip()
            elif part:
                params.setdefault("value", part)
        candidates.append((step_num, operator, params))

    for m in finish_re.finditer(text):
        candidates.append((int(m.group(1)), "Finish", {"result": m.group(2).strip()}))

    if not candidates:
        return None, {}

    # Lowest step number first; prefer operator over Finish when tied.
    candidates.sort(key=lambda x: (x[0], 1 if x[1] == "Finish" else 0))
    _, operator, params = candidates[0]
    return operator, params


# ---------------------------------------------------------------------------
# Episodic reflection buffer
# ---------------------------------------------------------------------------
class ReflexionBuffer:
    """
    Token-bounded, FIFO episodic buffer of verbal failure post-mortems.

    Each entry is a plain string — the LLM's reflection after a failed episode.
    Total buffer length is capped at _BUFFER_MAX_CHARS characters to stay
    within the context window budget.
    """

    def __init__(
        self,
        max_chars: int = _BUFFER_MAX_CHARS,
        max_size: int = _BUFFER_MAX_SIZE,
    ) -> None:
        self._entries: List[str] = []
        self._max_chars = max_chars
        self._max_size = max_size

    # ------------------------------------------------------------------

    def add(self, reflection: str) -> None:
        """Append a reflection, enforcing FIFO eviction when over budget."""
        reflection = reflection.strip()
        if not reflection:
            return

        self._entries.append(reflection)

        # Enforce size cap first (fast path).
        while len(self._entries) > self._max_size:
            self._entries.pop(0)

        # Then enforce character budget (trim oldest entries).
        while self._total_chars() > self._max_chars and len(self._entries) > 1:
            self._entries.pop(0)

    def render(self) -> str:
        """
        Return the buffer as a formatted string ready to be injected into a
        system prompt.  Returns empty string when the buffer is empty.
        """
        if not self._entries:
            return ""
        lines = ["PAST FAILURE REFLECTIONS (learn from these to avoid repeating mistakes):"]
        for i, entry in enumerate(self._entries, 1):
            lines.append(f"[{i}] {entry}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def _total_chars(self) -> int:
        return sum(len(e) for e in self._entries)


# ---------------------------------------------------------------------------
# ReflexionAgent
# ---------------------------------------------------------------------------
class ReflexionAgent:
    """
    Reflexion baseline — Shinn et al., NeurIPS 2023.

    Episode execution uses the same T→A→O loop as ReAct.
    After every FAILED episode, a verbal post-mortem is generated via LLM
    and stored in ReflexionBuffer.  The buffer is prepended to the system
    prompt of every subsequent episode.

    No symbolic operators, no structured patch schema, no governance.
    """

    MAX_STEPS = 8  # maximum T→A→O steps per episode (matches ReAct)
    _ALLOWED_OPS = {
        "travel_planning": {"BookHotel", "BookFlight"},
        "travel_stochastic": {"BookHotel", "BookFlight"},
        "ecommerce": {"PlaceOrder", "ApplyPromoCode", "CalculateShipping", "ProcessRefund", "UpdateInventory"},
        "itsm": {"ProvisionAccess", "DeployPatch", "ResetCredentials", "CreateTicket"},
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        log_cfg = dict(config.get("logging", {}))
        log_cfg["level"] = log_cfg.get("level", "WARNING")
        self.logger = setup_logger(log_cfg, name="reflexion")

        self.logger.info("=" * 70)
        self.logger.info("Initializing Reflexion Baseline (Shinn et al., NeurIPS 2023)...")
        self.logger.info("=" * 70)

        # Scenario — identical to ANNEAL for fair comparison.
        scenario_name = config.get("scenario", {}).get("name", "travel_planning")
        scenario_class = SCENARIO_MAP.get(scenario_name, TravelPlanningScenario)
        self.scenario = scenario_class(config["scenario"])
        self.scenario_name = scenario_name
        self.allowed_operators = set(self._ALLOWED_OPS.get(scenario_name, {"BookHotel", "BookFlight"}))

        # Metrics — identical collector.
        self.metrics = MetricsCollector()

        # LLM configuration — read from fdka.propose_edit (same block as all
        # other baselines so a single config change propagates everywhere).
        pe_cfg = config.get("fdka", {}).get("propose_edit", {}) or {}
        self.llm_provider = pe_cfg.get("llm_provider", "mock")
        self.model        = pe_cfg.get("model", "gpt-4o-mini")
        self.temperature  = float(pe_cfg.get("temperature", 0.3))
        self.max_tokens   = 512
        self.timeout      = float(pe_cfg.get("timeout_sec", 60.0))
        self.metrics.set_run_metadata({
            "baseline": "reflexion",
            "difficulty": config.get("scenario", {}).get("difficulty"),
            "seed": config.get("scenario", {}).get("task_generation_seed"),
            "llm_provider": self.llm_provider,
            "model": self.model,
            "fdka_enabled": False,
        })

        # OpenAI client
        self._openai_client = None
        if self.llm_provider == "openai":
            self._init_openai()

        # Domain-specific few-shot and base system prompt.
        few_shot = (
            _ECOMMERCE_FEW_SHOT if "ecommerce" in scenario_name
            else _ITSM_FEW_SHOT if "itsm" in scenario_name
            else _TRAVEL_FEW_SHOT
        )
        action_block = _ACTION_BLOCKS.get(scenario_name, _ACTION_BLOCKS["travel_planning"])
        self._base_system = _BASE_SYSTEM.format(
            action_block=action_block,
            few_shot=few_shot,
        )

        # Episodic reflection buffer — the core Reflexion component.
        self.buffer = ReflexionBuffer()

        self.logger.info("✅ Reflexion baseline ready")
        self.logger.info(f"   LLM        : {self.model} (provider: {self.llm_provider})")
        self.logger.info(f"   Max steps  : {self.MAX_STEPS}")
        self.logger.info(f"   Buffer cap : {_BUFFER_MAX_SIZE} reflections / {_BUFFER_MAX_CHARS} chars")
        self.logger.info(f"   Reflection : failure-only, episode-boundary")
        self.logger.info("=" * 70)

    # ------------------------------------------------------------------
    # OpenAI init
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        try:
            import openai
            self._openai_client = openai.OpenAI()
            self.logger.info("REFLEXION: OpenAI client initialised.")
        except Exception as e:
            self.logger.warning(f"REFLEXION: OpenAI init failed ({e}). Falling back to mock.")
            self.llm_provider = "mock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self) -> MetricsCollector:
        print(f"\n{'=' * 70}")
        print("🪞 REFLEXION BASELINE EVALUATION")
        print(f"{'=' * 70}")

        for task_id in range(self.scenario.num_tasks):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self._run_episode(task_id, instruction)

        print(f"\n{'=' * 70}")
        print("✅ REFLEXION EVALUATION COMPLETE")
        print(f"{'=' * 70}")
        self.metrics.print_summary()
        output_dir = Path(self.config.get("output", {}).get("results_dir", "data/results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics.save(output_dir / "metrics.json")
        return self.metrics

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def _run_episode(self, task_id: int, instruction: str) -> None:
        """
        Run one episode (task) using the T→A→O loop.

        On failure:
          1. Generate a verbal post-mortem via LLM.
          2. Add it to the episodic buffer.
          (The buffer is prepended automatically to the *next* episode's prompt.)

        On success:
          No reflection is generated — this is the defining property of Reflexion
          versus LLM-Reflect, which can reflect on any episode outcome.
        """
        print(f"\n{'=' * 70}")
        print(f"Task {task_id + 1}: {instruction}")
        buffer_info = f"  [Buffer: {len(self.buffer)} reflection(s)]" if self.buffer else ""
        print(f"{'=' * 70}{buffer_info}")

        # Build the system prompt: base + current failure buffer.
        system_prompt = self._build_system_prompt()

        # Conversation context for this episode (fresh each task — key property).
        context: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Task: {instruction}\nThought 1:"},
        ]

        trace: List[Dict[str, Any]] = []
        episode_failed = False
        last_failure: Dict[str, Any] = {}

        for step in range(1, self.MAX_STEPS + 1):
            # ---- LLM call ----------------------------------------
            raw = self._call_llm(context, step)

            # Append the raw LLM output as an assistant turn.
            context.append({"role": "assistant", "content": raw})

            # ---- Parse action ------------------------------------
            operator, params = _parse_action(raw)

            if operator is None:
                print(f"   Step {step}: ⚠ Could not parse action — treating as failure.")
                episode_failed = True
                last_failure = {"operator": "unknown", "error": "parse_failure",
                                "message": "No parseable action in LLM output."}
                break

            # ---- Finish -----------------------------------------
            if operator == "Finish":
                result = params.get("result", "")
                if "failure" in result.lower():
                    print(f"   Step {step}: Finish[failure] — episode failed.")
                    episode_failed = True
                    last_failure = {"operator": "Finish", "error": "explicit_failure",
                                    "message": result}
                else:
                    print(f"   Step {step}: Finish[success] — episode complete.")
                    # Clear the flag: within-episode recovery succeeded.
                    # Reflexion only stores reflections when the EPISODE ends
                    # in failure — not when recovery happened mid-episode.
                    episode_failed = False
                break

            if operator not in self.allowed_operators:
                obs_text = (
                    f"Observation {step}: FAILED — InvalidAction: unsupported action '{operator}' "
                    f"for scenario '{self.scenario_name}'. Allowed actions: {', '.join(sorted(self.allowed_operators))}.\n"
                    f"Thought {step + 1}:"
                )
                print(f"   Step {step}: ✗ unsupported action '{operator}'")
                trace.append({
                    "step": step,
                    "operator": operator,
                    "params": params,
                    "success": False,
                    "error": "InvalidAction",
                    "message": obs_text,
                })
                episode_failed = True
                last_failure = {
                    "operator": operator,
                    "error": "InvalidAction",
                    "message": obs_text,
                    "instruction": instruction,
                }
                context.append({"role": "user", "content": obs_text})
                if step == self.MAX_STEPS:
                    break
                continue

            # ---- Execute operator --------------------------------
            print(f"   Step {step}: {operator}({params})")

            should_fail = self._scenario_should_fail(task_id, operator, params)

            if should_fail:
                failure_info = self._scenario_failure_details(task_id, operator, params)
                err_msg = failure_info.get("message", "Unknown error")
                err_type = failure_info.get("error", "unknown")

                obs_text = (
                    f"Observation {step}: FAILED — {err_type}: {err_msg}\n"
                    f"Thought {step + 1}:"
                )
                print(f"   ✗ {operator} failed: {err_type}")

                trace.append({
                    "step": step, "operator": operator, "params": params,
                    "success": False, "error": err_type, "message": err_msg,
                })
                episode_failed = True
                last_failure = {
                    "operator": operator, "error": err_type,
                    "message": err_msg, "instruction": instruction,
                }

                # Continue the loop — Reflexion does NOT retry within episode
                # by re-planning from scratch; the LLM sees the failure as an
                # Observation and may attempt a different action at the next step.
                context.append({"role": "user", "content": obs_text})

                # If we just used the last step, mark as failed.
                if step == self.MAX_STEPS:
                    break
                continue

            # Success
            obs_text = (
                f"Observation {step}: Success — {operator} completed.\n"
                f"Thought {step + 1}:"
            )
            trace.append({
                "step": step, "operator": operator, "params": params,
                "success": True,
            })
            print(f"   ✓ {operator} succeeded.")
            context.append({"role": "user", "content": obs_text})

        # ---- Record outcome ------------------------------------
        if episode_failed:
            self.metrics.record_task(task_id, False, trace)
            print("❌ Task FAILED")

            # --- Core Reflexion mechanism: generate and store post-mortem ---
            reflection = self._generate_reflection(instruction, trace, last_failure)
            if reflection:
                self.buffer.add(reflection)
                print(f"🪞 Reflection stored [{len(self.buffer)}/{_BUFFER_MAX_SIZE}]: "
                      f"{reflection[:70]}{'...' if len(reflection) > 70 else ''}")
        else:
            self.metrics.record_task(task_id, True, trace)
            print("✅ Task SUCCEEDED — no reflection generated.")

    def _scenario_should_fail(self, task_id: int, operator: str, params: Dict[str, Any]) -> bool:
        """Support both travel-style and ecommerce-style scenario failure APIs."""
        if hasattr(self.scenario, "should_inject_failure"):
            return bool(self.scenario.should_inject_failure(task_id, operator, params, None))
        if hasattr(self.scenario, "should_fail"):
            return bool(self.scenario.should_fail(task_id, operator, params=params, state=None))
        return False

    def _scenario_failure_details(self, task_id: int, operator: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self.scenario, "get_failure_details"):
            return self.scenario.get_failure_details(task_id, operator, params=params, state=None) or {}
        return {}

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """
        Assemble the full system prompt for this episode.
        If the episodic buffer is non-empty, prepend it to the base prompt.
        """
        buffer_text = self.buffer.render()
        if buffer_text:
            return f"{buffer_text}\n\n{self._base_system}"
        return self._base_system

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, context: List[Dict[str, str]], step: int) -> str:
        """
        Call the LLM with the current conversation context.

        Stop sequences truncate generation after one Action line — the same
        technique used in react.py to prevent the model from hallucinating a
        full trajectory in one response.
        """
        if self.llm_provider == "openai" and self._openai_client:
            try:
                stop = [f"\nObservation {step}:", f"Observation {step}:"]
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=context,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    stop=stop,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                self.logger.warning(f"REFLEXION: LLM call failed ({e}). Using mock.")

        # Mock fallback — deterministic, no API call.
        return self._mock_action(context, step)

    def _mock_action(self, context: List[Dict[str, str]], step: int) -> str:
        """
        Deterministic mock when no LLM is available.
        Inspects task instruction and buffer content to choose an action.
        """
        # Extract the original task instruction from the user turn.
        task_text = ""
        for msg in context:
            if msg["role"] == "user" and "Task:" in msg["content"]:
                task_text = msg["content"]
                break
        task_lower = task_text.lower()

        # Check buffer for warnings about past failures.
        buffer_warns_card = (
            "corporate" in self.buffer.render().lower()
            and "blocked" in self.buffer.render().lower()
        )

        if step == 1:
            payment = "PersonalCard:PC-1134" if buffer_warns_card else "CorporateCard:CC-5512"
            if "hotel" in task_lower:
                return (
                    f"I need to book a hotel. Using {'personal card due to past failures' if buffer_warns_card else 'corporate card'}.\n"
                    f"Action {step}: BookHotel[location=Chicago, dates=May 3-5, payment={payment}]"
                )
            if "flight" in task_lower:
                return (
                    f"I need to book a flight.\n"
                    f"Action {step}: BookFlight[origin=Newark, destination=Chicago, date=May 3, payment={payment}]"
                )
            # E-commerce
            if "order" in task_lower or "laptop" in task_lower:
                return (
                    f"I need to place an order.\n"
                    f"Action {step}: PlaceOrder[product=laptop, quantity=2, payment_method=credit_card]"
                )

        # Default: finish
        return f"Task appears complete.\nAction {step}: Finish[success]"

    # ------------------------------------------------------------------
    # Reflection generation
    # ------------------------------------------------------------------

    def _generate_reflection(
        self,
        instruction: str,
        trace: List[Dict[str, Any]],
        last_failure: Dict[str, Any],
    ) -> str:
        """
        Generate a verbal post-mortem for a failed episode.

        This is the verbal reinforcement signal in Reflexion: a concise,
        actionable diagnosis of what went wrong and what to do differently.
        Fires ONLY on episode failure (never on success).

        When a real LLM is available, uses a targeted reflection prompt.
        Falls back to a rule-based template for mock provider.
        """
        operator  = last_failure.get("operator", "unknown")
        err_type  = last_failure.get("error",    "unknown")
        err_msg   = last_failure.get("message",  "")

        # Summarise trace for the reflection prompt (last 3 steps).
        trace_summary = "; ".join(
            f"Step {t['step']}: {t['operator']}({'✓' if t.get('success') else '✗ ' + t.get('error', '')})"
            for t in trace[-3:]
        ) or "No steps recorded."

        if self.llm_provider == "openai" and self._openai_client:
            prompt = (
                "An agent failed to complete a task. "
                "Write a concise reflection (2-3 sentences) that:\n"
                "  1. Identifies the specific cause of failure.\n"
                "  2. States clearly what the agent should do differently next time.\n\n"
                f"Task       : {instruction}\n"
                f"Failed at  : {operator}\n"
                f"Error      : {err_type} — {err_msg}\n"
                f"Trajectory : {trace_summary}\n\n"
                "Reflection:"
            )
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=128,
                    timeout=self.timeout,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception as e:
                self.logger.warning(f"REFLEXION: Reflection LLM call failed ({e}). Using template.")

        # Rule-based template fallback.
        msg_lower = err_msg.lower()
        if "blocked" in msg_lower and "corporate" in msg_lower:
            return (
                f"The {operator} action failed because the corporate card is blocked "
                f"on blackout dates. Next time, use a personal payment card (e.g. PersonalCard:PC-1134) "
                f"when booking during restricted periods."
            )
        if "timeout" in msg_lower or "503" in msg_lower:
            return (
                f"The {operator} action failed due to an API timeout or transient error. "
                f"Next time, retry the operation or check for deprecated API endpoints before calling."
            )
        if "schema" in msg_lower or "deprecated" in msg_lower or "v2" in msg_lower:
            return (
                f"The {operator} action failed due to an API schema change. "
                f"Next time, verify the API endpoint version and update parameters accordingly."
            )
        if "invalid" in msg_lower or "expired" in msg_lower:
            return (
                f"The {operator} action failed because the payment method is invalid or expired. "
                f"Next time, switch to an alternative payment method."
            )
        # Generic fallback.
        short_msg = err_msg[:100] if err_msg else err_type
        return (
            f"The {operator} action failed with error '{short_msg}'. "
            f"Next time, reconsider the parameters passed to {operator} "
            f"and check preconditions before executing."
        )


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Reflexion Baseline (standalone)")
    print("=" * 70)

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config_loader import load_config

    config = load_config("config.yaml")
    config["scenario"]["num_tasks"] = 10
    config["logging"]["level"] = "INFO"

    agent = ReflexionAgent(config)
    metrics = agent.run_evaluation()

    summary = metrics.get_summary()
    print(f"\nSuccess Rate : {summary['success_rate']:.1%}")
    print(f"RFR_term     : {summary.get('terminal_rfr', summary.get('repeat_failure_rate', 0)):.1%}")
    print(f"Buffer size  : {len(agent.buffer)} reflection(s)")

    print("\n[Stored Reflections]")
    for i, r in enumerate(agent.buffer._entries, 1):
        print(f"  {i}. {r[:80]}{'...' if len(r) > 80 else ''}")

    print("\nExpected: SR above LLM-Reflect but RFR_term > 0 (no structural repair).")
    print("✅ Reflexion standalone test complete.")
