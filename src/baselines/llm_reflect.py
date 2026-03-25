# src/baselines/llm_reflect.py
"""
LLM + Textual Reflection Baseline (LLM-Reflect)
Corresponds to manuscript Section XII.2 baseline definition.

This baseline represents ReAct/Reflexion-style agents:
- Uses LLM for planning and execution decisions
- Maintains textual memory of past failures
- Reflects on errors using natural language reasoning
- NO symbolic operators: all reasoning is text-based
- NO verify-before-act: LLM handles constraint checking implicitly

Key difference from ANNEAL:
- Reflection = text summaries stored in memory (not symbolic patches)
- No formal precondition checking or symbolic state
- Decisions based on LLM prompt with history, not explicit rules

Expected behavior: Better than Static-NS but worse than ANNEAL
- Some adaptation via memory (~65% success vs ~89%)
- Slower learning (text is less precise than symbolic patches)
- No formal verification (higher constraint violations)
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..core.state import SymbolicState
from ..scenarios import SCENARIO_MAP
from ..scenarios.travel_planning import TravelPlanningScenario
from ..utils.metrics import MetricsCollector
from ..utils.logger import setup_logger


class LLMReflectAgent:
    """
    LLM-based agent with textual reflection.

    Architecture:
    1. LLM plans actions based on instruction + memory
    2. Execute actions (simulate)
    3. On failure: Generate textual reflection
    4. Store reflection in memory for future tasks

    No symbolic state, no explicit preconditions, no formal verification.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM-Reflect baseline.

        Args:
            config: System configuration
        """
        self.config = config

        # Setup logger
        log_config = config.get('logging', {})
        log_config['level'] = log_config.get('level', 'WARNING')
        self.logger = setup_logger(log_config, name="llm_reflect")

        self.logger.info("=" * 70)
        self.logger.info("Initializing LLM-Reflect Baseline...")
        self.logger.info("=" * 70)

        # Minimal symbolic state (for scenario compatibility)
        self.state = SymbolicState()

        # Scenario — support travel and ecommerce (same as ANNEAL)
        scenario_name = config.get('scenario', {}).get('name', 'travel_planning')
        scenario_class = SCENARIO_MAP.get(scenario_name, TravelPlanningScenario)
        self.scenario = scenario_class(config['scenario'])
        self.scenario_name = scenario_name

        # Metrics collector
        self.metrics = MetricsCollector()

        # LLM configuration
        pe_cfg = config.get('fdka', {}).get('propose_edit', {}) or {}
        self.llm_provider = pe_cfg.get('llm_provider', 'mock')
        self.llm_model = pe_cfg.get('model', 'gpt-4o-mini')
        self.temperature = float(pe_cfg.get('temperature', 0.3))
        self.max_tokens = 512
        self.timeout = float(pe_cfg.get('timeout_sec', 60.0))

        # Initialise OpenAI client when needed
        self._openai_client = None
        if self.llm_provider == 'openai':
            self._init_openai()

        # Textual memory (key difference from ANNEAL — persists across tasks)
        self.reflection_memory: List[Dict[str, str]] = []
        self.max_memory_size = 20

        # Execution history for context
        self.execution_history: List[Dict] = []

        self.logger.info("✅ LLM-Reflect baseline ready (textual reasoning + cross-episode memory)")
        self.logger.info(f"   LLM: {self.llm_model} (provider: {self.llm_provider})")
        self.logger.info(f"   Cross-episode memory: ENABLED (max {self.max_memory_size} reflections)")
        self.logger.info("=" * 70)

    # ------------------------------------------------------------------
    # OpenAI init
    # ------------------------------------------------------------------

    def _init_openai(self) -> None:
        try:
            import openai
            self._openai_client = openai.OpenAI()
            self.logger.info("LLM-REFLECT: OpenAI client initialised.")
        except Exception as e:
            self.logger.warning(f"LLM-REFLECT: OpenAI init failed ({e}). Falling back to mock.")
            self.llm_provider = 'mock'

    def run_evaluation(self) -> MetricsCollector:
        """
        Run evaluation using LLM-based reflection.

        Returns:
            MetricsCollector with results
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 LLM-REFLECT BASELINE EVALUATION")
        print(f"{'=' * 70}")

        for task_id in range(self.scenario.num_tasks):
            instruction = self.scenario.get_task(task_id)
            if instruction:
                self._run_task_with_reflection(task_id, instruction)

        print(f"\n{'=' * 70}")
        print(f"✅ LLM-REFLECT EVALUATION COMPLETE")
        print(f"{'=' * 70}")

        # Print summary
        self.metrics.print_summary()

        return self.metrics

    def run_task(self, task_id: int, instruction: str) -> Dict[str, Any]:
        """
        Run a single task and return structured outcome (for evaluation scripts).

        Args:
            task_id: Task identifier
            instruction: Natural language instruction

        Returns:
            Dict with 'status', 'trace', 'metrics'
        """
        self._run_task_with_reflection(task_id, instruction)

        # Determine status from metrics
        task_data = next((t for t in self.metrics.tasks if t.get('task_id') == task_id), None)
        if task_data and task_data.get('success'):
            status = 'success'
        else:
            status = 'failure'

        return {
            'status': status,
            'trace': task_data.get('trace', []) if task_data else [],
            'metrics': self.metrics.get_summary()
        }

    def _run_task_with_reflection(self, task_id: int, instruction: str) -> None:
        """
        Run a single task using LLM reasoning with textual reflection.

        Process:
        1. Generate plan using LLM (with memory context)
        2. Execute plan (simulate with failure injection)
        3. If failure: Generate reflection and store in memory
        4. Retry with updated memory

        Args:
            task_id: Task identifier
            instruction: Natural language instruction
        """
        print(f"\n{'=' * 70}")
        print(f"Task {task_id + 1}: {instruction}")
        print(f"{'=' * 70}")

        # Update state from instruction (minimal, for scenario compatibility)
        self.state.update_from_instruction(instruction)

        # Build context from memory
        memory_context = self._build_memory_context(instruction)

        max_attempts = 3
        final_trace = []

        for attempt in range(max_attempts):
            print(f"\n--- Attempt #{attempt + 1} ---")

            # Step 1: LLM generates plan (with memory)
            plan = self._llm_plan(instruction, memory_context, attempt)

            if not plan:
                print("❌ LLM planning failed")
                break

            # Step 2: Execute plan
            trace, success, failure_info = self._execute_llm_plan(
                plan, task_id, instruction
            )
            final_trace = trace or final_trace

            if success:
                print("✅ Task Succeeded")
                self.metrics.record_task(task_id, True, trace)

                # Store successful execution in history
                self._update_execution_history(instruction, trace, success=True)
                return

            # Step 3: On failure, generate reflection
            print(f"❌ Execution Failed: {failure_info.get('error', 'Unknown')}")
            print("💭 Generating textual reflection...")

            reflection = self._generate_reflection(instruction, trace, failure_info)
            self._store_reflection(reflection)

            # Update memory context for next attempt
            memory_context = self._build_memory_context(instruction)

            print(f"   Reflection stored: {reflection['summary'][:60]}...")

        # All attempts exhausted
        print("❌ Task Failed after all attempts")
        self.metrics.record_task(task_id, False, final_trace)
        self._update_execution_history(instruction, final_trace, success=False)

    def _llm_plan(self, instruction: str, memory_context: str, attempt: int) -> Optional[List[Dict]]:
        """
        Use LLM to generate a full upfront execution plan (one-shot planning).

        Key distinction from ReAct:
        - Plans ALL steps before execution begins (not step-by-step)
        - Memory context contains cross-episode reflections from past failures
        - LLM can reason about prior constraints when building the plan
        """
        prompt = self._build_planning_prompt(instruction, memory_context, attempt)

        if self.llm_provider == 'openai' and self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                raw = response.choices[0].message.content or ""
                plan = self._parse_llm_plan(raw, instruction)
                if plan:
                    return plan
                self.logger.warning("LLM-REFLECT: Could not parse plan from LLM output. Falling back to mock.")
            except Exception as e:
                self.logger.warning(f"LLM-REFLECT: LLM call failed ({e}). Falling back to mock.")

        # Mock fallback — used when provider='mock' or LLM call fails
        return self._mock_llm_plan(instruction, memory_context, attempt)

    def _build_planning_prompt(self, instruction: str, memory_context: str, attempt: int) -> str:
        """
        Build LLM prompt for one-shot upfront planning.
        Domain-aware: handles both travel and ecommerce scenarios.
        Memory context contains cross-episode reflections — the key advantage
        of LLM-Reflect over ReAct.
        """
        is_ecommerce = 'ecommerce' in self.scenario_name
        is_itsm = 'itsm' in self.scenario_name

        if is_ecommerce:
            actions_hint = "PlaceOrder, ApplyPromoCode, CalculateShipping, ProcessRefund, UpdateInventory"
            action_format = (
                "PlaceOrder[product=laptop, quantity=2, payment_method=credit_card]\n"
                "ApplyPromoCode[promo_code=SAVE10]"
            )
            domain_hint = "e-commerce assistant"
        elif is_itsm:
            actions_hint = (
                "ProvisionAccess, DeployPatch, ResetCredentials, CreateTicket"
            )
            action_format = (
                "ProvisionAccess[user=alice.johnson, resource=billing-system, "
                "role=readonly, requester=manager.taylor]\n"
                "DeployPatch[system=web-servers, patch_id=KB-2024-0891, "
                "environment=staging, approved_by=lead.chen]"
            )
            domain_hint = "IT service management assistant"
        else:
            actions_hint = "BookHotel, BookFlight"
            action_format = (
                "BookHotel[location=Chicago, dates=May 3-5, payment=CorporateCard:CC-5512]\n"
                "BookFlight[origin=Newark, destination=Chicago, date=May 3, payment=CorporateCard:CC-5512]"
            )
            domain_hint = "travel planning assistant"

        retry_note = f"\nThis is RETRY ATTEMPT #{attempt + 1}. Previous attempts failed. Adjust your plan based on past experiences above.\n" if attempt > 0 else ""

        prompt = f"""You are a {domain_hint}. Generate an upfront execution plan for the task below.

TASK: {instruction}

PAST EXPERIENCES (cross-episode memory — use these to avoid known failures):
{memory_context if memory_context else "None recorded yet."}
{retry_note}
OUTPUT FORMAT — list each action on its own line, exactly like this:
{action_format}

Available actions: {actions_hint}
Rules:
- Only include actions required by the task.
- If past experiences warn about a constraint (e.g. blocked payment), adjust parameters accordingly.
- Do NOT include explanations — output action lines only.

Plan:"""
        return prompt

    def _parse_llm_plan(self, llm_output: str, instruction: str) -> Optional[List[Dict]]:
        """
        Parse LLM one-shot plan output into a list of action dicts.
        Accepts lines of the form: OperatorName[key=val, key=val]
        Falls back to None if no valid actions found.
        """
        action_re = re.compile(r"(\w+)\[([^\]]*)\]")
        plan: List[Dict] = []

        for line in llm_output.splitlines():
            line = line.strip()
            # Skip blank lines, markdown bullets, and "Plan:" header
            if not line or line.lower().startswith("plan:"):
                continue
            # Strip leading "- " or "N. " list markers
            line = re.sub(r"^[-\d]+[.)]\s*", "", line).strip()

            m = action_re.search(line)
            if not m:
                continue

            operator = m.group(1).strip()
            raw_params = m.group(2).strip()

            # Only accept known operators
            known = {
                "BookHotel", "BookFlight",
                "PlaceOrder", "ApplyPromoCode", "CalculateShipping",
                "ProcessRefund", "UpdateInventory",
                "ProvisionAccess", "DeployPatch", "ResetCredentials", "CreateTicket",
            }
            if operator not in known:
                continue

            params: Dict[str, Any] = {}
            if raw_params:
                for part in raw_params.split(","):
                    part = part.strip()
                    if "=" in part:
                        k, _, v = part.partition("=")
                        params[k.strip()] = v.strip()

            plan.append({"action": operator, "params": params})

        return plan if plan else None

    def _mock_llm_plan(self, instruction: str, memory_context: str, attempt: int) -> List[Dict]:
        """
        Mock LLM planning (PoC implementation).

        Simulates how an LLM would respond to reflections in memory.
        """
        plan = []
        instruction_lower = instruction.lower()

        # ---- ITSM domain ----
        if 'itsm' in self.scenario_name:
            if "provision" in instruction_lower or "access" in instruction_lower:
                role = "admin" if "admin" in instruction_lower else "readonly"
                # Memory: if past failures mention approval_code, switch to readonly
                if "approval" in memory_context.lower() and role == "admin":
                    role = "readonly"
                    print("   [LLM-Reflect: Memory suggests avoiding admin role]")
                user = "alice.johnson"
                for word in instruction_lower.split():
                    if "." in word and "@" not in word:
                        user = word
                        break
                plan.append({"action": "ProvisionAccess",
                             "params": {"user": user, "resource": "billing-system",
                                        "role": role, "requester": "manager.taylor"}})
            elif "deploy" in instruction_lower or "patch" in instruction_lower:
                env = "prod" if "prod" in instruction_lower else "staging"
                if "change_window" in memory_context.lower() or "window" in memory_context.lower():
                    env = "staging"
                    print("   [LLM-Reflect: Memory suggests deploying to staging]")
                plan.append({"action": "DeployPatch",
                             "params": {"system": "web-servers", "patch_id": "KB-2024-0891",
                                        "environment": env, "approved_by": "lead.chen"}})
            elif "reset" in instruction_lower or "credential" in instruction_lower:
                plan.append({"action": "ResetCredentials",
                             "params": {"user": "alice.johnson", "system": "VPN",
                                        "method": "email"}})
            elif "ticket" in instruction_lower or "incident" in instruction_lower:
                priority = "critical" if "critical" in instruction_lower else "high"
                if "escalation" in memory_context.lower() or "access denied" in memory_context.lower():
                    priority = "high"
                    print("   [LLM-Reflect: Memory suggests using high instead of critical priority]")
                plan.append({"action": "CreateTicket",
                             "params": {"category": "network-outage", "priority": priority,
                                        "description": "Service degradation detected"}})
            return plan if plan else None

        # ---- Travel domain ----
        if "hotel" in instruction_lower:
            action = {
                "action": "BookHotel",
                "params": self._extract_hotel_params(instruction)
            }
            # Check if memory suggests avoiding corporate card
            if "corporate" in memory_context.lower() and "blocked" in memory_context.lower():
                action["params"]["payment"] = "PersonalCard:PC-1134"
                print("   [LLM: Memory suggests avoiding corporate card]")
            plan.append(action)

        if "flight" in instruction_lower:
            action = {
                "action": "BookFlight",
                "params": self._extract_flight_params(instruction)
            }
            plan.append(action)

        return plan if plan else None

    def _execute_llm_plan(self, plan: List[Dict], task_id: int, instruction: str) -> Tuple[List, bool, Dict]:
        """
        Execute the LLM-generated plan.

        Simulates execution with failure injection (no formal verification).

        Returns:
            (trace, success, failure_info)
        """
        trace = []

        for i, step in enumerate(plan):
            action_name = step.get("action")
            params = step.get("params", {})

            print(f"   Executing: {action_name}")

            # Simulate execution (check for injected failures)
            should_fail = self.scenario.should_inject_failure(
                task_id, action_name, params, self.state
            )

            if should_fail:
                # Execution failed
                failure_info = self.scenario.get_failure_details(
                    task_id, action_name, params, self.state
                )

                trace.append({
                    "step": i,
                    "action": action_name,
                    "params": params,
                    "success": False,
                    "error": failure_info
                })

                return trace, False, failure_info

            # Success
            trace.append({
                "step": i,
                "action": action_name,
                "params": params,
                "success": True
            })
            print(f"   ✓ {action_name} succeeded")

        return trace, True, {}

    def _generate_reflection(self, instruction: str, trace: List[Dict], failure_info: Dict) -> Dict[str, str]:
        """
        Generate textual reflection on failure.

        When a real LLM is available, ask it to produce a nuanced reflection
        so the memory context carries genuine reasoning about the constraint.
        Falls back to rule-based reflection for mock provider.
        """
        error_type = failure_info.get("error", "Unknown")
        operator = failure_info.get("operator", "Unknown")
        message = failure_info.get("message", "")

        # Attempt real LLM reflection when available
        if self.llm_provider == 'openai' and self._openai_client:
            try:
                reflection_prompt = (
                    f"A task failed. Produce a concise reflection (3 sentences max) "
                    f"explaining what went wrong and how to avoid it next time.\n\n"
                    f"Task: {instruction}\n"
                    f"Failed action: {operator}\n"
                    f"Error: {error_type} — {message}\n\n"
                    f"Reflection:"
                )
                response = self._openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": reflection_prompt}],
                    temperature=0.3,
                    max_tokens=128,
                    timeout=self.timeout,
                )
                llm_reflection = (response.choices[0].message.content or "").strip()
                if llm_reflection:
                    summary = llm_reflection.split(".")[0].strip()
                    lesson = llm_reflection
                    constraint = f"{operator}: {error_type}"
                    return {
                        "instruction_pattern": self._get_instruction_pattern(instruction),
                        "failed_action": operator,
                        "error_type": error_type,
                        "summary": summary,
                        "lesson": lesson,
                        "constraint": constraint,
                        "timestamp": trace[-1].get("timestamp", "") if trace else "",
                        "hash": hashlib.md5(f"{operator}:{error_type}".encode()).hexdigest()[:8],
                    }
            except Exception as e:
                self.logger.warning(f"LLM-REFLECT: Reflection LLM call failed ({e}). Using rule-based.")

        # Rule-based fallback (mock provider or LLM failure)
        if "corporate" in message.lower() and "blocked" in message.lower():
            summary = "Corporate cards are blocked during certain date periods"
            lesson = "When booking hotels, check date restrictions and consider using personal payment methods"
            constraint = "Corporate card policy: blocked on blackout dates"
        elif "invalid" in message.lower() or "expired" in message.lower():
            summary = "Payment method is invalid or expired"
            lesson = "Switch to an alternative payment method (e.g. PersonalCard) when corporate card is rejected"
            constraint = f"{operator}: payment validation failed"
        elif "timeout" in message.lower() or "api" in message.lower() or "deprecated" in message.lower():
            summary = "Booking API error (timeout or schema change)"
            lesson = "Retry the operation or use the updated API endpoint if deprecated"
            constraint = "API reliability: check endpoint version and retry on timeout"
        else:
            summary = f"{operator} failed: {message[:60]}"
            lesson = f"Avoid triggering {operator} under these conditions: {message[:60]}"
            constraint = f"Constraint in {operator}: {error_type}"

        return {
            "instruction_pattern": self._get_instruction_pattern(instruction),
            "failed_action": operator,
            "error_type": error_type,
            "summary": summary,
            "lesson": lesson,
            "constraint": constraint,
            "timestamp": trace[-1].get("timestamp", "") if trace else "",
            "hash": hashlib.md5(f"{operator}:{error_type}".encode()).hexdigest()[:8],
        }

    def _store_reflection(self, reflection: Dict[str, str]) -> None:
        """
        Store reflection in memory.

        Uses FIFO with max size to prevent memory overflow.
        """
        # Check for duplicates
        reflection_hash = reflection.get("hash")
        existing = [r for r in self.reflection_memory if r.get("hash") == reflection_hash]

        if existing:
            print("   [Memory: Similar reflection already exists, skipping]")
            return

        # Add to memory
        self.reflection_memory.append(reflection)

        # Enforce max size (FIFO)
        if len(self.reflection_memory) > self.max_memory_size:
            removed = self.reflection_memory.pop(0)
            print(f"   [Memory: Evicted old reflection: {removed['summary'][:40]}...]")

        print(f"   [Memory: Stored reflection #{len(self.reflection_memory)}]")

    def _build_memory_context(self, instruction: str) -> str:
        """
        Build relevant memory context for current instruction.

        Retrieves reflections that match the instruction pattern.

        Args:
            instruction: Current task instruction

        Returns:
            Formatted string with relevant reflections
        """
        if not self.reflection_memory:
            return ""

        # Get instruction pattern
        pattern = self._get_instruction_pattern(instruction)

        # Retrieve relevant reflections
        relevant = []
        for reflection in self.reflection_memory:
            # Simple matching: if patterns overlap
            if pattern in reflection.get("instruction_pattern", ""):
                relevant.append(reflection)

        # If no exact matches, use most recent
        if not relevant and self.reflection_memory:
            relevant = self.reflection_memory[-3:]  # Last 3

        # Format for prompt
        if not relevant:
            return ""

        context_lines = []
        for i, r in enumerate(relevant, 1):
            context_lines.append(f"{i}. {r['summary']}")
            context_lines.append(f"   Lesson: {r['lesson']}")
            context_lines.append(f"   Constraint: {r['constraint']}")

        return "\n".join(context_lines)

    def _get_instruction_pattern(self, instruction: str) -> str:
        """
        Extract pattern from instruction for memory retrieval.

        Simple keyword extraction (PoC).
        """
        keywords = []
        instruction_lower = instruction.lower()

        if "hotel" in instruction_lower:
            keywords.append("hotel")
        if "flight" in instruction_lower:
            keywords.append("flight")
        if "book" in instruction_lower:
            keywords.append("booking")

        # Extract location
        locations = ["san francisco", "new york", "chicago", "seattle", "boston"]
        for loc in locations:
            if loc in instruction_lower:
                keywords.append(loc)

        return " ".join(keywords) if keywords else "general"

    def _extract_hotel_params(self, instruction: str) -> Dict[str, str]:
        """Extract hotel booking parameters from instruction."""
        import re

        params = {}

        # Location
        loc_match = re.search(r'in\s([\w\s]+?)(?=\sfor|\son|$)', instruction, re.IGNORECASE)
        if loc_match:
            params['location'] = loc_match.group(1).strip()

        # Dates
        date_match = re.search(r'(\w+\s\d+-\d+)', instruction)
        if date_match:
            params['dates'] = date_match.group(1)

        # Payment (default to corporate, memory may override)
        params['payment'] = "CorporateCard:CC-5512"

        return params

    def _extract_flight_params(self, instruction: str) -> Dict[str, str]:
        """Extract flight booking parameters from instruction."""
        import re

        params = {}

        # Origin
        origin_match = re.search(r'from\s([\w\s]+?)(?=\sto|\son|$)', instruction, re.IGNORECASE)
        if origin_match:
            params['origin'] = origin_match.group(1).strip()

        # Destination
        dest_match = re.search(r'to\s([\w\s]+?)(?=\sfor|\son|$)', instruction, re.IGNORECASE)
        if dest_match:
            params['destination'] = dest_match.group(1).strip()

        return params

    def _update_execution_history(self, instruction: str, trace: List[Dict], success: bool) -> None:
        """Store execution in history for analysis."""
        self.execution_history.append({
            "instruction": instruction,
            "trace": trace,
            "success": success
        })

        # Keep last 50
        if len(self.execution_history) > 50:
            self.execution_history.pop(0)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test LLM-Reflect baseline in isolation.
    Should show some adaptation via memory but slower than ANNEAL.
    """
    print("=" * 70)
    print("Testing LLM-Reflect Baseline")
    print("=" * 70)

    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config_loader import load_config

    config = load_config("config.yaml")

    # Override for quick test
    config['scenario']['num_tasks'] = 10
    config['logging']['level'] = 'INFO'

    # Run baseline
    agent = LLMReflectAgent(config)
    metrics = agent.run_evaluation()

    # Print results
    print("\n" + "=" * 70)
    print("LLM-REFLECT TEST RESULTS")
    print("=" * 70)

    summary = metrics.get_summary()
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"RFR: {summary['repeat_failure_rate']:.1%}")
    print(f"TTA: {summary['time_to_adapt']}")
    print(f"Memory Size: {len(agent.reflection_memory)}")

    # Show stored reflections
    print("\n[Stored Reflections]")
    for i, r in enumerate(agent.reflection_memory, 1):
        print(f"{i}. {r['summary']}")

    # Expected: Better than Static-NS, worse than ANNEAL
    print("\nExpected behavior:")
    print("  - Success rate: ~60-70% (learns via memory, but imprecise)")
    print("  - RFR: ~15-20% (some adaptation, but slower than symbolic)")
    print("  - TTA: ~20-30 tasks (textual reasoning is slower)")

    print("\n✅ LLM-Reflect baseline test complete")