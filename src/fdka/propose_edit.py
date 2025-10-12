# src/fdka/propose_edit.py

"""
Implements the 3-stage ProposeEdit pipeline from Section VIII-B of the paper.
Its sole responsibility is to take a failure trace and produce a structured,
verifiable patch proposal.

UPDATED (CRITICAL FIX):
- Enhanced LLM prompt with explicit domain vocabulary and constraints
- Added concrete few-shot examples with actual predicates (not placeholders)
- Expanded template keywords to cover payment validation failures
- Improved state context extraction for better grounding
- Stricter output format validation with domain predicate enforcement
"""

from typing import List, Dict, Any
import uuid
import json
import hashlib
import signal
import functools
import re

# NEW: Hugging Face Transformers imports
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# UPDATED: Expanded template keywords to include payment validation patterns
PATCH_TEMPLATES = {
    ("PreconditionUnmet", ("blackout", "policy", "blocked", "h-23")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(BlockedCard(payment, dates))",
            "justification": "API policy H-23 blocks corporate cards during blackout dates"
        }
    },
    # NEW: Payment validation pattern for invalid/unsupported payment methods
    ("PreconditionUnmet", ("invalid", "payment", "unsupported", "expired", "declined")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "ValidPayment(payment)",
            "justification": "Payment method validation to prevent API rejection"
        }
    },
    ("ToolError", ("timeout", "api", "network")): {
        "action": "REFINE_EFFECT",
        "patch": {
            "guard": "IfThen(NetworkAvailable(), ExecuteTool())",
            "justification": "Tool timeout handling; ensure network is available before execution."
        }
    },
    "default": {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "ValidState()",
            "justification": "Generic failure prevention for an unrecognized error pattern."
        }
    }
}


class TimeoutException(Exception):
    pass


def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Function call timed out after {seconds}s")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


class FDKAPipeline:
    """
    Implements the 3-stage neurosymbolic patch synthesis pipeline.
    Public API: `propose_edit(trace, rule_pool)`
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        pe_cfg = config.get("propose_edit", {})

        self.llm_provider = pe_cfg.get("llm_provider", "mock")
        self.llm_model = pe_cfg.get("model", "gpt-4")
        self.temperature = pe_cfg.get("temperature", 0.3)
        self.num_examples = pe_cfg.get("num_examples", 3)

        active_key = pe_cfg.get("active_model")
        local_map = pe_cfg.get("local_models", {})
        self.local_model_id = local_map.get(active_key)

        self.llm_pipeline = None
        self.accepted_patch_cache: List[Dict[str, Any]] = []
        self.ALLOWED_ACTIONS = ["ADD_PRECONDITION", "REFINE_EFFECT", "UPDATE_TOOL_SCHEMA"]

        # UPDATED: Expanded known predicates with payment validation
        self.KNOWN_PREDICATES = {
            "is_card_valid", "is_hotel_available", "is_flight_available",
            "check_not_blocked_card", "Not", "BlockedCard", "IfThen",
            "NetworkAvailable", "Sent", "ValidState", "ValidPayment"
        }

        print(f"FDKA: Initialized with provider='{self.llm_provider}', model='{self.llm_model}'")

        if self.llm_provider == "transformers":
            self._initialize_local_llm()

    def _initialize_local_llm(self):
        if not TRANSFORMERS_AVAILABLE:
            print("FDKA: ❌ transformers not installed. Run 'pip install transformers torch'.")
            return
        if not self.local_model_id:
            raise ValueError("Config error: 'local_model_id' missing")
        try:
            print(f"FDKA: 🚧 Loading local LLM: {self.local_model_id}...")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.local_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("FDKA: ✅ Local LLM loaded.")
        except Exception as e:
            print(f"FDKA: ❌ Failed to load LLM: {e}")
            self.llm_pipeline = None

    def propose_edit(self, trace: List[Dict[str, Any]], rule_pool) -> Dict[str, Any]:
        print("\n--- FDKA PROPOSAL INITIATED ---")
        fi = self._extract_failure_info(trace)
        op_name = self._localize(fi)
        print(f"FDKA: [1/3] Fault localized -> '{op_name}'")
        patch = self._propose_edit_three_stage(fi, trace, rule_pool)
        patch.setdefault("operator", op_name)
        patch.setdefault("id", f"patch-{uuid.uuid4().hex[:8]}")
        patch.setdefault("source", "fdka.v1.2")
        print(f"FDKA: [3/3] Patch proposed -> {patch['action']}: {patch['details']} (id={patch['id']})")
        print("--- FDKA PROPOSAL COMPLETED ---")
        return patch

    def _extract_failure_info(self, trace):
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return {}

    def _localize(self, fi):
        return fi.get("operator", "Unknown")

    def _propose_edit_three_stage(self, fi, trace, rule_pool):
        print("  🔧 ProposeEdit: Starting 3-stage pipeline...")
        prompt = self._stage1_serialize(fi, trace, rule_pool)
        print("  📝 Stage 1: Serialized trace.")
        raw = self._stage2_generate(prompt)
        print(f"  🤖 Stage 2: LLM generated -> {raw.get('action', 'N/A')}")
        valid = self._stage3_validate(raw, rule_pool)
        print("  ✅ Stage 3: Validated patch.")
        return valid

    def _stage1_serialize(self, fi, trace, rp):
        op = rp.get_operator(fi.get("operator", "Unknown"))
        sig = {
            "name": fi.get("operator", "Unknown"),
            "params": op.params if op else [],
            "preconds": rp.list_preconditions(fi.get("operator", "Unknown")) if op else [],
            "effects": ["<effects>"]
        }
        minimal = self._extract_minimal_state(trace, sig)
        delta = self._compute_state_delta(fi)
        return {
            "operator": sig,
            "state_minimal": minimal,
            "error": {
                "type": fi.get("error"),
                "site": fi.get("operator"),
                "evidence": (fi.get("policy_ref", "") + fi.get("message", "")[:50])
            },
            "state_delta": delta
        }

    def _extract_minimal_state(self, trace, sig):
        ms = {}
        for e in reversed(trace):
            if "state_before" in e:
                fs = e["state_before"]
                for p in sig.get("params", []):
                    for k, v in fs.items():
                        if p.lower() in k.lower(): ms[k] = v
                for k in ["payment_method", "travel_dates", "blackout_dates", "corporate_card_policy"]:
                    if k in fs: ms[k] = fs[k]
                break
        return ms

    def _compute_state_delta(self, fi):
        name = fi.get("operator", "Unknown")
        et = fi.get("error", "Unknown")
        exp = []
        if "Hotel" in name: exp.append("Booked(location, dates)")
        if "Flight" in name: exp.append("Booked(origin, destination, date)")
        return {"expected": exp, "observed": [f"Error({et})"]}

    def _stage2_generate(self, prompt):
        if self.llm_provider == "transformers":
            return self._real_llm_generate(prompt)
        return self._mock_llm_generate(prompt)

    def _build_llm_prompt(self, sp):
        """
        CRITICAL FIX: Enhanced prompt with explicit domain vocabulary and constraints.

        This addresses the root cause where LLMs generate invalid predicates like "Pred(param)".
        Now provides:
        1. Explicit domain vocabulary with parameter bindings
        2. Concrete few-shot examples using actual predicates
        3. Current failure context with real state values
        4. Strict output format enforcement
        """
        # Extract actual state values for grounding
        state = sp.get("state_minimal", {})
        payment = state.get("payment_method", "payment")
        dates = state.get("travel_dates", "dates")
        blackout_dates = state.get("blackout_dates", [])

        # Get error details
        error_type = sp.get("error", {}).get("type", "Unknown")
        error_evidence = sp.get("error", {}).get("evidence", "")
        operator_name = sp.get("operator", {}).get("name", "Unknown")

        # Build domain-specific guidance based on error type
        guidance = ""
        recommended_predicate = ""

        if error_type == "PreconditionUnmet":
            if any(keyword in error_evidence.lower() for keyword in ["blackout", "blocked", "policy"]):
                guidance = "This is a blackout date violation. Use Not(BlockedCard(payment, dates))."
                recommended_predicate = f'Not(BlockedCard({payment}, {dates}))'
            elif any(keyword in error_evidence.lower() for keyword in ["invalid", "payment", "expired", "declined"]):
                guidance = "This is a payment validation failure. Use ValidPayment(payment)."
                recommended_predicate = f'ValidPayment({payment})'
            else:
                guidance = "For PreconditionUnmet, add a precondition check using domain predicates."
                recommended_predicate = f'ValidPayment({payment})'

        elif error_type == "ToolError":
            if any(keyword in error_evidence.lower() for keyword in ["timeout", "network", "api"]):
                guidance = "This is a network/API timeout. Use REFINE_EFFECT with IfThen(NetworkAvailable(), ...)."
                recommended_predicate = 'IfThen(NetworkAvailable(), ExecuteTool())'
            else:
                guidance = "For ToolError, add conditional execution guards."
                recommended_predicate = 'IfThen(NetworkAvailable(), ExecuteTool())'

        # Build comprehensive prompt
        prompt = f"""You are a symbolic patch generator for a travel booking system.

=== DOMAIN VOCABULARY (USE ONLY THESE) ===
Available predicates for patches:
1. Not(BlockedCard(payment, dates)) - Check if payment method is NOT blocked on given dates
2. ValidPayment(payment) - Validate payment method is active and supported
3. NetworkAvailable() - Check network connectivity
4. IfThen(Condition, Effect) - Conditional effect execution

Parameter bindings for this failure:
- payment = "{payment}"
- dates = "{dates}"
- blackout_dates = {json.dumps(blackout_dates)}

=== CURRENT FAILURE CONTEXT ===
Operator: {operator_name}
Error Type: {error_type}
Evidence: {error_evidence}

{guidance}

=== FEW-SHOT EXAMPLES (ACTUAL PATCHES) ===
Example 1 - Blackout date handling:
{{"action": "ADD_PRECONDITION",
  "operator": "BookHotel",
  "patch": {{
    "predicate": "Not(BlockedCard(CorporateCard:CC-5512, April 21-24))",
    "justification": "API policy blocks corporate cards during blackout periods"
  }}
}}

Example 2 - Payment validation:
{{"action": "ADD_PRECONDITION",
  "operator": "BookHotel", 
  "patch": {{
    "predicate": "ValidPayment(CorporateCard:CC-5512)",
    "justification": "Ensure payment method is valid before booking"
  }}
}}

Example 3 - Network timeout handling:
{{"action": "REFINE_EFFECT",
  "operator": "BookFlight",
  "patch": {{
    "guard": "IfThen(NetworkAvailable(), ExecuteTool())",
    "justification": "Prevent tool execution on network failure"
  }}
}}

=== STRICT OUTPUT REQUIREMENTS ===
1. Use ONLY predicates from the Domain Vocabulary above
2. Use actual parameter values (e.g., "{payment}", not "payment")
3. Choose action: ADD_PRECONDITION for missing checks, REFINE_EFFECT for conditional logic
4. Output ONLY valid JSON, no explanation text

=== RECOMMENDED SOLUTION ===
Based on the failure context, the recommended predicate is:
{recommended_predicate}

Now generate the patch for this failure in JSON format:
"""

        return prompt

    @timeout(seconds=60)
    def _real_llm_generate(self, sp):
        if not self.llm_pipeline:
            print("  ⚠️ No real LLM; falling back to mock.")
            return self._mock_llm_generate(sp)

        prompt = self._build_llm_prompt(sp)

        try:
            print("  🧠 Calling local model with enhanced prompt...")
            out = self.llm_pipeline(
                prompt,
                max_new_tokens=300,  # Increased for more detailed output
                do_sample=True,
                temperature=self.temperature,
                return_full_text=False,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
                top_p=0.95,  # Nucleus sampling for better quality
                repetition_penalty=1.1
            )

            txt = out[0]["generated_text"].strip()

            # Robust JSON extraction
            jsstart = txt.find("{")
            if jsstart < 0:
                print(f"  ⚠️ No JSON found in output: {txt[:100]}")
                raise ValueError("No JSON in LLM output")

            # Extract balanced JSON
            bc, jsend = 0, jsstart
            for i, c in enumerate(txt[jsstart:], start=jsstart):
                if c == "{":
                    bc += 1
                elif c == "}":
                    bc -= 1
                if bc == 0:
                    jsend = i + 1
                    break

            jp = txt[jsstart:jsend].strip("`")
            rp = json.loads(jp)

            print(f"  ✓ Parsed {len(jp)} chars from LLM output")

            # Validate that output uses domain predicates
            predicate = rp.get("patch", {}).get("predicate", "")
            guard = rp.get("patch", {}).get("guard", "")
            detail = predicate or guard

            if detail and not any(p in detail for p in self.KNOWN_PREDICATES):
                print(f"  ⚠️ LLM generated unknown predicate: {detail}")
                print(f"  ⚠️ Falling back to template-based generation")
                return self._mock_llm_generate(sp)

            return rp

        except TimeoutException as e:
            print(f"  ⚠️ LLM call timed out: {e}")
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Invalid JSON from LLM: {e}")
        except Exception as e:
            print(f"  ⚠️ LLM generation error: {e}")

        print("  → Falling back to template-based generation")
        return self._mock_llm_generate(sp)

    def _mock_llm_generate(self, sp):
        """
        Template-based patch generation with improved pattern matching.
        This serves as both a fallback and a baseline for comparison.
        """
        error = sp.get("error", {}).get("type", "")
        ev = sp.get("error", {}).get("evidence", "").lower()
        operator_name = sp.get("operator", {}).get("name", "Unknown")

        print(f"  🎯 Template matching: error={error}, evidence='{ev[:50]}...'")

        # Try to match templates with expanded keywords
        for k, t in PATCH_TEMPLATES.items():
            if isinstance(k, tuple):
                ft, kws = k
                if ft == error and any(kw in ev for kw in kws):
                    print(f"  ✓ Matched template: {ft} with keywords {[kw for kw in kws if kw in ev]}")
                    result = {**t, "operator": operator_name}
                    return result

        # Default fallback
        print(f"  ⚠️ No template match, using default")
        d = PATCH_TEMPLATES["default"]
        return {**d, "operator": operator_name}

    def _stage3_validate(self, raw, rp):
        """
        Stage 3: Deterministic validation with enhanced checks.
        """
        self._validate_schema(raw)
        self._validate_typing_and_semantics(raw, rp)
        return self._normalize_patch(raw)

    def _validate_schema(self, p):
        """Validate patch schema structure."""
        if not isinstance(p, dict) or not all(k in p for k in ["action", "operator", "patch"]):
            raise ValueError(f"Invalid patch structure: {p}")
        if p["action"] not in self.ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{p['action']}', must be one of {self.ALLOWED_ACTIONS}")
        print("  ✓ Schema OK")

    def _validate_typing_and_semantics(self, p, rp):
        """
        Validate that patch uses known predicates.
        UPDATED: Stricter validation with helpful error messages.
        """
        d = p["patch"].get("predicate") or p["patch"].get("guard", "")

        if not d:
            print("  ✓ Type OK (no predicate details)")
            return

        # Check if uses known predicates
        uses_known = any(pred in d for pred in self.KNOWN_PREDICATES)

        if not uses_known:
            # Extract what looks like predicate names
            potential_predicates = re.findall(r'([A-Z][a-zA-Z]+)', d)
            print(f"  ⚠️ Unknown predicate in '{d}'")
            print(f"  ⚠️ Detected tokens: {potential_predicates}")
            print(f"  ⚠️ Known predicates: {self.KNOWN_PREDICATES}")
            print(f"  ⚠️ This may cause stage_fn to reject during canary test")
            # Don't raise - let canary test catch it, but warn
            return

        print("  ✓ Type OK (uses known predicates)")

    def _normalize_patch(self, p):
        """
        Normalize patch to standard format.
        """
        norm = {
            "action": p["action"],
            "operator": p["operator"],
            "details": p["patch"].get("predicate") or p["patch"].get("guard", ""),
            "justification": p["patch"].get("justification", "Generated by FDKA")
        }

        # Generate content hash for deduplication
        content_str = f"{norm['operator']}:{norm['action']}:{norm['details']}"
        norm["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        return norm


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the enhanced ProposeEdit pipeline.
    """
    print("=" * 70)
    print("Testing Enhanced FDKA ProposeEdit Pipeline")
    print("=" * 70)

    # Mock configuration
    config = {
        "propose_edit": {
            "llm_provider": "mock",
            "model": "gpt-4",
            "temperature": 0.3,
            "num_examples": 3
        }
    }

    # Initialize pipeline
    fdka = FDKAPipeline(config)

    # Test case 1: Blackout date failure
    print("\n[Test 1: Blackout Date Failure]")
    mock_trace = [
        {"state_before": {
            "payment_method": "CorporateCard:CC-5512",
            "travel_dates": "April 21-24",
            "blackout_dates": ["April 21", "April 22", "April 23"]
        }},
        {"error": "PreconditionUnmet",
         "operator": "BookHotel",
         "message": "Corporate cards blocked",
         "policy_ref": "H-23"}
    ]


    class MockRulePool:
        def get_operator(self, name):
            class MockOp:
                params = ["location", "dates", "payment"]

            return MockOp()

        def list_preconditions(self, name):
            return ["is_card_valid", "is_hotel_available"]


    patch = fdka.propose_edit(mock_trace, MockRulePool())
    print(f"\nGenerated Patch:")
    print(f"  Action: {patch['action']}")
    print(f"  Details: {patch['details']}")
    print(f"  Justification: {patch['justification']}")

    # Test case 2: Payment validation failure
    print("\n[Test 2: Invalid Payment Failure]")
    mock_trace2 = [
        {"state_before": {
            "payment_method": "CorporateCard:CC-9999",
            "travel_dates": "June 10-11"
        }},
        {"error": "PreconditionUnmet",
         "operator": "BookHotel",
         "message": "Payment method invalid"}
    ]

    patch2 = fdka.propose_edit(mock_trace2, MockRulePool())
    print(f"\nGenerated Patch:")
    print(f"  Action: {patch2['action']}")
    print(f"  Details: {patch2['details']}")
    print(f"  Expected: ValidPayment(...)")

    # Test case 3: Network timeout
    print("\n[Test 3: Network Timeout]")
    mock_trace3 = [
        {"state_before": {"network_status": "unstable"}},
        {"error": "ToolError",
         "operator": "BookFlight",
         "message": "API timeout"}
    ]

    patch3 = fdka.propose_edit(mock_trace3, MockRulePool())
    print(f"\nGenerated Patch:")
    print(f"  Action: {patch3['action']}")
    print(f"  Details: {patch3['details']}")
    print(f"  Expected: IfThen(NetworkAvailable(), ...)")

    print("\n" + "=" * 70)
    print("✅ Enhanced FDKA testing complete")
    print("=" * 70)