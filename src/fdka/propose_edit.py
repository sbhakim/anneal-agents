# src/fdka/propose_edit.py
"""
Implements the 3-stage ProposeEdit pipeline from Section VIII-B of the paper.

✅ UPDATED: Fixed provider initialization to work with flat config structure.
"""

from typing import List, Dict, Any, Optional
import uuid
import json
import hashlib
import signal
import functools
import re
import os

# API providers
try:
    from .llm_providers.openai_provider import OpenAIProvider

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from .llm_providers.deepseek_provider import DeepSeekProvider

    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

# Hugging Face Transformers
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

PATCH_TEMPLATES = {
    ("PreconditionUnmet", ("blackout", "policy", "blocked", "h-23")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(BlockedCard(payment, dates))",
            "justification": "API policy H-23 blocks corporate cards during blackout dates"
        }
    },
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


def timeout(seconds=60):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Function call timed out after {seconds}s")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.name == 'nt':
                return func(*args, **kwargs)
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

    def __init__(self, config: Dict[str, Any], metrics_collector: Optional[Any] = None):
        """
        Initialize FDKA pipeline with dynamic LLM provider selection.

        ✅ FIXED: Now correctly passes flat config structure to providers.
        """
        # Extract the 'propose_edit' section from fdka config
        self.config = config.get("propose_edit", {})
        self.metrics_collector = metrics_collector

        self.llm_provider_type = self.config.get("llm_provider", "mock")
        self.llm = None

        print(f"FDKA: Initializing with provider='{self.llm_provider_type}'")

        # ✅ FIXED: Pass the entire config (flat structure) to providers
        if self.llm_provider_type == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI provider selected but not available. Run: pip install openai>=1.0.0")

            # ✅ Pass self.config directly (contains model, temperature, etc.)
            self.llm = OpenAIProvider(self.config)
            print(f"FDKA: ✅ OpenAI provider initialized (model={self.config.get('model', 'gpt-4o-mini')})")

        elif self.llm_provider_type == 'deepseek':
            if not DEEPSEEK_AVAILABLE:
                raise ImportError("DeepSeek provider not available. Ensure deepseek_provider.py exists.")

            # ✅ Pass self.config directly
            self.llm = DeepSeekProvider(self.config)
            print(f"FDKA: ✅ DeepSeek provider initialized (model={self.config.get('model', 'deepseek-chat')})")

        elif self.llm_provider_type == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers provider selected but not available. Run: pip install transformers torch")
            self._initialize_local_llm()
            print(f"FDKA: ✅ Local Transformers model initialized")

        else:  # mock provider
            print("FDKA: Using mock LLM (rule-based fallback, no API calls).")

        self.accepted_patch_cache: List[Dict[str, Any]] = []
        self.ALLOWED_ACTIONS = ["ADD_PRECONDITION", "REFINE_EFFECT", "UPDATE_TOOL_SCHEMA"]
        self.KNOWN_PREDICATES = {
            "is_card_valid", "is_hotel_available", "is_flight_available",
            "check_not_blocked_card", "Not", "BlockedCard", "IfThen",
            "NetworkAvailable", "Sent", "ValidState", "ValidPayment"
        }

    def _initialize_local_llm(self):
        """Initializes a local Hugging Face transformers pipeline."""
        # ✅ For transformers, look for nested config (backwards compatibility)
        transformers_cfg = self.config.get('transformers_config', {})

        # If no nested config, use flat config values
        if not transformers_cfg:
            transformers_cfg = self.config

        active_model_key = transformers_cfg.get('active_model')
        local_models_map = transformers_cfg.get('local_models', {})

        # Fallback: If no active_model specified, use 'model' field directly
        if not active_model_key and 'model' in transformers_cfg:
            self.local_model_id = transformers_cfg['model']
        else:
            self.local_model_id = local_models_map.get(active_model_key)

        if not self.local_model_id:
            raise ValueError(f"Config error: Could not determine model ID from config: {transformers_cfg}")

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
            print(f"FDKA: ❌ Failed to load local LLM: {e}")
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
        print(
            f"FDKA: [3/3] Patch proposed -> {patch.get('action', 'N/A')}: {patch.get('details', 'N/A')} (id={patch.get('id')})")
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
        raw_patch_json = self._stage2_generate(prompt)
        print(f"  🤖 Stage 2: LLM generated -> {raw_patch_json.get('action', 'N/A')}")
        valid_patch = self._stage3_validate(raw_patch_json, rule_pool)
        print("  ✅ Stage 3: Validated patch.")
        return valid_patch

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
            "operator": sig, "state_minimal": minimal,
            "error": {"type": fi.get("error"), "site": fi.get("operator"),
                      "evidence": (fi.get("policy_ref", "") + fi.get("message", "")[:50])},
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
        name, et = fi.get("operator", "Unknown"), fi.get("error", "Unknown")
        exp = []
        if "Hotel" in name: exp.append("Booked(location, dates)")
        if "Flight" in name: exp.append("Booked(origin, destination, date)")
        return {"expected": exp, "observed": [f"Error({et})"]}

    def _stage2_generate(self, serialized_prompt):
        prompt_str = self._build_llm_prompt(serialized_prompt)

        if self.llm_provider_type in ['openai', 'deepseek']:
            return self._generate_with_api(prompt_str)
        elif self.llm_provider_type == 'transformers':
            return self._generate_with_transformers(prompt_str)
        else:  # mock
            return self._mock_llm_generate(serialized_prompt)

    @timeout(seconds=60)
    def _generate_with_api(self, prompt: str) -> Dict:
        if not self.llm:
            print("  ⚠️ API provider not initialized. Falling back to mock.")
            return self._mock_llm_generate({"prompt": prompt})

        print(f"  🧠 Calling {self.llm_provider_type} API with enhanced prompt...")
        result = self.llm.generate(prompt)

        # ✅ Track efficiency metrics
        if self.metrics_collector:
            self.metrics_collector.record_llm_call(
                tokens=result.get('tokens_used', 0),
                latency_sec=result.get('latency_sec', 0),
                model=result.get('model', self.llm_provider_type)
            )

        if 'error' in result or not result['text']:
            print(f"  ⚠️ API call failed. Reason: {result.get('error', 'Empty response')}. Falling back.")
            return self._mock_llm_generate({"prompt": prompt})

        raw_text = result['text']
        try:
            json_start = raw_text.find('{')
            json_end = raw_text.rfind('}') + 1
            if json_start != -1:
                json_str = raw_text[json_start:json_end]
                return json.loads(json_str)
            raise ValueError("No JSON object found in API response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️ Could not parse JSON from API response: {e}. Falling back.")
            return self._mock_llm_generate({"prompt": prompt})

    @timeout(seconds=60)
    def _generate_with_transformers(self, prompt: str) -> Dict:
        if not self.llm_pipeline:
            print("  ⚠️ No local LLM; falling back to mock.")
            return self._mock_llm_generate({"prompt": prompt})

        try:
            print("  🧠 Calling local model with enhanced prompt...")
            # Use flat config or nested transformers_config
            cfg = self.config.get('transformers_config', self.config)

            out = self.llm_pipeline(
                prompt, max_new_tokens=cfg.get('max_tokens', 300), do_sample=True,
                temperature=cfg.get('temperature', 0.3), return_full_text=False,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id, top_p=0.95,
                repetition_penalty=cfg.get('repetition_penalty', 1.1)
            )
            txt = out[0]["generated_text"].strip()

            jsstart = txt.find("{")
            if jsstart < 0: raise ValueError("No JSON in LLM output")

            bc, jsend = 0, jsstart
            for i, c in enumerate(txt[jsstart:], start=jsstart):
                if c == "{":
                    bc += 1
                elif c == "}":
                    bc -= 1
                if bc == 0: jsend = i + 1; break

            jp = txt[jsstart:jsend].strip("`")
            return json.loads(jp)

        except (TimeoutException, json.JSONDecodeError, Exception) as e:
            print(f"  ⚠️ Local LLM generation error: {e}. Falling back.")
            return self._mock_llm_generate({"prompt": prompt})

    def _build_llm_prompt(self, sp):
        state = sp.get("state_minimal", {})
        payment = state.get("payment_method", "payment")
        dates = state.get("travel_dates", "dates")
        error_type = sp.get("error", {}).get("type", "Unknown")
        error_evidence = sp.get("error", {}).get("evidence", "")
        operator_name = sp.get("operator", {}).get("name", "Unknown")

        guidance, recommended_predicate = "", ""
        if error_type == "PreconditionUnmet":
            if any(k in error_evidence.lower() for k in ["blackout", "blocked"]):
                guidance = "This is a blackout date violation. Use Not(BlockedCard(payment, dates))."
                recommended_predicate = f'Not(BlockedCard({payment}, {dates}))'
            elif any(k in error_evidence.lower() for k in ["invalid", "payment"]):
                guidance = "This is a payment validation failure. Use ValidPayment(payment)."
                recommended_predicate = f'ValidPayment({payment})'
        elif error_type == "ToolError" and any(k in error_evidence.lower() for k in ["timeout", "network"]):
            guidance = "This is a network/API timeout. Use REFINE_EFFECT with IfThen(NetworkAvailable(), ...)."
            recommended_predicate = 'IfThen(NetworkAvailable(), ExecuteTool())'

        return f"""You are a symbolic patch generator for a travel booking system.

=== DOMAIN VOCABULARY (USE ONLY THESE) ===
Predicates: Not(BlockedCard(payment, dates)), ValidPayment(payment), NetworkAvailable(), IfThen(Condition, Effect)

=== CURRENT FAILURE CONTEXT ===
Operator: {operator_name}
Error Type: {error_type}
Evidence: {error_evidence}
{guidance}

=== FEW-SHOT EXAMPLES ===
1. {{"action": "ADD_PRECONDITION", "operator": "BookHotel", "patch": {{"predicate": "Not(BlockedCard(CorporateCard:CC-5512, April 21-24))", "justification": "API policy blocks corporate cards during blackout periods"}}}}
2. {{"action": "ADD_PRECONDITION", "operator": "BookHotel", "patch": {{"predicate": "ValidPayment(CorporateCard:CC-5512)", "justification": "Ensure payment method is valid before booking"}}}}
3. {{"action": "REFINE_EFFECT", "operator": "BookFlight", "patch": {{"guard": "IfThen(NetworkAvailable(), ExecuteTool())", "justification": "Prevent tool execution on network failure"}}}}

=== STRICT OUTPUT REQUIREMENTS ===
1. Use ONLY predicates from the Domain Vocabulary.
2. Use actual parameter values (e.g., "{payment}").
3. Output ONLY valid JSON.

=== RECOMMENDED SOLUTION ===
Recommended Predicate: {recommended_predicate}

Generate the patch JSON for the current failure:
"""

    def _mock_llm_generate(self, sp):
        error = sp.get("error", {}).get("type", "")
        ev = sp.get("error", {}).get("evidence", "").lower()
        operator_name = sp.get("operator", {}).get("name", "Unknown")
        print(f"  🎯 Mock template matching: error={error}, evidence='{ev[:50]}...'")
        for k, t in PATCH_TEMPLATES.items():
            if isinstance(k, tuple):
                ft, kws = k
                if ft == error and any(kw in ev for kw in kws):
                    return {**t, "operator": operator_name}
        return {**PATCH_TEMPLATES["default"], "operator": operator_name}

    def _stage3_validate(self, raw, rp):
        """Stage 3: Deterministic validation."""
        self._validate_schema(raw)
        self._validate_typing_and_semantics(raw, rp)
        return self._normalize_patch(raw)

    def _validate_schema(self, p):
        if not isinstance(p, dict) or not all(k in p for k in ["action", "operator", "patch"]):
            raise ValueError(f"Invalid patch structure: {p}")
        if p["action"] not in self.ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{p['action']}', must be one of {self.ALLOWED_ACTIONS}")
        print("  ✓ Schema OK")

    def _validate_typing_and_semantics(self, p, rp):
        d = p["patch"].get("predicate") or p["patch"].get("guard", "")
        if d and not any(pred in d for pred in self.KNOWN_PREDICATES):
            print(f"  ⚠️ Unknown predicate in '{d}'. This may cause issues.")
        print("  ✓ Type OK")

    def _normalize_patch(self, p):
        """Normalize patch to a standard internal format."""
        norm = {
            "action": p["action"], "operator": p["operator"],
            "details": p["patch"].get("predicate") or p["patch"].get("guard", ""),
            "justification": p["patch"].get("justification", "Generated by FDKA")
        }
        content_str = f"{norm['operator']}:{norm['action']}:{norm['details']}"
        norm["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()[:12]
        return norm