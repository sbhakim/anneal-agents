# src/fdka/propose_edit.py
"""
Implements the 3-stage ProposeEdit pipeline from Section VIII-B of the paper.
Its sole responsibility is to take a failure trace and produce a structured,
verifiable patch proposal.

UPDATED: The monolithic `run_fdka_cycle` has been removed. Orchestration is now
handled by SelfEvolveSystem. This class now exposes a single public method:
`propose_edit`, which executes the 3-stage synthesis pipeline.
"""
from typing import List, Dict, Any, Tuple, Optional
import uuid
import json
import hashlib

# NEW: Add imports for Hugging Face Transformers to run a real offline LLM
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# A dictionary that maps failure patterns to patch templates for the mock generator.
# This data-driven approach replaces a rigid if/elif structure, making it
# easier to add new "learned" repair strategies. It simulates the knowledge
# of a constrained, fine-tuned model.
PATCH_TEMPLATES = {
    # CORRECTED: Dictionary keys now use tuples instead of lists.
    # Pattern: (error_type, tuple_of_keywords_in_evidence) -> patch_template
    ("PreconditionUnmet", ("blackout", "policy", "blocked", "h-23")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(BlockedCard(payment, dates))",
            "justification": "API policy H-23 blocks corporate cards during blackout dates"
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


class FDKAPipeline:
    """
    Implements the 3-stage neurosymbolic patch synthesis pipeline.
    Localize -> Serialize -> Constrained Generate -> Validate.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        propose_edit_config = config.get("propose_edit", {})

        # ProposeEdit configuration (Section VIII-B)
        self.llm_provider = propose_edit_config.get("llm_provider", "mock")
        self.llm_model = propose_edit_config.get("model", "gpt-4")
        self.temperature = propose_edit_config.get("temperature", 0.3)
        self.num_examples = propose_edit_config.get("num_examples", 3)

        # --- UPDATED LOGIC TO READ NEW CONFIG STRUCTURE ---
        # Get the nickname of the active model (e.g., "gemma")
        active_model_key = propose_edit_config.get("active_model")
        # Get the dictionary that maps nicknames to full model IDs
        local_models_map = propose_edit_config.get("local_models", {})
        # Look up the full model ID using the active key
        self.local_model_id = local_models_map.get(active_model_key)
        # --- END OF UPDATED LOGIC ---

        # LLM pipeline for local model
        self.llm_pipeline = None

        # Cache for few-shot examples (provenance IDs)
        self.accepted_patch_cache: List[Dict[str, Any]] = []

        # Closed schema for patch actions (from Section VIII-B, Stage 2)
        self.ALLOWED_ACTIONS = [
            "ADD_PRECONDITION",
            "REFINE_EFFECT",
            "UPDATE_TOOL_SCHEMA"
        ]

        # Known predicates for type checking (from rule pool ontology)
        self.KNOWN_PREDICATES = {
            "is_card_valid", "is_hotel_available", "is_flight_available",
            "check_not_blocked_card", "Not", "BlockedCard", "IfThen",
            "NetworkAvailable", "Sent", "ValidState"
        }

        print(f"FDKA: Initialized with provider='{self.llm_provider}', model='{self.llm_model}'")

        # Pre-load the local model if configured to do so
        if self.llm_provider == "transformers":
            self._initialize_local_llm()

    def _initialize_local_llm(self):
        """Loads the local LLM from the cache using the transformers pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            print(
                "FDKA: ❌ 'transformers' library not installed. Cannot use local LLM. Run 'pip install transformers torch'.")
            return

        # This error message is what you saw. It now correctly checks the looked-up value.
        if not self.local_model_id:
            raise ValueError(
                "Configuration error: 'local_model_id' could not be determined from the 'active_model' key in config.yaml.")

        try:
            print(f"FDKA: 🚧 Initializing local LLM: {self.local_model_id}. This may take a moment...")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.local_model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on modern GPUs
                device_map="auto"  # Automatically use GPU if available
            )
            print("FDKA: ✅ Local LLM initialized successfully.")
        except Exception as e:
            print(f"FDKA: ❌ Failed to load local LLM. Check model ID and dependencies (e.g., 'accelerate'). Error: {e}")
            self.llm_pipeline = None

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def propose_edit(self, trace: List, rule_pool) -> Dict[str, Any]:
        """
        This is the new public entry point for this class.
        It runs the full 3-stage patch proposal pipeline.
        """
        print("\n--- FDKA PROPOSAL INITIATED ---")

        # 1) Localize the faulty operator
        failure_info = self._extract_failure_info(trace)
        responsible_operator_name = self._localize(failure_info)
        print(f"FDKA: [1/3] Fault localized to operator -> '{responsible_operator_name}'")

        # 2) Propose a patch using the 3-stage neurosymbolic pipeline
        patch = self._propose_edit_three_stage(failure_info, trace, rule_pool)

        # 3) Finalize the patch object with metadata
        patch.setdefault("operator", responsible_operator_name)
        patch.setdefault("id", f"patch-{uuid.uuid4().hex[:8]}")
        patch.setdefault("source", "fdka.v1.2")

        print(f"FDKA: [3/3] Patch proposed -> {patch.get('action')}: {patch.get('details', 'N/A')} (id={patch['id']})")
        print("--- FDKA PROPOSAL COMPLETED ---")

        return patch

    def _extract_failure_info(self, trace: List) -> Dict[str, Any]:
        """Extracts the last error entry from a trace."""
        for entry in reversed(trace or []):
            if isinstance(entry, dict) and "error" in entry:
                return entry
        return {}

    # ========================================================================
    # STAGE 0: Localization (Section VIII-A)
    # ========================================================================

    def _localize(self, failure_info: Dict[str, Any]) -> str:
        """For PoC, extracts operator directly from failure info."""
        return failure_info.get("operator", "Unknown")

    # ========================================================================
    # 3-STAGE PROPOSEEDIT PIPELINE (Section VIII-B)
    # ========================================================================

    def _propose_edit_three_stage(
            self,
            failure_info: Dict[str, Any],
            trace: List[Dict[str, Any]],
            rule_pool
    ) -> Dict[str, Any]:
        """Implements the complete 3-stage ProposeEdit pipeline."""
        print("  🔧 ProposeEdit: Starting 3-stage pipeline...")
        structured_prompt = self._stage1_serialize(failure_info, trace, rule_pool)
        print(f"  📝 Stage 1: Serialized trace to structured prompt.")
        raw_patch_json = self._stage2_generate(structured_prompt)
        print(f"  🤖 Stage 2: LLM generated patch -> {raw_patch_json.get('action', 'N/A')}")
        validated_patch = self._stage3_validate(raw_patch_json, rule_pool)
        print(f"  ✅ Stage 3: Validated and normalized patch.")
        return validated_patch

    def _stage1_serialize(
            self,
            failure_info: Dict[str, Any],
            trace: List[Dict[str, Any]],
            rule_pool
    ) -> Dict[str, Any]:
        operator_name = failure_info.get("operator", "Unknown")
        operator = rule_pool.get_operator(operator_name)
        operator_signature = {
            "name": operator_name,
            "params": operator.params if operator else [],
            "preconditions": rule_pool.list_preconditions(operator_name) if operator else [],
            "effects": ["<effect_functions>"]  # Placeholder for effects
        }
        state_minimal = self._extract_minimal_state(trace, operator_signature)
        state_delta = self._compute_state_delta(failure_info)
        return {
            "operator": operator_signature, "state_minimal": state_minimal,
            "error": {"type": failure_info.get("error"), "site": operator_name,
                      "evidence": failure_info.get("policy_ref") or failure_info.get("message", "")[:50]},
            "state_delta": state_delta
        }

    def _extract_minimal_state(self, trace: List[Dict[str, Any]], op_sig: Dict[str, Any]) -> Dict[str, Any]:
        minimal_state = {}
        # Find the last known state before the failure
        for entry in reversed(trace):
            if "state_before" in entry:
                full_state = entry["state_before"]
                # Include symbols relevant to the operator's signature
                for param in op_sig.get("params", []):
                    for key, value in full_state.items():
                        if param.lower() in key.lower(): minimal_state[key] = value
                # Include globally relevant symbols
                for key in ["payment_method", "travel_dates", "blackout_dates", "corporate_card_policy"]:
                    if key in full_state: minimal_state[key] = full_state[key]
                break
        return minimal_state

    def _compute_state_delta(self, failure_info: Dict[str, Any]) -> Dict[str, List[str]]:
        op_name = failure_info.get("operator", "Unknown")
        error_type = failure_info.get("error", "Unknown")
        expected = []
        if "Hotel" in op_name:
            expected.append("Booked(location, dates)")
        elif "Flight" in op_name:
            expected.append("Booked(origin, destination, date)")
        return {"expected": expected, "observed": [f"Error({error_type})"]}

    def _stage2_generate(self, structured_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        UPDATED: This method now acts as a router.
        It calls the appropriate generator based on the config.
        """
        if self.llm_provider == "transformers":
            return self._real_llm_generate(structured_prompt)
        else:  # Defaults to "mock"
            return self._mock_llm_generate(structured_prompt)

    def _build_llm_prompt(self, structured_prompt: Dict[str, Any]) -> str:
        """Builds the detailed text prompt with error-type hints."""
        error_type = structured_prompt.get("error", {}).get("type", "")

        # Add error-specific guidance to nudge the LLM
        error_guidance = ""
        if error_type == "ToolError":
            error_guidance = "\nNOTE: For a ToolError (like a timeout or API error), the best fix is usually to `REFINE_EFFECT` with a conditional guard, such as `IfThen(NetworkAvailable(), ...)`. Avoid adding unrelated preconditions."
        elif error_type == "PreconditionUnmet":
            error_guidance = "\nNOTE: For a PreconditionUnmet failure (like a policy violation), the best fix is to `ADD_PRECONDITION` with a `Not(...)` predicate to prevent the failure in the future."

        # Provide a concrete example in the schema and clear instructions.
        prompt_template = f"""You are a symbolic patch generator for operator repair.
Given a failure trace, propose a typed patch in the JSON schema below.
For the "action" field, you must choose ONE of the following values: "ADD_PRECONDITION", "REFINE_EFFECT", or "UPDATE_TOOL_SCHEMA".
Use only predicates defined in the operator's signature or the minimal state. Do not invent new symbols.{error_guidance}

[Schema Definition and Example]
{{
  "action": "ADD_PRECONDITION",
  "operator": "ExampleOperatorName",
  "patch": {{
    "predicate": "PredicateName(param1, param2)",
    "guard": null,
    "justification": "A brief, factual reason for the patch based on evidence."
  }}
}}

[Failure Trace]
{json.dumps(structured_prompt, indent=2)}

Output your patch in JSON:
"""
        return prompt_template

    def _real_llm_generate(self, structured_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Generates patch using real LLM with robust JSON extraction."""
        if not self.llm_pipeline:
            print("  ⚠️ LLM Generation: Real LLM not available. Falling back to mock.")
            return self._mock_llm_generate(structured_prompt)

        prompt = self._build_llm_prompt(structured_prompt)

        try:
            print("  🧠 LLM Generation: Calling local model...")
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=250,  # Increase from 150
                do_sample=True,
                temperature=self.temperature,
                return_full_text=False,  # Only get generated text
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            generated_text = outputs[0]['generated_text'].strip()

            # **FIX 1**: Find FIRST complete JSON object (handles trailing text)
            json_start = generated_text.find('{')
            if json_start == -1:
                raise ValueError("No JSON found")

            # Stack-based brace matching for nested objects
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(generated_text[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            json_part = generated_text[json_start:json_end]

            # **FIX 2 (IMPROVED)**: Clean markdown wrappers (only if they bookend the JSON)
            if json_part.startswith('```json'):
                json_part = json_part[7:]
            if json_part.endswith('```'):
                json_part = json_part[:-3]
            json_part = json_part.strip()

            raw_patch = json.loads(json_part)
            print(f"  ✓ LLM Generation: Successfully parsed {len(json_part)} chars")
            return raw_patch

        except Exception as e:
            print(f"  ⚠️ LLM Generation Error: {str(e)[:100]}")
            print(
                f"     Raw output (first 200 chars): {generated_text[:200] if 'generated_text' in locals() else 'N/A'}")
            return self._mock_llm_generate(structured_prompt)

    def _mock_llm_generate(self, structured_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a patch using a data-driven mock for deterministic testing.
        """
        operator = structured_prompt.get("operator", {}).get("name", "")
        evidence = structured_prompt.get("error", {}).get("evidence", "")

        template = self._get_patch_template_for_failure(structured_prompt)
        # Deep copy to avoid modifying the original template
        patch_data = json.loads(json.dumps(template))

        # Personalize the justification if possible
        if evidence and "justification" in patch_data["patch"]:
            patch_data["patch"]["justification"] = evidence[:80]

        return {**patch_data, "operator": operator}

    def _get_patch_template_for_failure(self, structured_prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selects a patch template based on the failure pattern.
        This simulates a constrained, fine-tuned model's pattern recognition.
        """
        error_type = structured_prompt.get("error", {}).get("type", "")
        evidence = structured_prompt.get("error", {}).get("evidence", "").lower()

        # Iterate through specific patterns first
        for key, template in PATCH_TEMPLATES.items():
            if isinstance(key, tuple):
                fail_type, keywords = key
                if fail_type == error_type and any(kw in evidence for kw in keywords):
                    return template

        # Return the default fallback if no specific pattern matches
        return PATCH_TEMPLATES["default"]

    def _stage3_validate(self, raw_patch: Dict[str, Any], rule_pool) -> Dict[str, Any]:
        """
        Performs deterministic validation: schema, typing, and normalization.
        Corresponds to Section 8.2.3 of the paper.
        """
        self._validate_schema(raw_patch)
        self._validate_typing_and_semantics(raw_patch, rule_pool)
        return self._normalize_patch(raw_patch)

    def _validate_schema(self, patch: Dict[str, Any]) -> None:
        """Ensures the patch conforms to the required JSON schema."""
        if not isinstance(patch, dict) or not all(k in patch for k in ["action", "operator", "patch"]):
            raise ValueError(f"Invalid patch structure or missing required fields: {patch}")
        if patch["action"] not in self.ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{patch['action']}'")
        print("  ✓ Schema validation passed")

    def _validate_typing_and_semantics(self, patch: Dict[str, Any], rule_pool) -> None:
        """
        Ensures predicates are known and variables are well-typed.
        A full implementation would consult a type lattice and operator signatures.
        """
        details = patch["patch"].get("predicate") or patch["patch"].get("guard", "")
        if not details:  # Some patches might not have these details, which is valid
            print("  ✓ Type and semantic checks passed (no details to check).")
            return

        # For PoC, we do a simple check if any known predicate is mentioned.
        found_known_predicate = any(p in details for p in self.KNOWN_PREDICATES)
        if not found_known_predicate:
            raise ValueError(f"Patch contains unknown predicates or symbols: {details}")
        print("  ✓ Type and semantic checks passed")

    def _normalize_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the raw patch into a standardized internal format."""
        normalized = {
            "action": patch["action"],
            "operator": patch["operator"],
            "details": patch["patch"].get("predicate") or patch["patch"].get("guard", ""),
            "justification": patch["patch"].get("justification", "Generated by FDKA")
        }
        content = f"{normalized['operator']}:{normalized['action']}:{normalized['details']}"
        normalized["content_hash"] = hashlib.sha256(content.encode()).hexdigest()[:12]
        return normalized