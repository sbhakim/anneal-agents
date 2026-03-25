# src/fdka/propose_edit.py
"""
Implements the 3-stage ProposeEdit pipeline from Section VIII-B of the paper.

UPDATED:
- Integrated fixes for local Transformers models (device mapping, dtype, tokenizer pad id).
- Robust prompt & JSON parsing for all providers (OpenAI/DeepSeek/Transformers/Mock).
- Action token normalization + aliasing (e.g., "ADD_PRE CONDITION" → "ADD_PRECONDITION").
- Fail-soft behavior: schema/type errors return {action:"REJECTED", details:...} instead of raising.
- Deterministic validation after normalization; helpful warnings on unknown predicates.
- Timeouts for LLM calls; graceful fallbacks to mock generation.
- NEW: Return and preserve a top-level `usage` dict (tokens, latency, model).

MINIMUM MANUSCRIPT/RESULTS UPDATE (2026-01):
- Add privacy-safe provenance hooks: hash of (system+user prompt) + hash of raw model text.
  This makes runs auditable/reproducible without logging the raw prompt/response.

MINIMUM TOOL-DRIFT UPDATE (2026-01):
- Allow UPDATE_TOOL_SCHEMA patches without predicate/guard.
- Deterministic router for deprecation/version drift signals (v1→v2) so canary/commit can trigger.
"""

from typing import List, Dict, Any, Optional
import uuid
import json
import hashlib
import signal
import functools
import re
import os
import time

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
    # Keep TorchDynamo quiet on odd graphs
    import torch._dynamo  # type: ignore
    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _hash_text(x: Any) -> str:
    """Stable sha256 hash for audit trails (no raw prompt logging)."""
    try:
        if isinstance(x, (dict, list, tuple)):
            s = json.dumps(x, sort_keys=True, default=str)
        else:
            s = str(x)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(repr(x).encode("utf-8")).hexdigest()


def _is_tool_drift(evidence: str) -> bool:
    ev = (evidence or "").lower()
    drift_markers = (
        "deprecated",
        "deprecation",
        "/v1/",
        "v1/book",
        "use /v2",
        "use v2",
        "moved to",
        "endpoint changed",
        "no longer supported",
        "unsupported endpoint",
    )
    return any(m in ev for m in drift_markers)


def _build_tool_drift_patch(operator_name: str, evidence: str) -> Dict[str, Any]:
    ev = (evidence or "").lower()
    auth_markers = (
        "auth_token",
        "bearer",
        "signed_session_token",
        "session token",
        "authentication schema",
        "oauth",
        "jwt",
    )
    # Keep details as a string because RulePool stores schema updates as metadata text.
    if any(m in ev for m in auth_markers):
        details = (
            "Replace legacy auth_token bearer flow with signed_session_token request field "
            "(authentication schema drift)."
        )
    elif "/v1/" in ev or "v1/book" in ev:
        details = "Switch tool endpoint from /v1/book to /v2/book (deprecation/tool-drift)."
    else:
        details = "Update tool schema/endpoint to the latest supported version (tool-drift)."
    return {
        "action": "UPDATE_TOOL_SCHEMA",
        "operator": operator_name,
        "patch": {
            "schema_update": details,
            "justification": f"Observed tool/endpoint drift: {evidence[:120]}",
        },
    }


def _is_non_patchable_refund_window(error_type: str, evidence: str, violated: str) -> bool:
    if error_type != "PreconditionUnmet":
        return False
    text = f"{evidence or ''} {violated or ''}".lower()
    window_markers = (
        "withinrefundwindow",
        "is_refund_within_window",
        "returnwindowexceeded",
        "refund window",
        "return window",
        "within refund window",
    )
    if any(m in text for m in window_markers):
        return True
    if ("refund" in text or "return" in text) and ("window" in text or "within" in text):
        return True
    return False


# ---------------------------
# Templates for mock fallback
# ---------------------------
PATCH_TEMPLATES = {
    ("PreconditionUnmet", ("blackout", "policy", "blocked", "h-23")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(BlockedCard(payment, dates))",
            "justification": "API policy H-23 blocks corporate cards during blackout dates"
        }
    },
    ("PreconditionUnmet", ("return window", "refund", "returnwindowexceeded", "withinrefundwindow", "is_refund_within_window")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "WithinRefundWindow(days_since_purchase)",
            "justification": "Refunds must be within the allowed return window"
        }
    },
    ("PreconditionUnmet", ("shipping", "apo", "fpo", "po_box", "restricted", "shippingallowed", "is_shipping_allowed")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(RestrictedAddress(location))",
            "justification": "Shipping is restricted for certain address types"
        }
    },
    ("PreconditionUnmet", ("price changed", "price_changed", "pricechanged", "cart", "no_price_change")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "Not(PriceChanged(order))",
            "justification": "Order price must match current catalog pricing"
        }
    },
    ("PreconditionUnmet", ("invalid", "payment", "unsupported", "expired", "declined")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "ValidPayment(payment)",
            "justification": "Payment method validation to prevent API rejection"
        }
    },
    ("ToolError", ("deprecated", "/v1", "v1/book", "use /v2", "use v2")): {
        "action": "UPDATE_TOOL_SCHEMA",
        "patch": {
            "schema_update": "Switch tool endpoint from /v1/book to /v2/book (deprecation/tool-drift).",
            "justification": "Fix endpoint deprecation by migrating to v2."
        }
    },
    ("ToolError", ("timeout", "unavailable", "network")): {
        "action": "REFINE_EFFECT",
        "patch": {
            "guard": "ApiTimeoutRetry(service)",
            "justification": "Add timeout retry recovery to handle transient API failures."
        }
    },
    ("ToolError", ("payment validation", "invalid", "unsupported payment")): {
        "action": "ADD_PRECONDITION",
        "patch": {
            "predicate": "ValidPayment(payment)",
            "justification": "Validate payment method before API call to prevent rejection."
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


# ---------------------------
# Action normalization helpers
# ---------------------------
ALLOWED_ACTIONS = {"ADD_PRECONDITION", "REFINE_EFFECT", "UPDATE_TOOL_SCHEMA", "SKIP_CONSTRAINT"}
ACTION_ALIASES = {
    "ADD_PRE CONDITION": "ADD_PRECONDITION",
    "ADD_PRE-CONDITION": "ADD_PRECONDITION",
    "ADD PRECONDITION": "ADD_PRECONDITION",
    "ADD_PRECONDITIONS": "ADD_PRECONDITION",
    "REFINE EFFECT": "REFINE_EFFECT",
    "UPDATE_TOOL-SCHEMA": "UPDATE_TOOL_SCHEMA",
    "UPDATE TOOL SCHEMA": "UPDATE_TOOL_SCHEMA",
}


def _normalize_action(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # fast alias on raw
    if s in ACTION_ALIASES:
        return ACTION_ALIASES[s]
    t = s.replace("-", "_").replace(" ", "_").upper()
    # collapse multiple underscores
    while "__" in t:
        t = t.replace("__", "_")
    # alias after normalization
    if t in ACTION_ALIASES:
        return ACTION_ALIASES[t]
    return t


# ---------------------------
# Timeout decorator
# ---------------------------
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
        """
        self.root_config = config or {}
        self.config = self.root_config.get("propose_edit", self.root_config) or {}
        self.metrics_collector = metrics_collector
        self.llm_provider_type = self.config.get("llm_provider", "mock")
        self.llm = None
        print(f"FDKA: Initializing with provider='{self.llm_provider_type}'")

        if self.llm_provider_type == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI provider selected but not available. Run: pip install openai>=1.0.0")
            self.llm = OpenAIProvider(self.config)
            print(f"FDKA: ✅ OpenAI provider initialized (model={self.config.get('model', 'gpt-4o-mini')})")

        elif self.llm_provider_type == 'deepseek':
            if not DEEPSEEK_AVAILABLE:
                raise ImportError("DeepSeek provider not available. Ensure deepseek_provider.py exists.")
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
        self.KNOWN_PREDICATES = {
            "is_card_valid", "is_hotel_available", "is_flight_available",
            "check_not_blocked_card", "Not", "BlockedCard", "IfThen",
            "NetworkAvailable", "Sent", "ValidState", "ValidPayment", "ExecuteTool"
        }

    # ---------------------------
    # Local Transformers bootstrap
    # ---------------------------
    def _initialize_local_llm(self):
        """Initializes a local Hugging Face transformers pipeline with config hints."""
        transformers_cfg = self.config.get('transformers_config', self.config) or {}
        self.local_model_id = transformers_cfg.get('model') or self.config.get('model')
        if not self.local_model_id:
            raise ValueError(f"Config error: 'model' not specified for transformers provider.")

        try:
            print(f"FDKA: 🚧 Loading local LLM: {self.local_model_id}...")

            # Device selection: explicit device_map or auto CUDA/CPU
            device_map = transformers_cfg.get("device_map", self.config.get("device_map", "auto"))
            dtype_pref = (transformers_cfg.get("dtype") or self.config.get("dtype") or "auto")
            trust_remote = bool(transformers_cfg.get("trust_remote_code", self.config.get("trust_remote_code", True)))

            # Resolve torch dtype
            if dtype_pref == "auto":
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            elif str(dtype_pref).lower() in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif str(dtype_pref).lower() in ("fp16", "float16", "half"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # Pipeline device: -1 for CPU, 0 for first GPU
            device = 0 if torch.cuda.is_available() else -1
            print(f"FDKA: ⚙️  Setting device for local model pipeline to: {'cuda:0' if device == 0 else 'cpu'}")

            # Tokenizer & model (explicit so we can set pad token safely)
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_id, trust_remote_code=trust_remote)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            # Let HF pick the best loader strategy; pipeline handles weights loading.
            self.llm_pipeline = pipeline(
                task="text-generation",
                model=self.local_model_id,
                tokenizer=tokenizer,
                torch_dtype=torch_dtype,
                device=device
            )
            print("FDKA: ✅ Local LLM loaded.")
        except Exception as e:
            print(f"FDKA: ❌ Failed to load local LLM: {e}")
            self.llm_pipeline = None

    # ---------------------------
    # Public entrypoint
    # ---------------------------
    def propose_edit(self, trace: List[Dict[str, Any]], rule_pool) -> Dict[str, Any]:
        print("\n--- FDKA PROPOSAL INITIATED ---")
        try:
            fi = self._extract_failure_info(trace)
            op_name = self._localize(fi)
            print(f"FDKA: [1/3] Fault localized -> '{op_name}'")

            patch = self._propose_edit_three_stage(fi, trace, rule_pool)
            patch.setdefault("operator", op_name)
            patch.setdefault("id", f"patch-{uuid.uuid4().hex[:8]}")
            patch.setdefault("source", "fdka.v1.3")

            # Always present: used by metrics + provenance.
            patch.setdefault("usage", {"tokens_used": 0, "latency_sec": 0.0, "model": self.llm_provider_type})
            patch.setdefault(
                "llm_trace",
                {
                    "provider": self.llm_provider_type,
                    "model": patch.get("usage", {}).get("model", self.llm_provider_type),
                    "prompt_hash": "",
                    "response_hash": "",
                },
            )

            print(
                f"FDKA: [3/3] Patch proposed -> {patch.get('action', 'N/A')}: {patch.get('details', 'N/A')} (id={patch.get('id')})")
            print("--- FDKA PROPOSAL COMPLETED ---")
            return patch
        except Exception as e:
            msg = f"propose_edit_error: {e.__class__.__name__}: {e}"
            print(f"FDKA: ❌ {msg}")
            return {
                "action": "REJECTED",
                "details": msg,
                "id": f"patch-{uuid.uuid4().hex[:8]}",
                "source": "fdka.v1.3",
                "usage": {"tokens_used": 0, "latency_sec": 0.0, "model": self.llm_provider_type},
                "llm_trace": {
                    "provider": self.llm_provider_type,
                    "model": self.llm_provider_type,
                    "prompt_hash": "",
                    "response_hash": "",
                },
            }

    # ---------------------------
    # Stage 1: Serialize context
    # ---------------------------
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

        # Deterministic router for common tool-drift (prevents "no provable improvement" dead-ends).
        evidence = (prompt.get("error", {}).get("evidence", "") or "")
        violated = (prompt.get("error", {}).get("violated", "") or "")
        op_name = (prompt.get("operator", {}) or {}).get("name", fi.get("operator", "Unknown"))
        if _is_non_patchable_refund_window(fi.get("error", ""), evidence, violated):
            return {"action": "REJECTED", "details": "non_patchable_refund_window"}
        if fi.get("error") == "ToolError" and _is_tool_drift(evidence):
            return self._stage3_validate(_build_tool_drift_patch(op_name, evidence), rule_pool)

        raw_patch_json = self._stage2_generate(prompt)
        if isinstance(raw_patch_json, dict) and raw_patch_json.get("action") == "REJECTED":
            return raw_patch_json
        action_seen = (raw_patch_json or {}).get("action", "N/A")
        print(f"  🤖 Stage 2: LLM generated -> {action_seen}")
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
        violated = fi.get("violated") or fi.get("violated_predicate") or ""
        ev = ((fi.get("policy_ref") or "") + " " + (fi.get("message", "") or "") + " " + str(violated)).strip()
        return {
            "operator": sig, "state_minimal": minimal,
            "error": {"type": fi.get("error"), "site": fi.get("operator"),
                      "evidence": ev[:200], "violated": violated},
            "state_delta": delta
        }

    def _extract_minimal_state(self, trace, sig):
        ms = {}
        for e in reversed(trace):
            if "state_before" in e:
                fs = e["state_before"]
                for p in sig.get("params", []):
                    for k, v in fs.items():
                        if isinstance(p, str) and p.lower() in str(k).lower():
                            ms[k] = v
                for k in ["payment_method", "travel_dates", "blackout_dates", "corporate_card_policy"]:
                    if k in fs:
                        ms[k] = fs[k]
                break
        return ms

    def _compute_state_delta(self, fi):
        name, et = fi.get("operator", "Unknown"), fi.get("error", "Unknown")
        exp = []
        if "Hotel" in name:
            exp.append("Booked(location, dates)")
        if "Flight" in name:
            exp.append("Booked(origin, destination, date)")
        return {"expected": exp, "observed": [f"Error({et})"]}

    # ---------------------------
    # Stage 2: Generate candidate
    # ---------------------------
    def _stage2_generate(self, serialized_prompt):
        prompt_dict = self._build_llm_prompt(serialized_prompt)
        if self.llm_provider_type in ['openai', 'deepseek']:
            return self._generate_with_api(prompt_dict)
        elif self.llm_provider_type == 'transformers':
            return self._generate_with_transformers(prompt_dict)
        else:
            return self._mock_llm_generate(serialized_prompt)

    @timeout(seconds=60)
    def _generate_with_api(self, prompt_dict: Dict) -> Dict:
        if not self.llm:
            print("  ⚠️ API provider not initialized. Falling back to mock.")
            return self._mock_llm_generate(prompt_dict.get("serialized_prompt", {}))

        print(f"  🧠 Calling {self.llm_provider_type} API...")
        messages = [
            {"role": "system", "content": prompt_dict["system"]},
            {"role": "user", "content": prompt_dict["user"]}
        ]

        prompt_hash = prompt_dict.get("prompt_hash") or _hash_text([m.get("role") + ":" + m.get("content", "") for m in messages])

        for attempt in range(2):
            t0 = time.time()
            result = self.llm.generate(messages=messages)
            latency = max(0.0, time.time() - t0)
            usage = {
                "tokens_used": result.get('tokens_used', 0),
                "latency_sec": result.get('latency_sec', latency),
                "model": result.get('model', self.llm_provider_type)
            }

            if self.metrics_collector:
                self.metrics_collector.record_llm_call(
                    tokens=usage["tokens_used"],
                    latency_sec=usage["latency_sec"],
                    model=usage["model"]
                )

            if 'error' in result or not result.get('text'):
                print(f"  ⚠️ OpenAI error or no text: error={result.get('error')}, text_len={len(result.get('text', ''))}")
                continue

            raw_text = result['text']
            print(f"  ✅ OpenAI returned {len(raw_text)} chars: {raw_text[:100]}...")
            response_hash = _hash_text(raw_text)

            parsed = self._extract_json_from_text(raw_text)
            if parsed is not None:
                parsed["usage"] = usage
                parsed["llm_trace"] = {
                    "provider": self.llm_provider_type,
                    "model": usage.get("model", self.llm_provider_type),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                }
                return parsed

            print(f"  ⚠️ Attempt {attempt + 1}: Could not parse JSON. Retrying with correction prompt...")
            messages = [
                {"role": "system",
                 "content": "You are a JSON generator. Respond ONLY with the corrected JSON object (no markdown)."},
                {"role": "user",
                 "content": f"Fix the malformed JSON below and output ONLY valid JSON (no prose, no fences):\n\n{raw_text}"}
            ]
            time.sleep(1)

        print(f"  ❌ All attempts to parse JSON failed. Falling back to mock.")
        return self._mock_llm_generate(prompt_dict.get("serialized_prompt", {}))

    @timeout(seconds=120)
    def _generate_with_transformers(self, prompt_dict: Dict) -> Dict:
        if not self.llm_pipeline:
            print("  ⚠️ No local LLM; falling back to mock.")
            return self._mock_llm_generate(prompt_dict.get("serialized_prompt", {}))
        try:
            print("  🧠 Calling local model...")
            full_prompt = f"{prompt_dict['system']}\n\n{prompt_dict['user']}"
            prompt_hash = prompt_dict.get("prompt_hash") or _hash_text(full_prompt)

            cfg = self.config.get('transformers_config', self.config) or {}

            t0 = time.time()
            out = self.llm_pipeline(
                full_prompt,
                max_new_tokens=int(cfg.get('max_tokens', 300)),
                do_sample=True,
                temperature=float(cfg.get('temperature', 0.3)),
                return_full_text=False,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id,
                top_p=float(cfg.get('top_p', 0.95)),
                repetition_penalty=float(cfg.get('repetition_penalty', 1.1)),
            )
            latency = max(0.0, time.time() - t0)
            txt = (out[0]["generated_text"] if out and isinstance(out, list) and "generated_text" in out[0] else "").strip()
            response_hash = _hash_text(txt)

            parsed = self._extract_json_from_text(txt)

            try:
                tokens_used = len(self.llm_pipeline.tokenizer.encode(full_prompt)) + \
                              len(self.llm_pipeline.tokenizer.encode(txt))
            except Exception:
                tokens_used = 0

            usage = {
                "tokens_used": tokens_used,
                "latency_sec": latency,
                "model": getattr(self, "local_model_id", "transformers")
            }

            if self.metrics_collector:
                self.metrics_collector.record_llm_call(
                    tokens=usage["tokens_used"],
                    latency_sec=usage["latency_sec"],
                    model=usage["model"]
                )

            if parsed is not None:
                parsed["usage"] = usage
                parsed["llm_trace"] = {
                    "provider": "transformers",
                    "model": usage.get("model", "transformers"),
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                }
                return parsed
            raise ValueError("No valid JSON found in local LLM output")
        except (TimeoutException, json.JSONDecodeError, Exception) as e:
            print(f"  ⚠️ Local LLM generation error: {e}. Falling back.")
            return self._mock_llm_generate(prompt_dict.get("serialized_prompt", {}))

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a single JSON object from free text or fenced blocks."""
        if not text:
            return None
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            bal = 0
            last = None
            for i, ch in enumerate(text[start:end + 1], start=start):
                if ch == '{':
                    bal += 1
                elif ch == '}':
                    bal -= 1
                    if bal == 0:
                        last = i
                        # Continue to find the LAST balanced closing brace.
            if last is None:
                last = end
            try:
                return json.loads(text[start:last + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _build_llm_prompt(self, sp: Dict) -> Dict:
        """
        Builds the prompt for the LLM, now explicitly asking for the 'operator' key.
        Enhanced with constraint awareness and better examples to reduce canary failures.
        Also returns the serialized_prompt for mock fallback.
        """
        error_type = sp.get("error", {}).get("type", "Unknown")
        error_evidence = sp.get("error", {}).get("evidence", "")
        operator_name = sp.get("operator", {}).get("name", "Unknown")

        # Extract full error trace for context
        error_msg = sp.get("error", {}).get("message", "")
        violated_pred = sp.get("error", {}).get("violated", "")
        full_context = f"{error_evidence} {error_msg} {violated_pred}".strip()

        system_prompt = (
            "You are an expert system that generates symbolic patches for operator failures.\n"
            "Your patches will be tested via canary simulation before deployment.\n"
            "Output EXACTLY ONE valid JSON object. No explanations, no markdown, no text outside the JSON."
        )

        # Build constraint warning based on evidence
        # Strip operator name from evidence before checking — operator names like
        # "ApplyPromoCode" contain substrings ("promo") that falsely trigger the
        # constraint warning on patchable ToolError failures.
        constraint_warning = ""
        evidence_for_constraint_check = full_context.lower()
        for name_fragment in [operator_name.lower(), operator_name.replace(" ", "").lower()]:
            evidence_for_constraint_check = evidence_for_constraint_check.replace(name_fragment, "")
        if any(pattern in evidence_for_constraint_check for pattern in [
            'refund', 'return_window', 'purchased', 'days ago',
            'apo', 'fpo', 'po_box', 'shipping_restriction',
            'out of stock', 'inventory', 'insufficient',
            'promo', 'expired', 'policy', 'violation'
        ]):
            constraint_warning = """
⚠️ WARNING: Evidence suggests this may be a BUSINESS CONSTRAINT, not a patchable bug.
Business constraints (refund windows, shipping restrictions, inventory limits, policy violations)
should NOT be patched. If you detect a constraint, return:
{{"action": "SKIP_CONSTRAINT", "operator": "{operator}", "patch": {{"reason": "business_constraint"}}}}
"""

        user_prompt = f"""
A failure occurred. Generate a JSON patch to fix it, or identify if it's an unpatchable constraint.

FAILURE CONTEXT:
- Operator: {operator_name}
- Error Type: {error_type}
- Full Evidence: {full_context[:500]}
{constraint_warning}

DOMAIN AXIOMS (Must preserve):
- ExpiredPayment -> InvalidPayment
- InvalidPayment -> Not(ValidPayment)
- BlockedCard -> Not(ValidPayment)
- Not(NetworkAvailable) -> ToolError

PATCH STRUCTURE:
Required keys: "action", "operator", "patch"
- "action" must be one of: {sorted(list(ALLOWED_ACTIONS))}
- "operator" must equal "{operator_name}"

COMMON PATTERNS (Choose the best match):

1. INVALID/EXPIRED PAYMENT (PreconditionUnmet with payment error):
   {{"action": "ADD_PRECONDITION", "operator": "{operator_name}", "patch": {{"predicate": "ValidPayment(payment)"}}}}

2. BLOCKED CARD/BLACKOUT DATE:
   {{"action": "ADD_PRECONDITION", "operator": "{operator_name}", "patch": {{"predicate": "Not(BlockedCard(payment, dates))"}}}}

3. NETWORK/TIMEOUT/SERVICE UNAVAILABLE:
   {{"action": "REFINE_EFFECT", "operator": "{operator_name}", "patch": {{"guard": "ApiTimeoutRetry(service)"}}}}

4. TOOL DRIFT/DEPRECATION (e.g., API version change /v1 → /v2):
   {{"action": "UPDATE_TOOL_SCHEMA", "operator": "{operator_name}", "patch": {{"schema_update": "Switch endpoint to new version"}}}}

5. AUTH TOKEN / SESSION TOKEN SCHEMA DRIFT:
   {{"action": "UPDATE_TOOL_SCHEMA", "operator": "{operator_name}", "patch": {{"schema_update": "Replace legacy auth_token with signed_session_token"}}}}

6. MISSING REQUIRED FIELD:
   {{"action": "ADD_PRECONDITION", "operator": "{operator_name}", "patch": {{"predicate": "HasRequiredField(data, 'field_name')"}}}}

CRITICAL RULES:
- DO NOT create patches that change business logic (pricing, inventory, policies)
- DO NOT add preconditions that will always fail
- DO NOT modify operator effects unless you're fixing a bug
- If evidence mentions constraints (refund policy, shipping restrictions), return SKIP_CONSTRAINT
- Ensure your patch actually addresses the root cause in the evidence

EXAMPLES OF GOOD PATCHES:
✓ Adding ValidPayment check when payment is expired/invalid
✓ Updating schema when API endpoint changed (tool drift)
✓ Adding network availability guard for timeout errors

EXAMPLES OF BAD PATCHES (Will fail canary):
✗ Adding inventory checks (that's a business constraint, not a bug)
✗ Changing refund window logic (immutable business policy)
✗ Generic patches that don't address the specific error
✗ Patches that make the operator too restrictive

Analyze the failure evidence carefully and generate the MOST APPROPRIATE patch.
Return ONLY the JSON object."""

        return {
            "system": system_prompt,
            "user": user_prompt,
            "serialized_prompt": sp,
            "prompt_hash": _hash_text(system_prompt + "\n\n" + user_prompt),
        }

    def _mock_llm_generate(self, sp):
        error = sp.get("error", {}).get("type", "")
        violated = sp.get("error", {}).get("violated", "")
        ev_parts = [
            sp.get("error", {}).get("evidence", "") or "",
            error or "",
            violated or "",
        ]
        ev = " ".join(ev_parts).lower()
        operator_name = sp.get("operator", {}).get("name", "Unknown")
        prompt_hash = _hash_text(sp)

        if error == "ToolError" and _is_tool_drift(ev):
            out = _build_tool_drift_patch(operator_name, ev)
            out["usage"] = {"tokens_used": 0, "latency_sec": 0.0, "model": "mock"}
            out["llm_trace"] = {
                "provider": "mock",
                "model": "mock",
                "prompt_hash": prompt_hash,
                "response_hash": _hash_text(out),
            }
            return out

        print(f"  🎯 Mock template matching: error={error}, evidence='{ev[:50]}...'")
        for k, t in PATCH_TEMPLATES.items():
            if isinstance(k, tuple):
                ft, kws = k
                if ft == error and any(kw in ev for kw in kws):
                    out = {**t, "operator": operator_name}
                    out["usage"] = {"tokens_used": 0, "latency_sec": 0.0, "model": "mock"}
                    out["llm_trace"] = {
                        "provider": "mock",
                        "model": "mock",
                        "prompt_hash": prompt_hash,
                        "response_hash": _hash_text(out),
                    }
                    return out

        out = {**PATCH_TEMPLATES["default"], "operator": operator_name}
        out["usage"] = {"tokens_used": 0, "latency_sec": 0.0, "model": "mock"}
        out["llm_trace"] = {
            "provider": "mock",
            "model": "mock",
            "prompt_hash": prompt_hash,
            "response_hash": _hash_text(out),
        }
        return out

    # ---------------------------
    # Stage 3: Validate + Normalize
    # ---------------------------
    def _stage3_validate(self, raw, rp):
        """Stage 3: Deterministic validation + normalization."""
        try:
            if not isinstance(raw, dict):
                raw = {"action": "REJECTED", "details": "non_dict_patch"}
                return raw

            act = _normalize_action(raw.get("action", ""))
            raw["action"] = act

            if not raw.get("operator"):
                raw["operator"] = (rp and getattr(rp, "last_operator", None)) or "Unknown"

            self._validate_schema(raw)
            self._validate_typing_and_semantics(raw, rp)
            return self._normalize_patch(raw)
        except Exception as e:
            return {
                "action": "REJECTED",
                "details": f"validation_error: {e}",
                "usage": raw.get("usage") if isinstance(raw, dict) else None,
                "llm_trace": raw.get("llm_trace") if isinstance(raw, dict) else None,
            }

    def _validate_schema(self, p):
        if not isinstance(p, dict) or not all(k in p for k in ["action", "operator", "patch"]):
            raise ValueError(f"Invalid patch structure: {p}")
        if p["action"] not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{p['action']}', must be one of {sorted(ALLOWED_ACTIONS)}")
        if not isinstance(p["patch"], dict):
            raise ValueError("Patch 'patch' must be a JSON object")

        if p["action"] == "SKIP_CONSTRAINT":
            # SKIP_CONSTRAINT is a meta-action indicating LLM detected a business constraint
            if "reason" not in p["patch"]:
                raise ValueError("SKIP_CONSTRAINT requires 'reason' in patch")
        elif p["action"] == "UPDATE_TOOL_SCHEMA":
            if not any(k in p["patch"] for k in ("schema_update", "schema", "tool_schema", "endpoint", "update")):
                raise ValueError("UPDATE_TOOL_SCHEMA requires 'schema_update' (or equivalent) in patch")
        else:
            if not any(k in p["patch"] for k in ("predicate", "guard")):
                raise ValueError("Patch must include either 'predicate' or 'guard'")

        print("  ✓ Schema OK")

    def _validate_typing_and_semantics(self, p, rp):
        act = p.get("action", "")
        if act == "UPDATE_TOOL_SCHEMA":
            d = (
                p["patch"].get("schema_update")
                or p["patch"].get("schema")
                or p["patch"].get("tool_schema")
                or p["patch"].get("endpoint")
                or ""
            )
        else:
            d = p["patch"].get("predicate") or p["patch"].get("guard", "")

        if d and act != "UPDATE_TOOL_SCHEMA" and not any(pred in d for pred in self.KNOWN_PREDICATES):
            print(f"  ⚠️ Unknown predicate in '{d}'. This may cause issues.")
        print("  ✓ Type OK")

    def _normalize_patch(self, p):
        """Normalize patch to a standard internal format for downstream scoring/governance."""
        act = p["action"]

        # Special handling for SKIP_CONSTRAINT (LLM detected business constraint)
        if act == "SKIP_CONSTRAINT":
            print(f"  🔍 LLM detected business constraint: {p['patch'].get('reason', 'unknown')}")
            return {
                "action": "SKIP_CONSTRAINT",
                "operator": p["operator"],
                "details": p["patch"].get("reason", "llm_detected_constraint"),
                "justification": "LLM identified this as a business constraint, not a patchable bug",
                "content_hash": "skip_constraint"
            }

        if act == "UPDATE_TOOL_SCHEMA":
            details = (
                p["patch"].get("schema_update")
                or p["patch"].get("schema")
                or p["patch"].get("tool_schema")
                or p["patch"].get("endpoint")
                or ""
            )
        else:
            details = p["patch"].get("predicate") or p["patch"].get("guard", "")

        norm = {
            "action": act,
            "operator": p["operator"],
            "details": details,
            "justification": p["patch"].get("justification", "Generated by FDKA")
        }
        content_str = f"{norm['operator']}:{norm['action']}:{norm['details']}"
        norm["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        if isinstance(p, dict) and "usage" in p:
            norm["usage"] = p["usage"]

        if isinstance(p, dict) and "llm_trace" in p and isinstance(p["llm_trace"], dict):
            norm["llm_trace"] = p["llm_trace"]

        return norm
