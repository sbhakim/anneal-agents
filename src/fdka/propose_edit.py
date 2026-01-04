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


# ---------------------------
# Action normalization helpers
# ---------------------------
ALLOWED_ACTIONS = {"ADD_PRECONDITION", "REFINE_EFFECT", "UPDATE_TOOL_SCHEMA"}
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
            # Fail-soft: never crash the evaluation loop due to a malformed patch
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
        return {
            "operator": sig, "state_minimal": minimal,
            "error": {"type": fi.get("error"), "site": fi.get("operator"),
                      "evidence": (fi.get("policy_ref", "") + (fi.get("message", "") or "")[:50])},
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
            # mock path expects the same structure as serialized_prompt
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

        for attempt in range(2):  # Try up to 2 times
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
                continue

            raw_text = result['text']
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

            # retry prompt to fix JSON
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

            # time the call for latency
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

            # estimate tokens (best-effort) using tokenizer
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
        # try fenced ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # try any single top-level {...}
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            # attempt bracket balancing
            bal = 0
            last = start
            for i, ch in enumerate(text[start:end + 1], start=start):
                if ch == '{':
                    bal += 1
                elif ch == '}':
                    bal -= 1
                    if bal == 0:
                        last = i
                        break
            try:
                return json.loads(text[start:last + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _build_llm_prompt(self, sp: Dict) -> Dict:
        """
        Builds the prompt for the LLM, now explicitly asking for the 'operator' key.
        Also returns the serialized_prompt for mock fallback.
        """
        error_type = sp.get("error", {}).get("type", "Unknown")
        error_evidence = sp.get("error", {}).get("evidence", "")
        operator_name = sp.get("operator", {}).get("name", "Unknown")

        system_prompt = (
            "You are an expert system that outputs exactly one valid JSON object for a symbolic patch.\n"
            "Do not add explanations or markdown. Output ONLY the JSON object."
        )

        user_prompt = f"""
A failure occurred. Generate a JSON patch to fix it.

FAILURE CONTEXT:
- Operator: {operator_name}
- Error Type: {error_type}
- Evidence: {error_evidence}

RULES:
- Keys required: "action", "operator", "patch".
- "action" must be one of: {sorted(list(ALLOWED_ACTIONS))}
- "operator" must equal "{operator_name}"
- For invalid/expired payment: {{"action": "ADD_PRECONDITION", "operator": "{operator_name}", "patch": {{"predicate": "ValidPayment(payment)"}}}}
- For blocked card/blackout: {{"action": "ADD_PRECONDITION", "operator": "{operator_name}", "patch": {{"predicate": "Not(BlockedCard(payment, dates))"}}}}
- For network/timeout: {{"action": "REFINE_EFFECT", "operator": "{operator_name}", "patch": {{"guard": "IfThen(NetworkAvailable(), ExecuteTool())"}}}}

Return only the JSON object, no prose."""
        return {
            "system": system_prompt,
            "user": user_prompt,
            "serialized_prompt": sp,
            "prompt_hash": _hash_text(system_prompt + "\n\n" + user_prompt),
        }

    def _mock_llm_generate(self, sp):
        error = sp.get("error", {}).get("type", "")
        ev = (sp.get("error", {}).get("evidence", "") or "").lower()
        operator_name = sp.get("operator", {}).get("name", "Unknown")
        prompt_hash = _hash_text(sp)

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
            # Defensive: ensure dict
            if not isinstance(raw, dict):
                raw = {"action": "REJECTED", "details": "non_dict_patch"}
                return raw

            # Normalize action before validation
            act = _normalize_action(raw.get("action", ""))
            raw["action"] = act

            # If operator missing, attempt to infer from serialized prompt (no-op here)
            if not raw.get("operator"):
                raw["operator"] = (rp and getattr(rp, "last_operator", None)) or "Unknown"

            # Schema checks
            self._validate_schema(raw)

            # Typing/semantic hints (non-fatal)
            self._validate_typing_and_semantics(raw, rp)

            # Normalize final patch (preserve optional usage + llm_trace)
            return self._normalize_patch(raw)
        except Exception as e:
            # Fail-soft: preserve audit metadata if present
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
        # basic patch content presence
        if not isinstance(p["patch"], dict):
            raise ValueError("Patch 'patch' must be a JSON object")
        if not any(k in p["patch"] for k in ("predicate", "guard")):
            raise ValueError("Patch must include either 'predicate' or 'guard'")
        print("  ✓ Schema OK")

    def _validate_typing_and_semantics(self, p, rp):
        d = p["patch"].get("predicate") or p["patch"].get("guard", "")
        if d and not any(pred in d for pred in self.KNOWN_PREDICATES):
            print(f"  ⚠️ Unknown predicate in '{d}'. This may cause issues.")
        print("  ✓ Type OK")

    def _normalize_patch(self, p):
        """Normalize patch to a standard internal format for downstream scoring/governance."""
        norm = {
            "action": p["action"],
            "operator": p["operator"],
            "details": p["patch"].get("predicate") or p["patch"].get("guard", ""),
            "justification": p["patch"].get("justification", "Generated by FDKA")
        }
        content_str = f"{norm['operator']}:{norm['action']}:{norm['details']}"
        norm["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        # preserve optional usage payload for external logging/CSV
        if isinstance(p, dict) and "usage" in p:
            norm["usage"] = p["usage"]

        # preserve prompt/response hashes (privacy-safe provenance)
        if isinstance(p, dict) and "llm_trace" in p and isinstance(p["llm_trace"], dict):
            norm["llm_trace"] = p["llm_trace"]

        return norm
