# src/fdka/llm_providers/openai_provider.py

import os, time, logging, random
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
from openai import OpenAI
from openai import APIError, RateLimitError, BadRequestError, APITimeoutError, AuthenticationError

logger = logging.getLogger(__name__)


class GenResult(TypedDict, total=False):
    text: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    model: str
    finish_reason: str
    request_id: str
    error: str
    error_type: str
    error_code: Union[int, str, None]


@dataclass
class OpenAIProvider:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout_sec: float = 60.0
    base_url: Optional[str] = None
    max_retries: int = 4
    _client: OpenAI = field(init=False, repr=False)

    def __init__(self, config: Dict[str, Any]):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if api_key.startswith("sk-proj-...") or api_key == "sk-proj-...":
            raise ValueError("Please replace 'sk-proj-...' with your actual OpenAI API key")

        self.model = config.get("model", self.model)
        self.temperature = float(config.get("temperature", self.temperature))
        self.max_tokens = int(config.get("max_tokens", self.max_tokens))
        self.timeout_sec = float(config.get("timeout_sec", self.timeout_sec))
        self.base_url = config.get("base_url", None)
        self.max_retries = int(config.get("max_retries", self.max_retries))

        # Respect optional base_url (Azure / proxy / self-hosted)
        self._client = OpenAI(base_url=self.base_url, timeout=self.timeout_sec) if self.base_url else OpenAI(
            timeout=self.timeout_sec
        )
        logger.info(f"OpenAIProvider ready (model={self.model}, timeout={self.timeout_sec}s)")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(2 ** attempt, 16) + random.random() * 0.5
        time.sleep(delay)

    def _extract_text(self, resp) -> str:
        """
        Robustly extract text from a Responses API object.
        - Prefer resp.output_text when available.
        - Fallback to the first text block in resp.output.
        """
        try:
            txt = getattr(resp, "output_text", None)
            if txt:
                return txt
        except Exception:
            pass

        try:
            out = getattr(resp, "output", None) or []
            if out and isinstance(out, list):
                # Newer Responses API: output[0].content[*].text
                first = out[0]
                content = getattr(first, "content", None) or []
                for part in content:
                    t = getattr(part, "text", None)
                    if t:
                        return t
        except Exception:
            pass
        return ""

    def _format_ok(self, resp, start: float, model: str) -> GenResult:
        latency = time.time() - start
        request_id = getattr(resp, "id", "")
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        try:
            finish_reason = (resp.output[0].finish_reason if getattr(resp, "output", None) else "") or ""
        except Exception:
            finish_reason = ""

        text = self._extract_text(resp) or ""
        logger.info(
            "OPENAI_CALL",
            extra={
                "model": model,
                "request_id": request_id,
                "latency_sec": round(latency, 3),
                "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
                "finish_reason": finish_reason,
            },
        )
        return {
            "text": text,
            "tokens_used": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_sec": latency,
            "model": model,
            "finish_reason": finish_reason,
            "request_id": request_id,
        }

    def _format_err(self, e: Exception, start: float, model: str) -> GenResult:
        latency = time.time() - start
        err_type = e.__class__.__name__
        code = getattr(e, "status", None) or getattr(e, "code", None)
        request_id = getattr(e, "request_id", "")
        error_msg = str(getattr(e, "message", None) or e)  # ✅ FIXED: renamed from 'msg' to 'error_msg'

        # ✅ FIXED: Use 'error_message' in extra dict instead of 'msg'
        logger.error(
            "OPENAI_ERROR",
            extra={
                "type": err_type,
                "code": code,
                "request_id": request_id,
                "error_message": error_msg  # ✅ FIXED
            }
        )
        return {
            "text": "",
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_sec": latency,
            "model": model,
            "error": error_msg,
            "error_type": err_type,
            "error_code": code,
            "request_id": request_id,
        }

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> GenResult:
        """
        Unified generation via the OpenAI Responses API.

        Args:
            prompt: Plain text prompt (mutually exclusive with messages)
            messages: Chat-style messages list: [{"role":"system"|"user"|"assistant","content":"..."}]
            **kwargs: Optional passthrough settings (seed, top_p, stop, response_format, tools, etc.)

        Returns:
            GenResult dict with text, usage stats, and metadata (or error fields).
        """
        if (prompt is None) == (messages is None):
            raise ValueError("Provide exactly one of `prompt` or `messages`")

        # Build the "input" payload for Responses API
        input_payload: Union[str, List[Dict[str, str]]]
        if messages is not None:
            input_payload = messages
        else:
            input_payload = prompt  # type: ignore[assignment]

        # Accept both max_output_tokens and max_tokens (prefer explicit kwarg first)
        max_out = kwargs.pop("max_output_tokens", None)
        if max_out is None:
            max_out = kwargs.pop("max_tokens", self.max_tokens)
        max_out = int(max_out) if int(max_out) > 0 else self.max_tokens

        call_args: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "input": input_payload,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_output_tokens": max_out,
        }
        # Pass through a few useful/allowed extras
        for k in (
            "seed",
            "top_p",
            "stop",
            "response_format",
            "tools",
            "tool_choice",
            "user",
            "reasoning",
            "logprobs",
            "top_logprobs",
        ):
            if k in kwargs:
                call_args[k] = kwargs.pop(k)
        if kwargs:
            logger.debug(f"Ignored unsupported kwargs: {list(kwargs.keys())}")

        start = time.time()
        attempt = 0

        while True:
            try:
                resp = self._client.responses.create(**call_args)
                return self._format_ok(resp, start, call_args["model"])

            except AuthenticationError as e:
                # ✅ Do not retry auth errors
                return self._format_err(e, start, call_args["model"])

            except (RateLimitError, APITimeoutError) as e:
                if attempt >= self.max_retries:
                    return self._format_err(e, start, call_args["model"])
                attempt += 1
                self._sleep_backoff(attempt)

            except BadRequestError as e:
                # Most formatting/contract issues land here—surface and stop
                return self._format_err(e, start, call_args["model"])

            except APIError as e:
                status = getattr(e, "status", 0) or 0
                if 500 <= int(status) < 600 and attempt < self.max_retries:
                    attempt += 1
                    self._sleep_backoff(attempt)
                    continue
                return self._format_err(e, start, call_args["model"])

            except Exception as e:
                return self._format_err(e, start, call_args["model"])
