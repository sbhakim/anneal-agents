# src/fdka/llm_providers/deepseek_provider.py

import os, time, logging, random
from typing import Dict, Any, List, TypedDict, Union, Optional
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
class DeepSeekProvider:
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout_sec: float = 60.0
    base_url: str = "https://api.deepseek.com"
    max_retries: int = 4
    _client: OpenAI = field(init=False, repr=False)

    def __init__(self, config: Dict[str, Any]):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        if api_key.startswith("sk-...") or api_key == "sk-...":
            raise ValueError("Please replace 'sk-...' with your actual DeepSeek API key")

        self.model = config.get("model", self.model)
        self.temperature = float(config.get("temperature", self.temperature))
        self.max_tokens = int(config.get("max_tokens", self.max_tokens))
        self.timeout_sec = float(config.get("timeout_sec", self.timeout_sec))
        self.base_url = config.get("base_url", self.base_url)
        self.max_retries = int(config.get("max_retries", self.max_retries))
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout_sec)
        logger.info(f"DeepSeekProvider ready (model={self.model}, base_url={self.base_url})")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(2 ** attempt, 16) + random.random() * 0.5
        time.sleep(delay)

    def _format_ok(self, resp, start: float) -> GenResult:
        latency = time.time() - start
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

        choices = getattr(resp, "choices", None) or []
        first = choices[0] if choices else None
        finish_reason = getattr(first, "finish_reason", "") if first else ""
        message = getattr(first, "message", None) if first else None
        text = getattr(message, "content", "") if message else ""
        request_id = getattr(resp, "id", "") if hasattr(resp, "id") else ""

        logger.info(
            "DEEPSEEK_CALL",
            extra={
                "model": self.model,
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
            "model": self.model,
            "finish_reason": finish_reason,
            "request_id": request_id,
        }

    def _format_err(self, e: Exception, start: float) -> GenResult:
        latency = time.time() - start
        err_type = e.__class__.__name__
        code = getattr(e, "status", None) or getattr(e, "code", None)
        request_id = getattr(e, "request_id", "")
        error_msg = str(getattr(e, "message", None) or e)

        # Console visibility for debugging runs
        print(f"\n❌ DeepSeek API Error:")
        print(f"   Type: {err_type}")
        print(f"   Code: {code}")
        print(f"   Message: {error_msg}")
        print(f"   Request ID: {request_id}")

        logger.error(
            "DEEPSEEK_ERROR",
            extra={
                "type": err_type,
                "code": code,
                "request_id": request_id,
                "error_message": error_msg
            }
        )

        return {
            "text": "",
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_sec": latency,
            "model": self.model,
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
        if (prompt is None) == (messages is None):
            raise ValueError("Provide exactly one of `prompt` or `messages`")

        # Convert to Chat Completions format
        if messages is not None:
            messages_payload = messages
        else:
            messages_payload = [
                {
                    "role": "system",
                    "content": "You are an expert system that generates formal rule patches for autonomous agents."
                },
                {"role": "user", "content": prompt or ""}  # prompt is not None here by contract
            ]

        # Accept both max_output_tokens and max_tokens (mirror OpenAI provider behavior)
        max_out = kwargs.pop("max_output_tokens", None)
        if max_out is None:
            max_out = kwargs.pop("max_tokens", self.max_tokens)
        try:
            max_out = int(max_out)
        except Exception:
            max_out = self.max_tokens
        if max_out <= 0:
            max_out = self.max_tokens

        call_args: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages_payload,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": max_out,
        }

        for k in ("stop", "stream", "top_p", "frequency_penalty", "presence_penalty", "n", "user"):
            if k in kwargs:
                call_args[k] = kwargs.pop(k)

        if kwargs:
            # ✅ FIXED: removed stray bracket
            logger.debug(f"Ignored unsupported kwargs: {list(kwargs.keys())}")

        start = time.time()
        attempt = 0

        while True:
            try:
                resp = self._client.chat.completions.create(**call_args)
                return self._format_ok(resp, start)

            except AuthenticationError as e:
                # Do not retry auth errors
                return self._format_err(e, start)

            except (RateLimitError, APITimeoutError) as e:
                if attempt >= self.max_retries:
                    return self._format_err(e, start)
                attempt += 1
                logger.warning(f"DeepSeek retry {attempt}/{self.max_retries} after {e.__class__.__name__}")
                self._sleep_backoff(attempt)

            except BadRequestError as e:
                return self._format_err(e, start)

            except APIError as e:
                status = getattr(e, "status", 0) or 0
                if 500 <= int(status) < 600 and attempt < self.max_retries:
                    attempt += 1
                    logger.warning(f"DeepSeek retry {attempt}/{self.max_retries} after 5xx error")
                    self._sleep_backoff(attempt)
                    continue
                return self._format_err(e, start)

            except Exception as e:
                return self._format_err(e, start)
