# src/fdka/llm_providers/anthropic_provider.py
import os, time, logging
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError, AuthenticationError

logger = logging.getLogger(__name__)

class GenResult(TypedDict, total=False):
    text: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    model: str
    finish_reason: str
    error: str
    error_type: str

@dataclass
class AnthropicProvider:
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout_sec: float = 60.0
    max_retries: int = 4
    _client: Anthropic = field(init=False, repr=False)

    def __init__(self, config: Dict[str, Any]):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.model = config.get("model", self.model)
        self.temperature = float(config.get("temperature", self.temperature))
        self.max_tokens = int(config.get("max_tokens", self.max_tokens))
        self.timeout_sec = float(config.get("timeout_sec", self.timeout_sec))
        self.max_retries = int(config.get("max_retries", self.max_retries))

        self._client = Anthropic(api_key=api_key, timeout=self.timeout_sec)
        logger.info(f"AnthropicProvider ready (model={self.model})")

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> GenResult:
        if (prompt is None) == (messages is None):
            raise ValueError("Provide exactly one of `prompt` or `messages`")

        if prompt is not None:
            messages_for_api = [{"role": "user", "content": prompt}]
            system_prompt = None
        else:
            # Separate system prompt from messages for Anthropic API
            system_prompt = None
            messages_for_api = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                else:
                    messages_for_api.append(msg)

        start = time.time()
        
        try:
            api_kwargs = {
                "model": kwargs.get("model", self.model),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": messages_for_api
            }
            if system_prompt:
                api_kwargs["system"] = system_prompt
                
            resp = self._client.messages.create(**api_kwargs)
            
            latency = time.time() - start
            prompt_tokens = resp.usage.input_tokens
            completion_tokens = resp.usage.output_tokens
            text = resp.content[0].text if resp.content else ""
            
            return {
                "text": text,
                "tokens_used": prompt_tokens + completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_sec": latency,
                "model": self.model,
                "finish_reason": resp.stop_reason or "",
            }

        except Exception as e:
            latency = time.time() - start
            err_type = e.__class__.__name__
            return {
                "text": "",
                "tokens_used": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_sec": latency,
                "model": self.model,
                "error": str(e),
                "error_type": err_type,
            }
