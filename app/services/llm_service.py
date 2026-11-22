"""Provider-neutral OpenAI/Mistral chat completion adapter."""
from typing import Any


class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM provider is not configured."""


class LLMService:
    """Small synchronous adapter shared by OpenAI and Mistral providers."""

    def __init__(self, provider: str, *, api_key: str | None, model: str,
                 temperature: float = 0.0, max_tokens: int = 1200,
                 client: Any | None = None) -> None:
        if provider not in {"openai", "mistral"}:
            raise ValueError("provider must be 'openai' or 'mistral'")
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.provider, self.api_key, self.model = provider, api_key, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = client

    def _get_client(self) -> Any:
        if not self.api_key:
            raise LLMConfigurationError(
                f"API key for provider '{self.provider}' is not configured"
            )
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            else:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.api_key)
        return self._client

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            response = client.chat.complete(
                model=self.model, messages=messages, temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM provider returned an empty response")
        return content.strip()
