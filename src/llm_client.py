"""LLM client for Reddit Radar.

This module handles all LLM API interactions. Currently supports:
- DeepSeek (via OpenAI-compatible API)
- OpenAI (future)
- Anthropic (future)
"""

from dataclasses import dataclass
from typing import Generator

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

from src.config import get_config, LLMCredentials


# API endpoints for different providers
PROVIDER_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
}


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class APIKeyError(LLMClientError):
    """Raised when API key is missing or invalid."""
    pass


class LLMClient:
    """LLM client supporting multiple providers.

    Currently supports DeepSeek with OpenAI-compatible API.
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider (deepseek, openai).
            model: Model to use.
            api_key: API key. If None, loads from config.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.

        Raises:
            ImportError: If openai library is not installed.
            APIKeyError: If API key is not configured.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai library is required for LLM access. "
                "Install it with: pip install openai"
            )

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            config = get_config()
            api_key = config.llm_credentials.get_key_for_provider(provider)

        if not api_key:
            raise APIKeyError(
                f"API key not configured for provider '{provider}'. "
                f"Set the appropriate environment variable."
            )

        self._api_key = api_key
        self._client = self._create_client()

    def _create_client(self) -> "OpenAI":
        """Create OpenAI client configured for the provider.

        Returns:
            Configured OpenAI client.
        """
        base_url = PROVIDER_ENDPOINTS.get(self.provider)
        if base_url is None:
            raise LLMClientError(f"Unknown provider: {self.provider}")

        return OpenAI(
            api_key=self._api_key,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with generated content.

        Raises:
            LLMClientError: If generation fails.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            # Extract usage info
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=self.model,
                provider=self.provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            raise LLMClientError(f"LLM generation failed: {e}")

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Yields:
            Chunks of generated text.

        Raises:
            LLMClientError: If generation fails.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMClientError(f"LLM streaming generation failed: {e}")


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Get an LLM client instance.

    Args:
        provider: Optional provider override.
        model: Optional model override.

    Returns:
        Configured LLMClient.
    """
    config = get_config()

    return LLMClient(
        provider=provider or config.llm.provider,
        model=model or config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


# Convenience function for simple generation
def generate_text(
    prompt: str,
    system_prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Generate text from prompt.

    Args:
        prompt: User prompt.
        system_prompt: Optional system prompt.
        provider: Optional provider override.
        model: Optional model override.

    Returns:
        Generated text content.
    """
    client = get_llm_client(provider, model)
    response = client.generate(prompt, system_prompt)
    return response.content
