"""LLM-based sentence aggregation for recognized sign language words."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Ты помощник для агрегации "
    "распознанных жестов "
    "русского жестового языка. "
    "Преобразуй список слов "
    "в одно естественное предложение "
    "на русском языке. "
    "Исправляй только грамматику и порядок слов. "
    "Не добавляй новую фактическую информацию."
)


@runtime_checkable
class LLMProvider(Protocol):
    """Contract for pluggable LLM providers."""

    async def generate_sentence(self, words: Sequence[str]) -> str:
        """Generate a readable sentence from recognized words."""


@dataclass(slots=True, frozen=True)
class AggregatorConfig:
    """Runtime configuration for sentence aggregation."""

    timeout_seconds: float = 5.0


@dataclass(slots=True, frozen=True)
class OpenAIProviderConfig:
    """Configuration for OpenAI API-based aggregation."""

    model: str
    api_key: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    max_tokens: int = 96


@dataclass(slots=True, frozen=True)
class VLLMProviderConfig:
    """Configuration for local vLLM server aggregation."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    max_tokens: int = 96


class OpenAIAPIProvider:
    """LLM provider using the official OpenAI Python library."""

    def __init__(self, config: OpenAIProviderConfig) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImportError(
                "openai is required for OpenAIAPIProvider. "
                "Install with: pip install openai"
            ) from exc

        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key)

    async def generate_sentence(self, words: Sequence[str]) -> str:
        prompt = _build_user_prompt(words)
        response = await self._client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class VLLMProvider:
    """Provider for locally hosted vLLM server with OpenAI-compatible API."""

    def __init__(self, config: VLLMProviderConfig) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImportError(
                "openai is required for VLLMProvider. Install with: pip install openai"
            ) from exc

        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def generate_sentence(self, words: Sequence[str]) -> str:
        prompt = _build_user_prompt(words)
        response = await self._client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class SentenceAggregator:
    """Aggregate recognized words into a sentence using an LLM with fallback."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: AggregatorConfig | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._config = config or AggregatorConfig()

    async def aggregate(self, words: Sequence[str]) -> str:
        """Return an LLM-generated sentence or fallback raw text.

        Fallback is used when:
        - provider is not configured
        - provider times out
        - provider raises any error
        - provider returns an empty string
        """
        cleaned_words = [word.strip() for word in words if word and word.strip()]
        fallback_text = " ".join(cleaned_words)

        if not cleaned_words:
            return ""
        if self._llm_provider is None:
            return fallback_text

        try:
            result = await asyncio.wait_for(
                self._llm_provider.generate_sentence(cleaned_words),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "LLM aggregation timed out after %.2f seconds, using fallback.",
                self._config.timeout_seconds,
            )
            return fallback_text
        except Exception:  # pragma: no cover - depends on concrete provider.
            logger.exception("LLM aggregation failed, using fallback.")
            return fallback_text

        sentence = result.strip()
        return sentence if sentence else fallback_text


def _build_user_prompt(words: Sequence[str]) -> str:
    serialized = ", ".join(f'"{word}"' for word in words)
    return (
        "Преобразуй последовательность слов "
        "в одно корректное предложение. "
        "Верни только итоговое предложение "
        "без пояснений.\n"
        f"Слова: [{serialized}]"
    )
