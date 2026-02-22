"""LLM-based sentence aggregation for recognized sign language words."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

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


@dataclass(slots=True, frozen=True)
class LocalHFProviderConfig:
    """Configuration for in-process local Hugging Face model aggregation."""

    model: str
    device: Literal["auto", "cpu", "cuda"] = "auto"
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


class LocalHFProvider:
    """Provider that runs a local HF CausalLM directly inside the API process."""

    def __init__(self, config: LocalHFProviderConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise ImportError(
                "transformers and torch are required for LocalHFProvider. "
                "Install with: pip install transformers torch"
            ) from exc

        self._config = config
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        self._model = AutoModelForCausalLM.from_pretrained(config.model)
        self._device = _resolve_device(torch, requested=config.device)
        self._model.to(self._device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    async def generate_sentence(self, words: Sequence[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, words)

    def _generate_sync(self, words: Sequence[str]) -> str:
        prompt = _build_chat_prompt(
            tokenizer=self._tokenizer,
            system_prompt=self._config.system_prompt,
            words=words,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}

        do_sample = self._config.temperature > 0
        temperature = self._config.temperature if do_sample else 1.0
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_tokens:]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _postprocess_generated_text(generated_text)


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


def _build_chat_prompt(tokenizer: Any, system_prompt: str, words: Sequence[str]) -> str:
    user_prompt = _build_user_prompt(words)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}\n\nОтвет:"


def _postprocess_generated_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    return first_line or cleaned


def _resolve_device(torch_module: Any, requested: Literal["auto", "cpu", "cuda"]) -> Any:
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device requested for local LLM, but CUDA is unavailable.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
