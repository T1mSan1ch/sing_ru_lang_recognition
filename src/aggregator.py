"""Backward-compatible exports for aggregator module."""

from app.aggregator import (
    AggregatorConfig,
    LLMProvider,
    LocalHFProvider,
    LocalHFProviderConfig,
    OpenAIAPIProvider,
    OpenAIProviderConfig,
    SentenceAggregator,
    VLLMProvider,
    VLLMProviderConfig,
)

__all__ = [
    "AggregatorConfig",
    "LLMProvider",
    "LocalHFProvider",
    "LocalHFProviderConfig",
    "OpenAIAPIProvider",
    "OpenAIProviderConfig",
    "SentenceAggregator",
    "VLLMProvider",
    "VLLMProviderConfig",
]
