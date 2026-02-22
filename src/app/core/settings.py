"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="NONVERBAL_",
        env_file=".env",
        extra="ignore",
    )

    llm_provider: Literal["none", "openai", "vllm"] = "none"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str = "http://localhost:8000/v1"
    llm_timeout_seconds: float = 5.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
