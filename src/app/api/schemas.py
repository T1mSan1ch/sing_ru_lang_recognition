"""Pydantic schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecognizeBase64Request(BaseModel):
    """Request body for base64-encoded video payload."""

    video_base64: str = Field(..., min_length=1)


class RecognizeResponse(BaseModel):
    """Recognition API response."""

    words: list[str]
    sentence: str
