"""Application entrypoint for the nonverbal sign language service."""

from __future__ import annotations

import base64
import binascii
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from app.aggregator import (
    AggregatorConfig,
    LocalHFProvider,
    LocalHFProviderConfig,
    OpenAIAPIProvider,
    OpenAIProviderConfig,
    SentenceAggregator,
    VLLMProvider,
    VLLMProviderConfig,
)
from app.api.schemas import RecognizeBase64Request, RecognizeResponse
from app.core.settings import get_settings
from app.services.recognition_service import RecognitionService
from slovo_model import SlovoONNXModel

logger = logging.getLogger(__name__)


def _build_aggregator() -> SentenceAggregator:
    settings = get_settings()

    if settings.llm_provider == "openai":
        if not settings.llm_model or not settings.llm_api_key:
            raise RuntimeError(
                "Set NONVERBAL_LLM_MODEL and NONVERBAL_LLM_API_KEY for OpenAI mode."
            )
        provider = OpenAIAPIProvider(
            OpenAIProviderConfig(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
            )
        )
    elif settings.llm_provider == "vllm":
        if not settings.llm_model:
            raise RuntimeError("Set NONVERBAL_LLM_MODEL for vLLM mode.")
        provider = VLLMProvider(
            VLLMProviderConfig(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key or "not-needed",
            )
        )
    elif settings.llm_provider == "local":
        if not settings.llm_model:
            raise RuntimeError("Set NONVERBAL_LLM_MODEL for local mode.")
        logger.info(
            "Using local LLM provider with model=%s device=%s",
            settings.llm_model,
            settings.llm_local_device,
        )
        provider = LocalHFProvider(
            LocalHFProviderConfig(
                model=settings.llm_model,
                device=settings.llm_local_device,
                temperature=settings.llm_local_temperature,
                max_tokens=settings.llm_local_max_tokens,
            )
        )
    else:
        provider = None

    return SentenceAggregator(
        llm_provider=provider,
        config=AggregatorConfig(timeout_seconds=settings.llm_timeout_seconds),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and services once at application startup."""
    service: RecognitionService | None = None
    try:
        model = SlovoONNXModel()
        aggregator = _build_aggregator()
        service = RecognitionService(model=model, aggregator=aggregator)
    except Exception:  # pragma: no cover - defensive startup guard.
        logger.exception("Recognition service startup failed.")

    app.state.recognition_service = service
    yield


app = FastAPI(title="Nonverbal Sign Language Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: Request) -> RecognizeResponse:
    """Recognize words from video and return words with aggregated sentence."""
    video_bytes = await _extract_video_payload(request)
    service: RecognitionService | None = getattr(
        request.app.state, "recognition_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Recognition service is unavailable.",
        )

    try:
        result = await service.recognize_video_bytes(video_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard.
        logger.exception("Failed to process /recognize request.")
        raise HTTPException(status_code=500, detail="Recognition failed.") from exc

    return RecognizeResponse(words=result.words, sentence=result.sentence)


async def _extract_video_payload(
    request: Request,
) -> bytes:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("video")
        if uploaded is None:
            raise HTTPException(
                status_code=400,
                detail="Multipart request must contain file field `video`.",
            )
        if not hasattr(uploaded, "read"):
            raise HTTPException(
                status_code=400,
                detail="Field `video` must be a valid uploaded file.",
            )
        data = await uploaded.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
        return data

    if "application/json" in content_type:
        try:
            raw_payload = await request.json()
            payload = RecognizeBase64Request.model_validate(raw_payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON payload.",
            ) from exc

        encoded = payload.video_base64.strip()
        if "," in encoded and encoded.lower().startswith("data:"):
            encoded = encoded.split(",", maxsplit=1)[1]
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 payload.",
            ) from exc

        if not decoded:
            raise HTTPException(
                status_code=400,
                detail="Decoded base64 payload is empty.",
            )
        return decoded

    raise HTTPException(
        status_code=415,
        detail=(
            "Unsupported content type. Use multipart/form-data with file field "
            "`video` or application/json with field `video_base64`."
        ),
    )
