"""Business logic for sign-language recognition pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.aggregator import SentenceAggregator
from app.preprocessing.video import decode_video_bytes
from slovo_model import SlovoONNXModel


@dataclass(slots=True, frozen=True)
class RecognitionResult:
    """Result of full recognition pipeline."""

    words: list[str]
    sentence: str


class RecognitionService:
    """Run CV recognition and aggregate recognized words into sentence."""

    def __init__(self, model: SlovoONNXModel, aggregator: SentenceAggregator) -> None:
        self._model = model
        self._aggregator = aggregator

    async def recognize_video_bytes(self, video_bytes: bytes) -> RecognitionResult:
        """Recognize words from video bytes and aggregate them into sentence."""
        frames = await asyncio.to_thread(decode_video_bytes, video_bytes)
        inference = await asyncio.to_thread(self._model.infer, frames)
        words = self._extract_words(inference)
        sentence = await self._aggregator.aggregate(words)
        return RecognitionResult(words=words, sentence=sentence)

    @staticmethod
    def _extract_words(inference: dict[str, Any]) -> list[str]:
        direct_words = inference.get("words")
        if isinstance(direct_words, list):
            cleaned = [str(word).strip() for word in direct_words if str(word).strip()]
            if cleaned:
                return cleaned

        outputs = inference.get("outputs")
        if not outputs:
            return []

        logits = np.asarray(outputs[0])
        if logits.size == 0:
            return []

        if logits.ndim == 1:
            token_ids = [int(np.argmax(logits))]
        else:
            token_ids = [int(item) for item in logits.argmax(axis=-1).reshape(-1)]

        collapsed: list[int] = []
        previous: int | None = None
        for token_id in token_ids:
            if token_id != previous:
                collapsed.append(token_id)
            previous = token_id

        return [f"token_{token_id}" for token_id in collapsed]
