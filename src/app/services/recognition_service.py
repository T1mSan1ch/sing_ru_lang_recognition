"""Business logic for sign-language recognition pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.aggregator import SentenceAggregator
from app.preprocessing.video import (
    MVITV2_CLIP_LEN,
    MVITV2_HOP_SIZE,
    decode_video_bytes,
    prepare_mvitv2_small_32_2_clips,
)
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
        clip_batch = await asyncio.to_thread(
            prepare_mvitv2_small_32_2_clips,
            frames,
            MVITV2_HOP_SIZE,
        )
        inference = await asyncio.to_thread(
            self._infer_and_merge_clips,
            clip_batch.clips,
            clip_batch.clip_starts,
        )
        words = self._extract_words(inference)
        sentence = await self._aggregator.aggregate(words)
        return RecognitionResult(words=words, sentence=sentence)

    def _infer_and_merge_clips(
        self,
        clips: np.ndarray,
        clip_starts: np.ndarray,
    ) -> dict[str, Any]:
        clip_inferences = [
            self._model.infer(clip, preprocess=False)
            for clip in clips
        ]
        if not clip_inferences:
            return {"outputs": [], "confidence": None}

        merged_outputs = self._merge_outputs(
            clip_inferences=clip_inferences,
            clip_starts=clip_starts,
            clip_len=MVITV2_CLIP_LEN,
        )
        return {
            "outputs": merged_outputs,
            "confidence": self._estimate_confidence(merged_outputs),
            "preprocessed": False,
            "input_shape": tuple(clips.shape),
        }

    @staticmethod
    def _merge_outputs(
        *,
        clip_inferences: list[dict[str, Any]],
        clip_starts: np.ndarray,
        clip_len: int,
    ) -> list[np.ndarray]:
        if not clip_inferences:
            return []
        outputs_lists = [item.get("outputs") for item in clip_inferences]
        if any(not isinstance(outputs, list) for outputs in outputs_lists):
            return clip_inferences[0].get("outputs", [])

        min_count = min(len(outputs) for outputs in outputs_lists)
        merged: list[np.ndarray] = []
        for index in range(min_count):
            clip_outputs = [np.asarray(outputs[index]) for outputs in outputs_lists]
            merged.append(
                RecognitionService._merge_single_output(
                    outputs=clip_outputs,
                    clip_starts=clip_starts,
                    clip_len=clip_len,
                )
            )
        return merged

    @staticmethod
    def _merge_single_output(
        *,
        outputs: list[np.ndarray],
        clip_starts: np.ndarray,
        clip_len: int,
    ) -> np.ndarray:
        if len(outputs) == 1:
            return outputs[0]

        squeezed = [
            RecognitionService._drop_leading_batch_axis(item)
            for item in outputs
        ]
        same_shape = all(item.shape == squeezed[0].shape for item in squeezed)
        if same_shape and squeezed[0].ndim != 2:
            return np.mean(np.stack(squeezed, axis=0), axis=0)

        can_merge_temporally = (
            all(item.ndim == 2 for item in squeezed)
            and all(item.shape[1] == squeezed[0].shape[1] for item in squeezed)
            and len(squeezed) == int(clip_starts.shape[0])
        )
        if not can_merge_temporally:
            if same_shape:
                return np.mean(np.stack(squeezed, axis=0), axis=0)
            return outputs[0]

        classes = squeezed[0].shape[1]
        lengths = [item.shape[0] for item in squeezed]
        scaled_starts = [
            int(round(int(start) * length / clip_len))
            for start, length in zip(clip_starts, lengths, strict=True)
        ]
        merged_len = max(
            start + length
            for start, length in zip(scaled_starts, lengths, strict=True)
        )
        logits_sum = np.zeros((merged_len, classes), dtype=np.float32)
        logits_count = np.zeros((merged_len, 1), dtype=np.float32)

        for start, clip_logits in zip(scaled_starts, squeezed, strict=True):
            end = start + clip_logits.shape[0]
            logits_sum[start:end] += clip_logits.astype(np.float32, copy=False)
            logits_count[start:end] += 1.0

        logits_count[logits_count == 0.0] = 1.0
        return logits_sum / logits_count

    @staticmethod
    def _drop_leading_batch_axis(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array)
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        return result

    @staticmethod
    def _estimate_confidence(outputs: list[np.ndarray]) -> float | None:
        if not outputs:
            return None
        first = np.asarray(outputs[0])
        if first.size == 0:
            return None
        flat = first.reshape(-1).astype(np.float32)
        if flat.size == 1:
            return float(flat[0])

        shifted = flat - np.max(flat)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
        return float(np.max(probs))

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
