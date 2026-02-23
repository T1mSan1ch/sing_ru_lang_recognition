"""Video preprocessing helpers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

MVITV2_TARGET_SIZE: Final[tuple[int, int]] = (224, 224)
MVITV2_CLIP_LEN: Final[int] = 32
MVITV2_FRAME_STEP: Final[int] = 2
MVITV2_HOP_SIZE: Final[int] = 16
# PyTorchVideo defaults for MViT-based video backbones.
MVITV2_MEAN: Final[np.ndarray] = np.asarray([0.45, 0.45, 0.45], dtype=np.float32)
MVITV2_STD: Final[np.ndarray] = np.asarray([0.225, 0.225, 0.225], dtype=np.float32)


@dataclass(slots=True, frozen=True)
class MViTv2ClipBatch:
    """Prepared clips and metadata for multi-clip inference."""

    clips: np.ndarray
    clip_starts: np.ndarray


def decode_video_bytes(video_bytes: bytes) -> np.ndarray:
    """Decode a video binary payload into frames array (T, H, W, C)."""
    if not video_bytes:
        raise ValueError("Video payload is empty.")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "opencv-python-headless is required for video decoding. "
            "Install with: pip install opencv-python-headless"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(video_bytes)

    frames: list[np.ndarray] = []
    capture = cv2.VideoCapture(temp_path.as_posix())
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        capture.release()
        temp_path.unlink(missing_ok=True)

    if not frames:
        raise ValueError("Unable to decode video or no frames found.")

    return np.asarray(frames, dtype=np.uint8)


def prepare_mvitv2_small_32_2_input(frames: np.ndarray) -> np.ndarray:
    """Prepare RGB frames for MViTv2-small-32-2 ONNX inference.

    Expected input shape: ``(T, H, W, C)`` with ``uint8`` RGB frames.
    Returns tensor with shape ``(32, 3, 224, 224)`` and ``float32`` dtype.
    """
    data = np.asarray(frames)
    if data.ndim != 4:
        raise ValueError("Frames must have shape (T, H, W, C).")
    if data.shape[-1] != 3:
        raise ValueError("Frames must contain 3 color channels (RGB).")
    if data.shape[0] == 0:
        raise ValueError("Input frames are empty.")

    clip_batch = prepare_mvitv2_small_32_2_clips(frames=data, hop_size=MVITV2_CLIP_LEN)
    return clip_batch.clips[0]


def prepare_mvitv2_small_32_2_clips(
    frames: np.ndarray,
    hop_size: int = MVITV2_HOP_SIZE,
) -> MViTv2ClipBatch:
    """Prepare normalized clip windows for MViTv2-small-32-2.

    Returns:
        ``MViTv2ClipBatch`` with:
            clips: ``(N, 32, 3, 224, 224)`` float32
            clip_starts: start indices (in sampled frame space) for each clip
    """
    data = np.asarray(frames)
    if data.ndim != 4:
        raise ValueError("Frames must have shape (T, H, W, C).")
    if data.shape[-1] != 3:
        raise ValueError("Frames must contain 3 color channels (RGB).")
    if data.shape[0] == 0:
        raise ValueError("Input frames are empty.")
    if hop_size <= 0:
        raise ValueError("hop_size must be > 0.")

    sampled = data[::MVITV2_FRAME_STEP]
    if sampled.shape[0] == 0:
        sampled = data[:1]

    clip_starts = _build_clip_starts(
        total_frames=sampled.shape[0],
        clip_len=MVITV2_CLIP_LEN,
        hop_size=hop_size,
    )
    clips = np.asarray(
        [
            _normalize_rgb_frames(
                np.asarray(
                    [
                        _resize_with_aspect_ratio(frame, MVITV2_TARGET_SIZE)
                        for frame in _window_with_padding(
                            sampled,
                            start=start,
                            clip_len=MVITV2_CLIP_LEN,
                        )
                    ],
                    dtype=np.float32,
                ),
                mean=MVITV2_MEAN,
                std=MVITV2_STD,
            )
            for start in clip_starts
        ],
        dtype=np.float32,
    )
    return MViTv2ClipBatch(clips=clips, clip_starts=clip_starts)


def _fit_to_clip_length(frames: np.ndarray, clip_len: int) -> np.ndarray:
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0.")
    total = frames.shape[0]
    if total == clip_len:
        return frames
    if total > clip_len:
        indices = np.linspace(0, total - 1, num=clip_len, dtype=np.int32)
        return frames[indices]

    pad_count = clip_len - total
    pad = np.repeat(frames[-1:], repeats=pad_count, axis=0)
    return np.concatenate((frames, pad), axis=0)


def _window_with_padding(
    frames: np.ndarray,
    *,
    start: int,
    clip_len: int,
) -> np.ndarray:
    if start < 0:
        raise ValueError("start must be >= 0.")
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0.")

    end = start + clip_len
    window = frames[start:end]
    if window.shape[0] >= clip_len:
        return window
    return _fit_to_clip_length(window, clip_len)


def _build_clip_starts(
    *,
    total_frames: int,
    clip_len: int,
    hop_size: int,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0.")
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0.")
    if hop_size <= 0:
        raise ValueError("hop_size must be > 0.")
    if total_frames <= clip_len:
        return np.asarray([0], dtype=np.int32)

    starts = list(range(0, total_frames - clip_len + 1, hop_size))
    tail_start = total_frames - clip_len
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return np.asarray(starts, dtype=np.int32)


def _resize_with_aspect_ratio(
    frame_rgb: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "opencv-python-headless is required for video preprocessing. "
            "Install with: pip install opencv-python-headless"
        ) from exc

    target_h, target_w = target_size
    src_h, src_w = frame_rgb.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Frame dimensions must be > 0.")

    scale = min(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(frame_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=frame_rgb.dtype)
    top = (target_h - scaled_h) // 2
    left = (target_w - scaled_w) // 2
    canvas[top : top + scaled_h, left : left + scaled_w] = resized
    return canvas


def _normalize_rgb_frames(
    frames_rgb: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape (T, H, W, C).")
    if np.any(std == 0):
        raise ValueError("std values must be non-zero.")

    normalized = (frames_rgb / 255.0 - mean.reshape(1, 1, 1, 3)) / std.reshape(
        1, 1, 1, 3
    )
    return np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32, copy=False)
