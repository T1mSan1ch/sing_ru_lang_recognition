"""Video preprocessing helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


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
