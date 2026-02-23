"""Utilities for loading and running Slovo ONNX models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - import guard for lightweight envs.
    raise ImportError(
        "onnxruntime is required for SlovoONNXModel. "
        "Install it with: pip install onnxruntime-gpu or onnxruntime."
    ) from exc


@dataclass(slots=True, frozen=True)
class SlovoModelConfig:
    """Runtime config for Slovo ONNX model."""

    model_dir: Path = Path("models/slovo")
    model_name: str | None = None
    target_size: tuple[int, int] = (224, 224)
    target_fps: float = 25.0
    enable_preprocessing: bool = True
    providers: Sequence[str] | None = None


class SlovoONNXModel:
    """Load a Slovo ONNX model and run inference on frame sequences."""

    def __init__(self, config: SlovoModelConfig | None = None) -> None:
        self.config = config or SlovoModelConfig()
        self.model_path = self._resolve_model_path(
            model_dir=self.config.model_dir,
            model_name=self.config.model_name,
        )
        self.providers = self._resolve_providers(self.config.providers)
        self.session = ort.InferenceSession(
            self.model_path.as_posix(),
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    @staticmethod
    def list_available_models(model_dir: str | Path = "models/slovo") -> list[str]:
        """Return available ONNX models in the model directory."""
        model_root = Path(model_dir)
        if not model_root.exists():
            return []
        return sorted(path.name for path in model_root.glob("*.onnx"))

    def infer(
        self,
        frames: np.ndarray,
        *,
        input_fps: float | None = None,
        preprocess: bool | None = None,
    ) -> dict[str, Any]:
        """Run ONNX inference.

        Args:
            frames: Either raw frames ``(T, H, W, C)`` or prepared tensor.
            input_fps: FPS of incoming frames. Used only if preprocessing is enabled.
            preprocess: Overrides config-level preprocessing toggle.

        Returns:
            Dictionary with raw ONNX outputs and helper metadata.
        """
        should_preprocess = (
            self.config.enable_preprocessing if preprocess is None else preprocess
        )
        model_input = self._prepare_model_input(
            frames=frames,
            input_fps=input_fps,
            preprocess=should_preprocess,
        )
        outputs = self.session.run(None, {self.input_name: model_input})
        return {
            "outputs": outputs,
            "confidence": self._estimate_confidence(outputs),
            "preprocessed": should_preprocess,
            "input_shape": tuple(model_input.shape),
        }

    def _prepare_model_input(
        self,
        *,
        frames: np.ndarray,
        input_fps: float | None,
        preprocess: bool,
    ) -> np.ndarray:
        if frames.size == 0:
            raise ValueError("Input frames are empty.")

        data = np.asarray(frames)
        if preprocess:
            if data.ndim != 4:
                raise ValueError(
                    "Preprocessing expects raw frames in shape (T, H, W, C)."
                )
            data = self._preprocess_frames(
                frames=data,
                input_fps=input_fps,
                target_fps=self.config.target_fps,
                target_size=self.config.target_size,
            )
            data = np.transpose(data, (0, 3, 1, 2))  # (T, C, H, W)

        return self._to_model_layout(data)

    def _to_model_layout(self, data: np.ndarray) -> np.ndarray:
        source_dtype = data.dtype
        data = data.astype(np.float32, copy=False)
        if np.issubdtype(source_dtype, np.integer) and data.max() > 1.0:
            data /= 255.0

        if data.ndim == 5:
            return data

        if data.ndim != 4:
            raise ValueError(
                "Expected model input with 4 or 5 dims, got shape "
                f"{tuple(data.shape)}."
            )

        # input shape often looks like [B, C, T, H, W] or [B, T, C, H, W]
        second_dim = self.input_shape[1] if len(self.input_shape) >= 2 else None
        third_dim = self.input_shape[2] if len(self.input_shape) >= 3 else None

        if second_dim == 3:
            # data currently (T, C, H, W) -> (B, C, T, H, W)
            data = np.transpose(data, (1, 0, 2, 3))[np.newaxis, ...]
        elif third_dim == 3:
            # data currently (T, C, H, W) -> (B, T, C, H, W)
            data = data[np.newaxis, ...]
        else:
            # Fallback to the most common format.
            data = np.transpose(data, (1, 0, 2, 3))[np.newaxis, ...]

        return np.ascontiguousarray(data, dtype=np.float32)

    @staticmethod
    def _preprocess_frames(
        frames: np.ndarray,
        *,
        input_fps: float | None,
        target_fps: float,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        resized = SlovoONNXModel._resize_frames_nearest(frames, target_size)
        if input_fps is None or input_fps <= 0:
            return resized
        return SlovoONNXModel._resample_fps(
            frames=resized,
            input_fps=input_fps,
            target_fps=target_fps,
        )

    @staticmethod
    def _resample_fps(
        frames: np.ndarray,
        *,
        input_fps: float,
        target_fps: float,
    ) -> np.ndarray:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0.")
        if abs(input_fps - target_fps) < 1e-6:
            return frames

        total_frames = frames.shape[0]
        duration_sec = total_frames / input_fps
        target_count = max(1, int(round(duration_sec * target_fps)))
        indices = np.linspace(0, total_frames - 1, num=target_count, dtype=np.int32)
        return frames[indices]

    @staticmethod
    def _resize_frames_nearest(
        frames: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        target_h, target_w = target_size
        if target_h <= 0 or target_w <= 0:
            raise ValueError("target_size values must be > 0.")
        if frames.ndim != 4:
            raise ValueError("Frames must have shape (T, H, W, C).")

        _, src_h, src_w, _ = frames.shape
        if src_h == target_h and src_w == target_w:
            return frames

        y_idx = np.linspace(0, src_h - 1, num=target_h, dtype=np.int32)
        x_idx = np.linspace(0, src_w - 1, num=target_w, dtype=np.int32)
        return frames[:, y_idx][:, :, x_idx]

    @staticmethod
    def _resolve_model_path(model_dir: Path, model_name: str | None) -> Path:
        model_root = Path(model_dir)
        if not model_root.exists():
            raise FileNotFoundError(f"Model directory not found: {model_root}")

        if model_name:
            candidate = model_root / model_name
            if candidate.suffix != ".onnx":
                candidate = candidate.with_suffix(".onnx")
            if not candidate.exists():
                raise FileNotFoundError(f"Model not found: {candidate}")
            return candidate

        models = sorted(model_root.glob("*.onnx"))
        if not models:
            raise FileNotFoundError(f"No ONNX models found in: {model_root}")
        return models[0]

    @staticmethod
    def _resolve_providers(requested: Sequence[str] | None) -> list[str]:
        available = set(ort.get_available_providers())
        if requested:
            providers = [name for name in requested if name in available]
            if not providers:
                raise RuntimeError(
                    f"None of requested providers are available. "
                    f"requested={list(requested)}, available={sorted(available)}"
                )
            return providers

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

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
