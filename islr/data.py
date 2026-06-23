"""Video dataset reading from CSV manifests.

CSV format produced by `scripts/data/build_sampled_manifest.py`:
  video,label,split,user_id,begin,end[,length]

Where:
  video    - absolute path to the video file
  label    - human-readable class name (Russian word, ASL gloss, etc.)
  begin    - first frame index of the gesture (optional)
  end      - last frame index of the gesture (optional)

The dataset returns a `dict` per sample with:
  pixels  : (T, C, H, W) float32 tensor in [0, 1]
  label   : int class id
  span    : (begin, end) normalised to [0, 1] of the whole video, or (0, 1) if missing
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

VideoTransform = Callable[[torch.Tensor, dict], tuple[torch.Tensor, dict]]


@dataclass
class VideoSample:
    pixels: torch.Tensor  # (T, C, H, W) float32 in [0, 1]
    label: int
    span: tuple[float, float]  # normalised gesture span in the source video


class VideoCsvDataset(Dataset):
    """Reads videos lazily from a CSV manifest.

    Args:
      csv_path:    path to the manifest CSV
      label_to_id: mapping `{label_name: class_id}` from the label_map.csv
      clip_len:    number of frames to sample per clip
      transform:   optional callable `(video, meta) -> (video, meta)` that
                   receives an (T, C, H, W) tensor in [0, 1] and a meta dict
                   containing keys `span`, `total_frames`, `clip_len`,
                   `frame_indices`. It must return the (possibly modified)
                   video tensor and updated meta.
    """

    def __init__(
        self,
        csv_path: Path,
        label_to_id: dict[str, int],
        clip_len: int = 32,
        transform: VideoTransform | None = None,
    ) -> None:
        self.rows = self._read_csv(csv_path)
        self.label_to_id = label_to_id
        self.clip_len = clip_len
        self.transform = transform

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_video(self, path: str) -> torch.Tensor:
        """Read whole video into (T, C, H, W) float32 tensor in [0, 1].

        Tries backends in order: torchvision (needs PyAV), then OpenCV
        (preinstalled on Kaggle), then imageio. This avoids a hard
        dependency on PyAV, which is not available in the default Kaggle
        Python 3.12 image.
        """
        # 1) torchvision + PyAV (fast path if available)
        try:
            from torchvision.io import read_video

            video, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
            if video.numel() > 0:
                return video.float() / 255.0
        except Exception:
            pass

        # 2) OpenCV fallback (cv2 ships with Kaggle base image)
        try:
            import cv2  # type: ignore
            import numpy as np

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"cv2 cannot open: {path}")
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                raise RuntimeError(f"Empty video: {path}")
            arr = np.stack(frames, axis=0)  # (T, H, W, C)
            return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
        except Exception:
            pass

        # 3) imageio last-ditch
        import imageio.v3 as iio  # type: ignore

        arr = iio.imread(path)
        video = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if video.numel() == 0:
            raise RuntimeError(f"Empty/unreadable video: {path}")
        return video.float() / 255.0

    def _gesture_span(self, row: dict[str, str], total_frames: int) -> tuple[float, float]:
        begin = row.get("begin", "")
        end = row.get("end", "")
        try:
            b = max(0, int(float(begin))) if begin else 0
            e = min(total_frames, int(float(end))) if end else total_frames
            if e <= b:
                return (0.0, 1.0)
            return (b / total_frames, e / total_frames)
        except (TypeError, ValueError):
            return (0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        video = self._load_video(row["video"])
        total = video.shape[0]
        span = self._gesture_span(row, total)
        meta = {
            "span": span,
            "total_frames": total,
            "clip_len": self.clip_len,
            "frame_indices": None,  # populated by SampleFrames transform
        }
        if self.transform is not None:
            video, meta = self.transform(video, meta)
        return {
            "pixels": video,  # (T, C, H, W)
            "label": self.label_to_id[row["label"]],
            "span": torch.tensor(meta["span"], dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    """Default collate: stacks pixels, labels, spans into batched tensors."""
    return {
        "pixels": torch.stack([b["pixels"] for b in batch], dim=0),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "span": torch.stack([b["span"] for b in batch], dim=0),
    }
