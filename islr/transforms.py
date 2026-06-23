"""Video and image augmentations for ISLR — pure torchvision/PyTorch.

The contract: each transform takes `(video, meta)` where:
  * video is (T, C, H, W) float in [0, 1]
  * meta is a dict carrying `span` (gesture frames in [0, 1] of source video),
    `total_frames`, `clip_len`, `frame_indices`

and returns the (possibly modified) `(video, meta)`.

Compose them with `Compose([...])`.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

VideoTransform = Callable[[torch.Tensor, dict], tuple[torch.Tensor, dict]]


class Compose:
    def __init__(self, transforms: list[VideoTransform]) -> None:
        self.transforms = transforms

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        for t in self.transforms:
            video, meta = t(video, meta)
        return video, meta


# ----------------------------------------------------------------------------
# Frame sampling
# ----------------------------------------------------------------------------


class SampleFrames:
    """Pick `clip_len` frames from the source video at uniform stride.

    Args:
      clip_len:        target number of frames
      frame_interval:  stride between sampled frames
      train_jitter:    if True, randomly offset the start (training mode);
                       if False, take a centred deterministic window
      boundary_shift:  if >0, randomly shift the gesture span by [-K, +K] frames
                       (Random Boundary Shift augmentation from SberDevices)
    """

    def __init__(
        self,
        clip_len: int = 32,
        frame_interval: int = 2,
        train_jitter: bool = True,
        boundary_shift: int = 0,
    ) -> None:
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.train_jitter = train_jitter
        self.boundary_shift = boundary_shift

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        total = video.shape[0]
        b_norm, e_norm = meta["span"]
        b = int(b_norm * total)
        e = int(e_norm * total)
        # Random boundary shift
        if self.boundary_shift > 0:
            b += random.randint(-self.boundary_shift, self.boundary_shift)
            e += random.randint(-self.boundary_shift, self.boundary_shift)
        b = max(0, b)
        e = min(total, max(b + 1, e))

        window = self.clip_len * self.frame_interval
        gesture_len = e - b
        if gesture_len >= window:
            offset = random.randint(b, e - window) if self.train_jitter else (b + e - window) // 2
            start = offset
        else:
            # gesture is shorter than the window — centre it
            pad = (window - gesture_len) // 2
            start = max(0, b - pad)
            if start + window > total:
                start = max(0, total - window)

        indices = [min(total - 1, start + i * self.frame_interval) for i in range(self.clip_len)]
        indices_t = torch.tensor(indices, dtype=torch.long)
        meta["frame_indices"] = indices_t
        # Update span relative to the sampled clip (used by IoU loss)
        clip_start = start
        clip_end = min(total - 1, start + window)
        gesture_in_clip_start = max(0.0, (b - clip_start) / max(1, clip_end - clip_start))
        gesture_in_clip_end = min(1.0, (e - clip_start) / max(1, clip_end - clip_start))
        meta["span"] = (gesture_in_clip_start, gesture_in_clip_end)
        return video[indices_t], meta


# ----------------------------------------------------------------------------
# Video-level augmentations
# ----------------------------------------------------------------------------


class SpeedChange:
    """Replicate or skip frames inside the clip to simulate gesture speed."""

    def __init__(self, factor: float, p: float = 0.25) -> None:
        self.factor = factor
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        T = video.shape[0]
        new_T = max(2, int(round(T / self.factor)))
        idx = torch.linspace(0, T - 1, steps=new_T).round().long().clamp(0, T - 1)
        new_video = video[idx]
        # Resample back to T frames so downstream stays consistent
        idx2 = torch.linspace(0, new_T - 1, steps=T).round().long().clamp(0, new_T - 1)
        return new_video[idx2], meta


class RandomDrop:
    """Randomly drop a few frames and replace them by repeating neighbours."""

    def __init__(self, drop_ratio: float = 0.1, p: float = 0.5) -> None:
        self.drop_ratio = drop_ratio
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        T = video.shape[0]
        n_drop = max(1, int(T * self.drop_ratio))
        drop_idx = set(random.sample(range(T), n_drop))
        keep = [i for i in range(T) if i not in drop_idx]
        new_idx = []
        j = 0
        for i in range(T):
            if i in drop_idx:
                new_idx.append(keep[min(j, len(keep) - 1)])
            else:
                new_idx.append(i)
                j = min(j + 1, len(keep) - 1)
        return video[torch.tensor(new_idx, dtype=torch.long)], meta


class RandomAdd:
    """Randomly duplicate frames in-place (slows down portions of the clip)."""

    def __init__(self, add_ratio: float = 0.3, p: float = 0.25) -> None:
        self.add_ratio = add_ratio
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        T = video.shape[0]
        n_add = max(1, int(T * self.add_ratio))
        add_idx = sorted(random.sample(range(T), n_add))
        new_indices = list(range(T))
        for i in add_idx:
            new_indices.insert(i, i)
        new_indices = new_indices[:T]
        return video[torch.tensor(new_indices, dtype=torch.long)], meta


# ----------------------------------------------------------------------------
# Image-level augmentations (applied per-frame consistently across the clip)
# ----------------------------------------------------------------------------


class Resize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        # video is (T, C, H, W)
        return F.interpolate(
            video, size=(self.size, self.size), mode="bilinear", align_corners=False
        ), meta


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        H, W = video.shape[-2:]
        top = (H - self.size) // 2
        left = (W - self.size) // 2
        return video[:, :, top : top + self.size, left : left + self.size], meta


class RandomCrop:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        H, W = video.shape[-2:]
        if H <= self.size or W <= self.size:
            return F.interpolate(
                video, size=(self.size, self.size), mode="bilinear", align_corners=False
            ), meta
        top = random.randint(0, H - self.size)
        left = random.randint(0, W - self.size)
        return video[:, :, top : top + self.size, left : left + self.size], meta


class HorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() < self.p:
            return torch.flip(video, dims=[-1]), meta
        return video, meta


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.05,
        hue: float = 0.05,
        p: float = 0.5,
    ) -> None:
        self.b = brightness
        self.c = contrast
        self.s = saturation
        self.h = hue
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        # Apply same jitter to all frames for temporal consistency
        b = 1.0 + random.uniform(-self.b, self.b)
        c = 1.0 + random.uniform(-self.c, self.c)
        s = 1.0 + random.uniform(-self.s, self.s)
        h = random.uniform(-self.h, self.h)
        out = video
        out = TF.adjust_brightness(out, b)
        out = TF.adjust_contrast(out, c)
        out = TF.adjust_saturation(out, s)
        out = TF.adjust_hue(out.clamp(0, 1), h)
        return out, meta


class GaussianNoise:
    def __init__(self, std_range: tuple[float, float] = (0.001, 0.005), p: float = 0.5) -> None:
        self.std_range = std_range
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        std = random.uniform(*self.std_range)
        return (video + torch.randn_like(video) * std).clamp(0, 1), meta


class Sharpness:
    def __init__(self, factor_range: tuple[float, float] = (0.5, 2.0), p: float = 0.35) -> None:
        self.factor_range = factor_range
        self.p = p

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        factor = random.uniform(*self.factor_range)
        return TF.adjust_sharpness(video, factor), meta


class RandomErasing:
    def __init__(self, p: float = 0.25, scale: tuple[float, float] = (0.02, 0.15)) -> None:
        self.p = p
        self.scale = scale

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return video, meta
        T, C, H, W = video.shape
        area = H * W
        target_area = random.uniform(*self.scale) * area
        aspect = random.uniform(0.3, 3.3)
        eh = int(round((target_area * aspect) ** 0.5))
        ew = int(round((target_area / aspect) ** 0.5))
        if eh >= H or ew >= W or eh < 1 or ew < 1:
            return video, meta
        top = random.randint(0, H - eh)
        left = random.randint(0, W - ew)
        video[:, :, top : top + eh, left : left + ew] = torch.rand(C, eh, ew, device=video.device)
        return video, meta


class Normalize:
    """Apply per-channel normalization (ImageNet defaults)."""

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, video: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        return (video - self.mean.to(video.dtype)) / self.std.to(video.dtype), meta


# ----------------------------------------------------------------------------
# Pipeline builders for ablation
# ----------------------------------------------------------------------------


def build_train_pipeline(
    clip_len: int = 32,
    frame_interval: int = 2,
    image_size: int = 224,
    aug_video: bool = True,
    aug_image: bool = True,
    boundary_shift: int = 5,
    flip: bool = True,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Compose:
    transforms: list[VideoTransform] = [
        SampleFrames(
            clip_len=clip_len,
            frame_interval=frame_interval,
            train_jitter=True,
            boundary_shift=boundary_shift if aug_video else 0,
        ),
    ]
    if aug_video:
        transforms += [
            RandomDrop(drop_ratio=0.1, p=0.5),
            RandomAdd(add_ratio=0.3, p=0.25),
            SpeedChange(factor=2.0, p=0.25),
            SpeedChange(factor=0.5, p=0.25),
        ]
    transforms += [
        Resize(int(image_size * 1.15)),
        RandomCrop(image_size),
    ]
    if aug_image:
        transforms += [
            ColorJitter(brightness=0.1, contrast=0.005, saturation=0.0, hue=0.05, p=0.5),
            GaussianNoise(std_range=(0.001, 0.005), p=0.5),
            Sharpness(factor_range=(0.5, 2.0), p=0.35),
            RandomErasing(p=0.25),
        ]
    if flip and aug_image:
        transforms.append(HorizontalFlip(p=0.5))
    transforms.append(Normalize(mean=mean, std=std))
    return Compose(transforms)


def build_eval_pipeline(
    clip_len: int = 32,
    frame_interval: int = 2,
    image_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Compose:
    return Compose(
        [
            SampleFrames(
                clip_len=clip_len,
                frame_interval=frame_interval,
                train_jitter=False,
                boundary_shift=0,
            ),
            Resize(image_size),
            CenterCrop(image_size),
            Normalize(mean=mean, std=std),
        ]
    )


# ----------------------------------------------------------------------------
# Batch-level: MixUp + CutMix
# ----------------------------------------------------------------------------


def mixup_cutmix(
    pixels: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    p_each: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix to a batch with probability `p_each` each.

    Returns (pixels, soft_labels) where soft_labels has shape (B, num_classes).
    If neither is applied, returns one-hot labels.
    """
    B = pixels.shape[0]
    one_hot = torch.zeros(B, num_classes, device=pixels.device).scatter_(
        1, labels.unsqueeze(1), 1.0
    )

    use_mixup = mixup_alpha > 0 and random.random() < p_each
    use_cutmix = (not use_mixup) and cutmix_alpha > 0 and random.random() < p_each

    if not (use_mixup or use_cutmix):
        return pixels, one_hot

    perm = torch.randperm(B, device=pixels.device)
    if use_mixup:
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        mixed_pixels = lam * pixels + (1 - lam) * pixels[perm]
        mixed_labels = lam * one_hot + (1 - lam) * one_hot[perm]
        return mixed_pixels, mixed_labels

    # CutMix on (T, C, H, W) — same box across all frames
    lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
    _, _, _, H, W = pixels.shape
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    pixels[:, :, :, y1:y2, x1:x2] = pixels[perm][:, :, :, y1:y2, x1:x2]
    lam_eff = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    mixed_labels = lam_eff * one_hot + (1 - lam_eff) * one_hot[perm]
    return pixels, mixed_labels
