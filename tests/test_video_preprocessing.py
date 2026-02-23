from __future__ import annotations

import numpy as np

from app.preprocessing.video import (
    MVITV2_CLIP_LEN,
    prepare_mvitv2_small_32_2_clips,
    prepare_mvitv2_small_32_2_input,
)


def test_prepare_mvitv2_small_32_2_input_selects_every_second_frame() -> None:
    frames = np.stack(
        [
            np.full((120, 160, 3), fill_value=i, dtype=np.uint8)
            for i in range(64)
        ],
        axis=0,
    )

    prepared = prepare_mvitv2_small_32_2_input(frames)

    assert prepared.shape == (32, 3, 224, 224)
    assert prepared.dtype == np.float32

    restored = (prepared[:, 0, 112, 112] * 0.225 + 0.45) * 255.0
    expected = np.arange(0, 64, 2, dtype=np.float32)
    np.testing.assert_allclose(restored, expected, atol=1.0)


def test_prepare_mvitv2_small_32_2_input_keeps_aspect_ratio_with_padding() -> None:
    frames = np.full((2, 100, 200, 3), fill_value=255, dtype=np.uint8)

    prepared = prepare_mvitv2_small_32_2_input(frames)

    # Letterbox bars on top/bottom become black before normalization.
    np.testing.assert_allclose(prepared[0, :, 0, 112], np.array([-2.0, -2.0, -2.0]))
    # Center pixels stay from original white frame.
    expected_white = (1.0 - 0.45) / 0.225
    np.testing.assert_allclose(
        prepared[0, :, 112, 112],
        np.array([expected_white, expected_white, expected_white]),
        atol=1e-4,
    )


def test_prepare_mvitv2_small_32_2_clips_for_long_video_adds_tail_window() -> None:
    frames = np.stack(
        [
            np.full((96, 128, 3), fill_value=i, dtype=np.uint8)
            for i in range(200)
        ],
        axis=0,
    )

    batch = prepare_mvitv2_small_32_2_clips(frames, hop_size=16)

    assert batch.clips.shape == (6, MVITV2_CLIP_LEN, 3, 224, 224)
    np.testing.assert_array_equal(batch.clip_starts, np.array([0, 16, 32, 48, 64, 68]))
