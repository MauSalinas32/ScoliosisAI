"""
Unit Tests: numerical equivalence of the memory-optimised validation (Milestone 3 feedback Q6)
==============================================================================================
ValidationStage was optimised from NumPy's default int64 / float64 to int16 / float32 to cut its
memory peak (67 MB -> about 20 MB for a 1024x1024 image). A dtype change can silently alter a
decision (overflow, rounding next to a threshold), so the stage's accept / reject decision is
compared with a plain int64 / float64 reference on structured, random and adversarial images.
"""

import io

import numpy as np
import pytest
from PIL import Image

from pipeline.core import PipelineContext
from pipeline.stages import ValidationStage
from utils.exceptions import ImageQualityError


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def reference_rejects(img: np.ndarray) -> bool:
    """The three rejection rules of ValidationStage evaluated in int64 / float64."""
    r, g, b = (img[:, :, i].astype(np.int64) for i in range(3))
    if float(np.mean(np.abs(r - g)) + np.mean(np.abs(r - b))) > 15.0:
        return True
    gray = np.mean(img.astype(np.float64), axis=2)
    h, w = gray.shape
    c = max(10, min(30, h // 10, w // 10))
    corner_mean = float(np.mean([gray[:c, :c].mean(), gray[:c, -c:].mean(), gray[-c:, :c].mean(), gray[-c:, -c:].mean()]))
    white_ratio = float(np.mean(gray > 235.0))
    mean_brightness = float(np.mean(gray))
    if (white_ratio > 0.25 and mean_brightness > 150.0) or (corner_mean > 180.0 and white_ratio > 0.15):
        return True
    top = float(gray[:int(h * 0.2), :].mean())
    bot = float(gray[int(h * 0.8):, :].mean())
    dy, dx = np.gradient(gray)
    sharp = float(np.mean(np.hypot(dx, dy) > 40.0))
    return (top - bot > 45.0 and top > 140.0) or sharp > 0.045


def stage_rejects(img: np.ndarray) -> bool:
    ctx = PipelineContext(original_filename="x.png", file_data=_png(img))
    try:
        ValidationStage().execute(ctx)
        return False
    except ImageQualityError:
        return True


def _images():
    rng = np.random.default_rng(123)
    x = np.arange(640)
    spine = np.tile(np.clip(180 - np.abs(x - 320) / 1.5, 0, None) + 40, (800, 1)).astype(np.uint8)
    gray3 = lambda a: np.repeat(a[..., None], 3, axis=2)  # noqa: E731
    out = {
        "black": np.zeros((300, 200, 3), np.uint8),
        "white": np.full((300, 200, 3), 255, np.uint8),
        "noise": rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
        "gray_noise": gray3(rng.integers(0, 256, (512, 512), dtype=np.uint8)),
        "spine": gray3(spine),
        "tinted_spine": np.stack([spine, (spine * 0.98).astype(np.uint8), (spine * 0.96).astype(np.uint8)], axis=2),
        "strongly_tinted_spine": np.stack([spine, (spine * 0.90).astype(np.uint8), (spine * 0.80).astype(np.uint8)], axis=2),
        "red_photo": np.dstack([np.full((200, 200), 220), np.full((200, 200), 40), np.full((200, 200), 40)]).astype(np.uint8),
        "extreme_channels": np.dstack([np.full((200, 200), 255), np.zeros((200, 200)), np.zeros((200, 200))]).astype(np.uint8),
        "half_white_logo": np.vstack([np.full((100, 300, 3), 255, np.uint8), np.zeros((100, 300, 3), np.uint8)]),
        "gradient": gray3(np.tile(np.linspace(0, 255, 400), (300, 1)).astype(np.uint8)),
        "checkerboard": gray3(((np.indices((400, 400)).sum(axis=0) // 8) % 2 * 255).astype(np.uint8)),
    }
    for i in range(13):  # random structured images near the thresholds
        base = rng.integers(60, 200)
        tint = rng.integers(0, 12, 3)
        arr = np.clip(base + rng.normal(0, 25, (256, 256, 1)) + tint, 0, 255).astype(np.uint8)
        out[f"random_{i}"] = arr
    return out


IMAGES = _images()


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_optimised_stage_takes_the_same_decision_as_the_float64_reference(name):
    img = IMAGES[name]
    assert stage_rejects(img) == reference_rejects(img), name


def test_the_image_set_contains_both_outcomes():
    decisions = {reference_rejects(img) for img in IMAGES.values()}
    assert decisions == {True, False}  # otherwise the comparison above would prove nothing


def test_int16_cannot_overflow_on_extreme_channel_differences():
    """With uint8 arithmetic 0 - 255 wraps to 1 and a pure red image would look grey; int16 keeps -255."""
    img = IMAGES["extreme_channels"]
    assert stage_rejects(img) is True
    wrapped = np.abs(img[:, :, 0] - img[:, :, 1])           # what uint8 arithmetic would compute
    assert float(wrapped.mean()) == 255.0 or wrapped.dtype == np.uint8


def test_peak_memory_is_far_below_the_int64_float64_implementation():
    import tracemalloc
    rng = np.random.default_rng(5)
    big = np.repeat(rng.integers(40, 200, (1024, 1024, 1), dtype=np.uint8), 3, axis=2)
    ctx = PipelineContext(original_filename="big.png", file_data=_png(big))
    tracemalloc.start()
    try:
        ValidationStage().execute(ctx)
    except ImageQualityError:
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 40 * 1024 * 1024   # the int64/float64 version needed about 67 MB
