"""
Input-Sensitivity Probe
========================
A trained classifier that ignores its input (for example one that always answers
"C-curve") can still report a plausible accuracy when one class dominates the test
set. This probe detects that failure *without needing the dataset*: it feeds a
fixed set of very different synthetic images and checks that the output
probabilities actually change.

It is a sanity check, NOT a replacement for the evaluation on the held-out test
split (``ai/evaluate_model.py``). It is used in three places:

  1. ``InferenceService.load_model`` refuses to serve an artefact that fails it
     (even if ``metadata.json`` says ``"approved": true``);
  2. ``ai/train_model.py`` / ``passes_quality_gate`` reject a model that fails it;
  3. ``python ai/model_probe.py --version 1.0.0`` reproduces, from the repository,
     the probe table reported in the documentation.

Definition: the model is *input-insensitive* when, for every class, the
**trimmed range** of its probability (largest minus smallest value after dropping
the single most extreme image on each side, so one outlier input cannot hide a
constant model) is below ``MIN_PROBABILITY_RANGE`` across the probe images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# Allow `python ai/model_probe.py` (script) as well as `import ai.model_probe`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SIZE = (224, 224)
MIN_PROBABILITY_RANGE = 0.10


def build_probe_images(seed: int = 0) -> Tuple[np.ndarray, List[str]]:
    """Deterministic, very different synthetic inputs in the model contract [0, 1]."""
    rng = np.random.default_rng(seed)
    h, w = IMAGE_SIZE
    xs = np.arange(w)

    def spine(shift_px: float) -> np.ndarray:
        img = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            centre = w / 2 + shift_px * np.sin(y / h * np.pi)
            img[y] = np.clip(1.0 - np.abs(xs - centre) / 30.0, 0.0, 1.0) * 0.8
        return img

    gradient = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    checker = (((np.arange(h)[:, None] // 16) + (np.arange(w)[None, :] // 16)) % 2).astype(np.float32)
    gray_images = {
        "black": np.zeros((h, w), np.float32),
        "white": np.ones((h, w), np.float32),
        "uniform noise": rng.uniform(0, 1, (h, w)).astype(np.float32),
        "horizontal gradient": gradient,
        "checkerboard": checker,
        "straight spine": spine(0),
        "spine +30 px (C)": spine(30),
        "spine -30 px (C)": spine(-30),
    }
    names = list(gray_images)
    batch = np.stack([np.repeat(g[..., None], 3, axis=-1) for g in gray_images.values()])
    return batch.astype(np.float32), names


def probe_probabilities(predict_fn: Callable[[np.ndarray], np.ndarray], class_names: Sequence[str]) -> Dict:
    """Run the probe through ``predict_fn`` (batch -> (N, K) probabilities)."""
    batch, names = build_probe_images()
    probs = np.asarray(predict_fn(batch), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(names):
        raise ValueError(f"predict_fn must return (N={len(names)}, K) probabilities, got {probs.shape}")
    ordered = np.sort(probs, axis=0)
    full_ranges = ordered[-1] - ordered[0]
    ranges = ordered[-2] - ordered[1]            # trimmed: drop the most extreme image on each side
    argmax = probs.argmax(axis=1)
    max_range = float(ranges.max())
    return {
        "images": names,
        "probabilities": [[round(float(p), 4) for p in row] for row in probs],
        "class_names": list(class_names),
        "probability_full_range_per_class": [round(float(r), 4) for r in full_ranges],
        "probability_trimmed_range_per_class": [round(float(r), 4) for r in ranges],
        "max_probability_range": round(max_range, 4),
        "distinct_predicted_classes": int(len(set(argmax.tolist()))),
        "min_required_range": MIN_PROBABILITY_RANGE,
        "input_insensitive": bool(max_range < MIN_PROBABILITY_RANGE),
    }


def main() -> int:
    from config.settings import CLASS_LABELS, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, MODEL_DIR

    parser = argparse.ArgumentParser(description="Input-sensitivity probe of a model artefact")
    parser.add_argument("--version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--no-write", action="store_true", help="do not write probe_results.json")
    args = parser.parse_args()

    artifact_dir = MODEL_DIR / f"{DEFAULT_MODEL_NAME}_v{args.version}"
    model_path = artifact_dir / "model.keras"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    result = probe_probabilities(lambda x: model(x, training=False).numpy(), [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))])

    print(f"Probe of {DEFAULT_MODEL_NAME} v{args.version}")
    print(f"{'image':<22}" + "".join(f"{c[:18]:>20}" for c in result["class_names"]))
    for name, row in zip(result["images"], result["probabilities"]):
        print(f"{name:<22}" + "".join(f"{p:>20.4f}" for p in row))
    print(f"full range per class:    {result['probability_full_range_per_class']}")
    print(f"trimmed range per class: {result['probability_trimmed_range_per_class']} (a responsive model needs >= {MIN_PROBABILITY_RANGE} for at least one class)")
    print("RESULT:", "INPUT-INSENSITIVE (rejected)" if result["input_insensitive"] else "responds to its input")

    if not args.no_write:
        (artifact_dir / "probe_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {artifact_dir / 'probe_results.json'}")
    return 1 if result["input_insensitive"] else 0


if __name__ == "__main__":
    sys.exit(main())
