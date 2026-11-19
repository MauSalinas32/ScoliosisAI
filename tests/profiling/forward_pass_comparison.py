"""
Forward-pass comparison on the packaged model: model.predict() vs eager call vs compiled tf.function.

    python tests/profiling/forward_pass_comparison.py --output evidence/forward_pass.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config.settings import MODEL_DIR  # noqa: E402


def timed(fn, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": round(statistics.mean(samples), 2), "median_ms": round(statistics.median(samples), 2),
            "p95_ms": round(float(np.percentile(samples, 95)), 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.1.0")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    model = tf.keras.models.load_model(str(MODEL_DIR / f"scoliosis_classifier_v{args.version}" / "model.keras"), compile=False)
    x = np.random.default_rng(0).random((1, 224, 224, 3)).astype("float32")
    compiled = tf.function(lambda t: model(t, training=False), reduce_retracing=True)
    for fn in (lambda: model.predict(x, verbose=0), lambda: model(x, training=False), lambda: compiled(x)):
        fn()  # warm-up / tracing

    result = {
        "model_version": args.version, "repeats": args.repeats, "cpu_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "model_predict": timed(lambda: model.predict(x, verbose=0), args.repeats),
        "eager_call": timed(lambda: model(x, training=False), args.repeats),
        "tf_function": timed(lambda: compiled(x).numpy(), args.repeats),
    }
    result["speedup_tf_function_vs_predict"] = round(result["model_predict"]["mean_ms"] / result["tf_function"]["mean_ms"], 1)
    result["speedup_tf_function_vs_eager"] = round(result["eager_call"]["mean_ms"] / result["tf_function"]["mean_ms"], 1)
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
