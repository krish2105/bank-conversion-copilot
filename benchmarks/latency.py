"""Latency, throughput, cold-start, and memory benchmarking -- the
concrete evidence behind the "evaluate and optimize AI software
performance" learning outcome most teams skip.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src import config
from src.inference.predict import load_bundle


def _sample_row() -> pd.DataFrame:
    return pd.DataFrame([{spec.name: spec.default for spec in config.FIELD_SPECS}])


def measure_cold_start(model_path: Path) -> float:
    """Artefact deserialisation time, measured separately: this is the
    dominant term in first-request latency on free Space hardware, which
    is why app.py warms the bundle at import time instead of lazily."""
    start = time.perf_counter()
    load_bundle(model_path=model_path)
    return time.perf_counter() - start


def measure_single_row_latency(bundle: dict, n_repeats: int = 200) -> dict:
    row = _sample_row()
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        bundle["model"].predict_proba(row)
        durations.append(time.perf_counter() - start)
    array = np.array(durations)
    return {
        "p50_ms": float(np.percentile(array, 50) * 1000),
        "p95_ms": float(np.percentile(array, 95) * 1000),
        "p99_ms": float(np.percentile(array, 99) * 1000),
    }


def measure_batch_throughput(
    bundle: dict, batch_sizes: tuple[int, ...] = (1, 10, 100, 1000, 10000)
) -> dict:
    row = _sample_row().iloc[0].to_dict()
    results = {}
    for size in batch_sizes:
        frame = pd.DataFrame([row] * size)
        start = time.perf_counter()
        bundle["model"].predict_proba(frame)
        elapsed = time.perf_counter() - start
        results[str(size)] = {
            "elapsed_s": elapsed,
            "rows_per_second": size / elapsed if elapsed > 0 else float("inf"),
        }
    return results


def measure_peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmarks(model_path: Path | None = None) -> dict:
    model_path = model_path or config.MODEL_PATH
    cold_start_s = measure_cold_start(model_path)
    bundle = load_bundle(model_path=model_path)
    return {
        "cold_start_s": cold_start_s,
        "single_row_latency": measure_single_row_latency(bundle),
        "batch_throughput": measure_batch_throughput(bundle),
        "peak_rss_mb": measure_peak_rss_mb(),
        "artifact_size_bytes": model_path.stat().st_size,
        "measured_at": datetime.now(UTC).isoformat(),
    }


def _benchmark_onnx(bundle: dict) -> dict:
    """joblib vs ONNX Runtime, scoped to the fitted classifier stage only.

    Converting the full ColumnTransformer + OneHotEncoder +
    CalibratedClassifierCV chain has known skl2onnx compatibility gaps --
    exactly the install risk BUILD_PROMPT.md flags. This degrades to an
    explicit unavailable result rather than crashing the run when the
    optional packages aren't installed.
    """
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return {
            "available": False,
            "note": "Install skl2onnx and onnxruntime to run this comparison.",
        }

    pipeline = bundle["model"].calibrated_classifiers_[0].estimator.estimator
    classifier = pipeline.named_steps["classifier"]
    transformed = pipeline[:-1].transform(_sample_row())
    n_features = transformed.shape[1]

    onnx_model = convert_sklearn(
        classifier, initial_types=[("input", FloatTensorType([None, n_features]))]
    )
    session = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_input = transformed.astype(np.float32)

    n_repeats = 200
    sklearn_durations, onnx_durations = [], []
    for _ in range(n_repeats):
        start = time.perf_counter()
        classifier.predict_proba(transformed)
        sklearn_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        session.run(None, {"input": onnx_input})
        onnx_durations.append(time.perf_counter() - start)

    return {
        "available": True,
        "scope": "classifier stage only (post feature-pipeline transform)",
        "sklearn_classifier_p50_ms": float(np.percentile(sklearn_durations, 50) * 1000),
        "onnx_p50_ms": float(np.percentile(onnx_durations, 50) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Optional: also benchmark an ONNX export of the classifier stage.",
    )
    args = parser.parse_args()

    payload = run_benchmarks()
    if args.onnx:
        payload["onnx"] = _benchmark_onnx(load_bundle())

    output_dir = config.REPO_ROOT / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = payload["measured_at"].replace(":", "-")
    (output_dir / f"results_{timestamp}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
