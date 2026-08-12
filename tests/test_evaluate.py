"""MetricSet fields are constructed with explicit keywords everywhere in
this codebase (Trap 3) so a missing field surfaces at the call site, not
deep inside a training run."""
import numpy as np
import pytest

from src.models.evaluate import (
    MetricSet,
    compute_metrics,
    confusion_counts,
    expected_calibration_error,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)


@pytest.fixture
def toy_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.clip(y_true * 0.5 + rng.random(500) * 0.5, 0, 1)
    return y_true, y_prob


def test_metric_set_requires_all_keywords():
    with pytest.raises(TypeError):
        MetricSet(model_name="x")  # type: ignore[call-arg]


def test_compute_metrics_returns_populated_metric_set(toy_predictions):
    y_true, y_prob = toy_predictions
    metrics = compute_metrics("toy", y_true, y_prob, threshold=0.5)
    assert metrics.n_samples == 500
    assert metrics.n_positive == int(y_true.sum())
    assert 0.0 <= metrics.average_precision <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_confusion_counts_sum_to_n(toy_predictions):
    y_true, y_prob = toy_predictions
    counts = confusion_counts(y_true, y_prob, threshold=0.5)
    assert counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"] == len(y_true)


def test_confusion_counts_known_case():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1])
    counts = confusion_counts(y_true, y_prob, threshold=0.5)
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_reliability_curve_uses_quantile_bins(toy_predictions):
    y_true, y_prob = toy_predictions
    curve = reliability_curve(y_true, y_prob, n_bins=5)
    assert curve["count"].sum() == len(y_true)
    assert (curve["count"] > 0).all()


def test_expected_calibration_error_zero_for_perfect_calibration():
    y_true = np.array([1] * 100 + [0] * 100)
    y_prob = np.array([0.9] * 100 + [0.1] * 100)
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece < 0.15


def test_curve_points_shapes(toy_predictions):
    y_true, y_prob = toy_predictions
    pr = pr_curve_points(y_true, y_prob)
    roc = roc_curve_points(y_true, y_prob)
    assert {"precision", "recall", "threshold"} <= set(pr.columns)
    assert {"fpr", "tpr", "threshold"} <= set(roc.columns)
