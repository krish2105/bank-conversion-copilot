"""Threshold economics and drift metrics -- the two places a wrong number
costs real money or masks real decay."""

import numpy as np
import pandas as pd
import pytest

from src.config import CostMatrix
from src.models.threshold import (
    ThresholdResult,
    capacity_constrained_threshold,
    realized_net_value,
    search_cost_optimal_threshold,
)
from src.monitor.drift import (
    DriftReport,
    compute_drift_report,
    jensen_shannon_categorical,
    population_stability_index,
)


def test_realized_net_value_known_case():
    cost_matrix = CostMatrix(cost_per_call=8.0, revenue_per_subscription=120.0)
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1])
    value = realized_net_value(y_true, y_prob, threshold=0.5, cost_matrix=cost_matrix)
    assert value == pytest.approx(120 - 2 * 8)


def test_search_threshold_is_on_grid_and_beats_default_on_its_own_metric():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=2000)
    y_prob = np.clip(y_true * 0.4 + rng.random(2000) * 0.3, 0, 1)
    cost_matrix = CostMatrix()
    result = search_cost_optimal_threshold(
        y_true, y_prob, cost_matrix=cost_matrix, n_grid=201
    )
    assert isinstance(result, ThresholdResult)
    grid = np.linspace(0.0, 1.0, 201)
    assert np.isclose(grid, result.threshold).any()
    value_at_default = cost_matrix.net_value(y_prob, 0.5)
    assert result.expected_net_value >= value_at_default - 1e-9


def test_capacity_constrained_threshold_hits_requested_share():
    rng = np.random.default_rng(2)
    y_prob = rng.random(10_000)
    threshold = capacity_constrained_threshold(y_prob, capacity_fraction=0.10)
    called_share = (y_prob >= threshold).mean()
    assert called_share == pytest.approx(0.10, abs=0.01)


def test_capacity_constrained_threshold_rejects_bad_fraction():
    with pytest.raises(ValueError):
        capacity_constrained_threshold(np.array([0.1, 0.9]), capacity_fraction=0.0)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    psi = population_stability_index(reference, reference.copy())
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_mean_shift():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 2000)
    small_shift = reference + 0.2
    large_shift = reference + 2.0
    psi_small = population_stability_index(reference, small_shift)
    psi_large = population_stability_index(reference, large_shift)
    assert 0.0 < psi_small < psi_large


def test_psi_handles_constant_reference_without_raising():
    reference = np.full(500, 7.0)
    current_same = np.full(500, 7.0)
    psi_same = population_stability_index(reference, current_same)
    assert psi_same == pytest.approx(0.0, abs=1e-6)
    current_different = np.full(500, 99.0)
    psi_different = population_stability_index(reference, current_different)
    assert np.isfinite(psi_different)


def test_jensen_shannon_bounded_and_zero_for_identical():
    ref = pd.Series(["a", "a", "b", "c"] * 50)
    js_identical = jensen_shannon_categorical(ref, ref.copy())
    assert js_identical == pytest.approx(0.0, abs=1e-9)

    disjoint = pd.Series(["x", "y", "z"] * 50)
    js_disjoint = jensen_shannon_categorical(ref, disjoint)
    assert 0.0 <= js_disjoint <= 1.0
    assert js_disjoint > js_identical


def test_compute_drift_report_verdict_stable_for_identical_frames():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "num": rng.normal(0, 1, 1000),
            "cat": rng.choice(["a", "b", "c"], size=1000),
        }
    )
    report = compute_drift_report(
        frame, frame.copy(), numeric_columns=["num"], categorical_columns=["cat"]
    )
    assert isinstance(report, DriftReport)
    assert report.verdict == "STABLE"


def test_compute_drift_report_verdict_flags_large_shift():
    rng = np.random.default_rng(6)
    reference = pd.DataFrame(
        {
            "num": rng.normal(0, 1, 1000),
            "cat": rng.choice(["a", "b", "c"], size=1000, p=[0.8, 0.1, 0.1]),
        }
    )
    current = pd.DataFrame(
        {
            "num": rng.normal(6, 1, 1000),
            "cat": rng.choice(["a", "b", "c"], size=1000, p=[0.1, 0.1, 0.8]),
        }
    )
    report = compute_drift_report(
        reference, current, numeric_columns=["num"], categorical_columns=["cat"]
    )
    assert report.verdict in {"MONITOR", "RETRAIN RECOMMENDED"}
