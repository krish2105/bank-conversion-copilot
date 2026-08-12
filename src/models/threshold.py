"""Cost-optimal and capacity-constrained decision thresholds.

0.5 assumes a false positive and a false negative cost the same. Here one
costs 8 EUR and the other a 120 EUR opportunity, so the threshold that
maximises business value sits nowhere near 0.5 -- it sits near the
break-even probability, adjusted for how well-calibrated the model
actually is (which is why this is a grid search, not the analytic 8/120).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import CostMatrix, DEFAULT_COST_MATRIX


@dataclass
class ThresholdResult:
    threshold: float
    expected_net_value: float
    realized_net_value: float
    realized_net_value_at_default: float
    realized_net_value_call_everyone: float
    uplift_vs_default: float
    uplift_vs_call_everyone: float


def realized_net_value(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cost_matrix: CostMatrix
) -> float:
    y_true = np.asarray(y_true)
    call_mask = np.asarray(y_prob) >= threshold
    n_called = int(call_mask.sum())
    n_converted = int(y_true[call_mask].sum())
    return n_converted * cost_matrix.revenue_per_subscription - n_called * cost_matrix.cost_per_call


def search_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: CostMatrix = DEFAULT_COST_MATRIX,
    n_grid: int = 201,
) -> ThresholdResult:
    grid = np.linspace(0.0, 1.0, n_grid)
    expected_values = np.array([cost_matrix.net_value(y_prob, float(t)) for t in grid])
    best_idx = int(np.argmax(expected_values))
    best_threshold = float(grid[best_idx])

    realized_best = realized_net_value(y_true, y_prob, best_threshold, cost_matrix)
    realized_default = realized_net_value(y_true, y_prob, 0.5, cost_matrix)
    realized_everyone = realized_net_value(y_true, y_prob, 0.0, cost_matrix)

    return ThresholdResult(
        threshold=best_threshold,
        expected_net_value=float(expected_values[best_idx]),
        realized_net_value=realized_best,
        realized_net_value_at_default=realized_default,
        realized_net_value_call_everyone=realized_everyone,
        uplift_vs_default=realized_best - realized_default,
        uplift_vs_call_everyone=realized_best - realized_everyone,
    )


def capacity_constrained_threshold(y_prob: np.ndarray, capacity_fraction: float) -> float:
    """Threshold that calls roughly the top capacity_fraction of prospects.

    Real call centres are constrained by agent hours, not list quality, so
    this mode ignores cost economics entirely and just staffs the queue.
    """
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    return float(np.quantile(np.asarray(y_prob), 1 - capacity_fraction))
