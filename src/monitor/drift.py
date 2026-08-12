"""Drift monitoring from scratch: PSI for numerics, Jensen-Shannon for
categoricals. No external library -- the arithmetic is simple enough to
defend under viva questioning, and it is one fewer dependency in a
constrained free-tier container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

PSI_STABLE_MAX = 0.10
PSI_MONITOR_MAX = 0.25
JS_STABLE_MAX = 0.05
JS_MONITOR_MAX = 0.15


def compute_reference_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin edges from the REFERENCE distribution only.

    Recomputing edges from the current period would let it influence its
    own yardstick and understate drift.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        value = float(edges[0]) if len(edges) else 0.0
        edges = np.array([value - 1e-9, value + 1e-9])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = compute_reference_bins(reference, n_bins=n_bins)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(len(reference), 1) + eps
    cur_pct = cur_counts / max(len(current), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def jensen_shannon_categorical(reference: pd.Series, current: pd.Series, base: float = 2.0) -> float:
    """Symmetric, bounded [0,1], well-behaved when a level is absent from
    one side entirely -- unlike plain KL divergence, which blows up there."""
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    p = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)
    q = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask]) / np.log(base)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass
class DriftReport:
    numeric_psi: dict[str, float]
    categorical_js: dict[str, float]
    verdict: str


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> DriftReport:
    numeric_psi = {
        column: population_stability_index(
            reference[column].dropna().to_numpy(), current[column].dropna().to_numpy()
        )
        for column in numeric_columns
    }
    categorical_js = {
        column: jensen_shannon_categorical(reference[column], current[column])
        for column in categorical_columns
    }

    worst_psi = max(numeric_psi.values(), default=0.0)
    worst_js = max(categorical_js.values(), default=0.0)

    if worst_psi > PSI_MONITOR_MAX or worst_js > JS_MONITOR_MAX:
        verdict = "RETRAIN RECOMMENDED"
    elif worst_psi > PSI_STABLE_MAX or worst_js > JS_STABLE_MAX:
        verdict = "MONITOR"
    else:
        verdict = "STABLE"

    return DriftReport(numeric_psi=numeric_psi, categorical_js=categorical_js, verdict=verdict)
