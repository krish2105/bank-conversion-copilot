"""Evaluation utilities shared by training, the report generator, and the
model card. Every metric constructor uses explicit keywords (Trap 3): when
this dataclass gains a field, every call site breaks loudly at the call,
not silently deep inside a training run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class MetricSet:
    model_name: str
    average_precision: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    n_positive: int


def compute_metrics(
    model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> MetricSet:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return MetricSet(
        model_name=model_name,
        average_precision=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        n_samples=int(len(y_true)),
        n_positive=int(np.sum(y_true)),
    )


def confusion_counts(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, int]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Quantile-binned, not uniform: most predictions cluster below 0.2 on
    this dataset, and uniform bins would leave the upper bins empty."""
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    frame["bin"] = pd.qcut(frame["y_prob"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        mean_predicted=("y_prob", "mean"),
        mean_observed=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return grouped.reset_index()


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    weights = curve["count"] / curve["count"].sum()
    gaps = (curve["mean_predicted"] - curve["mean_observed"]).abs()
    return float((weights * gaps).sum())


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "threshold": thresholds}
    )


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
