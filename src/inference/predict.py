"""The serving layer. Both app.py and streamlit_app.py import only this
module -- there is exactly one implementation of "what does the model
say", so the two UIs cannot disagree, and these tests exercise the shared
path rather than either UI's glue code."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.explain.shap_engine import explain_prediction
from src.models.threshold import capacity_constrained_threshold


@dataclass
class ScoreResult:
    probability: float
    verdict: str
    threshold: float
    expected_value_eur: float
    confidence_band: str
    drivers: list[tuple[str, float]]
    explanation_method: str
    explanation_reliable: bool
    explanation_note: str


@dataclass
class BatchResult:
    scored: pd.DataFrame
    warnings: list[str]


def load_bundle(model_path: Path | None = None) -> dict:
    path = model_path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.models.train` "
            f"(or `make train`) before scoring."
        )
    import joblib

    bundle = joblib.load(path)
    installed = sklearn.__version__
    recorded = bundle.get("sklearn_version")
    if recorded and recorded != installed:
        warnings.warn(
            f"Model was trained with scikit-learn {recorded}, but {installed} "
            f"is installed. Predictions may be unreliable.",
            stacklevel=2,
        )
    return bundle


def _confidence_band(probability: float, threshold: float) -> str:
    distance = abs(probability - threshold)
    if distance < 0.02:
        return "Borderline"
    if distance < 0.08:
        return "Moderate"
    return "Clear"


def _fields_to_frame(fields: dict) -> pd.DataFrame:
    row = {spec.name: fields.get(spec.name, spec.default) for spec in config.FIELD_SPECS}
    return pd.DataFrame([row])[list(config.APP_FIELD_ORDER)]


def score_one(
    bundle: dict,
    fields: dict,
    cost_matrix: config.CostMatrix = config.DEFAULT_COST_MATRIX,
) -> ScoreResult:
    row = _fields_to_frame(fields)
    probability = float(bundle["model"].predict_proba(row)[:, 1][0])
    threshold = float(bundle["threshold"])
    verdict = "CALL" if probability >= threshold else "SKIP"
    expected_value = (
        probability * cost_matrix.revenue_per_subscription - cost_matrix.cost_per_call
    )
    explanation = explain_prediction(bundle, row)
    return ScoreResult(
        probability=probability,
        verdict=verdict,
        threshold=threshold,
        expected_value_eur=expected_value,
        confidence_band=_confidence_band(probability, threshold),
        drivers=explanation.contributions,
        explanation_method=explanation.method,
        explanation_reliable=explanation.reliable,
        explanation_note=explanation.note,
    )


def _tolerant_ingest(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Never let an operations user see a stack trace: fill what's missing,
    drop what's leaky, and report every assumption made along the way."""
    notes: list[str] = []
    clean = frame.copy()

    leaky_present = [c for c in config.LEAKAGE_DENYLIST if c in clean.columns]
    if leaky_present:
        clean = clean.drop(columns=leaky_present)
        notes.append(f"Dropped leaky column(s) {leaky_present} — never used for scoring.")

    for spec in config.FIELD_SPECS:
        if spec.name not in clean.columns:
            clean[spec.name] = spec.default
            notes.append(
                f"Column '{spec.name}' missing — filled with default {spec.default!r}."
            )
        elif spec.kind == "categorical":
            normalized = clean[spec.name].astype(str).str.strip().str.lower()
            unseen = sorted(set(normalized) - set(spec.levels))
            if unseen:
                notes.append(
                    f"Column '{spec.name}' contains unseen value(s) {unseen} — "
                    f"mapped to the model's infrequent-category bucket."
                )

    extra = [c for c in clean.columns if c not in config.APP_FIELD_ORDER]
    if extra:
        clean = clean.drop(columns=extra)
        notes.append(f"Ignored unexpected column(s) {extra}.")

    return clean[list(config.APP_FIELD_ORDER)], notes


def score_batch(
    bundle: dict,
    frame: pd.DataFrame,
    capacity_fraction: float | None = None,
    cost_matrix: config.CostMatrix = config.DEFAULT_COST_MATRIX,
) -> BatchResult:
    clean, notes = _tolerant_ingest(frame)

    probabilities = bundle["model"].predict_proba(clean)[:, 1]
    threshold = float(bundle["threshold"])
    if capacity_fraction is not None:
        # Rank-based, not a >= threshold comparison: a threshold cutoff is
        # vulnerable to ties (isotonic calibration on a small validation
        # set can collapse many rows onto the same probability plateau),
        # which could call far more than the requested share. "Capacity"
        # means an exact headcount the call centre can staff, so select
        # exactly that many top-ranked rows instead.
        threshold = capacity_constrained_threshold(probabilities, capacity_fraction)
        n_call = max(1, round(capacity_fraction * len(probabilities)))
        order = np.argsort(-probabilities, kind="stable")
        call_mask = np.zeros(len(probabilities), dtype=bool)
        call_mask[order[:n_call]] = True
    else:
        call_mask = probabilities >= threshold

    result = clean.copy()
    result["probability"] = probabilities
    result["verdict"] = np.where(call_mask, "CALL", "SKIP")
    result["expected_value_eur"] = (
        probabilities * cost_matrix.revenue_per_subscription - cost_matrix.cost_per_call
    )
    result["confidence_band"] = [_confidence_band(p, threshold) for p in probabilities]
    result = result.sort_values("probability", ascending=False).reset_index(drop=True)
    result["priority_rank"] = np.arange(1, len(result) + 1)

    return BatchResult(scored=result, warnings=notes)
