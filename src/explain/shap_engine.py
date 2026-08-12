"""Three-stage degrading explainer.

SHAP's TreeExplainer support for sklearn's HistGradientBoostingClassifier
has varied by version, and a deployed app must not crash trying to explain
a prediction. Each stage after the first is a documented downgrade, not a
silent one: reliable=False always ships with a note the UI must show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExplanationResult:
    method: str
    reliable: bool
    contributions: list[tuple[str, float]]
    note: str


def _normalize_shap_shape(raw) -> np.ndarray:
    """SHAP returns (n, f), (n, f, 2), or a two-element [neg, pos] list
    depending on version and model type. Always resolve to (n, f) for the
    positive class."""
    if isinstance(raw, list):
        return np.asarray(raw[-1])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def _unwrap_pipeline(calibrated):
    try:
        frozen = calibrated.calibrated_classifiers_[0].estimator
        return frozen.estimator
    except (AttributeError, IndexError):
        return None


def _try_shap(pipeline, transformed: np.ndarray, feature_names: list[str]):
    try:
        import shap
    except ImportError:
        return None
    classifier = pipeline.named_steps.get("classifier")
    try:
        explainer = shap.TreeExplainer(classifier)
        raw = explainer.shap_values(transformed)
        values = _normalize_shap_shape(raw)
        contributions = list(zip(feature_names, values[0].tolist()))
        contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return contributions
    except Exception:
        return None


def _try_linear(pipeline, transformed: np.ndarray, feature_names: list[str]):
    classifier = pipeline.named_steps.get("classifier")
    coefs = getattr(classifier, "coef_", None)
    if coefs is None:
        return None
    coefs = np.ravel(coefs)
    if len(coefs) != transformed.shape[1]:
        return None
    contributions = list(zip(feature_names, (coefs * transformed[0]).tolist()))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions


def explain_prediction(bundle: dict, x_row: pd.DataFrame) -> ExplanationResult:
    calibrated = bundle["model"]
    feature_names = bundle["feature_names"]
    pipeline = _unwrap_pipeline(calibrated)

    transformed = None
    if pipeline is not None:
        try:
            transformed = pipeline[:-1].transform(x_row)
        except Exception:
            transformed = None

    if transformed is not None:
        shap_contribs = _try_shap(pipeline, transformed, feature_names)
        if shap_contribs is not None:
            return ExplanationResult(
                method="shap", reliable=True, contributions=shap_contribs,
                note="Exact per-prediction SHAP contributions.",
            )

        linear_contribs = _try_linear(pipeline, transformed, feature_names)
        if linear_contribs is not None:
            return ExplanationResult(
                method="linear_coefficients", reliable=True, contributions=linear_contribs,
                note="Exact for the logistic model: coefficient x standardised value.",
            )

    fallback = list(bundle.get("global_importances", []))
    return ExplanationResult(
        method="permutation_importance",
        reliable=False,
        contributions=fallback,
        note=(
            "SHAP and linear explanations were unavailable for this model. "
            "Showing GLOBAL feature importance instead -- this ranks "
            "features overall, not for this specific prediction, and does "
            "not indicate direction."
        ),
    )
