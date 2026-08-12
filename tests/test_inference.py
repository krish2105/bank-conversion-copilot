"""Three-stage explainer tests, plus the shared serving layer's end-to-end
contract: train tiny model -> save -> reload via the real load_bundle ->
score_one / score_batch / explain."""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.loader import DatasetBundle, audit_quality, build_period_index, split_xy, temporal_split
from src.explain.shap_engine import ExplanationResult, _normalize_shap_shape, explain_prediction
from src.features.pipeline import build_feature_pipeline, get_feature_names
from src.inference.predict import BatchResult, ScoreResult, load_bundle, score_batch, score_one
from src.models.train import train_and_save


def _fit_bundle(synthetic_frame, classifier, scale_numeric):
    x, y = split_xy(synthetic_frame)
    pipeline = build_feature_pipeline(scale_numeric=scale_numeric)
    pipeline.steps.append(("classifier", classifier))
    pipeline.fit(x, y)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
    calibrated.fit(x, y)
    return {
        "model": calibrated,
        "feature_names": get_feature_names(pipeline),
        "global_importances": [(name, 0.01) for name in x.columns],
    }, x.iloc[[0]]


def test_normalize_shap_shape_handles_all_three_forms():
    two_d = np.ones((1, 5))
    assert _normalize_shap_shape(two_d).shape == (1, 5)

    three_d = np.ones((1, 5, 2))
    assert _normalize_shap_shape(three_d).shape == (1, 5)

    two_element_list = [np.zeros((1, 5)), np.ones((1, 5))]
    assert _normalize_shap_shape(two_element_list).shape == (1, 5)


def test_explain_prediction_linear_stage_for_logistic_model(synthetic_frame):
    bundle, row = _fit_bundle(
        synthetic_frame,
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=500),
        scale_numeric=True,
    )
    result = explain_prediction(bundle, row)
    assert isinstance(result, ExplanationResult)
    assert result.method == "linear_coefficients"
    assert result.reliable is True
    assert len(result.contributions) > 0


def test_explain_prediction_degrades_gracefully_for_tree_model(synthetic_frame):
    bundle, row = _fit_bundle(
        synthetic_frame,
        HistGradientBoostingClassifier(max_iter=20, random_state=0),
        scale_numeric=False,
    )
    result = explain_prediction(bundle, row)
    assert result.method in {"shap", "permutation_importance"}
    assert result.reliable == (result.method == "shap")
    if not result.reliable:
        assert "GLOBAL" in result.note


def test_explain_prediction_falls_back_when_model_shape_is_unrecognised():
    bundle = {
        "model": object(),
        "feature_names": ["a", "b"],
        "global_importances": [("a", 0.5), ("b", 0.1)],
    }
    row = pd.DataFrame({"a": [1]})
    result = explain_prediction(bundle, row)
    assert result.method == "permutation_importance"
    assert result.reliable is False
    assert result.contributions == [("a", 0.5), ("b", 0.1)]


def test_contributions_sorted_by_magnitude(synthetic_frame):
    bundle, row = _fit_bundle(
        synthetic_frame,
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=500),
        scale_numeric=True,
    )
    result = explain_prediction(bundle, row)
    magnitudes = [abs(v) for _, v in result.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


@pytest.fixture
def trained_bundle_path(synthetic_frame, tmp_artifact_paths):
    quality = audit_quality(synthetic_frame)
    periods = build_period_index(synthetic_frame)
    train_df, valid_df, test_df = temporal_split(synthetic_frame, periods)
    dataset = DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)
    train_and_save(
        dataset,
        model_path=tmp_artifact_paths / "model.joblib",
        metrics_path=tmp_artifact_paths / "metrics.json",
        model_card_path=tmp_artifact_paths / "model_card.md",
        drift_path=tmp_artifact_paths / "drift.json",
    )
    return tmp_artifact_paths / "model.joblib"


def test_load_bundle_error_names_the_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.models.train"):
        load_bundle(model_path=tmp_path / "missing.joblib")


def test_end_to_end_score_one(trained_bundle_path):
    bundle = load_bundle(model_path=trained_bundle_path)
    fields = {spec.name: spec.default for spec in config.FIELD_SPECS}
    result = score_one(bundle, fields)
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.verdict in {"CALL", "SKIP"}
    assert result.confidence_band in {"Borderline", "Moderate", "Clear"}
    assert len(result.drivers) > 0


def test_end_to_end_score_batch_tolerant_ingestion(trained_bundle_path, synthetic_frame):
    bundle = load_bundle(model_path=trained_bundle_path)
    # duration is deliberately kept (a batch upload that still carries it,
    # e.g. exported straight from the source system) and job is dropped,
    # exercising both tolerant-ingestion paths at once.
    incomplete = synthetic_frame.drop(columns=["job"]).head(20)
    batch = score_batch(bundle, incomplete)
    assert isinstance(batch, BatchResult)
    assert "priority_rank" in batch.scored.columns
    probs = batch.scored["probability"].to_numpy()
    assert (np.diff(probs) <= 1e-12).all()
    assert any("job" in w for w in batch.warnings)
    assert any("duration" in w or "leaky" in w.lower() for w in batch.warnings)


def test_score_batch_capacity_mode_hits_requested_share(trained_bundle_path, synthetic_frame):
    bundle = load_bundle(model_path=trained_bundle_path)
    batch = score_batch(bundle, synthetic_frame.drop(columns=["duration"]), capacity_fraction=0.10)
    called_share = (batch.scored["verdict"] == "CALL").mean()
    assert called_share == pytest.approx(0.10, abs=0.03)
