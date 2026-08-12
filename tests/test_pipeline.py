"""Feature engineering happens entirely inside the sklearn Pipeline so
serving and training can never disagree about how a value was derived."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.loader import split_xy
from src.features.pipeline import (
    DomainFeatureBuilder,
    build_feature_pipeline,
    get_feature_names,
)


def test_domain_feature_builder_sentinel_flag_and_nan():
    frame = pd.DataFrame({
        "pdays": [999, 5, 999, 12],
        "campaign": [1, 2, 3, 4],
        "previous": [0, 1, 0, 2],
        "job": ["admin.", "  RETIRED  ", "unknown", "student"],
        "marital": ["married", "single", "unknown", "single"],
        "education": ["high.school", "unknown", "unknown", "basic.9y"],
        "default": ["no", "no", "unknown", "no"],
        "housing": ["yes", "no", "unknown", "yes"],
        "loan": ["no", "no", "unknown", "no"],
    })
    out = DomainFeatureBuilder().fit_transform(frame)
    assert list(out[config.FEATURE_NEVER_CONTACTED]) == [1, 0, 1, 0]
    assert out["pdays"].iloc[0] != out["pdays"].iloc[0]  # NaN
    assert out["pdays"].iloc[1] == 5
    assert out["job"].iloc[1] == "retired"  # case/whitespace normalised


def test_domain_feature_builder_unknown_count_exact():
    frame = pd.DataFrame({
        "pdays": [999], "campaign": [1], "previous": [0],
        "job": ["unknown"], "marital": ["unknown"], "education": ["unknown"],
        "default": ["no"], "housing": ["yes"], "loan": ["no"],
    })
    out = DomainFeatureBuilder().fit_transform(frame)
    assert out[config.FEATURE_N_UNKNOWN].iloc[0] == 3


def test_domain_feature_builder_contact_intensity_exact():
    frame = pd.DataFrame({
        "pdays": [999, 5], "campaign": [10, 4], "previous": [4, 1],
        "job": ["admin.", "admin."], "marital": ["married", "married"],
        "education": ["high.school", "high.school"], "default": ["no", "no"],
        "housing": ["yes", "yes"], "loan": ["no", "no"],
    })
    out = DomainFeatureBuilder().fit_transform(frame)
    assert out[config.FEATURE_CONTACT_INTENSITY].iloc[0] == pytest.approx(10 / 5)
    assert out[config.FEATURE_CONTACT_INTENSITY].iloc[1] == pytest.approx(4 / 2)


@pytest.mark.parametrize("scale_numeric", [True, False])
def test_pipeline_output_dense_finite_and_names_match_width(synthetic_frame, scale_numeric):
    x, _ = split_xy(synthetic_frame)
    pipeline = build_feature_pipeline(scale_numeric=scale_numeric)
    transformed = pipeline.fit_transform(x)
    assert isinstance(transformed, np.ndarray)
    assert np.isfinite(transformed).all()
    names = get_feature_names(pipeline)
    assert len(names) == transformed.shape[1]


def test_pipeline_handles_unseen_category_without_raising(synthetic_frame):
    x, _ = split_xy(synthetic_frame)
    pipeline = build_feature_pipeline(scale_numeric=False)
    pipeline.fit(x)
    novel = x.iloc[[0]].copy()
    novel["job"] = "astronaut"  # never seen during fit
    transformed = pipeline.transform(novel)
    assert np.isfinite(transformed).all()
