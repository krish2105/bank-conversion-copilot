"""The most valuable tests in the repo: prove the model cannot see the
future (duration) and cannot see across a split boundary (temporal split)."""
import pandas as pd
import pytest

from src import config
from src.data import loader
from src.data.loader import split_xy


def test_validate_schema_raises_on_missing_column():
    frame = pd.DataFrame({"age": [1]})
    with pytest.raises(ValueError, match="Missing"):
        loader.validate_schema(frame)


def test_period_index_monotonic_and_wraps_year(synthetic_frame):
    periods = loader.build_period_index(synthetic_frame)
    assert (periods.diff().dropna() >= 0).all()
    # oct(2008) -> ... -> dec(2008) -> mar(2009): year must have incremented
    assert periods.max() - periods.min() >= 12 - (11 - 2)  # dec=11 idx, mar=2 idx


def test_period_index_rejects_unknown_month():
    bad = pd.DataFrame({"month": ["notamonth"]})
    with pytest.raises(ValueError, match="Unknown month"):
        loader.build_period_index(bad)


def test_temporal_split_disjoint_ordered_exhaustive(synthetic_frame):
    periods = loader.build_period_index(synthetic_frame)
    train, valid, test = loader.temporal_split(synthetic_frame, periods)
    assert len(train) + len(valid) + len(test) == len(synthetic_frame)
    train_periods = set(periods.iloc[: len(train)])
    valid_periods = set(periods.iloc[len(train): len(train) + len(valid)])
    test_periods = set(periods.iloc[len(train) + len(valid):])
    assert train_periods.isdisjoint(valid_periods)
    assert valid_periods.isdisjoint(test_periods)
    assert train_periods.isdisjoint(test_periods)
    assert train.index.max() < valid.index.min() if len(valid) else True
    assert valid.index.max() < test.index.min() if len(test) else True


def test_split_xy_drops_duration_and_target(synthetic_frame):
    x, y = loader.split_xy(synthetic_frame)
    assert "duration" not in x.columns
    assert config.TARGET_COLUMN not in x.columns
    assert set(y.unique()) <= {0, 1}
    assert len(x) == len(y) == len(synthetic_frame)


def test_quality_audit_counts(synthetic_frame):
    audit = loader.audit_quality(synthetic_frame)
    assert audit.n_rows == len(synthetic_frame)
    assert audit.n_duplicate_rows >= 2
    for column in config.UNKNOWN_MARKER_COLUMNS:
        assert audit.unknown_counts[column] == int((synthetic_frame[column] == "unknown").sum())
    assert 0.0 < audit.pdays_sentinel_share < 1.0


def test_offline_load_raises_helpful_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        loader.load_raw_frame(offline=True, local_csv=tmp_path / "nope.csv")


def test_leakage_guard_raises_when_duration_reintroduced(synthetic_frame):
    from src.features.pipeline import build_feature_pipeline

    x, _ = split_xy(synthetic_frame)
    x_with_duration = x.copy()
    x_with_duration["duration"] = synthetic_frame["duration"]
    pipeline = build_feature_pipeline(scale_numeric=False)
    with pytest.raises(ValueError, match="duration"):
        pipeline.fit(x_with_duration)


def test_duration_absent_from_encoder_output_names(synthetic_frame):
    from src.features.pipeline import build_feature_pipeline, get_feature_names

    x, _ = split_xy(synthetic_frame)
    pipeline = build_feature_pipeline(scale_numeric=False)
    pipeline.fit(x)
    names = get_feature_names(pipeline)
    assert not any("duration" in name for name in names)
