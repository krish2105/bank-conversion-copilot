"""The fixture must be structurally faithful: same columns/dtypes/sentinels
as the real dataset, without mimicking its statistics."""

from src import config


def test_columns_match_raw_schema(synthetic_frame):
    assert set(synthetic_frame.columns) == set(config.RAW_INPUT_COLUMNS) | {
        config.TARGET_COLUMN
    }


def test_no_nulls_but_unknown_markers_present(synthetic_frame):
    assert synthetic_frame.isna().sum().sum() == 0
    for column in config.UNKNOWN_MARKER_COLUMNS:
        assert (synthetic_frame[column] == "unknown").sum() > 0


def test_pdays_sentinel_present_and_dominant(synthetic_frame):
    sentinel_share = (synthetic_frame["pdays"] == config.PDAYS_SENTINEL).mean()
    assert sentinel_share > 0.5


def test_month_sequence_contains_a_wrap(synthetic_frame):
    order = {m: i for i, m in enumerate(config.MONTH_ORDER)}
    idx = synthetic_frame["month"].map(order).to_numpy()
    assert any(idx[i + 1] < idx[i] for i in range(len(idx) - 1))


def test_positive_rate_near_eleven_percent(synthetic_frame):
    rate = (synthetic_frame[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean()
    assert 0.05 < rate < 0.20


def test_contains_exact_duplicates(synthetic_frame):
    assert synthetic_frame.duplicated().sum() >= 2


def test_signal_is_learnable_but_weak(synthetic_frame):
    success_rate = (
        synthetic_frame.loc[
            synthetic_frame["poutcome"] == "success", config.TARGET_COLUMN
        ]
        == "yes"
    ).mean()
    overall_rate = (synthetic_frame[config.TARGET_COLUMN] == "yes").mean()
    assert success_rate > overall_rate
    assert success_rate < 0.95
