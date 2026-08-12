"""Config is the single source of truth; these tests pin its public contract."""

import numpy as np

from src import config


def test_leakage_denylist_has_reasons():
    assert "duration" in config.LEAKAGE_DENYLIST
    for column in config.LEAKAGE_DENYLIST:
        assert column in config.LEAKAGE_REASONS
        assert len(config.LEAKAGE_REASONS[column]) > 10


def test_field_specs_exclude_duration_and_count_19():
    names = [f.name for f in config.FIELD_SPECS]
    assert "duration" not in names
    assert len(names) == 19
    assert names == list(config.APP_FIELD_ORDER)


def test_field_specs_cover_all_expected_columns():
    expected = {
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    }
    assert {f.name for f in config.FIELD_SPECS} == expected


def test_categorical_field_specs_have_levels():
    for spec in config.FIELD_SPECS:
        if spec.kind == "categorical":
            assert spec.levels is not None and len(spec.levels) >= 2
            assert spec.default in spec.levels
        else:
            assert spec.min_value is not None and spec.max_value is not None
            assert spec.min_value <= spec.default <= spec.max_value


def test_cost_matrix_breakeven():
    cm = config.CostMatrix(cost_per_call=8.0, revenue_per_subscription=120.0)
    assert abs(cm.breakeven_probability - (8.0 / 120.0)) < 1e-9


def test_cost_matrix_net_value_all_called_vs_none_called():
    cm = config.CostMatrix(
        cost_per_call=8.0, revenue_per_subscription=120.0, cost_of_missed_customer=0.0
    )
    probs = np.array([0.9, 0.9, 0.01, 0.01])
    call_high_conf_only = cm.net_value(probs, threshold=0.5)
    call_everyone = cm.net_value(probs, threshold=0.0)
    # calling the two low-probability rows destroys value (0.01*120 - 8 < 0)
    assert call_high_conf_only > call_everyone


def test_paths_resolve_under_repo_root():
    assert config.ARTIFACTS_DIR == config.REPO_ROOT / "artifacts"
    assert config.MODEL_PATH.name == "model.joblib"


def test_month_order_and_unknown_columns():
    assert config.MONTH_ORDER[0] == "jan" and config.MONTH_ORDER[-1] == "dec"
    assert len(config.MONTH_ORDER) == 12
    assert set(config.UNKNOWN_MARKER_COLUMNS) == {
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
    }
    assert config.PDAYS_SENTINEL == 999
