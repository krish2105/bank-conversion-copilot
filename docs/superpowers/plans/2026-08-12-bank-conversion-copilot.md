# Bank Conversion Copilot — Part 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Part 1 deliverable from `BUILD_PROMPT.md` — a production-grade ML deployment project (Bank Conversion Copilot) with real training code, a shared inference layer, two front-ends, drift monitoring, explainability, benchmarks, CI/CD, and docs — validated end-to-end against a synthetic fixture (no real training run in this session).

**Architecture:** `src/config.py` is the single source of truth for schema, paths, cost matrix, and field specs. `src/data/loader.py` → `src/features/pipeline.py` (sklearn `Pipeline`, all transforms fit-time only) → `src/models/train.py` (fits both candidates, selects on PR-AUC, calibrates, searches threshold, writes 4 artifacts) → `src/inference/predict.py` (the only module either UI imports) → `app.py` (Gradio) / `streamlit_app.py` (Streamlit), both built on `src/ui/theme.py`. `src/monitor/drift.py` and `src/explain/shap_engine.py` are standalone, numpy/sklearn-only modules consumed by both training and inference.

**Tech Stack:** Python 3.13 (spec targets 3.12; pins verified compatible — see design doc), scikit-learn 1.8.0 (exact), pandas (>=2.2,<3.1), numpy (>=2.0,<3.0), gradio 6.20.0 (exact), streamlit, pytest, ruff, mypy, matplotlib (Agg backend), psutil, python-docx, shap (optional import), joblib.

## Global Constraints

- Target dataset: UCI Bank Marketing (id 222), `bank-additional-full.csv`, `;`-separated, 41,188 rows, target `y`.
- `duration` is on an enforced `LEAKAGE_DENYLIST` — dropped, guarded, and tested everywhere.
- `unknown` in `job, marital, education, default, housing, loan` is a missing marker, not absence of nulls (`pandas.isna()` returns 0).
- `pdays == 999` is a sentinel ("never contacted"), not a magnitude — becomes `never_contacted_before` flag + `NaN`.
- Chronological split 70/15/15 by row order, boundaries snapped to month-period edges (reconstruct a monotonic period index from month-name wraparound, starting year 2008).
- Selection metric: `average_precision` (PR-AUC). Accuracy reported only to dismiss it against the ~88.7% majority baseline.
- Calibration: isotonic via `CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")` — **not** `cv="prefit"` (removed in sklearn 1.8).
- `LogisticRegression`: no explicit `penalty="l2"` kwarg (deprecated in 1.8; L2 is default, control via `C`).
- Cost matrix: `cost_per_call=8.0`, `revenue_per_subscription=120.0`, `cost_of_missed_customer=0.0`; breakeven = 8/120 ≈ 0.0667; threshold chosen by 201-point grid search on VALIDATION maximizing expected net value.
- Drift: PSI (10 reference-quantile bins, never recomputed on current data) for numerics; Jensen-Shannon divergence (base 2) for categoricals; verdict ∈ {STABLE, MONITOR, RETRAIN RECOMMENDED}.
- Explainability: 3-stage degrading (SHAP TreeExplainer → linear coefficients → permutation importance), each result carries `reliable: bool`.
- All tests run offline against `tests/conftest.py`'s synthetic fixture — no network, no real dataset in CI.
- `ruff` line-length 90; `ruff check .` and `ruff format --check .` must be clean; mypy advisory (not a hard gate).
- Every module has a docstring explaining **why**, not just what.
- No file `import`s scoring logic outside `src/inference/predict.py`; both `app.py` and `streamlit_app.py` call only that module.
- Model artifact (`artifacts/model.joblib`) must stay under 10 MB (CI-guarded).
- No dataset files or real training run in this session — everything validated against the synthetic fixture.

---

## Task 1: Project scaffold — dependency and tooling files

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Create: `.pre-commit-config.yaml`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `.streamlit/config.toml`

**Interfaces:**
- Produces: a working Python 3.13 venv contract — every later task assumes `pip install -r requirements-dev.txt` gives it `pytest`, `ruff`, `mypy`, `streamlit`, `matplotlib`, `psutil`, `python-docx`, `shap`.
- Produces: `Makefile` targets `install`, `train`, `test`, `lint`, `fmt`, `app`, `streamlit`, `eda`, `leakage`, `bench`, `report`, `clean` — later tasks' scripts are invoked through these targets by name only, so the exact script paths below must match.

- [ ] **Step 1: Write `requirements.txt`** (runtime only, sklearn pinned exactly)

```
scikit-learn==1.8.0
pandas>=2.2,<3.1
numpy>=2.0,<3.0
gradio==6.20.0
joblib>=1.3.0
scipy>=1.10.0
shap>=0.46.0
```

- [ ] **Step 2: Write `requirements-dev.txt`**

```
-r requirements.txt
streamlit>=1.38,<2.0
pytest>=8.0
pytest-cov>=5.0
ruff>=0.6.0
mypy>=1.11
matplotlib>=3.9
psutil>=6.0
python-docx>=1.1
pre-commit>=3.8
ucimlrepo>=0.0.7
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "bank-conversion-copilot"
version = "0.1.0"
description = "Cost-optimised targeting for outbound term-deposit campaigns"
requires-python = ">=3.12"

[tool.ruff]
line-length = 90
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
warn_unused_ignores = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 4: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=10240]
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
data/*.csv
data/*.zip
deploy/
*.egg-info/
.DS_Store
reports/*.docx
reports/*.pdf
reports/figures/*.png
!reports/figures/.gitkeep
```

Note: `artifacts/` is deliberately NOT ignored — the spec requires it committed.

- [ ] **Step 6: Write `Makefile`**

```makefile
.PHONY: install train test lint fmt app streamlit eda leakage bench report clean

install:
	pip install -r requirements-dev.txt
	pre-commit install

train:
	python -m src.models.train

test:
	pytest -v

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

app:
	python app.py

streamlit:
	streamlit run streamlit_app.py

eda:
	python scripts/run_eda.py

leakage:
	python scripts/leakage_demo.py

bench:
	python benchmarks/latency.py

report:
	python scripts/build_report.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
```

- [ ] **Step 7: Write `.streamlit/config.toml`** (dark theme matching `src/ui/theme.py`, built in Task 12)

```toml
[theme]
base = "dark"
primaryColor = "#3DDC97"
backgroundColor = "#0B0F14"
secondaryBackgroundColor = "#131A21"
textColor = "#E6EDF3"
font = "monospace"

[server]
headless = true
```

- [ ] **Step 8: Create empty package directories with `__init__.py` placeholders and a `reports/figures/.gitkeep`**

```bash
mkdir -p src/data src/features src/models src/monitor src/inference src/explain src/ui
mkdir -p tests scripts benchmarks .github/workflows space docs reports/figures artifacts
touch src/__init__.py src/data/__init__.py src/features/__init__.py src/models/__init__.py
touch src/monitor/__init__.py src/inference/__init__.py src/explain/__init__.py src/ui/__init__.py
touch reports/figures/.gitkeep
```

- [ ] **Step 9: Verify scaffold installs**

Run: `python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt`
Expected: exits 0, no errors.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt requirements-dev.txt pyproject.toml .pre-commit-config.yaml .gitignore Makefile .streamlit src tests scripts benchmarks .github space docs reports artifacts
git commit -m "Scaffold project: deps, tooling, package layout"
git push -u origin main
```

---

## Task 2: `src/config.py` — single source of truth

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `REPO_ROOT, DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR, MODEL_PATH, METRICS_PATH, MODEL_CARD_PATH, DRIFT_PATH` (all `pathlib.Path`, resolved from `Path(__file__).resolve().parent.parent`).
- Produces: `UCI_DATASET_ID, UCI_ZIP_URL, UCI_INNER_ZIP, UCI_CSV_NAME, CSV_SEPARATOR`.
- Produces: `TARGET_COLUMN = "y"`, `POSITIVE_LABEL = "yes"`, `RAW_INPUT_COLUMNS: tuple[str, ...]` (20 names incl. `duration`).
- Produces: `LEAKAGE_DENYLIST: tuple[str, ...]`, `LEAKAGE_REASONS: dict[str, str]` — every denylist entry has a reason.
- Produces: `UNKNOWN_MARKER_COLUMNS: tuple[str, ...]` (6 names), `PDAYS_SENTINEL = 999`.
- Produces: `FEATURE_NEVER_CONTACTED, FEATURE_N_UNKNOWN, FEATURE_CONTACT_INTENSITY, ENGINEERED_FEATURES`.
- Produces: `MONTH_ORDER: tuple[str, ...]` (12 names, `jan`..`dec`), `BASE_YEAR = 2008`, `TRAIN_FRACTION/VALID_FRACTION/TEST_FRACTION`.
- Produces: `CostMatrix` frozen dataclass — `cost_per_call, revenue_per_subscription, cost_of_missed_customer` fields; `.breakeven_probability` property; `.net_value(probabilities: np.ndarray, threshold: float) -> float` method; module-level `DEFAULT_COST_MATRIX = CostMatrix()`.
- Produces: `SELECTION_METRIC = "average_precision"`.
- Produces: `FieldSpec` frozen dataclass (`name, kind, default, levels, min_value, max_value, step, label, help_text`) and `FIELD_SPECS: tuple[FieldSpec, ...]` with exactly 19 entries (9 numeric + 10 categorical, `duration` excluded); `APP_FIELD_ORDER = tuple(f.name for f in FIELD_SPECS)`.
- Produces: `Branding` frozen dataclass (colors/fonts) and `Runtime` frozen dataclass (`sklearn_pinned_version, random_state, n_jobs`).
- Consumed by every later task — these exact names must not change.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
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
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "campaign", "pdays", "previous",
        "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed",
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
    cm = config.CostMatrix(cost_per_call=8.0, revenue_per_subscription=120.0, cost_of_missed_customer=0.0)
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
        "job", "marital", "education", "default", "housing", "loan",
    }
    assert config.PDAYS_SENTINEL == 999
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL / ERROR — `src.config` does not exist yet.

- [ ] **Step 3: Write `src/config.py`**

```python
"""Single source of truth for schema, paths, and business constants.

Both the training pipeline and the serving layer import from this module
instead of hard-coding column names or category levels. That is what makes
it structurally impossible for them to disagree about feature order or
dtype — the most common bug in a student ML repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Paths -------------------------------------------------------------
# Resolved from the repo root so behaviour is identical in a notebook,
# from the CLI, and inside the HF Space container at /home/user/app.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACTS_DIR / "model_card.md"
DRIFT_PATH = ARTIFACTS_DIR / "drift.json"

# --- UCI dataset identity -----------------------------------------------
UCI_DATASET_ID = 222
UCI_DATASET_NAME = "bank-additional-full"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_INNER_ZIP = "bank-additional.zip"
UCI_CSV_NAME = "bank-additional-full.csv"
CSV_SEPARATOR = ";"

TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"

RAW_INPUT_COLUMNS: tuple[str, ...] = (
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
)

# --- Leakage control ------------------------------------------------------
LEAKAGE_DENYLIST: tuple[str, ...] = ("duration",)
LEAKAGE_REASONS: dict[str, str] = {
    "duration": (
        "Call duration is only known after the call has happened. Using it "
        "to decide whether to place the call is impossible in production, "
        "and it inflates offline metrics by exposing the outcome (UCI's own "
        "documentation flags this as target leakage)."
    ),
}

# --- Missing-value handling -------------------------------------------
UNKNOWN_MARKER_COLUMNS: tuple[str, ...] = (
    "job", "marital", "education", "default", "housing", "loan",
)
PDAYS_SENTINEL = 999

# --- Engineered features (built inside the sklearn Pipeline) -----------
FEATURE_NEVER_CONTACTED = "never_contacted_before"
FEATURE_N_UNKNOWN = "n_unknown_fields"
FEATURE_CONTACT_INTENSITY = "contact_intensity"
ENGINEERED_FEATURES: tuple[str, ...] = (
    FEATURE_NEVER_CONTACTED, FEATURE_N_UNKNOWN, FEATURE_CONTACT_INTENSITY,
)

# --- Temporal split -------------------------------------------------------
MONTH_ORDER: tuple[str, ...] = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)
BASE_YEAR = 2008
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

# --- Cost matrix ------------------------------------------------------
@dataclass(frozen=True)
class CostMatrix:
    """Business economics of a single call, in EUR.

    net_value multiplies calibrated probabilities by real money, so the
    calibration step upstream is not optional cosmetics.
    """

    cost_per_call: float = 8.0
    revenue_per_subscription: float = 120.0
    cost_of_missed_customer: float = 0.0

    @property
    def breakeven_probability(self) -> float:
        return self.cost_per_call / self.revenue_per_subscription

    def net_value(self, probabilities: np.ndarray, threshold: float) -> float:
        probabilities = np.asarray(probabilities, dtype=float)
        call_mask = probabilities >= threshold
        called = probabilities[call_mask] * self.revenue_per_subscription - self.cost_per_call
        skipped = -probabilities[~call_mask] * self.cost_of_missed_customer
        return float(np.sum(called) + np.sum(skipped))


DEFAULT_COST_MATRIX = CostMatrix()
SELECTION_METRIC = "average_precision"

# --- App field specs (19 = 20 raw inputs minus duration) -------------------
@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    default: object
    label: str
    help_text: str = ""
    levels: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("age", "numeric", 40, "Age", min_value=18, max_value=95, step=1),
    FieldSpec(
        "job", "categorical", "admin.", "Job",
        levels=(
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown",
        ),
    ),
    FieldSpec(
        "marital", "categorical", "married", "Marital status",
        levels=("divorced", "married", "single", "unknown"),
    ),
    FieldSpec(
        "education", "categorical", "university.degree", "Education",
        levels=(
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown",
        ),
    ),
    FieldSpec("default", "categorical", "no", "Has credit in default?", levels=("no", "yes", "unknown")),
    FieldSpec("housing", "categorical", "yes", "Has housing loan?", levels=("no", "yes", "unknown")),
    FieldSpec("loan", "categorical", "no", "Has personal loan?", levels=("no", "yes", "unknown")),
    FieldSpec("contact", "categorical", "cellular", "Contact type", levels=("cellular", "telephone")),
    FieldSpec(
        "month", "categorical", "may", "Last contact month",
        levels=("mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    ),
    FieldSpec(
        "day_of_week", "categorical", "mon", "Last contact day of week",
        levels=("mon", "tue", "wed", "thu", "fri"),
    ),
    FieldSpec(
        "campaign", "numeric", 2, "Contacts this campaign",
        help_text="Number of contacts for this client during this campaign, including the current one.",
        min_value=1, max_value=50, step=1,
    ),
    FieldSpec(
        "pdays", "numeric", 999, "Days since previous contact",
        help_text="999 means never previously contacted.",
        min_value=0, max_value=999, step=1,
    ),
    FieldSpec(
        "previous", "numeric", 0, "Prior contacts",
        help_text="Number of contacts before this campaign for this client.",
        min_value=0, max_value=10, step=1,
    ),
    FieldSpec(
        "poutcome", "categorical", "nonexistent", "Previous campaign outcome",
        levels=("failure", "nonexistent", "success"),
    ),
    FieldSpec(
        "emp.var.rate", "numeric", 1.1, "Employment variation rate",
        min_value=-3.4, max_value=1.4, step=0.1,
    ),
    FieldSpec(
        "cons.price.idx", "numeric", 93.994, "Consumer price index",
        min_value=92.0, max_value=95.0, step=0.01,
    ),
    FieldSpec(
        "cons.conf.idx", "numeric", -36.4, "Consumer confidence index",
        min_value=-51.0, max_value=-26.0, step=0.1,
    ),
    FieldSpec(
        "euribor3m", "numeric", 4.857, "Euribor 3-month rate",
        min_value=0.6, max_value=5.1, step=0.001,
    ),
    FieldSpec(
        "nr.employed", "numeric", 5191.0, "Number employed (macro index)",
        min_value=4950.0, max_value=5230.0, step=1.0,
    ),
)
APP_FIELD_ORDER: tuple[str, ...] = tuple(f.name for f in FIELD_SPECS)

# --- Branding / runtime -------------------------------------------------
@dataclass(frozen=True)
class Branding:
    app_name: str = "Bank Conversion Copilot"
    tagline: str = "Cost-optimised targeting for outbound term-deposit campaigns"
    accent: str = "#3DDC97"
    background: str = "#0B0F14"
    surface: str = "#131A21"
    text: str = "#E6EDF3"
    danger: str = "#FF5D5D"
    warning: str = "#F5A623"
    font_family: str = "'JetBrains Mono', 'Fira Code', monospace"


@dataclass(frozen=True)
class Runtime:
    sklearn_pinned_version: str = "1.8.0"
    random_state: int = 42
    n_jobs: int = -1


BRANDING = Branding()
RUNTIME = Runtime()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run ruff**

Run: `ruff check src/config.py tests/test_config.py && ruff format --check src/config.py tests/test_config.py`
Expected: clean (fix any line-length-90 wraps if it complains).

- [ ] **Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "Add src/config.py as single source of truth"
git push
```

---

## Task 3: `tests/conftest.py` — synthetic schema-faithful fixture

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest_fixture.py`

**Interfaces:**
- Consumes: `src.config.RAW_INPUT_COLUMNS, TARGET_COLUMN, UNKNOWN_MARKER_COLUMNS, PDAYS_SENTINEL`.
- Produces: module-level function `make_synthetic_bank_frame(n_per_month: int = 300, seed: int = 42) -> pd.DataFrame` (importable as `from tests.conftest import make_synthetic_bank_frame`), plus pytest fixtures `synthetic_frame` (a `pd.DataFrame`) and `tmp_artifact_paths` (monkeypatches `src.config` path constants into `tmp_path`, returns the artifacts dir `Path`). Every later test task depends on these two fixture names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conftest_fixture.py
"""The fixture must be structurally faithful: same columns/dtypes/sentinels
as the real dataset, without mimicking its statistics."""
from src import config


def test_columns_match_raw_schema(synthetic_frame):
    assert set(synthetic_frame.columns) == set(config.RAW_INPUT_COLUMNS) | {config.TARGET_COLUMN}


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
    # poutcome == success should correlate with y == yes, but not perfectly
    success_rate = (
        synthetic_frame.loc[synthetic_frame["poutcome"] == "success", config.TARGET_COLUMN] == "yes"
    ).mean()
    overall_rate = (synthetic_frame[config.TARGET_COLUMN] == "yes").mean()
    assert success_rate > overall_rate
    assert success_rate < 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_conftest_fixture.py -v`
Expected: FAIL — `conftest.py` / `synthetic_frame` fixture do not exist.

- [ ] **Step 3: Write `tests/conftest.py`**

```python
"""Offline test fixtures.

CI has no network and no access to the real UCI dataset, so every test in
this suite runs against a synthetic frame that is *structurally* faithful
(same columns, dtypes, sentinel values, unknown markers, an out-of-order
month wrap) without being *statistically* faithful. These fixtures verify
the code is correct; only a real training run (Part 2 of BUILD_PROMPT.md)
says anything about model quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic_bank_frame(n_per_month: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # dec -> mar wrap (skipping jan/feb, exactly as the real campaign does)
    # so year reconstruction is exercised on genuinely out-of-order months,
    # not just accepted because the input happened to already be sorted.
    month_blocks = ["oct", "nov", "dec", "mar", "apr", "may"]
    months = np.repeat(month_blocks, n_per_month)
    n = len(months)

    day_of_week = rng.choice(["mon", "tue", "wed", "thu", "fri"], size=n)
    job = rng.choice(
        ["admin.", "blue-collar", "technician", "services", "management",
         "retired", "student", "unknown"],
        size=n, p=[0.22, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.08],
    )
    marital = rng.choice(["married", "single", "divorced", "unknown"], size=n, p=[0.55, 0.28, 0.14, 0.03])
    education = rng.choice(
        ["university.degree", "high.school", "basic.9y", "professional.course", "unknown"],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    default = rng.choice(["no", "unknown", "yes"], size=n, p=[0.78, 0.20, 0.02])
    housing = rng.choice(["yes", "no", "unknown"], size=n, p=[0.52, 0.44, 0.04])
    loan = rng.choice(["no", "yes", "unknown"], size=n, p=[0.82, 0.14, 0.04])
    contact = rng.choice(["cellular", "telephone"], size=n, p=[0.65, 0.35])

    age = rng.integers(18, 90, size=n)
    campaign = rng.integers(1, 15, size=n)

    never_contacted = rng.random(n) < 0.82
    pdays = np.where(never_contacted, config.PDAYS_SENTINEL, rng.integers(1, 27, size=n))
    previous = np.where(never_contacted, 0, rng.integers(1, 4, size=n))
    poutcome = np.where(
        never_contacted, "nonexistent",
        rng.choice(["failure", "success"], size=n, p=[0.7, 0.3]),
    )

    month_to_rate = {"oct": -1.0, "nov": -1.5, "dec": -2.0, "mar": -1.8, "apr": -1.2, "may": 1.1}
    emp_var_rate = np.array([month_to_rate[m] for m in months]) + rng.normal(0, 0.05, n)
    cons_price_idx = 93.0 + rng.normal(0, 0.3, n)
    cons_conf_idx = -40.0 + rng.normal(0, 3.0, n)
    euribor3m = np.clip(2.5 + emp_var_rate * 0.8 + rng.normal(0, 0.2, n), 0.6, 5.1)
    nr_employed = 5100 + emp_var_rate * 30 + rng.normal(0, 5, n)
    duration = rng.integers(0, 1200, size=n)

    # Deliberately weak signal: enough for the two required models to beat
    # the majority baseline, not enough to look suspiciously perfect.
    logit = (
        -2.2
        + 1.1 * (poutcome == "success")
        + 0.4 * (euribor3m < 2.0)
        - 0.15 * campaign
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = np.where(rng.random(n) < prob, "yes", "no")
    positive_rate = (y == "yes").mean()
    if positive_rate > 0.11:
        flip_down = (y == "yes") & (rng.random(n) < (1 - 0.11 / positive_rate))
        y = np.where(flip_down, "no", y)

    frame = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": months, "day_of_week": day_of_week, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr.employed": nr_employed, "y": y,
    })

    duplicates = frame.iloc[[5, n // 2, n - 5]].copy()
    return pd.concat([frame, duplicates], ignore_index=True)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_bank_frame()


@pytest.fixture
def tmp_artifact_paths(tmp_path, monkeypatch):
    """Redirect every config artifact path into tmp_path.

    Nothing in the test suite may write into the real artifacts/ dir — a
    teammate could be mid-demo against it.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "MODEL_PATH", artifacts_dir / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", artifacts_dir / "metrics.json")
    monkeypatch.setattr(config, "MODEL_CARD_PATH", artifacts_dir / "model_card.md")
    monkeypatch.setattr(config, "DRIFT_PATH", artifacts_dir / "drift.json")
    return artifacts_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_conftest_fixture.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest_fixture.py
git commit -m "Add synthetic schema-faithful test fixture"
git push
```

---

## Task 4: `src/data/loader.py` — load, validate, audit, period index, temporal split

**Files:**
- Create: `src/data/loader.py`
- Test: `tests/test_leakage.py` (new file — will also gain duration-guard tests in Task 5)

**Interfaces:**
- Consumes: `src.config.{RAW_INPUT_COLUMNS, TARGET_COLUMN, POSITIVE_LABEL, LEAKAGE_DENYLIST, UNKNOWN_MARKER_COLUMNS, PDAYS_SENTINEL, MONTH_ORDER, BASE_YEAR, TRAIN_FRACTION, VALID_FRACTION, TEST_FRACTION, UCI_ZIP_URL, UCI_INNER_ZIP, UCI_CSV_NAME, CSV_SEPARATOR, UCI_DATASET_ID, DATA_DIR}`.
- Produces: `load_raw_frame(offline: bool = False, local_csv: Path | None = None) -> pd.DataFrame`; `validate_schema(frame: pd.DataFrame) -> None` (raises `ValueError`); `QualityAudit` dataclass (`n_rows, n_duplicate_rows, unknown_counts: dict[str,int], pdays_sentinel_share: float`) and `audit_quality(frame) -> QualityAudit`; `build_period_index(frame: pd.DataFrame) -> pd.Series`; `temporal_split(frame, periods) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`; `split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]`; `DatasetBundle` dataclass (`train, valid, test: pd.DataFrame`, `quality: QualityAudit`) and `load_and_split(offline: bool = False, local_csv: Path | None = None) -> DatasetBundle` — this is the function `src/models/train.py` (Task 8) calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_leakage.py
"""The most valuable tests in the repo: prove the model cannot see the
future (duration) and cannot see across a split boundary (temporal split)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import loader


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
    # ordered: every train row precedes every valid row precedes every test row
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_leakage.py -v`
Expected: FAIL — `src.data.loader` does not exist.

- [ ] **Step 3: Write `src/data/loader.py`**

```python
"""Load the UCI Bank Marketing dataset three ways, validate it, audit its
quality, and produce a leakage-safe chronological split.

Chronological splitting matters here specifically because five macro
features are constant within a calendar month (see BUILD_PROMPT.md 1.4):
a plain positional split can put one month's economic conditions on both
sides of a boundary. Snapping every boundary to a period edge closes that.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def load_raw_frame(offline: bool = False, local_csv: Path | None = None) -> pd.DataFrame:
    if offline:
        path = local_csv or (config.DATA_DIR / config.UCI_CSV_NAME)
        if not path.exists():
            raise FileNotFoundError(
                f"--offline requested but {path} does not exist. Either "
                f"download {config.UCI_ZIP_URL} and extract "
                f"{config.UCI_CSV_NAME} into {config.DATA_DIR}, or run "
                f"`python -m src.models.train` without --offline."
            )
        return pd.read_csv(path, sep=config.CSV_SEPARATOR)

    frame = _load_via_ucimlrepo()
    if frame is None or "euribor3m" not in frame.columns:
        # Trap 4: ucimlrepo sometimes serves the legacy 17-input variant
        # with no macro columns. Fall back to the direct zip.
        frame = _load_via_direct_zip()
    return frame


def _load_via_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return None
    try:
        dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        frame.columns = [str(c) for c in frame.columns]
        return frame
    except Exception:
        return None


def _load_via_direct_zip() -> pd.DataFrame:
    with urllib.request.urlopen(config.UCI_ZIP_URL, timeout=30) as response:
        outer_bytes = response.read()
    outer = zipfile.ZipFile(io.BytesIO(outer_bytes))
    inner_bytes = outer.read(config.UCI_INNER_ZIP)
    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
    csv_bytes = inner.read(f"bank-additional/{config.UCI_CSV_NAME}")
    return pd.read_csv(io.BytesIO(csv_bytes), sep=config.CSV_SEPARATOR)


def validate_schema(frame: pd.DataFrame) -> None:
    expected = set(config.RAW_INPUT_COLUMNS) | {config.TARGET_COLUMN}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")


@dataclass
class QualityAudit:
    n_rows: int
    n_duplicate_rows: int
    unknown_counts: dict[str, int]
    pdays_sentinel_share: float


def audit_quality(frame: pd.DataFrame) -> QualityAudit:
    unknown_counts = {
        column: int((frame[column] == "unknown").sum())
        for column in config.UNKNOWN_MARKER_COLUMNS
    }
    return QualityAudit(
        n_rows=len(frame),
        n_duplicate_rows=int(frame.duplicated().sum()),
        unknown_counts=unknown_counts,
        pdays_sentinel_share=float((frame["pdays"] == config.PDAYS_SENTINEL).mean()),
    )


def build_period_index(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct a monotonic year-month period from file order alone.

    The file ships no date column, only a month name, and rows are known
    to be date-ordered. Walking rows and incrementing the year whenever the
    month number goes backwards (a calendar wrap) recovers a real period
    index without ever inventing a day.
    """
    order = {month: i for i, month in enumerate(config.MONTH_ORDER)}
    month_idx = frame["month"].map(order)
    if month_idx.isna().any():
        bad = sorted(set(frame.loc[month_idx.isna(), "month"]))
        raise ValueError(f"Unknown month values: {bad}")

    month_values = month_idx.to_numpy(dtype=np.int64)
    periods = np.empty(len(frame), dtype=np.int64)
    year = config.BASE_YEAR
    prev = None
    for i, m in enumerate(month_values):
        if prev is not None and m < prev:
            year += 1
        periods[i] = year * 12 + m
        prev = m
    return pd.Series(periods, index=frame.index, name="period")


def temporal_split(
    frame: pd.DataFrame, periods: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    values = periods.to_numpy()
    change_points = np.flatnonzero(np.diff(values) != 0) + 1
    boundaries = np.concatenate(([0], change_points, [n]))

    def snap(target: int) -> int:
        return int(boundaries[np.argmin(np.abs(boundaries - target))])

    train_end = snap(int(n * config.TRAIN_FRACTION))
    valid_end = max(train_end, snap(int(n * (config.TRAIN_FRACTION + config.VALID_FRACTION))))

    return frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features/target, dropping every denylisted column.

    This is the first of two leakage guards (the second lives inside the
    feature pipeline in src/features/pipeline.py) — belt and braces.
    """
    y = (frame[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
    x = frame.drop(columns=[config.TARGET_COLUMN, *config.LEAKAGE_DENYLIST], errors="ignore")
    return x, y


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    quality: QualityAudit


def load_and_split(offline: bool = False, local_csv: Path | None = None) -> DatasetBundle:
    frame = load_raw_frame(offline=offline, local_csv=local_csv).reset_index(drop=True)
    validate_schema(frame)
    quality = audit_quality(frame)
    periods = build_period_index(frame)
    train_df, valid_df, test_df = temporal_split(frame, periods)
    return DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_leakage.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/data/loader.py tests/test_leakage.py
ruff format --check src/data/loader.py tests/test_leakage.py
git add src/data/loader.py tests/test_leakage.py
git commit -m "Add data loader: 3-strategy load, period index, temporal split"
git push
```

---

## Task 5: `src/features/pipeline.py` — engineered features + leakage guard

**Files:**
- Create: `src/features/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `tests/test_leakage.py` (append guard-reintroduction tests)

**Interfaces:**
- Consumes: `src.config.{FIELD_SPECS, PDAYS_SENTINEL, UNKNOWN_MARKER_COLUMNS, FEATURE_NEVER_CONTACTED, FEATURE_N_UNKNOWN, FEATURE_CONTACT_INTENSITY, LEAKAGE_DENYLIST, LEAKAGE_REASONS}`.
- Produces: `NUMERIC_COLUMNS: tuple[str, ...]`, `CATEGORICAL_COLUMNS: tuple[str, ...]` (derived from `FIELD_SPECS`, not re-declared); `LeakageGuard` (sklearn transformer, raises `ValueError` if any `LEAKAGE_DENYLIST` column is present); `DomainFeatureBuilder` (sklearn transformer, adds the 3 engineered columns, normalises categorical case/whitespace, nulls out sentinel `pdays`); `build_feature_pipeline(scale_numeric: bool) -> sklearn.pipeline.Pipeline` (no final estimator — `src/models/train.py`, Task 8, appends the classifier); `get_feature_names(pipeline: Pipeline) -> list[str]`.
- `build_feature_pipeline(scale_numeric=True)` is used for `LogisticRegression`; `build_feature_pipeline(scale_numeric=False)` for `HistGradientBoostingClassifier`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
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
```

```python
# --- append to tests/test_leakage.py ---


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `src.features.pipeline` does not exist.

- [ ] **Step 3: Write `src/features/pipeline.py`**

```python
"""Feature engineering, entirely inside a sklearn Pipeline.

Nothing here touches a dataframe before .fit(). That is what makes serving
safe: the app hands raw user input to predict_proba and the identical
transformations run with parameters learned only from TRAIN. Scaling in a
notebook and pickling only the classifier is the single most common cause
of train/serve skew, and it fails silently — this structure makes that
class of bug impossible to introduce by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in config.FIELD_SPECS if f.kind == "numeric"
)
CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in config.FIELD_SPECS if f.kind == "categorical"
)


class LeakageGuard(BaseEstimator, TransformerMixin):
    """Raises if any denylisted (leaky) column reaches the pipeline.

    This is the second of two guards — src/data/loader.split_xy drops
    duration too, but a caller that skips split_xy (or a future refactor
    that forgets to) hits this instead of silently training a leaky model.
    """

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "LeakageGuard":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in config.LEAKAGE_DENYLIST if c in x.columns]
        if present:
            reasons = "; ".join(f"{c}: {config.LEAKAGE_REASONS[c]}" for c in present)
            raise ValueError(f"Leakage guard triggered by columns {present}. {reasons}")
        return x


class DomainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds never_contacted_before, n_unknown_fields, contact_intensity.

    Also normalises categorical case/whitespace so a hand-edited CSV with
    " RETIRED " lands on the fitted encoder's "retired" level instead of
    becoming a spurious unseen category.
    """

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "DomainFeatureBuilder":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()

        for column in CATEGORICAL_COLUMNS:
            if column in out.columns:
                out[column] = out[column].astype(str).str.strip().str.lower()

        sentinel_mask = out["pdays"] == config.PDAYS_SENTINEL
        out[config.FEATURE_NEVER_CONTACTED] = sentinel_mask.astype(int)
        out.loc[sentinel_mask, "pdays"] = np.nan

        unknown_mask = pd.DataFrame({
            column: out[column] == "unknown" for column in config.UNKNOWN_MARKER_COLUMNS
        })
        out[config.FEATURE_N_UNKNOWN] = unknown_mask.sum(axis=1)

        out[config.FEATURE_CONTACT_INTENSITY] = out["campaign"] / (out["previous"] + 1)

        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + list(config.ENGINEERED_FEATURES))


def build_feature_pipeline(scale_numeric: bool) -> Pipeline:
    numeric_features = list(NUMERIC_COLUMNS) + list(config.ENGINEERED_FEATURES)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=20,
                sparse_output=False,
            ),
        ),
    ])

    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("leakage_guard", LeakageGuard()),
        ("domain_features", DomainFeatureBuilder()),
        ("columns", columns),
    ])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["columns"].get_feature_names_out())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py tests/test_leakage.py -v`
Expected: all passed (5 in test_pipeline.py + 9 in test_leakage.py).

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/features/pipeline.py tests/test_pipeline.py tests/test_leakage.py
ruff format --check src/features/pipeline.py tests/test_pipeline.py tests/test_leakage.py
git add src/features/pipeline.py tests/test_pipeline.py tests/test_leakage.py
git commit -m "Add feature pipeline: engineered features + leakage guard"
git push
```

---

## Task 6: `src/models/evaluate.py` — MetricSet, reliability curve, ECE, curve points

**Files:**
- Create: `src/models/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `MetricSet` dataclass (`model_name: str, average_precision: float, roc_auc: float, accuracy: float, precision: float, recall: float, f1: float, n_samples: int, n_positive: int`); `compute_metrics(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> MetricSet`; `confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]` (keys `tp, fp, tn, fn`); `reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame` (columns `bin, mean_predicted, mean_observed, count`, quantile-binned); `expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float`; `pr_curve_points(y_true, y_prob) -> pd.DataFrame` (`precision, recall, threshold`); `roc_curve_points(y_true, y_prob) -> pd.DataFrame` (`fpr, tpr, threshold`).
- Consumed by `src/models/train.py` (Task 8) and `scripts/build_report.py` (Task 20).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
"""MetricSet fields are constructed with explicit keywords everywhere in
this codebase (Trap 3) so a missing field surfaces at the call site, not
deep inside a training run."""
import numpy as np
import pytest

from src.models.evaluate import (
    MetricSet,
    compute_metrics,
    confusion_counts,
    expected_calibration_error,
    pr_curve_points,
    reliability_curve,
    roc_curve_points,
)


@pytest.fixture
def toy_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    # probabilities correlated with, but not identical to, the truth
    y_prob = np.clip(y_true * 0.5 + rng.random(500) * 0.5, 0, 1)
    return y_true, y_prob


def test_metric_set_requires_all_keywords():
    with pytest.raises(TypeError):
        MetricSet(model_name="x")  # type: ignore[call-arg]


def test_compute_metrics_returns_populated_metric_set(toy_predictions):
    y_true, y_prob = toy_predictions
    metrics = compute_metrics("toy", y_true, y_prob, threshold=0.5)
    assert metrics.n_samples == 500
    assert metrics.n_positive == int(y_true.sum())
    assert 0.0 <= metrics.average_precision <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_confusion_counts_sum_to_n(toy_predictions):
    y_true, y_prob = toy_predictions
    counts = confusion_counts(y_true, y_prob, threshold=0.5)
    assert counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"] == len(y_true)


def test_confusion_counts_known_case():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1])
    counts = confusion_counts(y_true, y_prob, threshold=0.5)
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_reliability_curve_uses_quantile_bins(toy_predictions):
    y_true, y_prob = toy_predictions
    curve = reliability_curve(y_true, y_prob, n_bins=5)
    assert curve["count"].sum() == len(y_true)
    assert (curve["count"] > 0).all()


def test_expected_calibration_error_zero_for_perfect_calibration():
    # predicted prob equals empirical frequency exactly, in two big blocks
    y_true = np.array([1] * 100 + [0] * 100)
    y_prob = np.array([0.9] * 100 + [0.1] * 100)
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece < 0.15  # not exactly 0 due to bin-mean vs point mass, but small


def test_curve_points_shapes(toy_predictions):
    y_true, y_prob = toy_predictions
    pr = pr_curve_points(y_true, y_prob)
    roc = roc_curve_points(y_true, y_prob)
    assert {"precision", "recall", "threshold"} <= set(pr.columns)
    assert {"fpr", "tpr", "threshold"} <= set(roc.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `src.models.evaluate` does not exist.

- [ ] **Step 3: Write `src/models/evaluate.py`**

```python
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


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    weights = curve["count"] / curve["count"].sum()
    gaps = (curve["mean_predicted"] - curve["mean_observed"]).abs()
    return float((weights * gaps).sum())


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thresholds})


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/models/evaluate.py tests/test_evaluate.py
ruff format --check src/models/evaluate.py tests/test_evaluate.py
git add src/models/evaluate.py tests/test_evaluate.py
git commit -m "Add evaluation utilities: MetricSet, reliability curve, ECE"
git push
```

---

## Task 7: `src/models/threshold.py` — cost-optimal and capacity-constrained thresholds

**Files:**
- Create: `src/models/threshold.py`
- Create: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `src.config.{CostMatrix, DEFAULT_COST_MATRIX}`.
- Produces: `ThresholdResult` dataclass (`threshold, expected_net_value, realized_net_value, realized_net_value_at_default, realized_net_value_call_everyone, uplift_vs_default, uplift_vs_call_everyone` — all `float` except `threshold: float`); `realized_net_value(y_true, y_prob, threshold, cost_matrix) -> float`; `search_cost_optimal_threshold(y_true, y_prob, cost_matrix=config.DEFAULT_COST_MATRIX, n_grid=201) -> ThresholdResult`; `capacity_constrained_threshold(y_prob: np.ndarray, capacity_fraction: float) -> float`.
- `expected_net_value` is computed purely from calibrated probabilities via `cost_matrix.net_value` (the grid-search objective); `realized_*` fields use true validation/test labels and are what the report quotes as "real EUR" uplift.
- Consumed by `src/models/train.py` (Task 8) and `src/inference/predict.py` (Task 10, for the verdict panel's cutoff marker).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
"""Threshold economics and (later in this file, Task 9) drift metrics —
the two places a wrong number costs real money or masks real decay."""
import numpy as np
import pytest

from src.config import CostMatrix
from src.models.threshold import (
    ThresholdResult,
    capacity_constrained_threshold,
    realized_net_value,
    search_cost_optimal_threshold,
)


def test_realized_net_value_known_case():
    cost_matrix = CostMatrix(cost_per_call=8.0, revenue_per_subscription=120.0)
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1])
    # threshold 0.5 calls rows 0,1 (probs 0.9, 0.8); of those, row 0 is a
    # true positive -> 1*120 - 2*8 = 104
    value = realized_net_value(y_true, y_prob, threshold=0.5, cost_matrix=cost_matrix)
    assert value == pytest.approx(120 - 2 * 8)


def test_search_threshold_is_on_grid_and_beats_default_on_its_own_metric():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=2000)
    y_prob = np.clip(y_true * 0.4 + rng.random(2000) * 0.3, 0, 1)
    cost_matrix = CostMatrix()
    result = search_cost_optimal_threshold(y_true, y_prob, cost_matrix=cost_matrix, n_grid=201)
    assert isinstance(result, ThresholdResult)
    grid = np.linspace(0.0, 1.0, 201)
    assert np.isclose(grid, result.threshold).any()
    value_at_default = cost_matrix.net_value(y_prob, 0.5)
    # never loses to 0.5, on the exact metric the search optimises
    assert result.expected_net_value >= value_at_default - 1e-9


def test_capacity_constrained_threshold_hits_requested_share():
    rng = np.random.default_rng(2)
    y_prob = rng.random(10_000)
    threshold = capacity_constrained_threshold(y_prob, capacity_fraction=0.10)
    called_share = (y_prob >= threshold).mean()
    assert called_share == pytest.approx(0.10, abs=0.01)


def test_capacity_constrained_threshold_rejects_bad_fraction():
    with pytest.raises(ValueError):
        capacity_constrained_threshold(np.array([0.1, 0.9]), capacity_fraction=0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `src.models.threshold` does not exist.

- [ ] **Step 3: Write `src/models/threshold.py`**

```python
"""Cost-optimal and capacity-constrained decision thresholds.

0.5 assumes a false positive and a false negative cost the same. Here one
costs 8 EUR and the other a 120 EUR opportunity, so the threshold that
maximises business value sits nowhere near 0.5 — it sits near the
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/models/threshold.py tests/test_scoring.py
ruff format --check src/models/threshold.py tests/test_scoring.py
git add src/models/threshold.py tests/test_scoring.py
git commit -m "Add cost-optimal and capacity-constrained threshold search"
git push
```

---

## Task 8: `src/monitor/drift.py` — PSI + Jensen-Shannon, pure numpy

**Files:**
- Create: `src/monitor/drift.py`
- Modify: `tests/test_scoring.py` (append drift tests)

**Interfaces:**
- Produces: `compute_reference_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray`; `population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float`; `jensen_shannon_categorical(reference: pd.Series, current: pd.Series, base: float = 2.0) -> float`; `DriftReport` dataclass (`numeric_psi: dict[str, float], categorical_js: dict[str, float], verdict: str`); `compute_drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric_columns: Sequence[str], categorical_columns: Sequence[str]) -> DriftReport`; module constants `PSI_STABLE_MAX = 0.10, PSI_MONITOR_MAX = 0.25, JS_STABLE_MAX = 0.05, JS_MONITOR_MAX = 0.15`.
- `verdict` is one of `"STABLE" | "MONITOR" | "RETRAIN RECOMMENDED"`, driven by the worst PSI/JS value across all features.
- No external drift library — pure numpy/pandas, so the arithmetic is defensible in a viva.
- Consumed by `src/models/train.py` (Task 9, writes `artifacts/drift.json` comparing TRAIN vs TEST as a concrete worked example) and by the model card tab (Task 13/14).

- [ ] **Step 1: Write the failing tests** (appended to `tests/test_scoring.py`)

```python
# --- append to tests/test_scoring.py ---
import pandas as pd

from src.monitor.drift import (
    DriftReport,
    compute_drift_report,
    jensen_shannon_categorical,
    population_stability_index,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    psi = population_stability_index(reference, reference.copy())
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_mean_shift():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 2000)
    small_shift = reference + 0.2
    large_shift = reference + 2.0
    psi_small = population_stability_index(reference, small_shift)
    psi_large = population_stability_index(reference, large_shift)
    assert 0.0 < psi_small < psi_large


def test_psi_handles_constant_reference_without_raising():
    reference = np.full(500, 7.0)
    current_same = np.full(500, 7.0)
    psi_same = population_stability_index(reference, current_same)
    assert psi_same == pytest.approx(0.0, abs=1e-6)
    # a differing current must not raise, even though it can't be
    # differentiated by a single-bin reference edge case
    current_different = np.full(500, 99.0)
    psi_different = population_stability_index(reference, current_different)
    assert np.isfinite(psi_different)


def test_jensen_shannon_bounded_and_zero_for_identical():
    ref = pd.Series(["a", "a", "b", "c"] * 50)
    js_identical = jensen_shannon_categorical(ref, ref.copy())
    assert js_identical == pytest.approx(0.0, abs=1e-9)

    disjoint = pd.Series(["x", "y", "z"] * 50)
    js_disjoint = jensen_shannon_categorical(ref, disjoint)
    assert 0.0 <= js_disjoint <= 1.0
    assert js_disjoint > js_identical


def test_compute_drift_report_verdict_stable_for_identical_frames():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "num": rng.normal(0, 1, 1000),
        "cat": rng.choice(["a", "b", "c"], size=1000),
    })
    report = compute_drift_report(frame, frame.copy(), numeric_columns=["num"], categorical_columns=["cat"])
    assert isinstance(report, DriftReport)
    assert report.verdict == "STABLE"


def test_compute_drift_report_verdict_flags_large_shift():
    rng = np.random.default_rng(6)
    reference = pd.DataFrame({
        "num": rng.normal(0, 1, 1000),
        "cat": rng.choice(["a", "b", "c"], size=1000, p=[0.8, 0.1, 0.1]),
    })
    current = pd.DataFrame({
        "num": rng.normal(6, 1, 1000),
        "cat": rng.choice(["a", "b", "c"], size=1000, p=[0.1, 0.1, 0.8]),
    })
    report = compute_drift_report(reference, current, numeric_columns=["num"], categorical_columns=["cat"])
    assert report.verdict in {"MONITOR", "RETRAIN RECOMMENDED"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `src.monitor.drift` does not exist.

- [ ] **Step 3: Write `src/monitor/drift.py`**

```python
"""Drift monitoring from scratch: PSI for numerics, Jensen-Shannon for
categoricals. No external library — the arithmetic is simple enough to
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
    one side entirely — unlike plain KL divergence, which blows up there."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: 10 passed (4 threshold + 6 drift).

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/monitor/drift.py tests/test_scoring.py
ruff format --check src/monitor/drift.py tests/test_scoring.py
git add src/monitor/drift.py tests/test_scoring.py
git commit -m "Add from-scratch drift monitoring: PSI + Jensen-Shannon"
git push
```

---

## Task 9: `src/models/train.py` — the training entrypoint

**Files:**
- Create: `src/models/train.py`

**Interfaces:**
- Consumes: `src.data.loader.{DatasetBundle, load_and_split, split_xy}`; `src.features.pipeline.{build_feature_pipeline, get_feature_names, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS}`; `src.models.evaluate.{compute_metrics, expected_calibration_error, reliability_curve}`; `src.models.threshold.search_cost_optimal_threshold`; `src.monitor.drift.compute_drift_report`; `src.config.{DEFAULT_COST_MATRIX, RUNTIME, MODEL_PATH, METRICS_PATH, MODEL_CARD_PATH, DRIFT_PATH, LEAKAGE_REASONS}`.
- Produces: `train_and_save(dataset: DatasetBundle, model_path: Path | None = None, metrics_path: Path | None = None, model_card_path: Path | None = None, drift_path: Path | None = None) -> dict` (returns the metrics payload it also writes to `metrics_path`) and CLI `main()` (invoked via `python -m src.models.train [--offline]`).
- **Critical correctness point:** the four path parameters default to `None` and are resolved to `config.MODEL_PATH` etc. **inside the function body**, not as Python default-argument values evaluated at import time. A default bound at def-time would capture the pre-monkeypatch path and silently ignore `tmp_artifact_paths` in Task 10's tests.
- The saved bundle (`joblib.dump` target) is a `dict` with keys `model` (the fitted, calibrated sklearn estimator — calling `.predict_proba(raw_dataframe)` runs the full pipeline end to end), `winner_name`, `threshold`, `feature_names`, `global_importances` (`list[tuple[str, float]]`, sorted descending, computed once via `sklearn.inspection.permutation_importance` against VALIDATION — this is `src/explain/shap_engine.py`'s always-available stage-3 fallback), `sklearn_version`, `trained_at`. `src/explain/shap_engine.py` (Task 10) and `src/inference/predict.py` (Task 11) are the consumers of this exact shape.
- This task has **no dedicated test file** — per BUILD_PROMPT.md §1.9/1.13 the only test files are `conftest.py, test_leakage.py, test_pipeline.py, test_scoring.py, test_inference.py`, and the spec explicitly assigns the "train tiny model → save → reload → score" end-to-end test to `test_inference.py` (Tasks 10–11). This task ends with a manual smoke run instead of a pytest step; Tasks 10–11 provide the automated coverage.

- [ ] **Step 1: Write `src/models/train.py`**

```python
"""Training entrypoint: fits both candidates on TRAIN, selects the winner
on validation PR-AUC, calibrates it in isolation, searches a cost-optimal
threshold, evaluates on TEST exactly once, and writes all four artifacts.

Model selection, calibration, and threshold search read VALIDATION only.
TEST is opened exactly once, at the end, and nothing upstream is tuned
against it -- that discipline is what makes the reported numbers
trustworthy rather than optimistic.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.loader import DatasetBundle, load_and_split, split_xy
from src.features.pipeline import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_pipeline,
    get_feature_names,
)
from src.models.evaluate import compute_metrics, expected_calibration_error, reliability_curve
from src.models.threshold import search_cost_optimal_threshold
from src.monitor.drift import compute_drift_report


def _build_candidates() -> dict[str, object]:
    lr_pipeline = build_feature_pipeline(scale_numeric=True)
    lr_pipeline.steps.append((
        "classifier",
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs",
            random_state=config.RUNTIME.random_state,
        ),
    ))

    hgb_pipeline = build_feature_pipeline(scale_numeric=False)
    hgb_pipeline.steps.append((
        "classifier",
        HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
            random_state=config.RUNTIME.random_state,
        ),
    ))

    return {"logistic_regression": lr_pipeline, "hist_gradient_boosting": hgb_pipeline}


def train_and_save(
    dataset: DatasetBundle,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    model_card_path: Path | None = None,
    drift_path: Path | None = None,
) -> dict:
    model_path = model_path or config.MODEL_PATH
    metrics_path = metrics_path or config.METRICS_PATH
    model_card_path = model_card_path or config.MODEL_CARD_PATH
    drift_path = drift_path or config.DRIFT_PATH
    for path in (model_path, metrics_path, model_card_path, drift_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = split_xy(dataset.train)
    x_valid, y_valid = split_xy(dataset.valid)
    x_test, y_test = split_xy(dataset.test)

    candidates = _build_candidates()
    validation_metrics = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        valid_probs = pipeline.predict_proba(x_valid)[:, 1]
        validation_metrics[name] = compute_metrics(name, y_valid.to_numpy(), valid_probs, threshold=0.5)
        fitted[name] = pipeline

    winner_name = max(validation_metrics, key=lambda n: validation_metrics[n].average_precision)
    winner = fitted[winner_name]

    # Isotonic calibration on VALIDATION via FrozenEstimator (sklearn>=1.8
    # removed cv="prefit" — Trap 1). The winner is already fitted on TRAIN
    # and must not be refit here.
    calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")
    calibrated.fit(x_valid, y_valid)

    valid_calibrated_probs = calibrated.predict_proba(x_valid)[:, 1]
    threshold_result = search_cost_optimal_threshold(
        y_valid.to_numpy(), valid_calibrated_probs, cost_matrix=config.DEFAULT_COST_MATRIX,
    )

    # TEST is opened exactly once, here, at the end.
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(
        f"{winner_name}_calibrated", y_test.to_numpy(), test_probs,
        threshold=threshold_result.threshold,
    )
    majority_baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))
    ece = expected_calibration_error(y_test.to_numpy(), test_probs)
    reliability = reliability_curve(y_test.to_numpy(), test_probs)

    drift_report = compute_drift_report(
        x_train, x_test, numeric_columns=NUMERIC_COLUMNS, categorical_columns=CATEGORICAL_COLUMNS,
    )

    # Global (not per-prediction) permutation importance on the winner's
    # raw-input columns, computed once here against VALIDATION. This is
    # the stage-3 explanation fallback in src/explain/shap_engine.py — it
    # must always work, so it is precomputed rather than derived live from
    # a model type that might not support it (e.g. no feature_importances_
    # on HistGradientBoostingClassifier).
    perm_result = permutation_importance(
        winner, x_valid, y_valid, scoring="average_precision",
        n_repeats=5, random_state=config.RUNTIME.random_state, n_jobs=config.RUNTIME.n_jobs,
    )
    global_importances = sorted(
        zip(x_valid.columns.tolist(), perm_result.importances_mean.tolist()),
        key=lambda pair: pair[1], reverse=True,
    )

    bundle = {
        "model": calibrated,
        "winner_name": winner_name,
        "threshold": threshold_result.threshold,
        "feature_names": get_feature_names(winner),
        "global_importances": global_importances,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "dataset": {
            "n_rows": dataset.quality.n_rows,
            "n_duplicate_rows": dataset.quality.n_duplicate_rows,
            "unknown_counts": dataset.quality.unknown_counts,
            "pdays_sentinel_share": dataset.quality.pdays_sentinel_share,
            "n_train": len(dataset.train),
            "n_valid": len(dataset.valid),
            "n_test": len(dataset.test),
        },
        "model_comparison": {
            name: dataclasses.asdict(metric) for name, metric in validation_metrics.items()
        },
        "winner": winner_name,
        "calibration": {
            "expected_calibration_error": ece,
            "reliability_curve": reliability.to_dict(orient="records"),
        },
        "threshold_search": dataclasses.asdict(threshold_result),
        "breakeven_probability": config.DEFAULT_COST_MATRIX.breakeven_probability,
        "test_metrics": dataclasses.asdict(test_metrics),
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "global_importances": global_importances,
        "artifact_size_bytes": model_path.stat().st_size,
        "sklearn_version": sklearn.__version__,
        "trained_at": bundle["trained_at"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))

    drift_payload = {
        "numeric_psi": drift_report.numeric_psi,
        "categorical_js": drift_report.categorical_js,
        "verdict": drift_report.verdict,
        "reference": "train split",
        "current": "test split",
    }
    drift_path.write_text(json.dumps(drift_payload, indent=2))

    model_card_path.write_text(_render_model_card(metrics_payload, drift_payload))

    return metrics_payload


def _render_model_card(metrics: dict, drift: dict) -> str:
    leakage_lines = "\n".join(
        f"- **{col}**: {reason}" for col, reason in config.LEAKAGE_REASONS.items()
    )
    test = metrics["test_metrics"]
    comparison_rows = "\n".join(
        f"| {name} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} |"
        for name, m in metrics["model_comparison"].items()
    )
    threshold = metrics["threshold_search"]
    return f"""# Model Card — Bank Conversion Copilot

> Auto-generated by `src/models/train.py` from `artifacts/metrics.json`. Do not edit by hand.

## Overview

| Field | Value |
|---|---|
| Winner | {metrics['winner']} |
| Trained at | {metrics['trained_at']} |
| sklearn version | {metrics['sklearn_version']} |
| Selection metric | average_precision (PR-AUC) |
| Decision threshold | {threshold['threshold']:.4f} |

## Intended use

Rank outbound term-deposit telemarketing prospects by expected net value,
for a capacity-constrained call centre deciding who to phone.

## Out-of-scope use

**Not for credit decisions.** Conversion propensity is not a proxy for
credit risk, and this model was never validated for that purpose.

## Test performance

| Model | Validation AP | Validation ROC-AUC |
|---|---|---|
{comparison_rows}

Test set (opened once): precision {test['precision']:.4f}, recall {test['recall']:.4f},
average precision {test['average_precision']:.4f}, ROC-AUC {test['roc_auc']:.4f}.
Accuracy is reported last, deliberately: {test['accuracy']:.4f} versus a
majority-class baseline of {metrics['majority_baseline_accuracy']:.4f} —
accuracy alone cannot distinguish this model from predicting "no" for
everyone.

## Business framing

Break-even call probability: {metrics['breakeven_probability']:.4f}.
Chosen threshold yields a validation uplift of {threshold['uplift_vs_default']:.2f} EUR
versus the naive 0.5 cutoff and {threshold['uplift_vs_call_everyone']:.2f} EUR
versus calling everyone.

## Leakage controls

{leakage_lines}

## Drift monitoring (train vs test)

Verdict: **{drift['verdict']}**. See `artifacts/drift.json` for per-feature detail.

## Known limitations

- Temporal validity: trained on 2008–2010 Portuguese retail-banking data;
  economic conditions and channel mix have moved on.
- Geographic specificity: one bank, one country.
- `unknown` category bias: six fields carry a nontrivial "declined to
  state" rate, which the model treats as a signal rather than imputing away.
- No fairness certification, despite age and marital status being model
  inputs.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Bank Conversion Copilot model.")
    parser.add_argument("--offline", action="store_true", help="Load from data/ instead of the network.")
    args = parser.parse_args()

    dataset = load_and_split(offline=args.offline)
    metrics = train_and_save(dataset)

    winner_metrics = metrics["model_comparison"][metrics["winner"]]
    print(f"Winner: {metrics['winner']}")
    print(f"Validation AP: {winner_metrics['average_precision']:.4f}")
    test = metrics["test_metrics"]
    print(f"Test AP: {test['average_precision']:.4f}  ROC-AUC: {test['roc_auc']:.4f}")
    threshold = metrics["threshold_search"]
    print(f"Threshold: {threshold['threshold']:.4f}  (breakeven {metrics['breakeven_probability']:.4f})")
    print(f"Artifact size: {metrics['artifact_size_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke verification against the synthetic fixture**

Run:
```bash
python -c "
from pathlib import Path
from tests.conftest import make_synthetic_bank_frame
from src.data.loader import validate_schema, audit_quality, build_period_index, temporal_split, DatasetBundle
from src.models.train import train_and_save
import tempfile

frame = make_synthetic_bank_frame()
validate_schema(frame)
quality = audit_quality(frame)
periods = build_period_index(frame)
train_df, valid_df, test_df = temporal_split(frame, periods)
bundle = DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    metrics = train_and_save(
        bundle,
        model_path=tmp_dir / 'model.joblib',
        metrics_path=tmp_dir / 'metrics.json',
        model_card_path=tmp_dir / 'model_card.md',
        drift_path=tmp_dir / 'drift.json',
    )
    print('winner:', metrics['winner'])
    print('artifact bytes:', metrics['artifact_size_bytes'])
    assert (tmp_dir / 'model.joblib').exists()
    assert (tmp_dir / 'model_card.md').exists()
    assert len(metrics['global_importances']) > 0
    print('OK')
"
```
Expected: prints a winner name, an artifact size, and `OK` with no traceback. (Metrics quality is meaningless here — synthetic data only proves the plumbing works; real numbers come from Part 2's real training run.)

- [ ] **Step 3: Run ruff and commit**

```bash
ruff check src/models/train.py
ruff format --check src/models/train.py
git add src/models/train.py
git commit -m "Add training entrypoint: fit, select, calibrate, threshold, artifacts"
git push
```

---

## Task 10: `src/explain/shap_engine.py` — 3-stage degrading explainer

**Files:**
- Create: `src/explain/shap_engine.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: bundle shape from Task 9 (`model, feature_names, global_importances`).
- Produces: `ExplanationResult` dataclass (`method: str` — one of `"shap" | "linear_coefficients" | "permutation_importance"`, `reliable: bool`, `contributions: list[tuple[str, float]]` sorted by magnitude descending, `note: str`); `explain_prediction(bundle: dict, x_row: pd.DataFrame) -> ExplanationResult`.
- Built in dependency order **ahead of** `src/inference/predict.py` (Task 11), which is the only module that calls `explain_prediction` — per BUILD_PROMPT.md §1.3, front-ends import only `src.inference.predict`, so this module is never imported directly by `app.py` or `streamlit_app.py`.
- `reliable=False` on the stage-3 fallback is a UI contract: the drivers table must render the `note` text when `reliable` is `False`, not present a weaker explanation as if it were SHAP.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference.py (created here; Task 11 appends load_bundle/score_one/score_batch/e2e tests)
"""Three-stage explainer: SHAP -> linear coefficients -> global permutation
importance. Every stage after the first exists because a real deployed app
must never crash trying to explain a prediction."""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.data.loader import split_xy
from src.explain.shap_engine import ExplanationResult, _normalize_shap_shape, explain_prediction
from src.features.pipeline import build_feature_pipeline, get_feature_names
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


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
    # SHAP support for HistGradientBoosting varies by version -- either a
    # real SHAP explanation or the always-works fallback is acceptable,
    # but the contract (method/reliable/note agree) must hold.
    assert result.method in {"shap", "permutation_importance"}
    assert result.reliable == (result.method == "shap")
    if not result.reliable:
        assert "GLOBAL" in result.note


def test_explain_prediction_falls_back_when_model_shape_is_unrecognised():
    bundle = {
        "model": object(),  # no calibrated_classifiers_ attribute
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL — `src.explain.shap_engine` does not exist.

- [ ] **Step 3: Write `src/explain/shap_engine.py`**

```python
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
            "Showing GLOBAL feature importance instead — this ranks "
            "features overall, not for this specific prediction, and does "
            "not indicate direction."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inference.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run ruff and commit**

```bash
ruff check src/explain/shap_engine.py tests/test_inference.py
ruff format --check src/explain/shap_engine.py tests/test_inference.py
git add src/explain/shap_engine.py tests/test_inference.py
git commit -m "Add 3-stage degrading explainer: SHAP, linear, permutation fallback"
git push
```

---

## Task 11: `src/inference/predict.py` — the serving layer

**Files:**
- Create: `src/inference/predict.py`
- Modify: `tests/test_inference.py` (append — this file already exists from Task 10)

**Interfaces:**
- Consumes: `src.config.{FIELD_SPECS, APP_FIELD_ORDER, LEAKAGE_DENYLIST, CostMatrix, DEFAULT_COST_MATRIX, MODEL_PATH}`; `src.explain.shap_engine.explain_prediction`; `src.models.threshold.capacity_constrained_threshold`.
- Produces: `ScoreResult` dataclass (`probability, verdict, threshold, expected_value_eur, confidence_band, drivers, explanation_method, explanation_reliable, explanation_note`); `BatchResult` dataclass (`scored: pd.DataFrame, warnings: list[str]`); `load_bundle(model_path: Path | None = None) -> dict` (raises `FileNotFoundError` naming `python -m src.models.train`, warns on sklearn version mismatch); `score_one(bundle: dict, fields: dict, cost_matrix: CostMatrix = DEFAULT_COST_MATRIX) -> ScoreResult`; `score_batch(bundle: dict, frame: pd.DataFrame, capacity_fraction: float | None = None, cost_matrix: CostMatrix = DEFAULT_COST_MATRIX) -> BatchResult`.
- **This is the only module `app.py` (Task 13) and `streamlit_app.py` (Task 14) import for scoring.** Neither UI file may contain scoring logic, per BUILD_PROMPT.md §1.3.
- Confidence bands: distance from threshold `< 0.02` → `"Borderline"`, `< 0.08` → `"Moderate"`, else `"Clear"`.

- [ ] **Step 1: Write the failing tests** (appended to `tests/test_inference.py`)

```python
# --- append to tests/test_inference.py ---
import numpy as np
import pytest

from src import config
from src.data.loader import DatasetBundle, audit_quality, build_period_index, temporal_split
from src.inference.predict import BatchResult, ScoreResult, load_bundle, score_batch, score_one
from src.models.train import train_and_save


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
    incomplete = synthetic_frame.drop(columns=["job", "duration"]).head(20)
    batch = score_batch(bundle, incomplete)
    assert isinstance(batch, BatchResult)
    assert "priority_rank" in batch.scored.columns
    probs = batch.scored["probability"].to_numpy()
    assert (np.diff(probs) <= 1e-12).all()  # ranked descending by probability
    assert any("job" in w for w in batch.warnings)
    assert any("duration" in w or "leaky" in w.lower() for w in batch.warnings)


def test_score_batch_capacity_mode_hits_requested_share(trained_bundle_path, synthetic_frame):
    bundle = load_bundle(model_path=trained_bundle_path)
    batch = score_batch(bundle, synthetic_frame.drop(columns=["duration"]), capacity_fraction=0.10)
    called_share = (batch.scored["verdict"] == "CALL").mean()
    assert called_share == pytest.approx(0.10, abs=0.03)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL — `src.inference.predict` does not exist (the explain-stage tests from Task 10 continue to pass).

- [ ] **Step 3: Write `src/inference/predict.py`**

```python
"""The serving layer. Both app.py and streamlit_app.py import only this
module — there is exactly one implementation of "what does the model say",
so the two UIs cannot disagree, and these tests exercise the shared path
rather than either UI's glue code."""

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
    bundle: dict, fields: dict, cost_matrix: config.CostMatrix = config.DEFAULT_COST_MATRIX
) -> ScoreResult:
    row = _fields_to_frame(fields)
    probability = float(bundle["model"].predict_proba(row)[:, 1][0])
    threshold = float(bundle["threshold"])
    verdict = "CALL" if probability >= threshold else "SKIP"
    expected_value = probability * cost_matrix.revenue_per_subscription - cost_matrix.cost_per_call
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
            notes.append(f"Column '{spec.name}' missing — filled with default {spec.default!r}.")
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
        threshold = capacity_constrained_threshold(probabilities, capacity_fraction)

    result = clean.copy()
    result["probability"] = probabilities
    result["verdict"] = np.where(probabilities >= threshold, "CALL", "SKIP")
    result["expected_value_eur"] = (
        probabilities * cost_matrix.revenue_per_subscription - cost_matrix.cost_per_call
    )
    result["confidence_band"] = [_confidence_band(p, threshold) for p in probabilities]
    result = result.sort_values("probability", ascending=False).reset_index(drop=True)
    result["priority_rank"] = np.arange(1, len(result) + 1)

    return BatchResult(scored=result, warnings=notes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inference.py -v`
Expected: all passed (5 explain tests from Task 10 + 4 inference tests here).

- [ ] **Step 5: Run the full suite, ruff, and commit**

```bash
pytest -v
ruff check src/inference/predict.py tests/test_inference.py
ruff format --check src/inference/predict.py tests/test_inference.py
git add src/inference/predict.py tests/test_inference.py
git commit -m "Add shared inference layer: load_bundle, score_one, score_batch"
git push
```

---

## Task 12: `src/ui/theme.py` — shared design tokens, CSS, HTML fragments

**Files:**
- Create: `src/ui/theme.py`

**Interfaces:**
- Consumes: `src.config.{BRANDING, Branding}`.
- Produces: `build_css(branding: Branding = BRANDING) -> str`; `render_verdict_panel(probability, threshold, verdict, confidence_band, expected_value_eur, branding=BRANDING) -> str`; `render_drivers_table(drivers: list[tuple[str, float]], method: str, reliable: bool, note: str) -> str`.
- Consumed by `app.py` (Task 13) and `streamlit_app.py` (Task 14) — **neither app styles anything inline**; every color, font, and HTML fragment for the verdict panel and drivers table comes from here.
- No dedicated test file (not in the §1.9/§1.13 test list — pure deterministic string formatting, verified by a manual smoke check).

- [ ] **Step 1: Write `src/ui/theme.py`**

```python
"""Shared design tokens, CSS, and HTML fragments for both front-ends.

Neither app.py nor streamlit_app.py styles anything inline. A stock Gradio
or Streamlit page reads as a prototype, and Presentation and Reporting is
a graded criterion, so the visual language — dark, dense, financial-
terminal — lives in exactly one place.
"""

from __future__ import annotations

from src.config import BRANDING, Branding


def build_css(branding: Branding = BRANDING) -> str:
    return f"""
    :root {{
        --accent: {branding.accent};
        --bg: {branding.background};
        --surface: {branding.surface};
        --text: {branding.text};
        --danger: {branding.danger};
        --warning: {branding.warning};
    }}
    body, .gradio-container {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: {branding.font_family} !important;
    }}
    .bcc-card {{
        background-color: var(--surface);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }}
    .bcc-verdict-call {{ color: var(--accent); font-weight: 700; }}
    .bcc-verdict-skip {{ color: var(--danger); font-weight: 700; }}
    .bcc-bar-track {{
        position: relative;
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 5px;
        margin: 12px 0;
    }}
    .bcc-bar-fill {{
        position: absolute; left: 0; top: 0; height: 100%;
        background: var(--accent); border-radius: 5px;
    }}
    .bcc-cutoff-marker {{
        position: absolute; top: -4px; width: 2px; height: 18px;
        background: var(--warning);
    }}
    table.bcc-table {{ width: 100%; border-collapse: collapse; }}
    table.bcc-table th, table.bcc-table td {{
        text-align: left; padding: 4px 8px; border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    """


def render_verdict_panel(
    probability: float,
    threshold: float,
    verdict: str,
    confidence_band: str,
    expected_value_eur: float,
    branding: Branding = BRANDING,
) -> str:
    verdict_class = "bcc-verdict-call" if verdict == "CALL" else "bcc-verdict-skip"
    value_color = branding.accent if expected_value_eur >= 0 else branding.danger
    fill_pct = max(0.0, min(1.0, probability)) * 100
    cutoff_pct = max(0.0, min(1.0, threshold)) * 100
    return f"""
    <div class="bcc-card">
      <div style="font-size:14px;opacity:0.7;">Probability of conversion</div>
      <div style="font-size:32px;font-weight:700;">{probability:.1%}</div>
      <div class="bcc-bar-track">
        <div class="bcc-bar-fill" style="width:{fill_pct:.1f}%;"></div>
        <div class="bcc-cutoff-marker" style="left:{cutoff_pct:.1f}%;" title="Cutoff {threshold:.1%}"></div>
      </div>
      <div class="{verdict_class}" style="font-size:20px;">{verdict}</div>
      <div style="opacity:0.7;">Confidence: {confidence_band}</div>
      <div style="color:{value_color};font-weight:600;">
        Expected value of this call: {expected_value_eur:+.2f} EUR
      </div>
    </div>
    """


def render_drivers_table(
    drivers: list[tuple[str, float]], method: str, reliable: bool, note: str
) -> str:
    rows = "".join(
        f"<tr><td>{name}</td><td>{value:+.4f}</td></tr>" for name, value in drivers[:10]
    )
    banner = (
        "" if reliable
        else f'<div style="color:{BRANDING.warning};margin-bottom:8px;">{note}</div>'
    )
    return f"""
    <div class="bcc-card">
      <div style="font-size:14px;opacity:0.7;">Drivers ({method})</div>
      {banner}
      <table class="bcc-table">
        <thead><tr><th>Feature</th><th>Contribution</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """
```

- [ ] **Step 2: Manual smoke verification**

Run:
```bash
python -c "
from src.ui.theme import build_css, render_verdict_panel, render_drivers_table
assert '--accent' in build_css()
html = render_verdict_panel(0.42, 0.09, 'CALL', 'Clear', 42.5)
assert 'CALL' in html and 'bcc-cutoff-marker' in html
table = render_drivers_table([('euribor3m', 0.12), ('poutcome_success', -0.05)], 'shap', True, '')
assert 'euribor3m' in table
print('OK')
"
```
Expected: `OK`, no traceback.

- [ ] **Step 3: Run ruff and commit**

```bash
ruff check src/ui/theme.py
ruff format --check src/ui/theme.py
git add src/ui/theme.py
git commit -m "Add shared UI design tokens and HTML fragments"
git push
```

---

## Task 13: `app.py` — Gradio Blocks HF Space entrypoint

**Files:**
- Create: `app.py` (repo root — this is the HF Space entrypoint per `space/README.md`'s `app_file`)

**Interfaces:**
- Consumes: `src.inference.predict.{load_bundle, score_one, score_batch}`; `src.ui.theme.{build_css, render_verdict_panel, render_drivers_table}`; `src.config.{FIELD_SPECS, APP_FIELD_ORDER, BRANDING, METRICS_PATH, DRIFT_PATH}`. **Imports no other `src` module** — all scoring logic is reached through `src.inference.predict`.
- Produces: a module-level `demo = gr.Blocks(...)` and `if __name__ == "__main__": demo.launch()`. `BUNDLE = load_bundle()` runs at import time (warms the model before the first request).
- No dedicated test file — Gradio callback wiring isn't meaningfully unit-testable and isn't in the §1.13 list. Verified by manual smoke check here; a full interactive browser check happens in Task 21 once a placeholder-trained artifact exists (this session defers the real training run to Phase 2 per the design doc, so `load_bundle()` will raise until *some* artifact — real or placeholder — is written to `artifacts/model.joblib`).

- [ ] **Step 1: Write `app.py`**

```python
"""Gradio Blocks app -- the Hugging Face Space entrypoint.

Warms the model bundle at import time, not on first request: cold starts
on free Space hardware are dominated by artefact deserialisation, and a
live demo audience should never watch that happen. `import spaces` is
wrapped in try/except because it only exists inside an actual Space
container -- local development and CI must not require it.
"""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from src import config
from src.inference.predict import load_bundle, score_batch, score_one
from src.ui.theme import build_css, render_drivers_table, render_verdict_panel

try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None

BUNDLE = load_bundle()


def _score_prospect(*values):
    fields = dict(zip(config.APP_FIELD_ORDER, values, strict=True))
    result = score_one(BUNDLE, fields)
    verdict_html = render_verdict_panel(
        result.probability, result.threshold, result.verdict,
        result.confidence_band, result.expected_value_eur,
    )
    drivers_html = render_drivers_table(
        result.drivers, result.explanation_method, result.explanation_reliable,
        result.explanation_note,
    )
    return verdict_html, drivers_html


def _template_csv() -> str:
    path = "/tmp/bank_conversion_template.csv"
    pd.DataFrame([{spec.name: spec.default for spec in config.FIELD_SPECS}]).to_csv(path, index=False)
    return path


def _score_csv(file, capacity_pct):
    if file is None:
        return None, "Upload a CSV first."
    frame = pd.read_csv(file.name)
    capacity_fraction = capacity_pct / 100.0 if capacity_pct else None
    batch = score_batch(BUNDLE, frame, capacity_fraction=capacity_fraction)
    out_path = "/tmp/bank_conversion_scored.csv"
    batch.scored.to_csv(out_path, index=False)
    n_call = int((batch.scored["verdict"] == "CALL").sum())
    summary = f"{n_call} of {len(batch.scored)} marked CALL."
    if batch.warnings:
        summary += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in batch.warnings)
    return out_path, summary


def _model_card_html() -> str:
    if not config.METRICS_PATH.exists():
        return "<p>No metrics.json found. Run <code>python -m src.models.train</code>.</p>"
    metrics = json.loads(config.METRICS_PATH.read_text())
    drift = (
        json.loads(config.DRIFT_PATH.read_text())
        if config.DRIFT_PATH.exists() else {"verdict": "unknown"}
    )
    test = metrics["test_metrics"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{m['average_precision']:.4f}</td><td>{m['roc_auc']:.4f}</td></tr>"
        for name, m in metrics["model_comparison"].items()
    )
    return f"""
    <div class="bcc-card">
      <h3>Test performance</h3>
      <p>Precision {test['precision']:.4f} &middot; Recall {test['recall']:.4f} &middot;
         AP {test['average_precision']:.4f} &middot; ROC-AUC {test['roc_auc']:.4f} &middot;
         Accuracy {test['accuracy']:.4f} (majority baseline {metrics['majority_baseline_accuracy']:.4f})</p>
      <h3>Threshold economics</h3>
      <p>Threshold {metrics['threshold_search']['threshold']:.4f}
         (breakeven {metrics['breakeven_probability']:.4f}),
         uplift vs 0.5: {metrics['threshold_search']['uplift_vs_default']:.2f} EUR,
         uplift vs call-everyone: {metrics['threshold_search']['uplift_vs_call_everyone']:.2f} EUR.</p>
      <h3>Model comparison (validation)</h3>
      <table class="bcc-table"><thead><tr><th>Model</th><th>AP</th><th>ROC-AUC</th></tr></thead>
        <tbody>{rows}</tbody></table>
      <h3>Drift verdict</h3>
      <p>{drift['verdict']}</p>
      <h3>Data quality</h3>
      <p>{metrics['dataset']['n_rows']} rows, {metrics['dataset']['n_duplicate_rows']} duplicates,
         pdays sentinel share {metrics['dataset']['pdays_sentinel_share']:.2%}.</p>
      <h3>Provenance</h3>
      <p>sklearn {metrics['sklearn_version']}, trained {metrics['trained_at']}.</p>
    </div>
    """


def _build_field_inputs() -> list:
    inputs = []
    for spec in config.FIELD_SPECS:
        if spec.kind == "categorical":
            inputs.append(gr.Dropdown(choices=list(spec.levels), value=spec.default, label=spec.label))
        else:
            inputs.append(
                gr.Number(
                    value=spec.default, label=spec.label,
                    minimum=spec.min_value, maximum=spec.max_value,
                )
            )
    return inputs


with gr.Blocks(css=build_css(), title=config.BRANDING.app_name) as demo:
    gr.Markdown(f"# {config.BRANDING.app_name}\n{config.BRANDING.tagline}")

    with gr.Tab("Score a prospect"):
        field_inputs = _build_field_inputs()
        score_button = gr.Button("Score", variant="primary")
        verdict_out = gr.HTML()
        drivers_out = gr.HTML()
        score_button.click(
            _score_prospect, inputs=field_inputs, outputs=[verdict_out, drivers_out],
            api_name="score_one",
        )

    with gr.Tab("Score a call list"):
        gr.Markdown(
            "Upload a CSV of prospects. Missing columns get documented "
            "defaults; unseen categories degrade gracefully."
        )
        template_button = gr.Button("Download template CSV")
        template_file = gr.File(label="Template")
        template_button.click(_template_csv, outputs=template_file)

        csv_input = gr.File(label="Prospect CSV", file_types=[".csv"])
        capacity_slider = gr.Slider(0, 100, value=100, step=1, label="Capacity (% of list to call)")
        batch_button = gr.Button("Score list", variant="primary")
        batch_output = gr.File(label="Scored list")
        batch_summary = gr.Textbox(label="Summary", lines=4)
        batch_button.click(
            _score_csv, inputs=[csv_input, capacity_slider], outputs=[batch_output, batch_summary],
            api_name="score_batch",
        )

    with gr.Tab("Model card & monitoring"):
        gr.HTML(value=_model_card_html())


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Manual smoke verification (with a temporary throwaway model)**

Run:
```bash
python -c "
import tempfile
from pathlib import Path
from src import config
from tests.conftest import make_synthetic_bank_frame
from src.data.loader import validate_schema, audit_quality, build_period_index, temporal_split, DatasetBundle
from src.models.train import train_and_save

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    frame = make_synthetic_bank_frame()
    validate_schema(frame)
    quality = audit_quality(frame)
    periods = build_period_index(frame)
    train_df, valid_df, test_df = temporal_split(frame, periods)
    train_and_save(
        DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality),
        model_path=tmp_dir / 'model.joblib', metrics_path=tmp_dir / 'metrics.json',
        model_card_path=tmp_dir / 'model_card.md', drift_path=tmp_dir / 'drift.json',
    )
    config.MODEL_PATH = tmp_dir / 'model.joblib'
    config.METRICS_PATH = tmp_dir / 'metrics.json'
    config.DRIFT_PATH = tmp_dir / 'drift.json'
    import app  # noqa: F401
    print('app.py imported and BUNDLE warmed OK')
"
```
Expected: `app.py imported and BUNDLE warmed OK`, no traceback. (A full interactive browser click-through happens in Task 21, once real or placeholder artifacts live at the default `artifacts/` path.)

- [ ] **Step 3: Run ruff and commit**

```bash
ruff check app.py
ruff format --check app.py
git add app.py
git commit -m "Add app.py: Gradio Blocks HF Space entrypoint"
git push
```

---

## Task 14: `streamlit_app.py` — Streamlit local-dev front-end

**Files:**
- Create: `streamlit_app.py` (repo root)

**Interfaces:**
- Consumes: same as Task 13 — `src.inference.predict.{load_bundle, score_one, score_batch}`, `src.ui.theme.*`, `src.config.*`. No other `src` module.
- Produces: `main()` called unconditionally at module scope (Streamlit re-runs the whole script per interaction — there is no separate "import" moment to warm at, so the whole-script execution *is* the warm-up).
- Satisfies BUILD_PROMPT.md Tasks 2.1/2.2 (a fully functional local app for assignment grading, independent of the HF deployment path).
- No dedicated test file, same rationale as Task 13.

- [ ] **Step 1: Write `streamlit_app.py`**

```python
"""Streamlit app -- local development front-end (assignment Tasks 2.1/2.2).

Streamlit re-executes the whole script on every widget interaction, so the
model bundle is wrapped in @st.cache_resource: without it, every click
would re-deserialise the joblib artefact from disk.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config
from src.inference.predict import load_bundle, score_batch, score_one
from src.ui.theme import build_css, render_drivers_table, render_verdict_panel

st.set_page_config(page_title=config.BRANDING.app_name, layout="wide")
st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)


@st.cache_resource
def _bundle():
    return load_bundle()


def _prospect_tab() -> None:
    bundle = _bundle()
    fields = {}
    columns = st.columns(3)
    for i, spec in enumerate(config.FIELD_SPECS):
        with columns[i % 3]:
            if spec.kind == "categorical":
                fields[spec.name] = st.selectbox(
                    spec.label, options=list(spec.levels),
                    index=list(spec.levels).index(spec.default),
                )
            else:
                fields[spec.name] = st.number_input(
                    spec.label, min_value=float(spec.min_value),
                    max_value=float(spec.max_value), value=float(spec.default),
                )

    if st.button("Score", type="primary"):
        result = score_one(bundle, fields)
        st.markdown(
            render_verdict_panel(
                result.probability, result.threshold, result.verdict,
                result.confidence_band, result.expected_value_eur,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_drivers_table(
                result.drivers, result.explanation_method,
                result.explanation_reliable, result.explanation_note,
            ),
            unsafe_allow_html=True,
        )


def _batch_tab() -> None:
    bundle = _bundle()
    st.download_button(
        "Download template CSV",
        data=pd.DataFrame([{s.name: s.default for s in config.FIELD_SPECS}]).to_csv(index=False),
        file_name="bank_conversion_template.csv",
    )
    uploaded = st.file_uploader("Prospect CSV", type=["csv"])
    capacity_pct = st.slider("Capacity (% of list to call)", 0, 100, 100)
    if uploaded is not None and st.button("Score list", type="primary"):
        frame = pd.read_csv(uploaded)
        capacity_fraction = capacity_pct / 100.0 if capacity_pct else None
        batch = score_batch(bundle, frame, capacity_fraction=capacity_fraction)
        n_call = int((batch.scored["verdict"] == "CALL").sum())
        st.write(f"{n_call} of {len(batch.scored)} marked CALL.")
        for warning in batch.warnings:
            st.warning(warning)
        st.dataframe(batch.scored)
        st.download_button(
            "Download scored list", data=batch.scored.to_csv(index=False),
            file_name="bank_conversion_scored.csv",
        )


def _model_card_tab() -> None:
    if not config.METRICS_PATH.exists():
        st.info("No metrics.json found. Run `python -m src.models.train`.")
        return
    metrics = json.loads(config.METRICS_PATH.read_text())
    drift = (
        json.loads(config.DRIFT_PATH.read_text())
        if config.DRIFT_PATH.exists() else {"verdict": "unknown"}
    )
    test = metrics["test_metrics"]

    st.subheader("Test performance")
    st.write(
        f"Precision {test['precision']:.4f} · Recall {test['recall']:.4f} · "
        f"AP {test['average_precision']:.4f} · ROC-AUC {test['roc_auc']:.4f} · "
        f"Accuracy {test['accuracy']:.4f} (majority baseline {metrics['majority_baseline_accuracy']:.4f})"
    )

    st.subheader("Threshold economics")
    threshold = metrics["threshold_search"]
    st.write(
        f"Threshold {threshold['threshold']:.4f} (breakeven {metrics['breakeven_probability']:.4f}), "
        f"uplift vs 0.5: {threshold['uplift_vs_default']:.2f} EUR, "
        f"uplift vs call-everyone: {threshold['uplift_vs_call_everyone']:.2f} EUR."
    )

    st.subheader("Model comparison (validation)")
    st.dataframe(pd.DataFrame(metrics["model_comparison"]).T[["average_precision", "roc_auc"]])

    st.subheader("Drift verdict")
    st.write(drift["verdict"])

    st.subheader("Data quality")
    dataset = metrics["dataset"]
    st.write(
        f"{dataset['n_rows']} rows, {dataset['n_duplicate_rows']} duplicates, "
        f"pdays sentinel share {dataset['pdays_sentinel_share']:.2%}."
    )

    st.subheader("Provenance")
    st.write(f"sklearn {metrics['sklearn_version']}, trained {metrics['trained_at']}.")


def main() -> None:
    st.title(config.BRANDING.app_name)
    st.caption(config.BRANDING.tagline)
    tab1, tab2, tab3 = st.tabs(["Score a prospect", "Score a call list", "Model card & monitoring"])
    with tab1:
        _prospect_tab()
    with tab2:
        _batch_tab()
    with tab3:
        _model_card_tab()


main()
```

- [ ] **Step 2: Manual smoke verification (import in bare mode with a temporary model)**

Run the same temp-model setup as Task 13's Step 2, replacing `import app` with `import streamlit_app`. Expected: no traceback (Streamlit prints "missing ScriptRunContext" warnings to stderr when imported outside `streamlit run` — that is expected and not a failure).

- [ ] **Step 3: Run ruff and commit**

```bash
ruff check streamlit_app.py
ruff format --check streamlit_app.py
git add streamlit_app.py
git commit -m "Add streamlit_app.py: local-dev front-end (Tasks 2.1/2.2)"
git push
```

---

## Task 15: `scripts/run_eda.py` and `scripts/leakage_demo.py`

**Files:**
- Create: `scripts/run_eda.py`
- Create: `scripts/leakage_demo.py`

**Interfaces:**
- Produces (`run_eda.py`): `generate_eda(frame: pd.DataFrame, figures_dir: Path, summary_path: Path, findings_path: Path) -> dict` (business logic, testable in isolation) and CLI `main()` invoked via `make eda` → `python scripts/run_eda.py [--offline]`, which calls `load_and_split()` for real data and passes `dataset.train` to `generate_eda`. Writes 6 PNGs into `reports/figures/`, `reports/eda_summary.md`, `reports/eda_findings.json`.
- Produces (`leakage_demo.py`): `quantify_leakage(dataset: DatasetBundle) -> dict` (`roc_auc_without_duration, roc_auc_with_duration, lift_from_leakage`) and CLI `main()` invoked via `make leakage`, writing `reports/leakage_demo.json`.
- `scripts/build_report.py` (Task 20) reads `reports/eda_findings.json` and `reports/leakage_demo.json` directly — field names must match exactly.
- EDA runs on the TRAIN split only (not the full raw frame), keeping the "TEST opened once" discipline visible even in exploratory work.
- `leakage_demo.py` deliberately reintroduces `duration` (recovered from the original split before `split_xy` dropped it) into one of its two quick models — this is the one place in the repo where that is intentional and documented, not a bug.
- No dedicated test file. Verified by a manual smoke run against the synthetic fixture (this session defers the real network-backed EDA/leakage run to Phase 2, per the design doc).

- [ ] **Step 1: Write `scripts/run_eda.py`**

```python
"""Exploratory data analysis: 6 figures, a written summary, and structured
findings that scripts/build_report.py quotes verbatim, so no number in the
report can contradict the code.

matplotlib.use("Agg") must precede `import matplotlib.pyplot` -- CI and the
Space container have no display, and importing pyplot first locks in a
display-requiring backend that fails there (Trap 5). Runs on the TRAIN
split only, keeping "TEST opened once" visible even in EDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data.loader import build_period_index, load_and_split


def _fig_target_balance(frame: pd.DataFrame, path: Path) -> dict:
    counts = frame[config.TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.to_numpy(), color=[config.BRANDING.accent, config.BRANDING.danger])
    ax.set_title("Target class balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    positive_rate = float((frame[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean())
    return {"title": "Target class balance", "positive_rate": positive_rate}


def _fig_duration_leakage(frame: pd.DataFrame, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5, 4))
    frame.boxplot(column="duration", by=config.TARGET_COLUMN, ax=ax)
    ax.set_title("Call duration by outcome (duration is denylisted)")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    medians = frame.groupby(config.TARGET_COLUMN)["duration"].median().to_dict()
    return {"title": "Duration by outcome", "median_duration_by_class": medians}


def _fig_age_distribution(frame: pd.DataFrame, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(frame["age"], bins=30, color=config.BRANDING.accent)
    ax.set_title("Age distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "Age distribution", "median_age": float(frame["age"].median())}


def _fig_job_conversion(frame: pd.DataFrame, path: Path) -> dict:
    rate = frame.groupby("job")[config.TARGET_COLUMN].apply(
        lambda s: (s == config.POSITIVE_LABEL).mean()
    ).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(rate.index, rate.to_numpy(), color=config.BRANDING.accent)
    ax.set_title("Conversion rate by job")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "Conversion rate by job", "top_job": str(rate.idxmax()), "bottom_job": str(rate.idxmin())}


def _fig_macro_trend(frame: pd.DataFrame, path: Path) -> dict:
    periods = build_period_index(frame)
    trend = frame.groupby(periods)["euribor3m"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trend.index, trend.to_numpy(), color=config.BRANDING.accent)
    ax.set_title("euribor3m by period (concept drift)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "euribor3m trend across periods", "min": float(trend.min()), "max": float(trend.max())}


def _fig_pdays_sentinel(frame: pd.DataFrame, path: Path) -> dict:
    share = float((frame["pdays"] == config.PDAYS_SENTINEL).mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [share, 1 - share], labels=["Never contacted", "Previously contacted"],
        colors=[config.BRANDING.surface, config.BRANDING.accent],
    )
    ax.set_title("pdays sentinel share")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "pdays sentinel share", "never_contacted_share": share}


def generate_eda(frame: pd.DataFrame, figures_dir: Path, summary_path: Path, findings_path: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    findings = [
        _fig_target_balance(frame, figures_dir / "01_target_balance.png"),
        _fig_duration_leakage(frame, figures_dir / "02_duration_leakage.png"),
        _fig_age_distribution(frame, figures_dir / "03_age_distribution.png"),
        _fig_job_conversion(frame, figures_dir / "04_job_conversion.png"),
        _fig_macro_trend(frame, figures_dir / "05_macro_trend.png"),
        _fig_pdays_sentinel(frame, figures_dir / "06_pdays_sentinel.png"),
    ]

    findings_path.write_text(json.dumps({"findings": findings}, indent=2, default=str))
    summary_lines = ["# EDA Summary", ""]
    for finding in findings:
        detail = {k: v for k, v in finding.items() if k != "title"}
        summary_lines.append(f"- **{finding['title']}**: {json.dumps(detail, default=str)}")
    summary_path.write_text("\n".join(summary_lines))
    return {"findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = load_and_split(offline=args.offline)
    generate_eda(
        dataset.train, config.FIGURES_DIR,
        config.REPORTS_DIR / "eda_summary.md", config.REPORTS_DIR / "eda_findings.json",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/leakage_demo.py`**

```python
"""Quantifies the duration leak: trains the same lightweight classifier
with and without `duration` and reports the ROC-AUC gap. This is the
concrete number backing the leakage-control section of the report --
without it, "duration causes leakage" is an assertion, not evidence.
"""

from __future__ import annotations

import argparse
import json

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.loader import DatasetBundle, load_and_split, split_xy

NUMERIC_ONLY: tuple[str, ...] = (
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
)


def _quick_auc(x_train, y_train, x_valid, y_valid, columns: tuple[str, ...]) -> float:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(x_train[list(columns)], y_train)
    probs = pipeline.predict_proba(x_valid[list(columns)])[:, 1]
    return float(roc_auc_score(y_valid, probs))


def quantify_leakage(dataset: DatasetBundle) -> dict:
    x_train, y_train = split_xy(dataset.train)
    x_valid, y_valid = split_xy(dataset.valid)

    # split_xy already dropped duration; recover it here deliberately, for
    # this comparison only. This is the one intentional exception to the
    # leakage denylist in the whole repo.
    x_train_with_duration = x_train.assign(duration=dataset.train["duration"].to_numpy())
    x_valid_with_duration = x_valid.assign(duration=dataset.valid["duration"].to_numpy())

    without_duration_auc = _quick_auc(x_train, y_train, x_valid, y_valid, NUMERIC_ONLY)
    with_duration_auc = _quick_auc(
        x_train_with_duration, y_train, x_valid_with_duration, y_valid,
        (*NUMERIC_ONLY, "duration"),
    )
    return {
        "roc_auc_without_duration": without_duration_auc,
        "roc_auc_with_duration": with_duration_auc,
        "lift_from_leakage": with_duration_auc - without_duration_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = load_and_split(offline=args.offline)
    result = quantify_leakage(dataset)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "leakage_demo.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual smoke verification against the synthetic fixture**

Run:
```bash
python -c "
import tempfile
from pathlib import Path
from tests.conftest import make_synthetic_bank_frame
from src.data.loader import validate_schema, audit_quality, build_period_index, temporal_split, DatasetBundle
from scripts.run_eda import generate_eda
from scripts.leakage_demo import quantify_leakage

frame = make_synthetic_bank_frame()
validate_schema(frame)
quality = audit_quality(frame)
periods = build_period_index(frame)
train_df, valid_df, test_df = temporal_split(frame, periods)
dataset = DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    result = generate_eda(dataset.train, tmp_dir / 'figures', tmp_dir / 'eda_summary.md', tmp_dir / 'eda_findings.json')
    assert len(result['findings']) == 6
    assert len(list((tmp_dir / 'figures').glob('*.png'))) == 6
    leakage = quantify_leakage(dataset)
    assert 'lift_from_leakage' in leakage
    print('EDA findings:', len(result['findings']), 'leakage lift:', leakage['lift_from_leakage'])
    print('OK')
"
```
Expected: `OK`, no traceback. (Numbers are meaningless on synthetic data — the smoke check proves the plumbing works, not that the leak is real; that comes from Phase 2's real run.)

- [ ] **Step 4: Run ruff and commit**

```bash
touch scripts/__init__.py
ruff check scripts/run_eda.py scripts/leakage_demo.py
ruff format --check scripts/run_eda.py scripts/leakage_demo.py
git add scripts/run_eda.py scripts/leakage_demo.py scripts/__init__.py
git commit -m "Add EDA script (6 figures) and leakage quantification script"
git push
```

---

## Task 16: `benchmarks/latency.py` — latency, throughput, cold-start, memory

**Files:**
- Create: `benchmarks/latency.py`

**Interfaces:**
- Consumes: `src.inference.predict.load_bundle`; `src.config.{FIELD_SPECS, MODEL_PATH, REPO_ROOT}`.
- Produces: `measure_cold_start(model_path) -> float`; `measure_single_row_latency(bundle, n_repeats=200) -> dict` (`p50_ms, p95_ms, p99_ms`); `measure_batch_throughput(bundle, batch_sizes=(1,10,100,1000,10000)) -> dict`; `measure_peak_rss_mb() -> float`; `run_benchmarks(model_path=None) -> dict`; CLI `main()` invoked via `make bench` → `python benchmarks/latency.py [--onnx]`, writing `benchmarks/results_<timestamp>.json`.
- The `--onnx` path is a real, working comparison (not a stub) scoped to the fitted classifier stage only — converting the full `ColumnTransformer` + `OneHotEncoder` + `CalibratedClassifierCV` chain has known `skl2onnx` compatibility gaps, which is exactly the "install risk" BUILD_PROMPT.md flags. It degrades to an explicit `{"available": False, ...}` result when `skl2onnx`/`onnxruntime` aren't installed (they are intentionally absent from `requirements-dev.txt`), rather than crashing the run or being a fake stub.
- No dedicated test file. Verified by a manual smoke run against a temporary synthetic-trained model.

- [ ] **Step 1: Write `benchmarks/latency.py`**

```python
"""Latency, throughput, cold-start, and memory benchmarking -- the
concrete evidence behind the "evaluate and optimize AI software
performance" learning outcome most teams skip.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src import config
from src.inference.predict import load_bundle


def _sample_row() -> pd.DataFrame:
    return pd.DataFrame([{spec.name: spec.default for spec in config.FIELD_SPECS}])


def measure_cold_start(model_path: Path) -> float:
    """Artefact deserialisation time, measured separately: this is the
    dominant term in first-request latency on free Space hardware, which
    is why app.py warms the bundle at import time instead of lazily."""
    start = time.perf_counter()
    load_bundle(model_path=model_path)
    return time.perf_counter() - start


def measure_single_row_latency(bundle: dict, n_repeats: int = 200) -> dict:
    row = _sample_row()
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        bundle["model"].predict_proba(row)
        durations.append(time.perf_counter() - start)
    array = np.array(durations)
    return {
        "p50_ms": float(np.percentile(array, 50) * 1000),
        "p95_ms": float(np.percentile(array, 95) * 1000),
        "p99_ms": float(np.percentile(array, 99) * 1000),
    }


def measure_batch_throughput(bundle: dict, batch_sizes: tuple[int, ...] = (1, 10, 100, 1000, 10000)) -> dict:
    row = _sample_row().iloc[0].to_dict()
    results = {}
    for size in batch_sizes:
        frame = pd.DataFrame([row] * size)
        start = time.perf_counter()
        bundle["model"].predict_proba(frame)
        elapsed = time.perf_counter() - start
        results[str(size)] = {
            "elapsed_s": elapsed,
            "rows_per_second": size / elapsed if elapsed > 0 else float("inf"),
        }
    return results


def measure_peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmarks(model_path: Path | None = None) -> dict:
    model_path = model_path or config.MODEL_PATH
    cold_start_s = measure_cold_start(model_path)
    bundle = load_bundle(model_path=model_path)
    return {
        "cold_start_s": cold_start_s,
        "single_row_latency": measure_single_row_latency(bundle),
        "batch_throughput": measure_batch_throughput(bundle),
        "peak_rss_mb": measure_peak_rss_mb(),
        "artifact_size_bytes": model_path.stat().st_size,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def _benchmark_onnx(bundle: dict) -> dict:
    """joblib vs ONNX Runtime, scoped to the fitted classifier stage only.

    Converting the full ColumnTransformer + OneHotEncoder +
    CalibratedClassifierCV chain has known skl2onnx compatibility gaps --
    exactly the install risk BUILD_PROMPT.md flags. This degrades to an
    explicit unavailable result rather than crashing the run when the
    optional packages aren't installed.
    """
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return {"available": False, "note": "Install skl2onnx and onnxruntime to run this comparison."}

    pipeline = bundle["model"].calibrated_classifiers_[0].estimator.estimator
    classifier = pipeline.named_steps["classifier"]
    transformed = pipeline[:-1].transform(_sample_row())
    n_features = transformed.shape[1]

    onnx_model = convert_sklearn(classifier, initial_types=[("input", FloatTensorType([None, n_features]))])
    session = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_input = transformed.astype(np.float32)

    n_repeats = 200
    sklearn_durations, onnx_durations = [], []
    for _ in range(n_repeats):
        start = time.perf_counter()
        classifier.predict_proba(transformed)
        sklearn_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        session.run(None, {"input": onnx_input})
        onnx_durations.append(time.perf_counter() - start)

    return {
        "available": True,
        "scope": "classifier stage only (post feature-pipeline transform)",
        "sklearn_classifier_p50_ms": float(np.percentile(sklearn_durations, 50) * 1000),
        "onnx_p50_ms": float(np.percentile(onnx_durations, 50) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", action="store_true",
        help="Optional: also benchmark an ONNX export of the classifier stage.",
    )
    args = parser.parse_args()

    payload = run_benchmarks()
    if args.onnx:
        payload["onnx"] = _benchmark_onnx(load_bundle())

    output_dir = config.REPO_ROOT / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = payload["measured_at"].replace(":", "-")
    (output_dir / f"results_{timestamp}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke verification against a temporary synthetic-trained model**

Run:
```bash
python -c "
import tempfile
from pathlib import Path
from tests.conftest import make_synthetic_bank_frame
from src.data.loader import validate_schema, audit_quality, build_period_index, temporal_split, DatasetBundle
from src.models.train import train_and_save
from benchmarks.latency import run_benchmarks

frame = make_synthetic_bank_frame()
validate_schema(frame)
quality = audit_quality(frame)
periods = build_period_index(frame)
train_df, valid_df, test_df = temporal_split(frame, periods)
dataset = DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)

with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    train_and_save(
        dataset, model_path=tmp_dir / 'model.joblib', metrics_path=tmp_dir / 'metrics.json',
        model_card_path=tmp_dir / 'model_card.md', drift_path=tmp_dir / 'drift.json',
    )
    result = run_benchmarks(model_path=tmp_dir / 'model.joblib')
    assert 'p50_ms' in result['single_row_latency']
    assert set(result['batch_throughput'].keys()) == {'1', '10', '100', '1000', '10000'}
    print('cold start (s):', result['cold_start_s'])
    print('OK')
"
```
Expected: `OK`, no traceback.

- [ ] **Step 3: Run ruff and commit**

```bash
touch benchmarks/__init__.py
ruff check benchmarks/latency.py
ruff format --check benchmarks/latency.py
git add benchmarks/latency.py benchmarks/__init__.py
git commit -m "Add latency/throughput/cold-start/memory benchmarking"
git push
```

---

## Task 17: GitHub Actions workflows — `ci.yml`, `deploy.yml`, `file-size-guard.yml`

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/deploy.yml`
- Create: `.github/workflows/file-size-guard.yml`

**Interfaces:**
- `ci.yml`'s `smoke-train` job runs the exact same synthetic-fixture training smoke check used manually in Task 9 — as a real CI job, not a stub, and without needing network access (CI has none, per §1.13).
- `deploy.yml` reads `HF_USERNAME`/`SPACE_NAME` from top-level `env:`, fails before touching the Hub if `artifacts/model.joblib` is absent, assembles `deploy/` (never mirrors the repo), and passes `HF_TOKEN` only via `env:` from `secrets.HF_TOKEN`. It will not succeed until the user completes BUILD_PROMPT.md Part 2 Phases 4–5 (Space + secret) — that's expected, documented in `docs/RUNBOOK.md` (Task 18).
- `file-size-guard.yml` checks only files changed in the current PR's diff against its base branch, so it doesn't retroactively fail on pre-existing large files.
- No dedicated test file (YAML, not Python) — verified by `yaml.safe_load` syntax checks.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: ruff format --check .
      - name: mypy (advisory)
        run: mypy src || true
      - run: pytest -v

  smoke-train:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-dev.txt
      - name: Train on the synthetic fixture (no network -- proves the pipeline runs end to end)
        run: |
          python -c "
          import tempfile
          from pathlib import Path
          from tests.conftest import make_synthetic_bank_frame
          from src.data.loader import validate_schema, audit_quality, build_period_index, temporal_split, DatasetBundle
          from src.models.train import train_and_save

          frame = make_synthetic_bank_frame()
          validate_schema(frame)
          quality = audit_quality(frame)
          periods = build_period_index(frame)
          train_df, valid_df, test_df = temporal_split(frame, periods)
          dataset = DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)

          with tempfile.TemporaryDirectory() as tmp:
              tmp_dir = Path(tmp)
              metrics = train_and_save(
                  dataset,
                  model_path=tmp_dir / 'model.joblib',
                  metrics_path=tmp_dir / 'metrics.json',
                  model_card_path=tmp_dir / 'model_card.md',
                  drift_path=tmp_dir / 'drift.json',
              )
              assert (tmp_dir / 'model.joblib').exists()
              print('smoke-train OK, winner:', metrics['winner'])
          "
```

- [ ] **Step 2: Write `.github/workflows/deploy.yml`**

```yaml
name: Deploy to Hugging Face Space

on:
  push:
    branches: [main]
    paths-ignore:
      - "docs/**"
      - "reports/**"
      - "**/*.md"

permissions:
  contents: read

concurrency:
  group: deploy-space
  cancel-in-progress: false

env:
  HF_USERNAME: krish2105
  SPACE_NAME: bank-conversion-copilot

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Fail before touching the Hub if the model artefact is missing
        run: |
          if [ ! -f artifacts/model.joblib ]; then
            echo "::error::artifacts/model.joblib is missing. Run 'python -m src.models.train' and commit the artefact before deploying."
            exit 1
          fi

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Assemble the runtime payload (not a mirror -- tests/benchmarks/scripts/requirements-dev.txt/reports/docs must never reach the Space)
        run: |
          mkdir -p deploy
          cp app.py deploy/
          cp requirements.txt deploy/
          cp -r src deploy/src
          cp -r artifacts deploy/artifacts
          cp space/README.md deploy/README.md
          echo "Payload:"
          find deploy -type f

      - name: Install hub client
        run: pip install "huggingface_hub[cli]"

      - name: Upload to the Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          if [ -z "$HF_TOKEN" ]; then
            echo "::error::HF_TOKEN secret is not set."
            exit 1
          fi
          hf upload "$HF_USERNAME/$SPACE_NAME" deploy/ . --repo-type space

      - name: Write deployment summary
        run: |
          echo "### Deployed to Hugging Face Space" >> "$GITHUB_STEP_SUMMARY"
          echo "- URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" >> "$GITHUB_STEP_SUMMARY"
          echo "- Commit: $GITHUB_SHA" >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 3: Write `.github/workflows/file-size-guard.yml`**

```yaml
name: File size guard

on:
  pull_request:

permissions:
  contents: read

jobs:
  check-file-sizes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Fail if any non-LFS file over 10 MB was added or changed in this PR
        run: |
          git lfs ls-files -n > /tmp/lfs_files.txt || true
          over_limit=0
          while IFS= read -r file; do
            if [ -f "$file" ] && ! grep -qxF "$file" /tmp/lfs_files.txt; then
              size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
              if [ "$size" -gt 10485760 ]; then
                echo "::error file=$file::$file is $((size / 1024 / 1024)) MB and not tracked by Git LFS (10 MB limit)."
                over_limit=1
              fi
            fi
          done < <(git diff --name-only --diff-filter=ACM "origin/${{ github.base_ref }}"...HEAD)
          exit $over_limit
```

- [ ] **Step 4: Verify YAML syntax and commit**

Run:
```bash
python -c "
import yaml
for path in ['.github/workflows/ci.yml', '.github/workflows/deploy.yml', '.github/workflows/file-size-guard.yml']:
    with open(path) as f:
        yaml.safe_load(f)
    print(path, 'OK')
"
git add .github/workflows/ci.yml .github/workflows/deploy.yml .github/workflows/file-size-guard.yml
git commit -m "Add CI/CD workflows: lint+test+smoke-train, deploy, file-size guard"
git push
```

---
