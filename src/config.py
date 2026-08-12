"""Single source of truth for schema, paths, and business constants.

Both the training pipeline and the serving layer import from this module
instead of hard-coding column names or category levels. That is what makes
it structurally impossible for them to disagree about feature order or
dtype -- the most common bug in a student ML repo.
"""

from __future__ import annotations

from dataclasses import dataclass
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
