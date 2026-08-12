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

    Nothing in the test suite may write into the real artifacts/ dir -- a
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
