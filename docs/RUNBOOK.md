# Runbook

## Current state of this repository

This scaffold was built against `BUILD_PROMPT.md` Part 1. It contains
real, tested code for every module in the file manifest, validated
against the synthetic fixture in `tests/conftest.py`, and a **placeholder**
trained model (`artifacts/model.joblib` etc.) produced from that same
synthetic fixture purely so the two apps can actually launch and be
demonstrated. It deliberately does not include:

- A real training run against the UCI dataset with trustworthy metrics.
- A Hugging Face Space, token, or GitHub secret.
- Screenshots or report content that depends on either of the above.

These are exactly `BUILD_PROMPT.md` Part 2's human steps. Follow Part 2
Phases 0-8 in order; Phase 0 is a blocking prerequisite, do it first.

**Before doing anything else, replace the placeholder artifacts with a
real training run** (see below) -- the numbers currently in
`artifacts/metrics.json` and `artifacts/model_card.md` are from synthetic
data and must not be quoted in the report.

## Local setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    pre-commit install
    python -m pytest -v

## Train on real data

    python -m src.models.train
    make eda
    make leakage
    make bench

Sanity-check test ROC-AUC lands ~0.78-0.81 (see `BUILD_PROMPT.md` §1.18).
If it's 0.92+, `duration` leaked back in -- run
`pytest tests/test_leakage.py -v`. If it's below 0.70, the temporal split
is wrong.

## Run the apps locally

    streamlit run streamlit_app.py     # http://localhost:8501
    python app.py                      # http://localhost:7860

Both require `artifacts/model.joblib` to exist first.

## Deploy

See `BUILD_PROMPT.md` Part 2 Phases 4-6: create the Space, add the
`HF_TOKEN` GitHub secret, push to `main`. `deploy.yml` fails loudly (not
silently) if `artifacts/model.joblib` is missing or the secret is unset.

## Troubleshooting

See `BUILD_PROMPT.md` Part 6 for the full table. Most common:

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` on the Space | `src/` missing from the deploy payload -- check the assemble step's file list in the Actions log. |
| Every prediction errors on the Space | `model.joblib` missing or sklearn version mismatch -- check the `load_bundle` warning in Space logs. |
| Deploy fails with 401/403 | `HF_TOKEN` lacks write scope on the Space, or the secret name isn't exactly `HF_TOKEN`. |
| Test ROC-AUC 0.90+ | `duration` is back -- run `pytest tests/test_leakage.py -v`. |
| `pytest` command not found but `python -m pytest` works | Use `python -m pytest` (or `make test`) -- this is what the Makefile and CI both use. |
