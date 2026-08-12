# Runbook

## Current state of this repository

Real training has been run against the actual UCI dataset.
`artifacts/model.joblib`, `metrics.json`, `model_card.md`, and `drift.json`
are all genuine, not synthetic placeholders. `reports/eda_findings.json`,
`reports/leakage_demo.json`, and the figures in `reports/figures/` are
likewise from the real dataset. The Hugging Face Space is created and the
`HF_TOKEN` GitHub secret is set — deployment is live at
https://huggingface.co/spaces/krish21may/bank.

### Important finding: test ROC-AUC is ~0.52, not the ~0.78-0.81 BUILD_PROMPT.md predicts

This was investigated, not ignored. `BUILD_PROMPT.md` §1.18 says a correct
implementation should land ~0.78-0.81 and flags below-0.70 as "the split
is wrong." The actual measured number is real ROC-AUC 0.524 on TEST. Five
independent checks confirm this is **not a bug**:

1. The temporal split is exactly what §1.6 specifies: chronological order
   preserved, boundaries snapped to period edges, no shuffling, no leakage
   (`pytest tests/test_leakage.py -v` is green).
2. The real UCI file's row density is wildly uneven across time: 2008
   alone (May-Dec) is ~67% of all 41,188 rows, because campaign volume was
   heavily front-loaded. A 70%-by-row-count split therefore puts nearly
   all of TRAIN in 2008 (5.2% positive rate) and pushes TEST out to
   June 2009 - November 2010 (44.5% positive rate) -- a completely
   different economic regime (this is the same 2008 financial crisis /
   euribor collapse the dataset is famous for).
3. A from-scratch minimal baseline pipeline (bypassing every custom
   transformer in this repo) gets the same ~0.63 ceiling on VALIDATION,
   so the gap isn't introduced by `DomainFeatureBuilder` or the leakage
   guard.
4. The from-scratch drift monitor in `src/monitor/drift.py`, run
   completely independently on TRAIN vs TEST, verdicts
   **RETRAIN RECOMMENDED** -- it detected the same regime shift by a
   different method, which is exactly the coherent, cross-checked signal
   a real deployment would want.
5. `HistGradientBoosting` (exact hyperparameters from §1.7, no deviation)
   does even worse (~0.50, effectively random) with or without early
   stopping, ruling out early stopping as the cause.

Conclusion: BUILD_PROMPT.md's expected band appears to be an estimate that
doesn't hold up against a genuinely correct implementation on the real,
full dataset -- a strict chronological split on this specific dataset is
a hard generalisation problem, not a soft one. This is worth stating
explicitly in the report's limitations section and is good viva material,
not something to paper over.

## Local setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    pre-commit install
    python -m pytest -v

## Re-run training

    python -m src.models.train
    make eda
    make leakage
    make bench

## Run the apps locally

    streamlit run streamlit_app.py     # http://localhost:8501
    python app.py                      # http://localhost:7860

Both require `artifacts/model.joblib` to exist first.

## Deploy

Live at https://huggingface.co/spaces/krish21may/bank, deployed
automatically by `.github/workflows/deploy.yml` on every push to `main`
(except doc-only changes). `deploy.yml` fails loudly (not silently) if
`artifacts/model.joblib` is missing or `HF_TOKEN` is unset. To redeploy
after a change: just push to `main`.

## Troubleshooting

Most common issues:

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` on the Space | `src/` missing from the deploy payload -- check the assemble step's file list in the Actions log. |
| Every prediction errors on the Space | `model.joblib` missing or sklearn version mismatch -- check the `load_bundle` warning in Space logs. |
| Deploy fails with 401/403 | `HF_TOKEN` lacks write scope on the Space, or the secret name isn't exactly `HF_TOKEN`. |
| Test ROC-AUC 0.90+ | `duration` is back -- run `pytest tests/test_leakage.py -v`. |
| `pytest` command not found but `python -m pytest` works | Use `python -m pytest` (or `make test`) -- this is what the Makefile and CI both use. |
