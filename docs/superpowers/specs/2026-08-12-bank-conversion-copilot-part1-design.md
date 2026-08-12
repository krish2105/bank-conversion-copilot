# Bank Conversion Copilot — Part 1 build design

**Date:** 2026-08-12
**Source spec:** [`BUILD_PROMPT.md`](../../../BUILD_PROMPT.md), Part 1 (§1.1–1.18)
**Status:** Approved by user, proceeding to implementation.

## Scope

Build the complete Part 1 deliverable exactly as specified in `BUILD_PROMPT.md`:
a graded university capstone ML deployment project (Bank Conversion Copilot —
cost-optimised targeting for outbound term-deposit campaigns). The source
document is itself a fully prescriptive architecture ("implement exactly
this," "do not re-litigate these") — this design doc records the delta
between that spec and this execution session, not an alternative design.

Full file manifest, data contract, modelling spec, leakage controls, UI
requirements, testing requirements, and CI/CD requirements are as written in
`BUILD_PROMPT.md` §1.2–1.18 and are not restated here.

## Decisions made for this session

1. **Python 3.13, not 3.12.** The target machine has Python 3.11 and 3.13
   available, not 3.12. Verified via `pip install --dry-run` that every
   pinned package in the spec resolves to a working cp313 wheel at the exact
   or bounded version requested: `scikit-learn==1.8.0` (exact),
   `pandas>=2.2,<3.1` (resolves 3.0.5), `numpy>=2.0,<3.0` (resolves 2.5.2),
   `gradio==6.20.0` (exact). No pin changes needed — only the interpreter
   version differs from the spec's stated target.

2. **No real training run in this session.** The full pipeline (loader,
   feature pipeline, both models, calibration, threshold search, drift
   monitor, explainer, benchmarks) will be built and validated against the
   synthetic schema-faithful fixture in `tests/conftest.py` only, per
   §1.13. Running `python -m src.models.train` against the real UCI dataset,
   sanity-checking test ROC-AUC against the 0.78–0.81 band, and committing
   real artifacts is left for the user (BUILD_PROMPT.md Part 2, Phase 2).
   Consequence: `make report` will run structurally but the generated
   `.docx` will reflect whatever `artifacts/metrics.json` etc. contain at
   that point (absent until a real training run — the report script's
   correctness is validated, not its numbers).

3. **No Hugging Face Space work.** No Space is created, no `HF_TOKEN` is
   requested or set. `deploy.yml` is built correctly per §1.14 but will not
   successfully run until the user completes Part 2 Phases 0, 4, and 5. This
   is expected and is called out in `docs/RUNBOOK.md`.

4. **Git workflow.** Repo was empty on GitHub at session start. `git init`
   was run locally, `origin` added pointing at
   `https://github.com/krish2105/bank-conversion-copilot.git`, and commits
   will be pushed directly to `main` as work lands, since there is no
   existing history to disrupt and this is the initial scaffold.

## Definition of done for this session

- `pytest` green against the synthetic fixture; `ruff check .` and
  `ruff format --check .` clean.
- Every path in the §1.9 manifest exists and is non-stub: `src/` modules,
  both front-end entrypoints, `tests/`, `scripts/`, `benchmarks/latency.py`,
  the three GitHub Actions workflows, `docs/ADR-001-deployment-target.md`,
  `docs/RUNBOOK.md`, `docs/PRESENTATION_OUTLINE.md`, `space/README.md`.
- `make eda`, `make leakage`, `make bench`, `make report` all execute
  successfully (against synthetic/placeholder data where no real run has
  happened yet).
- Model artifact size guard, leakage guard, and temporal-split tests are
  real and passing — these are the tests the spec calls "the most valuable
  in the repo."
- Everything committed and pushed to `main`.
- Real training, HF Space creation, secrets, screenshots, and report
  content are explicitly out of scope for this session and remain on the
  user per BUILD_PROMPT.md Part 2.
