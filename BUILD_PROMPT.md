# Bank Conversion Copilot — Master Build Prompt & Execution Plan

**For:** Krishna Mathur, Atharva Soundankar, Yash Petkar
**Course:** Software Development for AI Models — Final Group Project (30% of unit)
**Institution:** SP Jain School of Global Management, Dubai
**Deadline:** submission + presentation, 14 Aug 2026
**Target repo:** `https://github.com/krish2105/bank-conversion-copilot`

---

## How to use this document

1. Create an empty repo and clone it.
2. Save this file at the repo root as `BUILD_PROMPT.md`.
3. Open Claude Code in that directory and paste **Part 1** as your first message
   (or say: *"Read BUILD_PROMPT.md and execute Part 1 in full."*).
4. Work through **Part 2** yourself — it contains the human-only steps
   (Hugging Face account, tokens, screenshots) that Claude Code cannot do.

**Read Part 2 Phase 0 before you write a line of code.** It contains a hard
30-day gate that can sink the whole deployment if you discover it late.

---

---

# PART 1 — MASTER PROMPT (paste this into Claude Code)

You are building a complete, production-grade ML deployment project for a graded
university capstone. Build it end to end: real code, real tests, real CI/CD, no
placeholders, no `TODO`, no stub functions.

## 1.1 The assignment being satisfied

Five graded parts:

1. **Model development** — real-world dataset, EDA, preprocessing, at least two
   models compared, best model saved with joblib/pickle.
2. **Application** — build a user-friendly Streamlit app, test it locally.
3. **Deployment** — deploy to a Hugging Face Space.
4. **CI/CD** — GitHub repo; `HF_TOKEN` stored as a GitHub Actions secret and
   never hard-coded; a `deploy.yml` that runs on push to `main`; prove automated
   deployment with a visible change.
5. **Report** — max 15 pages, `.docx`, containing: problem statement, dataset +
   source, EDA, preprocessing, model comparison, model justification, app
   screenshots, repo structure, workflow explanation, successful-workflow
   screenshot, deployed-app screenshot, both links, and an explanation of how
   automated deployment works.

Rubric criteria: **Research Depth, Innovation, Application of Techniques,
Presentation and Reporting.** The unit learning outcome most teams ignore is
*"Evaluate and optimize AI software performance"* — this build deliberately
targets it via latency benchmarking, drift monitoring, and cost-based threshold
optimisation.

## 1.2 Project identity

**Name:** Bank Conversion Copilot
**Tagline:** Cost-optimised targeting for outbound term-deposit campaigns
**Problem:** A retail bank's call centre has finite agent hours and an ~11%
conversion rate on term-deposit telemarketing. Calling everyone wastes money;
calling too few leaves revenue unrealised. The question is *who to phone*, not
merely *who will convert*.

## 1.3 HARD CONSTRAINTS — verified platform facts, do not re-litigate these

These were researched and confirmed. Treat them as given.

| Fact | Consequence for this build |
|---|---|
| Hugging Face **deprecated the Streamlit SDK on 2025-04-30**. Streamlit now requires the Docker template. | Cannot deploy a Streamlit Space on the built-in SDK. |
| **Gradio and Docker Spaces require a paid plan** (PRO $9/mo personal). Static Spaces are free. | Docker route is blocked on a free account. |
| **Exception:** free personal accounts *in good standing* may host **up to 2 Gradio Spaces on ZeroGPU**. Good standing = verified email **AND** account older than 30 days. | This is the only free path. |
| **ZeroGPU is Gradio-SDK only.** | The deployed front-end must be Gradio. |
| Files **> 10 MB require Git-LFS** to sync to a Space. | Keep `model.joblib` small; add a CI size guard. |
| Academia Hub requires a **250-seat minimum**. | Not available to a 3-person team. |
| Free Spaces **sleep when idle**; cold start is 60–90s. | Warm the Space before the live demo; warm the model at import time. |

### Resulting architecture decision (implement exactly this)

- **`app.py`** — Gradio Blocks app. This is the Hugging Face Space entrypoint.
- **`streamlit_app.py`** — a *fully functional* Streamlit app for local
  development, satisfying assignment Tasks 2.1 and 2.2.
- **Both front-ends import ONLY `src.inference.predict`.** Neither contains any
  scoring logic. There is exactly one implementation of "what does the model
  say", so the two UIs cannot disagree, and the tests exercise that shared path
  rather than either UI.
- Document the deviation in `docs/ADR-001-deployment-target.md`, citing the
  sources below, and state it in the report rather than leaving a marker to
  notice it.

Sources to cite in the ADR:
- `https://huggingface.co/docs/hub/en/spaces-sdks-streamlit`
- `https://huggingface.co/docs/hub/en/spaces-overview`
- `https://huggingface.co/docs/hub/en/spaces-zerogpu`
- `https://huggingface.co/docs/hub/en/spaces-github-actions`
- `https://huggingface.co/docs/hub/main/spaces-changelog`
- `https://huggingface.co/docs/hub/en/academia-hub`

## 1.4 Dataset

**UCI Machine Learning Repository, Bank Marketing, dataset id 222** — the
`bank-additional-full` variant.

| Property | Value |
|---|---|
| Rows | 41,188 |
| Input columns | 20 |
| Separator | `;` (semicolon, not comma) |
| Period | May 2008 – November 2010, **rows are ordered by date** |
| Target | `y` ∈ {`yes`, `no`}, ~11.27% positive |
| Landing page | `https://archive.ics.uci.edu/dataset/222/bank+marketing` |
| Direct zip | `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` |
| Nested path | outer zip → `bank-additional.zip` → `bank-additional/bank-additional-full.csv` |

Citation: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to
predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.

### Four dataset properties that drive the entire design

1. **`duration` is documented target leakage.** UCI's own attribute
   documentation states it highly affects the target, is not known before a call
   is performed, and after the call the outcome is already known — so it must be
   discarded for a realistic predictive model. Including it lifts ROC-AUC by
   roughly 0.13 while making the model undeployable. **Put it on an enforced
   denylist.**
2. **`unknown` is a missing marker, not a level**, in six categorical columns:
   `job`, `marital`, `education`, `default`, `housing`, `loan`.
   `pandas.isna()` returns **zero** nulls for this dataset — that is the trap.
3. **`pdays == 999` is a sentinel** meaning "never previously contacted", not
   "999 days ago". It affects ~96% of rows. Left as a magnitude it gives the
   column a mean near 960 with near-zero variance, so standardisation crushes
   the genuine 0–27 day range into numerical noise.
4. **Five Banco de Portugal macro features** — `emp.var.rate`,
   `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` — span the 2008
   financial crisis. Euribor falls from ~5% to under 1%. This is a genuine
   concept-drift story, and these features are **constant within a calendar
   month**, which matters for splitting (see 1.6).

> **Alternative considered and rejected:** UCI 296 (Diabetes 130-US Hospitals,
> 101,766 encounters, 50 features) has a higher research-depth ceiling but needs
> substantial ICD-9 feature engineering — wrong risk profile for a 48-hour
> deadline. If you ever switch, only `src/config.py` and `src/data/loader.py`
> change.

## 1.5 KNOWN API TRAPS — these were hit and fixed during a trial build

Do not rediscover these. The target environment is **Python 3.12,
scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4**.

**Trap 1 — `CalibratedClassifierCV(cv="prefit")` was REMOVED in scikit-learn 1.8.**
It raises `InvalidParameterError`. The replacement is `FrozenEstimator`:

```python
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

# WRONG on sklearn >= 1.8:
#   CalibratedClassifierCV(winner, method="isotonic", cv="prefit")
# RIGHT:
calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")
calibrated.fit(x_valid, y_valid)
```

This matters semantically, not just syntactically: the base estimator is already
fitted on TRAIN and must **not** be refitted. Passing an integer `cv` would refit
it on the validation data and destroy the separation between fitting,
calibrating, and testing.

**Trap 2 — `LogisticRegression(penalty="l2")` is deprecated in 1.8** and emits a
`FutureWarning`. L2 is the default; control it through `C` alone.

**Trap 3 — dataclass construction.** When a metrics dataclass gains a field,
every constructor call must pass it. A missing keyword surfaces as
`TypeError: __init__() missing 1 required positional argument` deep inside a
training run. Construct with explicit keywords and keep them complete.

**Trap 4 — `ucimlrepo` sometimes serves the legacy 17-input variant** with no
macro columns. Detect this (`if "euribor3m" not in frame.columns`) and fall back
to the direct zip download.

**Trap 5 — `matplotlib.use("Agg")` must precede `import matplotlib.pyplot`**, or
figure generation fails in CI where there is no display.

## 1.6 Two anti-leakage measures — both are required

**Measure 1: drop `duration`**, enforced by a denylist in config, a guard that
raises inside the feature pipeline, and a test that fails the build.

**Measure 2: chronological split with boundaries snapped to month edges.**

The file is date-ordered but ships no date column — only a month name.
Reconstruct a monotonic period index by walking rows in file order and
incrementing the year whenever the month number goes backwards (a calendar
wrap), starting at 2008. Then split 70% train / 15% validation / 15% test **by
row order, with each boundary snapped to a period edge.**

Snapping is not cosmetic: macro features are constant within a month, so a
mid-month boundary would put that month's economic conditions on both sides of
the split. That is a subtle leak a plain `.iloc` split would introduce.

**Discipline to enforce everywhere:**
- Both candidate models fit on TRAIN only.
- Model selection, calibration, and threshold search all use VALIDATION only.
- TEST is opened exactly once, at the end. Nothing is tuned against it.

## 1.7 Modelling specification

**Two candidates, as the brief requires:**

| Model | Role | Config |
|---|---|---|
| `LogisticRegression` | baseline | `C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"`, **scaled numerics** |
| `HistGradientBoostingClassifier` | challenger | `learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30`, **unscaled** |

Justify in the report: logistic regression is the model a bank's model-risk
function will actually approve, and it sets the floor a complex model must beat
by enough to justify its opacity. HistGradientBoosting is chosen over
RandomForest specifically because the fitted artefact is roughly an order of
magnitude smaller — which matters against the 10 MB non-LFS Space limit.

**Selection metric: `average_precision` (PR-AUC).** With ~11% positives, ROC-AUC
is optimistic and accuracy is actively misleading. Report accuracy *only to
dismiss it* against the majority-class baseline (~88.7%).

**Calibration:** isotonic, fitted on validation via `FrozenEstimator`. Required
because the threshold logic multiplies probabilities by euros — a predicted 0.30
must mean a real 30%. Report a reliability curve with quantile bins (not uniform:
most predictions cluster below 0.2 and uniform bins leave the upper bins empty)
plus Expected Calibration Error.

## 1.8 The innovation layer — this is where the marks above a pass live

### Cost-optimal decision threshold

Do **not** use 0.5, and do not maximise F1. Define a cost matrix in config:

```python
cost_per_call = 8.0                # EUR, fully loaded agent + telephony
revenue_per_subscription = 120.0   # EUR, discounted contribution margin
cost_of_missed_customer = 0.0      # campaign is capacity-limited, not one-shot
```

Theoretical break-even = 8/120 = **0.0667**. Then sweep 201 thresholds over
[0,1] **on the validation set** and pick the one maximising expected net value.
Grid rather than analytic, because the analytic break-even assumes perfect
calibration and real calibration is imperfect even after isotonic regression.

Report three numbers: the optimal threshold, uplift versus the default 0.5, and
uplift versus the status-quo "call everyone". Also implement a
**capacity-constrained mode** (`call the top N% we can staff`) because real call
centres are constrained by agent hours, not by list quality.

### Drift monitoring, implemented from scratch in numpy

- **PSI** for numeric features: 10 **reference** quantile bins (never combined
  bins — that lets the current period influence its own yardstick and
  understates drift). Bands: <0.10 stable, 0.10–0.25 monitor, >0.25 significant.
- **Jensen-Shannon divergence, base 2** for categoricals (symmetric, bounded
  [0,1], well-behaved when a level is absent from one side, unlike plain KL).
- Return a single verdict: `STABLE` / `MONITOR` / `RETRAIN RECOMMENDED`.

No external drift library: one fewer dependency in a constrained container, and
the arithmetic is simple enough that writing it out makes it defensible under
viva questioning. If you cannot explain the number, do not quote it.

### Explanations with graceful degradation

Three stages, because SHAP's `TreeExplainer` support for sklearn's
`HistGradientBoosting` has varied by version and the deployed app must not crash:

1. SHAP `TreeExplainer` — exact signed per-prediction contributions
2. Linear coefficient × standardised value — exact for the logistic model
3. Global permutation importance — approximate, direction-free, always works

Stage 3 is genuinely worse; the UI **must say so** via a `reliable: bool` flag
and an honest note. Silently presenting a weaker explanation as if it were SHAP
is the wrong trade.

Normalise SHAP output shape defensively — depending on versions it returns
`(n, f)`, `(n, f, 2)`, or a two-element list.

### Latency and memory benchmarking (`benchmarks/latency.py`)

This is what hits the "evaluate and optimize AI software performance" outcome.
Measure and write `benchmarks/results_*.json`:

- p50 / p95 / p99 single-row latency
- batch throughput across batch sizes (1, 10, 100, 1000, 10000)
- **cold-start cost**: artefact deserialisation time, measured separately —
  this is the dominant term in first-request latency on free Space hardware,
  which is *why* `app.py` warms the bundle at import time
- peak RSS via `psutil`
- artefact size on disk
- optional `--onnx` flag comparing joblib vs ONNX Runtime (mark clearly optional;
  `skl2onnx` adds install risk)

### Confidence bands

Distinguish "0.42, comfortably above a 0.09 cutoff" from "0.091, a coin-flip
either side". Bands: `< 0.02` from cutoff = Borderline, `< 0.08` = Moderate,
else Clear. Showing a bare CALL/SKIP for both invites false confidence.

## 1.9 Full file manifest

```
bank-conversion-copilot/
├── app.py                          # Gradio Blocks — HF Space entrypoint
├── streamlit_app.py                # Streamlit — local dev (Tasks 2.1/2.2)
├── requirements.txt                # runtime only; sklearn pinned EXACTLY
├── requirements-dev.txt            # + streamlit, pytest, ruff, mypy, matplotlib, psutil, python-docx
├── pyproject.toml                  # ruff (line-length 90), mypy, pytest config
├── .pre-commit-config.yaml         # incl. check-added-large-files --maxkb=10240
├── Makefile                        # install/train/test/lint/fmt/app/streamlit/eda/leakage/bench/report/clean
├── .gitignore                      # data/*.csv ignored; artifacts/ COMMITTED
├── .streamlit/config.toml          # dark theme matching src/ui/theme.py
├── README.md                       # GitHub-facing, NO HF frontmatter
│
├── src/
│   ├── config.py                   # SINGLE SOURCE OF TRUTH (see 1.10)
│   ├── data/loader.py              # 3-strategy load, validate, quality audit, period index, temporal split
│   ├── features/pipeline.py        # DomainFeatureBuilder + ColumnTransformer + leakage guard
│   ├── models/train.py             # the entrypoint; emits all 4 artefacts
│   ├── models/threshold.py         # cost-optimal + capacity-constrained threshold
│   ├── models/evaluate.py          # MetricSet, reliability curve + ECE, curve points
│   ├── monitor/drift.py            # PSI + Jensen-Shannon, pure numpy
│   ├── inference/predict.py        # THE serving layer — both UIs import only this
│   ├── explain/shap_engine.py      # 3-stage degrading explainer
│   └── ui/theme.py                 # shared design tokens + CSS + HTML fragments
│
├── tests/
│   ├── conftest.py                 # schema-faithful synthetic fixture (offline CI)
│   ├── test_leakage.py             # THE most valuable tests in the repo
│   ├── test_pipeline.py            # sentinel, unknown count, unseen categories
│   ├── test_scoring.py             # threshold economics, metrics, drift
│   └── test_inference.py           # end-to-end serving contract
│
├── scripts/
│   ├── run_eda.py                  # 6 figures + eda_summary.md + eda_findings.json
│   ├── leakage_demo.py             # quantifies the duration leak
│   └── build_report.py             # generates the 15-page .docx from metrics.json
│
├── benchmarks/latency.py
│
├── .github/workflows/
│   ├── ci.yml                      # ruff, mypy (advisory), pytest matrix 3.11+3.12, smoke-train job
│   ├── deploy.yml                  # push to main → assemble payload → hf upload
│   └── file-size-guard.yml         # fails PR on >10 MB non-LFS files
│
├── space/README.md                 # HF Space card WITH yaml frontmatter
├── docs/
│   ├── RUNBOOK.md
│   ├── ADR-001-deployment-target.md
│   └── PRESENTATION_OUTLINE.md
│
├── artifacts/                      # model.joblib, metrics.json, model_card.md, drift.json
└── reports/figures/
```

## 1.10 `src/config.py` is the single source of truth

Everything imports from it; nothing hard-codes a column name twice. The most
common bug in a student ML repo is the training pipeline and the serving pipeline
disagreeing about feature order or dtype. Because both read this file, they
cannot drift apart.

It must contain: all paths (resolved from repo root so behaviour is identical in
a notebook, from the CLI, and inside the Space container at `/home/user/app`);
UCI identity constants; `LEAKAGE_DENYLIST` + `LEAKAGE_REASONS` (reason strings
get rendered into the model card and report — so every denylisted column needs
one); the column schema; `PDAYS_SENTINEL = 999`; engineered feature names;
temporal-split config + `MONTH_ORDER`; a frozen `CostMatrix` dataclass with
`net_value()` and `breakeven_probability`; `SELECTION_METRIC`; **19 `FieldSpec`
entries** for the app inputs with **verbatim UCI category levels** (the app must
offer exactly these strings or the fitted encoder treats the value as unseen); a
`Branding` dataclass carrying design tokens shared by both front-ends; and a
`Runtime` dataclass.

## 1.11 Engineered features (three, inside the Pipeline)

Every transformation lives **inside** the sklearn Pipeline. Nothing is done to
the dataframe before `fit`. That is what makes serving safe — the app hands raw
user input straight to `predict_proba` and the identical transformations run with
parameters learned only from TRAIN. The alternative (scaling in a notebook, then
pickling only the classifier) is the single most common cause of train/serve skew
and it is invisible until production predictions quietly go wrong.

1. `never_contacted_before` — from `pdays == 999`; then set `pdays` to `NaN` for
   the imputer.
2. `n_unknown_fields` — count of `'unknown'` across the six bearing columns. How
   many fields a client refused is itself informative.
3. `contact_intensity` = `campaign / (previous + 1)` — separates "hammering a
   cold prospect" from "following up a warm one"; raw counts cannot express the
   ratio and tree splits on them are less efficient.

Also: normalise categorical case/whitespace inside the transformer, so a
hand-edited CSV with `"  RETIRED  "` lands on the `retired` level.

`OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
sparse_output=False)` — a call-centre app that 500s because someone typed a new
job title is worse than one that degrades gracefully.

Keep `'unknown'` as its own category rather than imputing it: a client declining
to state their job is a signal, and imputing to the mode destroys it.

## 1.12 Front-end requirements

Both apps, three tabs each: **Score a prospect** / **Score a call list** /
**Model card & monitoring**.

Design direction: dark, dense, financial-terminal. Deliberately *not* the default
look of either framework — a stock Gradio or Streamlit page reads as a prototype,
and "Presentation and Reporting" is a graded criterion. One shared design system
in `src/ui/theme.py`; nothing styled inline in either app file.

Required UI behaviours:
- Verdict panel with a probability bar and a **visible cutoff marker** at the
  threshold, so the non-0.5 cutoff is self-evident.
- Expected value of *this* call in EUR, coloured by sign.
- Drivers table from the explainer, with the method named and unreliability
  flagged.
- Batch tab: CSV upload → ranked list with `priority_rank`, downloadable;
  template-CSV generator; capacity slider.
- Tolerant batch ingestion: missing columns filled with documented defaults,
  unseen categories mapped, leaky columns dropped — and **every assumption
  reported back as a warning**. An operations user should never see a stack
  trace.
- Model card tab: test metrics, the accuracy-vs-majority-baseline explanation,
  threshold economics, model comparison table, drift verdict, data-quality
  findings, split timeline, provenance (sklearn version, data content hash).
- `app.py` must warm the bundle at import time and wrap `import spaces` in
  `try/except`. Set `api_name` on both handlers so the Space also exposes a REST
  endpoint for free.
- `streamlit_app.py` must use `@st.cache_resource` for the bundle, since
  Streamlit re-executes the whole script on every widget change.

## 1.13 Testing requirements

The suite must run **offline with no dataset** — CI has no network. Build a
schema-faithful synthetic fixture in `conftest.py`: same column names and
dtypes, the same `'unknown'` strings in the same six columns, the 999 sentinel,
a **dec→mar month wrap** so year reconstruction is genuinely exercised, ~11%
positives, a handful of exact duplicates, and a **deliberately weak** learnable
signal (a fixture where AUC hits 0.99 hides bugs a realistic one exposes).

Faithful *structure*, not faithful *statistics*. Fixtures verify the code is
correct; they say nothing about model quality. Every quality claim in the report
comes from a real training run.

Must-have tests:
- `duration` absent after `split_xy`; guard raises when reintroduced; not in
  `APP_FIELD_ORDER`; not in encoder output names
- no period appears in two slices; splits ordered, disjoint, exhaustive
- year increments on month wrap; period index monotonic
- sentinel → flag + NaN; unknown count exact; contact intensity exact
- unseen category does not raise; output dense and finite; names match matrix width
- cost matrix arithmetic; confusion matrix at a known threshold; optimised
  threshold never loses to 0.5; capacity mode hits the requested share
- PSI zero for identical distributions, grows with mean shift, handles constants;
  JS bounded [0,1]
- end-to-end: train tiny model → save in production artefact format → reload via
  the real `load_bundle` → `score_one` / `score_batch` / explain. Redirect
  everything into `tmp_path` so the suite never overwrites artefacts a teammate
  is about to demo.
- `load_bundle` error message must name the command to run, not just "not found"

## 1.14 `deploy.yml` — assemble, do not mirror

The obvious approach is `huggingface/hub-sync@v0.1.0`, which mirrors the whole
repo. **Do not do that.** This repo contains things the Space must never receive:
`tests/`, `benchmarks/`, `scripts/`, `requirements-dev.txt` (which pulls in
streamlit, pytest, matplotlib), `reports/`, `docs/`.

A Space rebuild reinstalls every dependency it can see, and free-tier build
minutes are the slowest part of the deploy loop. So stage exactly the runtime
payload into `deploy/` — `app.py`, `requirements.txt`, `src/`, `artifacts/`, and
`space/README.md` → `README.md` — then `hf upload`. The staging manifest doubles
as documentation of what the Space actually needs.

Other requirements:
- `env: HF_USERNAME` / `SPACE_NAME` at the top, so there is one place to edit.
- **Fail before touching the Hub** if `artifacts/model.joblib` is missing.
  Deploying an app whose model is absent produces a Space that builds
  successfully and then errors for every user — the worst failure mode.
- Pass the token as `env: HF_TOKEN: ${{ secrets.HF_TOKEN }}` so it never appears
  on a command line where it could land in a process listing or log. Fail with a
  clear message if unset.
- `permissions: contents: read` (least privilege).
- `concurrency: group: deploy-space, cancel-in-progress: false` — never let two
  deploys race.
- `paths-ignore` for `docs/**`, `reports/**`, `**/*.md` so documentation edits
  don't rebuild the Space.
- Write the Space URL and commit SHA to `$GITHUB_STEP_SUMMARY`.

`space/README.md` frontmatter: `sdk: gradio`, `sdk_version: 6.20.0`,
`python_version: "3.12"`, `app_file: app.py`, plus title/emoji/colors/license.
Keep `sdk_version` and the `gradio==` pin in `requirements.txt` **identical**.

## 1.15 Pinning

Pin `scikit-learn==1.8.0` **exactly** — a joblib artefact is a pickle of
scikit-learn objects, and loading it under a different minor version is not
guaranteed to work and can fail silently rather than loudly. Bound but do not
freeze the rest (`pandas>=2.2,<3.1`, `numpy>=2.0,<3.0`) so security patches flow.
`gradio==6.20.0`. Have `load_bundle` warn when the runtime sklearn version
differs from the artefact's recorded version.

Keep `requirements.txt` short: every package there is installed on every rebuild.

## 1.16 Report generator (`scripts/build_report.py`)

Generate `reports/Final_Group_Project_Report.docx`, **≤15 pages, target ~14** to
leave trim room. Read numbers from `artifacts/metrics.json`,
`reports/eda_findings.json`, and `reports/leakage_demo.json` so **no number in
the report can contradict the code**. Insert clearly marked screenshot
placeholders in the app-screenshots and CI-evidence sections.

Use the npm `docx` library (preinstalled — `require('docx')` directly, do not
`npm install`). Gotchas: A4 is the default page size; tables need `columnWidths`
on the table **and** `width` on every cell, both in `WidthType.DXA`; use
`ShadingType.CLEAR` (never `SOLID`, which renders black); never insert `•`
literally — use a `numbering` config with `LevelFormat.BULLET`; never use `\n` —
use separate `Paragraph` elements; `PageBreak` must sit inside a `Paragraph`;
TOC requires built-in `HeadingLevel.*`.

Verify by converting and looking at it:
```bash
python /mnt/skills/public/docx/scripts/office/soffice.py --headless --convert-to pdf reports/Final_Group_Project_Report.docx
pdftoppm -jpeg -r 100 reports/Final_Group_Project_Report.pdf page
```

## 1.17 Auto-generated model card

`artifacts/model_card.md`, written by `train.py` so it can never contradict
`metrics.json`. Header must say *auto-generated, do not edit by hand*. Sections:
overview table, intended use, **out-of-scope use** (explicitly: not for credit
decisions — conversion propensity is not a proxy for credit risk and the model
was never validated for it), test performance with accuracy listed **last** and
the baseline beside it, business framing, leakage controls rendered from
`LEAKAGE_REASONS`, and known limitations (temporal validity, geographic
specificity, `unknown` category bias, **no fairness certification despite age
and marital status being inputs**).

## 1.18 Definition of done

- `pytest` green; `ruff check .` and `ruff format --check .` clean
- `python -m src.models.train` completes and writes all four artefacts
- **Sanity check: test ROC-AUC lands ~0.78–0.81.** If 0.92+, `duration` leaked
  back in. If below 0.70, the split is wrong.
- Model artefact < 10 MB
- Both apps launch and score locally
- `make eda`, `make leakage`, `make bench`, `make report` all succeed
- Every module has a docstring explaining **why**, not just what — the comments
  are part of the graded deliverable

Write real code. Explain trade-offs in comments where a marker or a viva examiner
would ask "why did you do it that way".

---

---

# PART 2 — EXECUTION PLAN (human steps)

Owner column: **K** = Krishna, **A** = Atharva, **Y** = Yash.
Total hands-on ≈ 3 hours plus report writing.

## Phase 0 — BLOCKING PREREQUISITE, do this first (K, 5 min)

A free HF personal account can host up to 2 Gradio Spaces on ZeroGPU only if it
is **in good standing: verified email AND older than 30 days.**

1. `https://huggingface.co/settings/account` → confirm email verified
2. Check the account creation date

**If any of the three accounts is >30 days old → use it. Proceed.**
**If none is:**

| Option | Cost | Action |
|---|---|---|
| Use an older account belonging to a classmate/senior via an org you're added to | Free | 10 min |
| One member buys PRO for a month → Docker SDK → deploy `streamlit_app.py` instead | $9, cancel after grading | 20 min |
| Deploy the Gradio app elsewhere and document that HF blocked you | Free | Weakest — loses Part 3 marks |

Do not discover this at 2 a.m.

## Phase 1 — Local setup (all, 15 min)

```bash
git clone https://github.com/krish2105/bank-conversion-copilot.git
cd bank-conversion-copilot
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
pytest -v
```

Tests run on the synthetic fixture — no network, no dataset needed. If this
fails, the environment is wrong, not the code.

## Phase 2 — Train on real data (K, 10 min)

```bash
python -m src.models.train
make eda
make leakage
make bench
git add artifacts reports && git commit -m "Train on UCI bank-additional-full" && git push
```

**Record these from the log — the report needs them:** both models' validation
AP and ROC-AUC, the winner, the cost-optimal threshold and its net value, test
AP / ROC-AUC / precision / recall, and artefact size.

Sanity-check against 1.18 before moving on.

## Phase 3 — Local app test (A, 15 min) — Task 2.2

```bash
streamlit run streamlit_app.py     # http://localhost:8501
python app.py                      # http://localhost:7860
```

Checklist:
- [ ] Defaults produce a probability and CALL/SKIP verdict
- [ ] `poutcome=success` + `pdays=6` → probability rises noticeably
- [ ] `campaign=25` → probability falls
- [ ] Drivers table populates with real feature names
- [ ] Download template CSV → re-upload → ranked list
- [ ] Upload CSV missing a column → warning appears, no crash
- [ ] Capacity mode at 10% → ~10% marked CALL
- [ ] Model card tab shows metrics, threshold, drift verdict

**Screenshots — name them exactly:**

| File | Shows |
|---|---|
| `01-app-single-prospect.png` | Verdict panel with probability bar + cutoff marker |
| `02-app-drivers.png` | Explanation table |
| `03-app-batch.png` | Ranked call list + summary tiles |
| `04-app-modelcard.png` | Metrics, threshold economics, drift verdict |

## Phase 4 — Create the Space (K, 10 min)

`https://huggingface.co/new-space` → owner = the account that passed Phase 0 →
name `bank-conversion-copilot` → **SDK: Gradio** → **Hardware: ZeroGPU** (this
is what makes it free; do not pick CPU basic) → Public → Create. Leave it empty.

If ZeroGPU is not offered, the account failed the good-standing check. Back to
Phase 0.

## Phase 5 — Wire up CI/CD (K, 15 min) — Tasks 4.2 & 4.3

**Token:** `https://huggingface.co/settings/tokens` → Fine-grained → name
`github-actions-bank-copilot` → **Write** access scoped to that Space only →
copy once.

**Secret:** repo → `Settings → Secrets and variables → Actions → New repository
secret` → name **exactly** `HF_TOKEN` → paste. Never put it in a file.
Screenshot the page with the value hidden.

**Check `deploy.yml` matches your account** — edit `HF_USERNAME` if it isn't
`krish2105`.

```bash
git commit --allow-empty -m "Trigger first Space deployment" && git push origin main
```

Actions goes green in 1–2 min; the Space rebuilds in another 1–3. Visit
`https://<username>-bank-conversion-copilot.hf.space`.

## Phase 6 — Prove the automation (K, 10 min) — Task 4.4

Change the tagline in `src/config.py`, commit, push to `main`.

| File | Shows |
|---|---|
| `05-actions-run-green.png` | Successful workflow run, all steps ticked |
| `06-actions-step-summary.png` | Deploy summary with Space URL |
| `07-space-updated.png` | Live app showing the NEW tagline |
| `08-github-secret.png` | `HF_TOKEN` listed, value hidden |
| `09-repo-structure.png` | Repo file tree on GitHub |

Screenshot 07 beside 01 is the proof the pipeline is real. Without it, Part 4 is
an unverified claim.

## Phase 7 — Report (all, 60–90 min)

```bash
make report
```

Then by hand: insert screenshots at the marked placeholders; write the EDA
interpretation (figures are generated, the reading of them is yours — 3–4
observations, each stating what you see *and why it matters*); trim to 15 pages
(cut future-work first, then the appendix); **verify both links work from a
logged-out browser.**

## Phase 8 — Presentation (all, 45 min)

Follow `docs/PRESENTATION_OUTLINE.md`. **Open the Space 15 minutes before you
present** — free Spaces sleep, and a cold start is 90 seconds of dead air in
front of the class. Keep `04-app-modelcard.png` open in a background tab as a
Wi-Fi fallback. Do not live-edit code.

---

# PART 3 — Report structure (15 pages, section → data source)

| § | Section | Pages | Numbers come from |
|---|---|---|---|
| 1 | Problem statement & business objective | 1 | narrative + `CostMatrix` |
| 2 | Dataset & source (with citation) | 1 | `metrics.json → dataset` |
| 3 | EDA & key observations | 2.5 | `eda_findings.json` + 6 figures |
| 4 | Preprocessing & feature engineering | 1.5 | `config.py` schema |
| 5 | **Leakage control** (the differentiator) | 1.5 | `leakage_demo.json` |
| 6 | Model comparison & justification | 2 | `metrics.json → model_comparison` |
| 7 | Threshold optimisation & business value | 1.5 | `metrics.json → threshold_search` |
| 8 | Application (screenshots 01–04) | 1.5 | manual insert |
| 9 | Repo structure, CI/CD, deployment (05–09) | 2 | manual insert + workflow text |
| 10 | Performance, drift, limitations, future work | 1 | `benchmarks/`, `drift.json` |

Non-negotiable inclusions: both links, the workflow explanation, the successful
workflow screenshot, the deployed-app screenshot.

---

# PART 4 — Anticipated viva questions

**"Why is your accuracy only ~90%?"** Because accuracy is the wrong question.
Predicting "no" for everyone scores 88.7% here. Our value is in the ranking,
which average precision measures and accuracy cannot.

**"Why drop a feature that improved the score?"** Because it improved it by
cheating. Call duration doesn't exist when you decide whether to place the call.
A model that needs it cannot be deployed.

**"Why not 0.5?"** Because 0.5 assumes a false positive and a false negative cost
the same. Here one costs EUR 8 and the other a EUR 120 opportunity. Break-even
is 0.067.

**"Why Gradio when the brief said Streamlit?"** HF deprecated the Streamlit SDK
in April 2025 and Docker Spaces now need a paid plan. We built both front-ends
over one shared inference module and deployed the one the free tier allows. See
ADR-001.

**"How do you know it still works next quarter?"** We measure it — PSI per
numeric feature against the training distribution, with a retrain verdict the app
displays. Not a guess.

**"Who wrote what?"** Check the commit history — that's why we used pull requests.

---

# PART 5 — Submission checklist

**Code**
- [ ] `pytest` passes · `ruff check .` clean
- [ ] All four artefacts committed
- [ ] Commits from **all three members** under their own names
- [ ] At least one merged PR with a teammate's review comment

**Deployment**
- [ ] Space public and loads for a logged-out visitor
- [ ] `HF_TOKEN` is a GitHub secret, appears nowhere in code
- [ ] Latest Actions run green
- [ ] Phase 6 visible change is live

**Report (15 pages max, .docx)** — all 14 required elements from §1.1 item 5.

---

# PART 6 — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Space: `ModuleNotFoundError: No module named 'src'` | `src/` missing from payload — check the assemble step's printed file list |
| Space build fails on `sdk_version` | Bump `sdk_version` in `space/README.md` **and** the `gradio==` pin together |
| App loads, every prediction errors | `model.joblib` missing or sklearn version mismatch — check the `load_bundle` warning in Space logs |
| Deploy 401/403 | Token lacks write permission, wrong scope, or secret misspelled — must be exactly `HF_TOKEN` |
| Test ROC-AUC 0.90+ | `duration` is back — `pytest tests/test_leakage.py -v` |
| `ucimlrepo` returns wrong variant | Loader auto-falls back to the zip; else download manually into `data/` and run `--offline` |
| `InvalidParameterError: 'cv' ... Got 'prefit'` | sklearn 1.8 — use `FrozenEstimator` (see 1.5) |
