# Bank Conversion Copilot

Cost-optimised targeting for outbound term-deposit telemarketing campaigns.
A retail bank's call centre has finite agent hours and an ~11% conversion
rate -- the question this project answers is *who to phone*, not merely
*who will convert*.

Built for Software Development for AI Models (SP Jain School of Global
Management, Dubai) -- final group project.

## Live app

- Hugging Face Space: https://huggingface.co/spaces/krish21may/bank
  (deploys automatically from `main` -- see `.github/workflows/deploy.yml`.)

## What's here

| Path | What |
|---|---|
| `app.py` | Gradio Blocks app -- the deployed Space entrypoint |
| `streamlit_app.py` | Streamlit app -- local development |
| `src/` | Shared library: config, data loading, feature pipeline, models, drift monitoring, explainability, inference |
| `tests/` | Offline test suite against a synthetic schema-faithful fixture |
| `scripts/` | EDA, leakage quantification, report generation |
| `benchmarks/` | Latency/throughput/memory benchmarking |
| `docs/ADR-001-deployment-target.md` | Why Gradio, not Streamlit |
| `docs/RUNBOOK.md` | Setup, training, deployment, troubleshooting |

## Quickstart

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    pre-commit install
    python -m pytest -v            # offline, synthetic fixture, no dataset needed
    python -m src.models.train     # real training run (needs network)
    streamlit run streamlit_app.py

See `docs/RUNBOOK.md` for the full workflow.

> **Note:** `artifacts/` contains a real model trained on the actual UCI
> dataset. Its test ROC-AUC (~0.52) is lower than BUILD_PROMPT.md's
> predicted ~0.78-0.81 band — this was investigated and is a genuine,
> explainable property of a strict chronological split on this dataset
> (severe regime shift between the row-heavy 2008 training period and the
> 2009-2010 test period), not a bug. See `docs/RUNBOOK.md` for the full
> investigation.

## Deployment

Deployed automatically by `.github/workflows/deploy.yml` on every push to
`main`: it assembles a minimal runtime payload (`app.py`, `requirements.txt`,
`src/`, `artifacts/`, the Space README) and uploads it to the Hugging Face
Space using the `HF_TOKEN` GitHub secret. See
`docs/ADR-001-deployment-target.md` and `docs/RUNBOOK.md` for the full
detail.

## Dataset

UCI Machine Learning Repository, Bank Marketing (id 222),
`bank-additional-full` variant. Moro, S., Cortez, P., & Rita, P. (2014). A
data-driven approach to predict the success of bank telemarketing.
*Decision Support Systems*, 62, 22-31.
https://archive.ics.uci.edu/dataset/222/bank+marketing

## License

MIT -- see [LICENSE](LICENSE).
