# Bank Conversion Copilot

Cost-optimised targeting for outbound term-deposit telemarketing campaigns.
A retail bank's call centre has finite agent hours and an ~11% conversion
rate -- the question this project answers is *who to phone*, not merely
*who will convert*.

Built for Software Development for AI Models (SP Jain School of Global
Management, Dubai) -- final group project.

## Live app

- Hugging Face Space: https://huggingface.co/spaces/krish2105/bank-conversion-copilot
  (deploys automatically from `main` -- see `.github/workflows/deploy.yml`.
  Not live until Phases 4-5 of `BUILD_PROMPT.md` Part 2 are complete.)

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

See `docs/RUNBOOK.md` for the full workflow, and `BUILD_PROMPT.md` for the
complete build specification this repository implements.

> **Note:** `artifacts/` contains a real model trained on the actual UCI
> dataset. Its test ROC-AUC (~0.52) is lower than BUILD_PROMPT.md's
> predicted ~0.78-0.81 band — this was investigated and is a genuine,
> explainable property of a strict chronological split on this dataset
> (severe regime shift between the row-heavy 2008 training period and the
> 2009-2010 test period), not a bug. See `docs/RUNBOOK.md` for the full
> investigation.

## Deploying to Hugging Face (what's left)

This needs your own Hugging Face login and cannot be automated by an
agent — see `BUILD_PROMPT.md` Part 2 Phases 0, 4, 5 for full detail.
Short version:

1. **Check account age** at https://huggingface.co/settings/account —
   needs a verified email and an account older than 30 days for free
   ZeroGPU Spaces.
2. **Create the Space**: https://huggingface.co/new-space → SDK **Gradio**
   → Hardware **ZeroGPU** → name `bank-conversion-copilot`.
3. **Create a token**: https://huggingface.co/settings/tokens → Fine-grained
   → Write access scoped to that Space.
4. **Add it as a GitHub secret**: repo Settings → Secrets and variables →
   Actions → New repository secret → name exactly `HF_TOKEN`.
5. **Trigger deploy**: `git commit --allow-empty -m "Deploy" && git push` —
   `.github/workflows/deploy.yml` runs automatically and uploads `app.py`,
   `src/`, `artifacts/`, and `requirements.txt` to the Space.

## Dataset

UCI Machine Learning Repository, Bank Marketing (id 222),
`bank-additional-full` variant. Moro, S., Cortez, P., & Rita, P. (2014). A
data-driven approach to predict the success of bank telemarketing.
*Decision Support Systems*, 62, 22-31.
https://archive.ics.uci.edu/dataset/222/bank+marketing

## License

MIT -- see [LICENSE](LICENSE).
