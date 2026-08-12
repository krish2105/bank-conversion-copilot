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

> **Note:** `artifacts/` currently contains a placeholder model trained on
> synthetic data, so the apps are demonstrable out of the box. Run
> `python -m src.models.train` against the real dataset before trusting
> any number in `artifacts/metrics.json` or the model card.

## Dataset

UCI Machine Learning Repository, Bank Marketing (id 222),
`bank-additional-full` variant. Moro, S., Cortez, P., & Rita, P. (2014). A
data-driven approach to predict the success of bank telemarketing.
*Decision Support Systems*, 62, 22-31.
https://archive.ics.uci.edu/dataset/222/bank+marketing

## License

MIT -- see [LICENSE](LICENSE).
