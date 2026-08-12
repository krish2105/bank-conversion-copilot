# ADR-001: Deployment target is Gradio on HF Spaces (ZeroGPU), not Streamlit

## Status

Accepted.

## Context

The assignment brief specifies a Streamlit app. The following platform
facts were verified before this build:

- Hugging Face deprecated the Streamlit SDK on 2025-04-30; Streamlit
  Spaces now require the Docker template.
- Gradio and Docker Spaces require a paid plan (PRO, $9/month personal).
  Static Spaces are free.
- Exception: free personal accounts in good standing (verified email AND
  account older than 30 days) may host up to 2 Gradio Spaces on ZeroGPU
  hardware at no cost.
- ZeroGPU is Gradio-SDK only.
- Files over 10 MB require Git LFS to sync to a Space.
- Free Spaces sleep when idle; cold start is 60-90 seconds.

Sources:

- https://huggingface.co/docs/hub/en/spaces-sdks-streamlit
- https://huggingface.co/docs/hub/en/spaces-overview
- https://huggingface.co/docs/hub/en/spaces-zerogpu
- https://huggingface.co/docs/hub/en/spaces-github-actions
- https://huggingface.co/docs/hub/main/spaces-changelog
- https://huggingface.co/docs/hub/en/academia-hub

## Decision

- `app.py` (Gradio Blocks) is the Hugging Face Space entrypoint.
- `streamlit_app.py` is a fully functional Streamlit app for local
  development, satisfying assignment Tasks 2.1 and 2.2.
- Both front-ends import only `src.inference.predict`; neither contains
  scoring logic, so the two UIs cannot disagree and the automated test
  suite exercises the shared path rather than either UI's glue code.

## Consequences

- The deployed, publicly reachable app is Gradio, not Streamlit -- a
  deviation from the literal assignment brief. This is stated explicitly
  in the report and in this ADR rather than left for a marker to notice.
- The free-tier constraint (ZeroGPU, good standing, 30-day account age) is
  a hard gate, documented in `BUILD_PROMPT.md` Part 2 Phase 0, that must
  be checked before any deployment work begins.
- `artifacts/model.joblib` must stay under 10 MB to avoid Git LFS; the
  model choice (HistGradientBoosting over RandomForest) and CI's
  file-size-guard workflow both exist partly because of this constraint.
