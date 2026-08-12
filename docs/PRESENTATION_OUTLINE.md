# Presentation Outline

Target: ~10 minutes, all three members present a section.

1. Problem & business framing (1 min) -- who to call, not just who
   converts; the cost matrix (EUR 8 per call vs EUR 120 per subscription).
2. Dataset & the four traps (1.5 min) -- duration leakage, `unknown`
   markers, the `pdays` sentinel, macro-feature concept drift.
3. Leakage control, live (1.5 min) -- show `reports/leakage_demo.json`'s
   AUC gap; this is the differentiator.
4. Modelling & calibration (1.5 min) -- two models, PR-AUC selection,
   isotonic calibration via `FrozenEstimator`.
5. Threshold optimisation (1.5 min) -- why not 0.5, the 201-point grid,
   EUR uplift versus the default and versus calling everyone.
6. Live demo (2 min) -- `app.py` on the deployed Space; score a prospect,
   then a batch CSV.
7. CI/CD proof (1 min) -- a green Actions run, the Space updating live
   after a push.
8. Drift monitoring & limitations (1 min) -- PSI/JS verdict, what would
   trigger a retrain, known limitations from the model card.

## Before presenting

- Open the Space 15 minutes early -- free Spaces sleep, cold start is
  ~90 seconds.
- Keep `04-app-modelcard.png` open in a background tab as a Wi-Fi
  fallback.
- Do not live-edit code.
