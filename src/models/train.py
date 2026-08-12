"""Training entrypoint: fits both candidates on TRAIN, selects the winner
on validation PR-AUC, calibrates it in isolation, searches a cost-optimal
threshold, evaluates on TEST exactly once, and writes all four artifacts.

Model selection, calibration, and threshold search read VALIDATION only.
TEST is opened exactly once, at the end, and nothing upstream is tuned
against it -- that discipline is what makes the reported numbers
trustworthy rather than optimistic.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.loader import DatasetBundle, load_and_split, split_xy
from src.features.pipeline import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_pipeline,
    get_feature_names,
)
from src.models.evaluate import (
    compute_metrics,
    expected_calibration_error,
    reliability_curve,
)
from src.models.threshold import search_cost_optimal_threshold
from src.monitor.drift import compute_drift_report


def _build_candidates() -> dict[str, object]:
    lr_pipeline = build_feature_pipeline(scale_numeric=True)
    lr_pipeline.steps.append(
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=config.RUNTIME.random_state,
            ),
        )
    )

    hgb_pipeline = build_feature_pipeline(scale_numeric=False)
    hgb_pipeline.steps.append(
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=400,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=config.RUNTIME.random_state,
            ),
        )
    )

    return {"logistic_regression": lr_pipeline, "hist_gradient_boosting": hgb_pipeline}


def train_and_save(
    dataset: DatasetBundle,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    model_card_path: Path | None = None,
    drift_path: Path | None = None,
) -> dict:
    model_path = model_path or config.MODEL_PATH
    metrics_path = metrics_path or config.METRICS_PATH
    model_card_path = model_card_path or config.MODEL_CARD_PATH
    drift_path = drift_path or config.DRIFT_PATH
    for path in (model_path, metrics_path, model_card_path, drift_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = split_xy(dataset.train)
    x_valid, y_valid = split_xy(dataset.valid)
    x_test, y_test = split_xy(dataset.test)

    candidates = _build_candidates()
    validation_metrics = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        valid_probs = pipeline.predict_proba(x_valid)[:, 1]
        validation_metrics[name] = compute_metrics(
            name, y_valid.to_numpy(), valid_probs, threshold=0.5
        )
        fitted[name] = pipeline

    winner_name = max(
        validation_metrics, key=lambda n: validation_metrics[n].average_precision
    )
    winner = fitted[winner_name]

    # Isotonic calibration on VALIDATION via FrozenEstimator (sklearn>=1.8
    # removed cv="prefit" -- Trap 1). The winner is already fitted on TRAIN
    # and must not be refit here.
    calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")
    calibrated.fit(x_valid, y_valid)

    valid_calibrated_probs = calibrated.predict_proba(x_valid)[:, 1]
    threshold_result = search_cost_optimal_threshold(
        y_valid.to_numpy(),
        valid_calibrated_probs,
        cost_matrix=config.DEFAULT_COST_MATRIX,
    )

    # TEST is opened exactly once, here, at the end.
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(
        f"{winner_name}_calibrated",
        y_test.to_numpy(),
        test_probs,
        threshold=threshold_result.threshold,
    )
    majority_baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))
    ece = expected_calibration_error(y_test.to_numpy(), test_probs)
    reliability = reliability_curve(y_test.to_numpy(), test_probs)

    drift_report = compute_drift_report(
        x_train,
        x_test,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )

    # Global (not per-prediction) permutation importance on the winner's
    # raw-input columns, computed once here against VALIDATION. This is
    # the stage-3 explanation fallback in src/explain/shap_engine.py -- it
    # must always work, so it is precomputed rather than derived live from
    # a model type that might not support it (e.g. no feature_importances_
    # on HistGradientBoostingClassifier).
    perm_result = permutation_importance(
        winner,
        x_valid,
        y_valid,
        scoring="average_precision",
        n_repeats=5,
        random_state=config.RUNTIME.random_state,
        n_jobs=config.RUNTIME.n_jobs,
    )
    global_importances = sorted(
        zip(x_valid.columns.tolist(), perm_result.importances_mean.tolist(), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )

    bundle = {
        "model": calibrated,
        "winner_name": winner_name,
        "threshold": threshold_result.threshold,
        "feature_names": get_feature_names(winner),
        "global_importances": global_importances,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "dataset": {
            "n_rows": dataset.quality.n_rows,
            "n_duplicate_rows": dataset.quality.n_duplicate_rows,
            "unknown_counts": dataset.quality.unknown_counts,
            "pdays_sentinel_share": dataset.quality.pdays_sentinel_share,
            "n_train": len(dataset.train),
            "n_valid": len(dataset.valid),
            "n_test": len(dataset.test),
        },
        "model_comparison": {
            name: dataclasses.asdict(metric)
            for name, metric in validation_metrics.items()
        },
        "winner": winner_name,
        "calibration": {
            "expected_calibration_error": ece,
            "reliability_curve": reliability.to_dict(orient="records"),
        },
        "threshold_search": dataclasses.asdict(threshold_result),
        "breakeven_probability": config.DEFAULT_COST_MATRIX.breakeven_probability,
        "test_metrics": dataclasses.asdict(test_metrics),
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "global_importances": global_importances,
        "artifact_size_bytes": model_path.stat().st_size,
        "sklearn_version": sklearn.__version__,
        "trained_at": bundle["trained_at"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))

    drift_payload = {
        "numeric_psi": drift_report.numeric_psi,
        "categorical_js": drift_report.categorical_js,
        "verdict": drift_report.verdict,
        "reference": "train split",
        "current": "test split",
    }
    drift_path.write_text(json.dumps(drift_payload, indent=2))

    model_card_path.write_text(_render_model_card(metrics_payload, drift_payload))

    return metrics_payload


def _render_model_card(metrics: dict, drift: dict) -> str:
    leakage_lines = "\n".join(
        f"- **{col}**: {reason}" for col, reason in config.LEAKAGE_REASONS.items()
    )
    test = metrics["test_metrics"]
    comparison_rows = "\n".join(
        f"| {name} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} |"
        for name, m in metrics["model_comparison"].items()
    )
    threshold = metrics["threshold_search"]
    return f"""# Model Card — Bank Conversion Copilot

> Auto-generated by `src/models/train.py` from `artifacts/metrics.json`.
> Do not edit by hand.

## Overview

| Field | Value |
|---|---|
| Winner | {metrics["winner"]} |
| Trained at | {metrics["trained_at"]} |
| sklearn version | {metrics["sklearn_version"]} |
| Selection metric | average_precision (PR-AUC) |
| Decision threshold | {threshold["threshold"]:.4f} |

## Intended use

Rank outbound term-deposit telemarketing prospects by expected net value,
for a capacity-constrained call centre deciding who to phone.

## Out-of-scope use

**Not for credit decisions.** Conversion propensity is not a proxy for
credit risk, and this model was never validated for that purpose.

## Test performance

| Model | Validation AP | Validation ROC-AUC |
|---|---|---|
{comparison_rows}

Test set (opened once): precision {test["precision"]:.4f}, recall {test["recall"]:.4f},
average precision {test["average_precision"]:.4f}, ROC-AUC {test["roc_auc"]:.4f}.
Accuracy is reported last, deliberately: {test["accuracy"]:.4f} versus a
majority-class baseline of {metrics["majority_baseline_accuracy"]:.4f} —
accuracy alone cannot distinguish this model from predicting "no" for
everyone.

## Business framing

Break-even call probability: {metrics["breakeven_probability"]:.4f}.
Chosen threshold yields a validation uplift of {threshold["uplift_vs_default"]:.2f} EUR
versus the naive 0.5 cutoff and {threshold["uplift_vs_call_everyone"]:.2f} EUR
versus calling everyone.

## Leakage controls

{leakage_lines}

## Drift monitoring (train vs test)

Verdict: **{drift["verdict"]}**. See `artifacts/drift.json` for per-feature detail.

## Known limitations

- Temporal validity: trained on 2008–2010 Portuguese retail-banking data;
  economic conditions and channel mix have moved on.
- Geographic specificity: one bank, one country.
- `unknown` category bias: six fields carry a nontrivial "declined to
  state" rate, which the model treats as a signal rather than imputing away.
- No fairness certification, despite age and marital status being model
  inputs.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Bank Conversion Copilot model."
    )
    parser.add_argument(
        "--offline", action="store_true", help="Load from data/ instead of the network."
    )
    args = parser.parse_args()

    dataset = load_and_split(offline=args.offline)
    metrics = train_and_save(dataset)

    winner_metrics = metrics["model_comparison"][metrics["winner"]]
    print(f"Winner: {metrics['winner']}")
    print(f"Validation AP: {winner_metrics['average_precision']:.4f}")
    test = metrics["test_metrics"]
    print(f"Test AP: {test['average_precision']:.4f}  ROC-AUC: {test['roc_auc']:.4f}")
    threshold = metrics["threshold_search"]
    breakeven = metrics["breakeven_probability"]
    print(f"Threshold: {threshold['threshold']:.4f}  (breakeven {breakeven:.4f})")
    print(f"Artifact size: {metrics['artifact_size_bytes'] / 1024:.1f} KB")


if __name__ == "__main__":
    main()
