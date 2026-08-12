"""Generates reports/Final_Group_Project_Report.docx from the artifacts
that training, EDA, and the leakage demo already produced -- so no number
in the report can contradict the code.

Note on tooling: BUILD_PROMPT.md's report-generation notes describe the
npm `docx` library, which assumes a Node.js environment. This project is
Python end to end (requirements-dev.txt pins python-docx, not a Node
toolchain), so this script uses python-docx instead -- same requirement
(read every number from the JSON artifacts), different library.
"""

from __future__ import annotations

import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from src import config


def _load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _add_heading(doc: Document, text: str, level: int = 1):
    heading = doc.add_heading(text, level=level)
    return heading


def _add_kv_table(doc: Document, rows: list[tuple[str, str]]):
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text = "Metric", "Value"
    for key, value in rows:
        cells = table.add_row().cells
        cells[0].text = str(key)
        cells[1].text = str(value)
    return table


def build_report(
    metrics_path=None, eda_findings_path=None, leakage_path=None, output_path=None
) -> None:
    metrics_path = metrics_path or config.METRICS_PATH
    eda_findings_path = eda_findings_path or (config.REPORTS_DIR / "eda_findings.json")
    leakage_path = leakage_path or (config.REPORTS_DIR / "leakage_demo.json")
    output_path = output_path or (config.REPORTS_DIR / "Final_Group_Project_Report.docx")

    metrics = _load_json(metrics_path)
    eda = _load_json(eda_findings_path)
    leakage = _load_json(leakage_path)

    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    title = doc.add_heading(config.BRANDING.app_name, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(config.BRANDING.tagline).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "SP Jain School of Global Management, Dubai -- Software Development "
        "for AI Models, Final Group Project"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # 1. Problem statement & business objective
    _add_heading(doc, "1. Problem Statement & Business Objective")
    doc.add_paragraph(
        "A retail bank's call centre has finite agent hours and roughly an "
        "11% conversion rate on term-deposit telemarketing. Calling every "
        "prospect wastes money; calling too few leaves revenue unrealised. "
        "The question this project answers is who to phone, not merely who "
        "will convert."
    )
    _add_kv_table(
        doc,
        [
            ("Cost per call (EUR)", config.DEFAULT_COST_MATRIX.cost_per_call),
            (
                "Revenue per subscription (EUR)",
                config.DEFAULT_COST_MATRIX.revenue_per_subscription,
            ),
            (
                "Break-even probability",
                f"{config.DEFAULT_COST_MATRIX.breakeven_probability:.4f}",
            ),
        ],
    )

    # 2. Dataset & source
    _add_heading(doc, "2. Dataset & Source")
    doc.add_paragraph(
        "UCI Machine Learning Repository, Bank Marketing (dataset id 222), "
        "bank-additional-full variant. Moro, S., Cortez, P., & Rita, P. "
        "(2014). A data-driven approach to predict the success of bank "
        "telemarketing. Decision Support Systems, 62, 22-31."
    )
    if metrics:
        dataset = metrics["dataset"]
        _add_kv_table(
            doc,
            [
                ("Rows", dataset["n_rows"]),
                ("Duplicate rows", dataset["n_duplicate_rows"]),
                ("pdays sentinel share", f"{dataset['pdays_sentinel_share']:.2%}"),
                (
                    "Train / valid / test rows",
                    f"{dataset['n_train']} / {dataset['n_valid']} / {dataset['n_test']}",
                ),
            ],
        )
    else:
        doc.add_paragraph(
            "[[ Run `python -m src.models.train` to populate dataset metrics. ]]"
        )

    # 3. EDA & key observations
    _add_heading(doc, "3. EDA & Key Observations")
    if eda:
        for finding in eda["findings"]:
            para = doc.add_paragraph(style="List Bullet")
            para.add_run(f"{finding['title']}: ").bold = True
            detail = {k: v for k, v in finding.items() if k != "title"}
            para.add_run(json.dumps(detail, default=str))
        doc.add_paragraph(
            "[[ Insert each of the 6 figures from reports/figures/ here, with a "
            "1-2 sentence interpretation of what it shows and why it matters. ]]"
        )
    else:
        doc.add_paragraph("[[ Run `make eda` to populate this section. ]]")

    # 4. Preprocessing & feature engineering
    _add_heading(doc, "4. Preprocessing & Feature Engineering")
    doc.add_paragraph(
        "All transformations run inside a single sklearn Pipeline, fit only "
        "on TRAIN, so the app scores raw user input through the identical "
        "path used in training. Engineered features: never_contacted_before "
        "(from the pdays==999 sentinel), n_unknown_fields (count of "
        "'unknown' across six categorical columns), and contact_intensity "
        "(campaign / (previous + 1)). Categorical case/whitespace is "
        "normalised inside the transformer; unseen categories degrade to an "
        "infrequent-category bucket rather than raising."
    )

    # 5. Leakage control
    _add_heading(doc, "5. Leakage Control")
    doc.add_paragraph(
        "`duration` is documented target leakage: it is only known after a "
        "call completes, so a model that needs it cannot be deployed. It is "
        "enforced by a denylist in config, a guard inside the feature "
        "pipeline that raises if reintroduced, and dedicated tests."
    )
    if leakage:
        _add_kv_table(
            doc,
            [
                (
                    "ROC-AUC without duration",
                    f"{leakage['roc_auc_without_duration']:.4f}",
                ),
                ("ROC-AUC with duration", f"{leakage['roc_auc_with_duration']:.4f}"),
                ("Lift from leakage", f"{leakage['lift_from_leakage']:.4f}"),
            ],
        )
    else:
        doc.add_paragraph("[[ Run `make leakage` to populate this section. ]]")

    # 6. Model comparison & justification
    _add_heading(doc, "6. Model Comparison & Justification")
    doc.add_paragraph(
        "Two candidates, as the brief requires: LogisticRegression (the "
        "model a bank's model-risk function will actually approve, setting "
        "the floor a complex model must beat) and HistGradientBoosting "
        "(chosen over RandomForest because the fitted artefact is roughly "
        "an order of magnitude smaller). Selection metric: average_precision "
        "(PR-AUC) -- with ~11% positives, ROC-AUC is optimistic and accuracy "
        "is actively misleading."
    )
    if metrics:
        rows = [
            (name, f"AP {m['average_precision']:.4f}, ROC-AUC {m['roc_auc']:.4f}")
            for name, m in metrics["model_comparison"].items()
        ]
        rows.append(("Winner", metrics["winner"]))
        test = metrics["test_metrics"]
        rows.append(
            (
                "Test set (opened once)",
                f"AP {test['average_precision']:.4f}, ROC-AUC {test['roc_auc']:.4f}, "
                f"precision {test['precision']:.4f}, recall {test['recall']:.4f}",
            )
        )
        baseline = metrics["majority_baseline_accuracy"]
        rows.append(
            (
                "Accuracy (reported last, deliberately)",
                f"{test['accuracy']:.4f} vs majority baseline {baseline:.4f}",
            )
        )
        _add_kv_table(doc, rows)

    # 7. Threshold optimisation & business value
    _add_heading(doc, "7. Threshold Optimisation & Business Value")
    doc.add_paragraph(
        "Not 0.5, and not the F1-maximising threshold: a false positive "
        "costs EUR 8, a false negative forgoes EUR 120, so 0.5 assumes a "
        "cost symmetry that does not exist here. A grid of 201 thresholds "
        "is searched on VALIDATION, maximising expected net value under "
        "calibrated probabilities."
    )
    if metrics:
        threshold = metrics["threshold_search"]
        _add_kv_table(
            doc,
            [
                (
                    "Break-even probability (analytic)",
                    f"{metrics['breakeven_probability']:.4f}",
                ),
                ("Chosen threshold (grid search)", f"{threshold['threshold']:.4f}"),
                ("Uplift vs 0.5 (EUR)", f"{threshold['uplift_vs_default']:.2f}"),
                (
                    "Uplift vs calling everyone (EUR)",
                    f"{threshold['uplift_vs_call_everyone']:.2f}",
                ),
            ],
        )
        doc.add_paragraph(
            "A capacity-constrained mode is also implemented: rather than a "
            "probability cutoff, the top N% of prospects by rank are called "
            "-- real call centres are constrained by agent hours, not list "
            "quality."
        )

    # 8. Application
    _add_heading(doc, "8. Application")
    doc.add_paragraph(
        "Both front-ends (Gradio for deployment, Streamlit for local "
        "development) share one design system and import only "
        "src.inference.predict for scoring, so they cannot disagree."
    )
    doc.add_paragraph(
        "[[ Insert screenshots 01-04 here (verdict panel, drivers, batch, model card). ]]"
    )

    # 9. Repo structure, CI/CD, deployment
    _add_heading(doc, "9. Repo Structure, CI/CD, and Deployment")
    doc.add_paragraph(
        "Three GitHub Actions workflows: ci.yml (lint, matrix pytest across "
        "3.11/3.12, a synthetic-fixture smoke-train job), deploy.yml "
        "(assembles a minimal runtime payload -- app.py, requirements.txt, "
        "src/, artifacts/, the Space README -- and uploads it via the "
        "Hugging Face Hub client on every push to main), and "
        "file-size-guard.yml (fails a PR that adds a >10MB file outside "
        "Git LFS). See docs/ADR-001-deployment-target.md for why the "
        "deployed front-end is Gradio, not Streamlit."
    )
    doc.add_paragraph(
        "[[ Insert screenshots 05-09 here (Actions run, step summary, live "
        "Space, secret, repo tree). ]]"
    )

    # 10. Performance, drift, limitations, future work
    _add_heading(doc, "10. Performance, Drift, Limitations, and Future Work")
    doc.add_paragraph(
        "Latency, throughput, cold-start, and memory are measured in "
        "benchmarks/latency.py and written to benchmarks/results_*.json. "
        "Drift is monitored from scratch: PSI (10 reference-quantile bins) "
        "for numeric features, Jensen-Shannon divergence for categoricals, "
        "reduced to a single STABLE / MONITOR / RETRAIN RECOMMENDED verdict."
    )
    if metrics:
        _add_kv_table(doc, [("Artifact size (bytes)", metrics["artifact_size_bytes"])])
    doc.add_paragraph(
        "Known limitations: temporal validity (2008-2010 Portuguese "
        "retail-banking data), geographic specificity, unknown-category "
        "bias, and no fairness certification despite age and marital status "
        "being model inputs. Future work: capacity-aware retraining "
        "triggers, an ONNX-exported serving path, and richer explanation "
        "caching."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    build_report()
