"""Gradio Blocks app -- the Hugging Face Space entrypoint.

Warms the model bundle at import time, not on first request: cold starts
on free Space hardware are dominated by artefact deserialisation, and a
live demo audience should never watch that happen. `import spaces` is
wrapped in try/except because it only exists inside an actual Space
container -- local development and CI must not require it.
"""

from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from src import config
from src.inference.predict import load_bundle, score_batch, score_one
from src.ui.theme import build_css, render_drivers_table, render_verdict_panel

try:
    import spaces  # noqa: F401
except ImportError:
    spaces = None

BUNDLE = load_bundle()


def _score_prospect(*values):
    fields = dict(zip(config.APP_FIELD_ORDER, values, strict=True))
    result = score_one(BUNDLE, fields)
    verdict_html = render_verdict_panel(
        result.probability,
        result.threshold,
        result.verdict,
        result.confidence_band,
        result.expected_value_eur,
    )
    drivers_html = render_drivers_table(
        result.drivers,
        result.explanation_method,
        result.explanation_reliable,
        result.explanation_note,
    )
    return verdict_html, drivers_html


def _template_csv() -> str:
    path = "/tmp/bank_conversion_template.csv"
    pd.DataFrame([{spec.name: spec.default for spec in config.FIELD_SPECS}]).to_csv(
        path, index=False
    )
    return path


def _score_csv(file, capacity_pct):
    if file is None:
        return None, "Upload a CSV first."
    frame = pd.read_csv(file.name)
    capacity_fraction = capacity_pct / 100.0 if capacity_pct else None
    batch = score_batch(BUNDLE, frame, capacity_fraction=capacity_fraction)
    out_path = "/tmp/bank_conversion_scored.csv"
    batch.scored.to_csv(out_path, index=False)
    n_call = int((batch.scored["verdict"] == "CALL").sum())
    summary = f"{n_call} of {len(batch.scored)} marked CALL."
    if batch.warnings:
        summary += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in batch.warnings)
    return out_path, summary


def _model_card_html() -> str:
    if not config.METRICS_PATH.exists():
        return (
            "<p>No metrics.json found. Run <code>python -m src.models.train</code>.</p>"
        )
    metrics = json.loads(config.METRICS_PATH.read_text())
    drift = (
        json.loads(config.DRIFT_PATH.read_text())
        if config.DRIFT_PATH.exists()
        else {"verdict": "unknown"}
    )
    test = metrics["test_metrics"]
    threshold = metrics["threshold_search"]
    dataset = metrics["dataset"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{m['average_precision']:.4f}</td>"
        f"<td>{m['roc_auc']:.4f}</td></tr>"
        for name, m in metrics["model_comparison"].items()
    )
    return f"""
    <div class="bcc-card">
      <h3>Test performance</h3>
      <p>Precision {test["precision"]:.4f} &middot; Recall {test["recall"]:.4f}
         &middot; AP {test["average_precision"]:.4f}
         &middot; ROC-AUC {test["roc_auc"]:.4f}
         &middot; Accuracy {test["accuracy"]:.4f}
         (majority baseline {metrics["majority_baseline_accuracy"]:.4f})</p>
      <h3>Threshold economics</h3>
      <p>Threshold {threshold["threshold"]:.4f}
         (breakeven {metrics["breakeven_probability"]:.4f}),
         uplift vs 0.5: {threshold["uplift_vs_default"]:.2f} EUR,
         uplift vs call-everyone: {threshold["uplift_vs_call_everyone"]:.2f} EUR.</p>
      <h3>Model comparison (validation)</h3>
      <table class="bcc-table">
        <thead><tr><th>Model</th><th>AP</th><th>ROC-AUC</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <h3>Drift verdict</h3>
      <p>{drift["verdict"]}</p>
      <h3>Data quality</h3>
      <p>{dataset["n_rows"]} rows, {dataset["n_duplicate_rows"]} duplicates,
         pdays sentinel share {dataset["pdays_sentinel_share"]:.2%}.</p>
      <h3>Provenance</h3>
      <p>sklearn {metrics["sklearn_version"]}, trained {metrics["trained_at"]}.</p>
    </div>
    """


def _build_field_inputs() -> list:
    inputs = []
    for spec in config.FIELD_SPECS:
        if spec.kind == "categorical":
            inputs.append(
                gr.Dropdown(
                    choices=list(spec.levels), value=spec.default, label=spec.label
                )
            )
        else:
            inputs.append(
                gr.Number(
                    value=spec.default,
                    label=spec.label,
                    minimum=spec.min_value,
                    maximum=spec.max_value,
                )
            )
    return inputs


with gr.Blocks(css=build_css(), title=config.BRANDING.app_name) as demo:
    gr.Markdown(f"# {config.BRANDING.app_name}\n{config.BRANDING.tagline}")

    with gr.Tab("Score a prospect"):
        field_inputs = _build_field_inputs()
        score_button = gr.Button("Score", variant="primary")
        verdict_out = gr.HTML()
        drivers_out = gr.HTML()
        score_button.click(
            _score_prospect,
            inputs=field_inputs,
            outputs=[verdict_out, drivers_out],
            api_name="score_one",
        )

    with gr.Tab("Score a call list"):
        gr.Markdown(
            "Upload a CSV of prospects. Missing columns get documented "
            "defaults; unseen categories degrade gracefully."
        )
        template_button = gr.Button("Download template CSV")
        template_file = gr.File(label="Template")
        template_button.click(_template_csv, outputs=template_file)

        csv_input = gr.File(label="Prospect CSV", file_types=[".csv"])
        capacity_slider = gr.Slider(
            0, 100, value=100, step=1, label="Capacity (% of list to call)"
        )
        batch_button = gr.Button("Score list", variant="primary")
        batch_output = gr.File(label="Scored list")
        batch_summary = gr.Textbox(label="Summary", lines=4)
        batch_button.click(
            _score_csv,
            inputs=[csv_input, capacity_slider],
            outputs=[batch_output, batch_summary],
            api_name="score_batch",
        )

    with gr.Tab("Model card & monitoring"):
        gr.HTML(value=_model_card_html())


if __name__ == "__main__":
    import os

    demo.launch(server_port=int(os.environ.get("PORT", 7860)))
