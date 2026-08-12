"""Streamlit app -- local development front-end (assignment Tasks 2.1/2.2).

Streamlit re-executes the whole script on every widget interaction, so the
model bundle is wrapped in @st.cache_resource: without it, every click
would re-deserialise the joblib artefact from disk.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config
from src.inference.predict import load_bundle, score_batch, score_one
from src.ui.theme import build_css, render_drivers_table, render_verdict_panel

st.set_page_config(page_title=config.BRANDING.app_name, layout="wide")
st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)


@st.cache_resource
def _bundle():
    return load_bundle()


def _prospect_tab() -> None:
    bundle = _bundle()
    fields = {}
    columns = st.columns(3)
    for i, spec in enumerate(config.FIELD_SPECS):
        with columns[i % 3]:
            if spec.kind == "categorical":
                fields[spec.name] = st.selectbox(
                    spec.label,
                    options=list(spec.levels),
                    index=list(spec.levels).index(spec.default),
                )
            else:
                fields[spec.name] = st.number_input(
                    spec.label,
                    min_value=float(spec.min_value),
                    max_value=float(spec.max_value),
                    value=float(spec.default),
                )

    if st.button("Score", type="primary"):
        result = score_one(bundle, fields)
        st.markdown(
            render_verdict_panel(
                result.probability,
                result.threshold,
                result.verdict,
                result.confidence_band,
                result.expected_value_eur,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_drivers_table(
                result.drivers,
                result.explanation_method,
                result.explanation_reliable,
                result.explanation_note,
            ),
            unsafe_allow_html=True,
        )


def _batch_tab() -> None:
    bundle = _bundle()
    st.download_button(
        "Download template CSV",
        data=pd.DataFrame([{s.name: s.default for s in config.FIELD_SPECS}]).to_csv(
            index=False
        ),
        file_name="bank_conversion_template.csv",
    )
    uploaded = st.file_uploader("Prospect CSV", type=["csv"])
    capacity_pct = st.slider("Capacity (% of list to call)", 0, 100, 100)
    if uploaded is not None and st.button("Score list", type="primary"):
        frame = pd.read_csv(uploaded)
        capacity_fraction = capacity_pct / 100.0 if capacity_pct else None
        batch = score_batch(bundle, frame, capacity_fraction=capacity_fraction)
        n_call = int((batch.scored["verdict"] == "CALL").sum())
        st.write(f"{n_call} of {len(batch.scored)} marked CALL.")
        for warning in batch.warnings:
            st.warning(warning)
        st.dataframe(batch.scored)
        st.download_button(
            "Download scored list",
            data=batch.scored.to_csv(index=False),
            file_name="bank_conversion_scored.csv",
        )


def _model_card_tab() -> None:
    if not config.METRICS_PATH.exists():
        st.info("No metrics.json found. Run `python -m src.models.train`.")
        return
    metrics = json.loads(config.METRICS_PATH.read_text())
    drift = (
        json.loads(config.DRIFT_PATH.read_text())
        if config.DRIFT_PATH.exists()
        else {"verdict": "unknown"}
    )
    test = metrics["test_metrics"]

    st.subheader("Test performance")
    baseline = metrics["majority_baseline_accuracy"]
    st.write(
        f"Precision {test['precision']:.4f} · Recall {test['recall']:.4f} · "
        f"AP {test['average_precision']:.4f} · ROC-AUC {test['roc_auc']:.4f} · "
        f"Accuracy {test['accuracy']:.4f} (majority baseline {baseline:.4f})"
    )

    st.subheader("Threshold economics")
    threshold = metrics["threshold_search"]
    breakeven = metrics["breakeven_probability"]
    st.write(
        f"Threshold {threshold['threshold']:.4f} (breakeven {breakeven:.4f}), "
        f"uplift vs 0.5: {threshold['uplift_vs_default']:.2f} EUR, "
        f"uplift vs call-everyone: {threshold['uplift_vs_call_everyone']:.2f} EUR."
    )

    st.subheader("Model comparison (validation)")
    st.dataframe(
        pd.DataFrame(metrics["model_comparison"]).T[["average_precision", "roc_auc"]]
    )

    st.subheader("Drift verdict")
    st.write(drift["verdict"])

    st.subheader("Data quality")
    dataset = metrics["dataset"]
    st.write(
        f"{dataset['n_rows']} rows, {dataset['n_duplicate_rows']} duplicates, "
        f"pdays sentinel share {dataset['pdays_sentinel_share']:.2%}."
    )

    st.subheader("Provenance")
    st.write(f"sklearn {metrics['sklearn_version']}, trained {metrics['trained_at']}.")


def main() -> None:
    st.title(config.BRANDING.app_name)
    st.caption(config.BRANDING.tagline)
    tab1, tab2, tab3 = st.tabs(
        ["Score a prospect", "Score a call list", "Model card & monitoring"]
    )
    with tab1:
        _prospect_tab()
    with tab2:
        _batch_tab()
    with tab3:
        _model_card_tab()


main()
