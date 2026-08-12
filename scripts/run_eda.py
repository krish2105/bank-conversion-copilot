"""Exploratory data analysis: 6 figures, a written summary, and structured
findings that scripts/build_report.py quotes verbatim, so no number in the
report can contradict the code.

matplotlib.use("Agg") must precede `import matplotlib.pyplot` -- CI and the
Space container have no display, and importing pyplot first locks in a
display-requiring backend that fails there. Runs on the TRAIN split only,
keeping "TEST opened once" visible even in EDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.loader import build_period_index, load_and_split  # noqa: E402


def _fig_target_balance(frame: pd.DataFrame, path: Path) -> dict:
    counts = frame[config.TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        counts.index,
        counts.to_numpy(),
        color=[config.BRANDING.accent, config.BRANDING.danger],
    )
    ax.set_title("Target class balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    positive_rate = float((frame[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean())
    return {"title": "Target class balance", "positive_rate": positive_rate}


def _fig_duration_leakage(frame: pd.DataFrame, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5, 4))
    frame.boxplot(column="duration", by=config.TARGET_COLUMN, ax=ax)
    ax.set_title("Call duration by outcome (duration is denylisted)")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    medians = frame.groupby(config.TARGET_COLUMN)["duration"].median().to_dict()
    return {"title": "Duration by outcome", "median_duration_by_class": medians}


def _fig_age_distribution(frame: pd.DataFrame, path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(frame["age"], bins=30, color=config.BRANDING.accent)
    ax.set_title("Age distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "Age distribution", "median_age": float(frame["age"].median())}


def _fig_job_conversion(frame: pd.DataFrame, path: Path) -> dict:
    rate = (
        frame.groupby("job")[config.TARGET_COLUMN]
        .apply(lambda s: (s == config.POSITIVE_LABEL).mean())
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(rate.index, rate.to_numpy(), color=config.BRANDING.accent)
    ax.set_title("Conversion rate by job")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "title": "Conversion rate by job",
        "top_job": str(rate.idxmax()),
        "bottom_job": str(rate.idxmin()),
    }


def _fig_macro_trend(frame: pd.DataFrame, path: Path) -> dict:
    periods = build_period_index(frame)
    trend = frame.groupby(periods)["euribor3m"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trend.index, trend.to_numpy(), color=config.BRANDING.accent)
    ax.set_title("euribor3m by period (concept drift)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "title": "euribor3m trend across periods",
        "min": float(trend.min()),
        "max": float(trend.max()),
    }


def _fig_pdays_sentinel(frame: pd.DataFrame, path: Path) -> dict:
    share = float((frame["pdays"] == config.PDAYS_SENTINEL).mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [share, 1 - share],
        labels=["Never contacted", "Previously contacted"],
        colors=[config.BRANDING.surface, config.BRANDING.accent],
    )
    ax.set_title("pdays sentinel share")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"title": "pdays sentinel share", "never_contacted_share": share}


def generate_eda(
    frame: pd.DataFrame, figures_dir: Path, summary_path: Path, findings_path: Path
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    findings = [
        _fig_target_balance(frame, figures_dir / "01_target_balance.png"),
        _fig_duration_leakage(frame, figures_dir / "02_duration_leakage.png"),
        _fig_age_distribution(frame, figures_dir / "03_age_distribution.png"),
        _fig_job_conversion(frame, figures_dir / "04_job_conversion.png"),
        _fig_macro_trend(frame, figures_dir / "05_macro_trend.png"),
        _fig_pdays_sentinel(frame, figures_dir / "06_pdays_sentinel.png"),
    ]

    findings_path.write_text(json.dumps({"findings": findings}, indent=2, default=str))
    summary_lines = ["# EDA Summary", ""]
    for finding in findings:
        detail = {k: v for k, v in finding.items() if k != "title"}
        summary_lines.append(
            f"- **{finding['title']}**: {json.dumps(detail, default=str)}"
        )
    summary_path.write_text("\n".join(summary_lines))
    return {"findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = load_and_split(offline=args.offline)
    generate_eda(
        dataset.train,
        config.FIGURES_DIR,
        config.REPORTS_DIR / "eda_summary.md",
        config.REPORTS_DIR / "eda_findings.json",
    )


if __name__ == "__main__":
    main()
