"""Quantifies the duration leak: trains the same lightweight classifier
with and without `duration` and reports the ROC-AUC gap. This is the
concrete number backing the leakage-control section of the report --
without it, "duration causes leakage" is an assertion, not evidence.
"""

from __future__ import annotations

import argparse
import json

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.loader import DatasetBundle, load_and_split, split_xy

NUMERIC_ONLY: tuple[str, ...] = (
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)


def _quick_auc(x_train, y_train, x_valid, y_valid, columns: tuple[str, ...]) -> float:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train[list(columns)], y_train)
    probs = pipeline.predict_proba(x_valid[list(columns)])[:, 1]
    return float(roc_auc_score(y_valid, probs))


def quantify_leakage(dataset: DatasetBundle) -> dict:
    x_train, y_train = split_xy(dataset.train)
    x_valid, y_valid = split_xy(dataset.valid)

    # split_xy already dropped duration; recover it here deliberately, for
    # this comparison only. This is the one intentional exception to the
    # leakage denylist in the whole repo.
    x_train_with_duration = x_train.assign(duration=dataset.train["duration"].to_numpy())
    x_valid_with_duration = x_valid.assign(duration=dataset.valid["duration"].to_numpy())

    without_duration_auc = _quick_auc(x_train, y_train, x_valid, y_valid, NUMERIC_ONLY)
    with_duration_auc = _quick_auc(
        x_train_with_duration,
        y_train,
        x_valid_with_duration,
        y_valid,
        (*NUMERIC_ONLY, "duration"),
    )
    return {
        "roc_auc_without_duration": without_duration_auc,
        "roc_auc_with_duration": with_duration_auc,
        "lift_from_leakage": with_duration_auc - without_duration_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dataset = load_and_split(offline=args.offline)
    result = quantify_leakage(dataset)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "leakage_demo.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
