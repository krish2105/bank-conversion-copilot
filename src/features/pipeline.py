"""Feature engineering, entirely inside a sklearn Pipeline.

Nothing here touches a dataframe before .fit(). That is what makes serving
safe: the app hands raw user input to predict_proba and the identical
transformations run with parameters learned only from TRAIN. Scaling in a
notebook and pickling only the classifier is the single most common cause
of train/serve skew, and it fails silently -- this structure makes that
class of bug impossible to introduce by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in config.FIELD_SPECS if f.kind == "numeric"
)
CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in config.FIELD_SPECS if f.kind == "categorical"
)


class LeakageGuard(BaseEstimator, TransformerMixin):
    """Raises if any denylisted (leaky) column reaches the pipeline.

    This is the second of two guards -- src/data/loader.split_xy drops
    duration too, but a caller that skips split_xy (or a future refactor
    that forgets to) hits this instead of silently training a leaky model.
    """

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> LeakageGuard:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in config.LEAKAGE_DENYLIST if c in x.columns]
        if present:
            reasons = "; ".join(f"{c}: {config.LEAKAGE_REASONS[c]}" for c in present)
            raise ValueError(f"Leakage guard triggered by columns {present}. {reasons}")
        return x


class DomainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds never_contacted_before, n_unknown_fields, contact_intensity.

    Also normalises categorical case/whitespace so a hand-edited CSV with
    " RETIRED " lands on the fitted encoder's "retired" level instead of
    becoming a spurious unseen category.
    """

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> DomainFeatureBuilder:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()

        for column in CATEGORICAL_COLUMNS:
            if column in out.columns:
                out[column] = out[column].astype(str).str.strip().str.lower()

        sentinel_mask = out["pdays"] == config.PDAYS_SENTINEL
        out[config.FEATURE_NEVER_CONTACTED] = sentinel_mask.astype(int)
        out.loc[sentinel_mask, "pdays"] = np.nan

        unknown_mask = pd.DataFrame(
            {column: out[column] == "unknown" for column in config.UNKNOWN_MARKER_COLUMNS}
        )
        out[config.FEATURE_N_UNKNOWN] = unknown_mask.sum(axis=1)

        out[config.FEATURE_CONTACT_INTENSITY] = out["campaign"] / (out["previous"] + 1)

        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + list(config.ENGINEERED_FEATURES))


def build_feature_pipeline(scale_numeric: bool) -> Pipeline:
    numeric_features = list(NUMERIC_COLUMNS) + list(config.ENGINEERED_FEATURES)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=False,
                ),
            ),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("leakage_guard", LeakageGuard()),
            ("domain_features", DomainFeatureBuilder()),
            ("columns", columns),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["columns"].get_feature_names_out())
