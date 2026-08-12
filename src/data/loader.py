"""Load the UCI Bank Marketing dataset three ways, validate it, audit its
quality, and produce a leakage-safe chronological split.

Chronological splitting matters here specifically because five macro
features are constant within a calendar month (see BUILD_PROMPT.md 1.4):
a plain positional split can put one month's economic conditions on both
sides of a boundary. Snapping every boundary to a period edge closes that.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def load_raw_frame(offline: bool = False, local_csv: Path | None = None) -> pd.DataFrame:
    if offline:
        path = local_csv or (config.DATA_DIR / config.UCI_CSV_NAME)
        if not path.exists():
            raise FileNotFoundError(
                f"--offline requested but {path} does not exist. Either "
                f"download {config.UCI_ZIP_URL} and extract "
                f"{config.UCI_CSV_NAME} into {config.DATA_DIR}, or run "
                f"`python -m src.models.train` without --offline."
            )
        return pd.read_csv(path, sep=config.CSV_SEPARATOR)

    frame = _load_via_ucimlrepo()
    if frame is None or "euribor3m" not in frame.columns:
        # Trap 4: ucimlrepo sometimes serves the legacy 17-input variant
        # with no macro columns. Fall back to the direct zip.
        frame = _load_via_direct_zip()
    return frame


def _load_via_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return None
    try:
        dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        frame.columns = [str(c) for c in frame.columns]
        return frame
    except Exception:
        return None


def _load_via_direct_zip() -> pd.DataFrame:
    with urllib.request.urlopen(config.UCI_ZIP_URL, timeout=30) as response:  # noqa: S310
        outer_bytes = response.read()
    outer = zipfile.ZipFile(io.BytesIO(outer_bytes))
    inner_bytes = outer.read(config.UCI_INNER_ZIP)
    inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
    csv_bytes = inner.read(f"bank-additional/{config.UCI_CSV_NAME}")
    return pd.read_csv(io.BytesIO(csv_bytes), sep=config.CSV_SEPARATOR)


def validate_schema(frame: pd.DataFrame) -> None:
    expected = set(config.RAW_INPUT_COLUMNS) | {config.TARGET_COLUMN}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")


@dataclass
class QualityAudit:
    n_rows: int
    n_duplicate_rows: int
    unknown_counts: dict[str, int]
    pdays_sentinel_share: float


def audit_quality(frame: pd.DataFrame) -> QualityAudit:
    unknown_counts = {
        column: int((frame[column] == "unknown").sum())
        for column in config.UNKNOWN_MARKER_COLUMNS
    }
    return QualityAudit(
        n_rows=len(frame),
        n_duplicate_rows=int(frame.duplicated().sum()),
        unknown_counts=unknown_counts,
        pdays_sentinel_share=float((frame["pdays"] == config.PDAYS_SENTINEL).mean()),
    )


def build_period_index(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct a monotonic year-month period from file order alone.

    The file ships no date column, only a month name, and rows are known
    to be date-ordered. Walking rows and incrementing the year whenever the
    month number goes backwards (a calendar wrap) recovers a real period
    index without ever inventing a day.
    """
    order = {month: i for i, month in enumerate(config.MONTH_ORDER)}
    month_idx = frame["month"].map(order)
    if month_idx.isna().any():
        bad = sorted(set(frame.loc[month_idx.isna(), "month"]))
        raise ValueError(f"Unknown month values: {bad}")

    month_values = month_idx.to_numpy(dtype=np.int64)
    periods = np.empty(len(frame), dtype=np.int64)
    year = config.BASE_YEAR
    prev = None
    for i, m in enumerate(month_values):
        if prev is not None and m < prev:
            year += 1
        periods[i] = year * 12 + m
        prev = m
    return pd.Series(periods, index=frame.index, name="period")


def temporal_split(
    frame: pd.DataFrame, periods: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    values = periods.to_numpy()
    change_points = np.flatnonzero(np.diff(values) != 0) + 1
    boundaries = np.concatenate(([0], change_points, [n]))

    def snap(target: int) -> int:
        return int(boundaries[np.argmin(np.abs(boundaries - target))])

    train_end = snap(int(n * config.TRAIN_FRACTION))
    valid_end = max(
        train_end, snap(int(n * (config.TRAIN_FRACTION + config.VALID_FRACTION)))
    )

    return frame.iloc[:train_end], frame.iloc[train_end:valid_end], frame.iloc[valid_end:]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features/target, dropping every denylisted column.

    This is the first of two leakage guards (the second lives inside the
    feature pipeline in src/features/pipeline.py) -- belt and braces.
    """
    y = (frame[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
    x = frame.drop(
        columns=[config.TARGET_COLUMN, *config.LEAKAGE_DENYLIST], errors="ignore"
    )
    return x, y


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    quality: QualityAudit


def load_and_split(offline: bool = False, local_csv: Path | None = None) -> DatasetBundle:
    frame = load_raw_frame(offline=offline, local_csv=local_csv).reset_index(drop=True)
    validate_schema(frame)
    quality = audit_quality(frame)
    periods = build_period_index(frame)
    train_df, valid_df, test_df = temporal_split(frame, periods)
    return DatasetBundle(train=train_df, valid=valid_df, test=test_df, quality=quality)
