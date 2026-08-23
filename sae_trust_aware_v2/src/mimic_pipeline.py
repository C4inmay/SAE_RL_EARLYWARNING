"""
Utilities for the REAL MIMIC/eICU stage.

This module deliberately does not guess your local MIMIC schema. It expects an
hourly patient-level table after SQL/BigQuery extraction.

Required conceptual columns:
  patient_id, stay_id, hour
  SAE onset / target information
  clinical variables used by the forecaster

The supplied notebooks show the expected schema and validate leakage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class CohortConfig:
    patient_col: str = "subject_id"
    stay_col: str = "stay_id"
    hour_col: str = "hour"
    onset_col: str = "sae_onset_hour"
    min_hours_before_onset: int = 1
    horizon: int = 6


def validate_hourly_table(df: pd.DataFrame, required: Sequence[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[[c for c in required if c in df.columns]].isnull().all().any():
        bad = df.columns[df.isnull().all()].tolist()
        raise ValueError(f"Completely empty columns detected: {bad}")

    if df.duplicated(["subject_id", "stay_id", "hour"]).any():
        raise ValueError("Duplicate patient/stay/hour rows detected.")


def make_hourly_labels(
    df: pd.DataFrame,
    cfg: CohortConfig,
) -> pd.DataFrame:
    """
    Create y_t = 1 when SAE onset is in (t, t+horizon].

    The onset timestamp must be determined independently from future predictor
    variables. Predictor construction should use only information available by t.
    """
    out = df.copy()
    delta = out[cfg.onset_col] - out[cfg.hour_col]
    out["y"] = (
        out[cfg.onset_col].notna()
        & (delta > 0)
        & (delta <= cfg.horizon)
    ).astype("int8")
    return out


def assert_no_future_rows(df: pd.DataFrame, cfg: CohortConfig):
    """
    Basic leakage check: after SAE onset, rows should not be used as early-warning
    examples. The default notebook filters to onset-or-before.
    """
    onset = df[cfg.onset_col]
    future = onset.notna() & (df[cfg.hour_col] >= onset)
    if future.any():
        raise ValueError(
            f"{int(future.sum())} rows occur at/after SAE onset. "
            "Filter these rows before model training."
        )


def add_time_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    group_cols=("subject_id", "stay_id"),
    history_hours: int = 3,
) -> pd.DataFrame:
    """Create current, delta and rolling-mean features without future leakage."""
    out = df.sort_values(list(group_cols) + ["hour"]).copy()
    g = out.groupby(list(group_cols), sort=False)

    for col in feature_cols:
        out[f"{col}_delta"] = g[col].diff().fillna(0.0)
        out[f"{col}_mean{history_hours}h"] = (
            g[col]
            .rolling(history_hours, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )

    return out


def patient_level_split(
    df: pd.DataFrame,
    group_cols=("subject_id", "stay_id"),
    train_frac=0.60,
    val_frac=0.20,
    seed=42,
):
    """Split whole ICU stays, not rows, to prevent patient/stay leakage."""
    keys = df[list(group_cols)].drop_duplicates().sample(
        frac=1.0, random_state=seed
    )
    n = len(keys)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_keys = keys.iloc[:n_train]
    val_keys = keys.iloc[n_train:n_train+n_val]
    test_keys = keys.iloc[n_train+n_val:]

    def take(keys_):
        return df.merge(keys_, on=list(group_cols), how="inner")

    return take(train_keys), take(val_keys), take(test_keys)


def event_level_summary(
    predictions: pd.DataFrame,
    pred_col="alarm",
    onset_col="sae_onset_hour",
):
    """
    One row per ICU stay:
      detected = any alarm before onset in the configured detection window.
    """
    rows = []
    for (subject_id, stay_id), g in predictions.groupby(
        ["subject_id", "stay_id"], sort=False
    ):
        onset = g[onset_col].iloc[0]
        alarms = g.loc[g[pred_col].astype(bool), "hour"].to_numpy()

        if pd.isna(onset):
            detected = False
            lead = np.nan
        else:
            valid = alarms[alarms < onset]
            detected = len(valid) > 0
            lead = float(onset - valid[0]) if detected else np.nan

        rows.append(
            {
                "subject_id": subject_id,
                "stay_id": stay_id,
                "event": not pd.isna(onset),
                "detected": detected,
                "lead_hours": lead,
            }
        )

    return pd.DataFrame(rows)


def load_hourly_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def save_hourly_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)
