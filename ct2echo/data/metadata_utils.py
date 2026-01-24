"""Utilities for extracting and normalizing patient-level metadata features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

SEX_BUCKETS: Sequence[str] = ("male", "female", "other")
RACE_BUCKETS: Sequence[str] = ("white", "black", "asian", "hispanic", "other")
NUMERIC_FIELDS: Sequence[str] = ("Patients_Age", "systolic_bp", "diastolic_bp", "heart_rate")


def _safe_numeric(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def _normalise_text(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip().lower()


def _map_sex(value: str) -> str:
    text = _normalise_text(value)
    if text in {"m", "male", "man"}:
        return "male"
    if text in {"f", "female", "woman"}:
        return "female"
    return "other"


def _map_race(value: str, fallback: str = "") -> str:
    text = _normalise_text(value)
    if not text:
        text = _normalise_text(fallback)
    if not text:
        return "other"
    if "white" in text:
        return "white"
    if "black" in text or "african" in text:
        return "black"
    if "asian" in text:
        return "asian"
    if "hispanic" in text or "latino" in text:
        return "hispanic"
    return "other"


@dataclass
class MetadataStats:
    mean: float
    std: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "MetadataStats":
        clean = _safe_numeric(series)
        if clean.empty:
            return cls(0.0, 1.0)
        mean = float(clean.mean(skipna=True))
        std = float(clean.std(skipna=True))
        if np.isnan(mean):
            mean = 0.0
        if np.isnan(std) or std < 1e-6:
            std = 1.0
        return cls(mean, std)

    def normalise(self, value) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return 0.0
        return float((numeric - self.mean) / self.std)


class MetadataProcessor:
    """Prepare and encode structured patient metadata for hypernetwork conditioning."""

    def __init__(self, dataframe: pd.DataFrame):
        self.stats = {field: MetadataStats.from_series(dataframe.get(field)) for field in NUMERIC_FIELDS}
        self.metadata_dim = len(SEX_BUCKETS) + len(RACE_BUCKETS) + len(NUMERIC_FIELDS)

    def encode_row(self, row) -> torch.Tensor:
        if isinstance(row, pd.DataFrame):
            if len(row) == 0:
                return torch.zeros(self.metadata_dim, dtype=torch.float32)
            row = row.iloc[0]

        sex_bucket = _map_sex(row.get("Patients_Sex"))
        race_bucket = _map_race(row.get("race_1"), fallback=row.get("ethnicity"))

        sex_vector: List[float] = [1.0 if sex_bucket == bucket else 0.0 for bucket in SEX_BUCKETS]
        race_vector: List[float] = [1.0 if race_bucket == bucket else 0.0 for bucket in RACE_BUCKETS]

        numeric_vector: List[float] = []
        for field in NUMERIC_FIELDS:
            stats = self.stats[field]
            value = row.get(field)
            numeric_vector.append(stats.normalise(value))

        full_vector = sex_vector + race_vector + numeric_vector
        return torch.tensor(full_vector, dtype=torch.float32)


__all__ = ["MetadataProcessor", "NUMERIC_FIELDS", "SEX_BUCKETS", "RACE_BUCKETS"]
