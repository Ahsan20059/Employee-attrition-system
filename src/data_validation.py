"""Strict schema checks with explicit normalization and a quality summary."""
import numpy as np
import pandas as pd

from src.config import CATEGORIES, EXPECTED, ID_FIELDS, NUMERICS, RATINGS, REQUIRED, TARGET

BINARY = {
    "1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1, "left": 1, "attrited": 1,
    "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0, "stayed": 0, "retained": 0,
}
TRAVEL = {
    "non travel": "Non-Travel", "travel rarely": "Travel_Rarely",
    "travel frequently": "Travel_Frequently",
}


def canonical_headers(frame):
    known = {"".join(c for c in name.lower() if c.isalnum()): name
             for name in (*EXPECTED, *ID_FIELDS)}
    headers = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    headers = [known.get("".join(c for c in name.lower() if c.isalnum()), name)
               for name in headers]
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate canonical column names found.")
    return frame.set_axis(headers, axis=1)


def normalize_target(series):
    mapped = series.astype("string").str.strip().str.casefold().map(BINARY)
    if mapped.isna().any():
        raise ValueError("Attrition contains missing or unsupported binary labels.")
    return mapped.astype("int8")


def validate(frame):
    frame = canonical_headers(frame).replace(r"^\s*$", np.nan, regex=True).copy()
    missing_required = sorted(set(REQUIRED) - set(frame))
    if missing_required or frame.empty:
        raise ValueError(f"Dataset is empty or missing required fields {missing_required}.")
    id_field = next((c for c in ID_FIELDS if c in frame), None)
    frame[TARGET] = normalize_target(frame[TARGET])
    for column in set(NUMERICS) & set(frame):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (~np.isfinite(numeric) | (numeric < 0))
        if column in RATINGS:
            lo, hi = RATINGS[column]
            invalid |= numeric.notna() & (~numeric.between(lo, hi) | (numeric % 1 != 0))
        if column == "Age":
            invalid |= numeric.notna() & ~numeric.between(16, 100)
        if invalid.any():
            raise ValueError(f"{column} contains {invalid.sum()} invalid numeric values.")
        frame[column] = numeric
    for column in set(CATEGORIES) & set(frame):
        cleaned = frame[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        if column == "OverTime":
            normalized = cleaned.str.casefold().map(BINARY)
            if (cleaned.notna() & normalized.isna()).any():
                raise ValueError("OverTime contains unsupported binary labels.")
            cleaned = normalized.map({0: "No", 1: "Yes"})
        elif column == "BusinessTravel":
            cleaned = cleaned.str.casefold().str.replace(r"[_-]+", " ", regex=True)
            if (cleaned.notna() & ~cleaned.isin(TRAVEL)).any():
                raise ValueError("BusinessTravel contains unsupported labels.")
            cleaned = cleaned.map(TRAVEL)
        else:
            keys = cleaned.str.casefold()
            labels = (pd.DataFrame({"key": keys, "label": cleaned}).dropna()
                      .groupby("key").label.min().to_dict())
            cleaned = keys.map(labels)
        frame[column] = cleaned.astype(object).where(cleaned.notna(), np.nan)

    # The source profile excludes identifiers and unknown metadata from duplicate detection.
    profile_columns = [c for c in EXPECTED if c in frame]
    duplicates = frame.duplicated(subset=profile_columns)
    frame = frame.loc[~duplicates].reset_index(drop=True)
    if frame[TARGET].nunique() != 2:
        raise ValueError("Attrition must contain both target classes.")
    quality = {
        "source_rows": len(frame) + int(duplicates.sum()), "rows": len(frame),
        "columns": len(frame.columns), "duplicates": int(duplicates.sum()),
        "missing_values": {c: int(frame[c].isna().sum()) for c in profile_columns},
        "missing_expected": sorted(set(EXPECTED) - set(frame)),
        "numeric_fields": [c for c in NUMERICS if c in frame],
        "categorical_fields": [c for c in CATEGORIES if c in frame],
        "types": {c: str(frame[c].dtype) for c in profile_columns},
        "identifier_field": id_field,
        "constant_columns": [c for c in profile_columns if frame[c].nunique() <= 1],
        "class_counts": {"stayed": int((frame[TARGET] == 0).sum()),
                         "left": int(frame[TARGET].sum())},
        "tenure_inconsistencies": {},
    }
    for a, b in (("YearsAtCompany", "TotalWorkingYears"),
                 ("YearsInCurrentRole", "YearsAtCompany"),
                 ("YearsSinceLastPromotion", "YearsAtCompany"),
                 ("YearsWithCurrManager", "YearsAtCompany")):
        if a in frame and b in frame:
            quality["tenure_inconsistencies"][f"{a} > {b}"] = int((frame[a] > frame[b]).sum())

    # Stable under row reordering. Identical normalized profiles were collapsed above.
    hashed = pd.util.hash_pandas_object(frame[profile_columns], index=False)
    if hashed.duplicated().any():
        raise ValueError("Derived record identifiers collided.")
    frame.insert(0, "Record", hashed.map(lambda value: f"R-{value:016X}"))
    return frame[["Record", *profile_columns]], quality
