"""Display risk bands are independent of the model operating threshold."""
import numpy as np
import pandas as pd

from src.config import RISK_LABELS

DEFAULT_BOUNDS = (0.30, 0.60)


def risk_bands(probabilities, lower=0.30, upper=0.60):
    if not 0 <= lower < upper <= 1:
        raise ValueError("Risk boundaries must satisfy 0 ≤ lower < upper ≤ 1.")
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Risk probabilities must be finite and between 0 and 1.")
    codes = np.where(p < lower, 0, np.where(p <= upper, 1, 2))
    return pd.Categorical.from_codes(codes, RISK_LABELS, ordered=True)


def score_records(frame, probabilities, bounds=DEFAULT_BOUNDS):
    return frame.assign(Probability=probabilities, RiskScore=100 * probabilities,
                        RiskBand=risk_bands(probabilities, *bounds))
