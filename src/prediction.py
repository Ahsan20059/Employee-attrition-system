"""Model bundle loading and inference paths used by the application."""
import joblib
import pandas as pd

from src.config import MODEL
from src.risk_scoring import score_records


def load_model(path=MODEL):
    if not path.is_file():
        raise FileNotFoundError("Trained model is missing. Run python scripts/run_training.py.")
    return joblib.load(path)


def probabilities(bundle, frame):
    return bundle["model"].predict_proba(frame.loc[:, bundle["features"]])[:, 1]


def predict(bundle, frame, bounds=(0.30, 0.60)):
    return score_records(frame, probabilities(bundle, frame), bounds)


def scenario(bundle, record, overrides):
    if len(record) != 1:
        raise ValueError("Scenario analysis expects a single baseline record.")
    unexpected = set(overrides) - set(bundle["features"])
    if unexpected:
        raise ValueError(f"Unsupported scenario fields {sorted(unexpected)}.")
    modified = record.assign(**overrides)
    baseline, alternative = probabilities(bundle, pd.concat([record, modified], ignore_index=True))
    return float(baseline), float(alternative)
