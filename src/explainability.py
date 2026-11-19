"""Permutation importance, training-reference perturbations and response profiles."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.prediction import probabilities


def global_importance(model, X, y, seed):
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=8, random_state=seed, n_jobs=1,
    )
    return pd.DataFrame({"Feature": X.columns, "AP decrease": result.importances_mean,
                         "Shuffle SD": result.importances_std}).sort_values("AP decrease", ascending=False)


def reference_values(X):
    reference = {}
    response_values = {}
    for col in X:
        series = X[col].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            reference[col] = float(series.median())
            response_values[col] = sorted(set(series.quantile(np.linspace(.05, .95, 10), interpolation="nearest").tolist()
                                              if series.nunique() > 12 else series.unique().tolist()))
        else:
            reference[col] = str(series.mode().iat[0])
            response_values[col] = sorted(series.unique().tolist())
    return reference, response_values


def local_factors(bundle, record):
    feature_names = list(bundle["reference"])
    alternatives = [record.assign(**{name: bundle["reference"][name]}) for name in feature_names]
    scores = probabilities(bundle, pd.concat([record, *alternatives], ignore_index=True))
    result = pd.DataFrame({
        "Feature": feature_names, "Change (pp)": 100 * (scores[0] - scores[1:]),
        "Observed": [str(record[name].iat[0]) for name in feature_names],
        "Reference": [str(bundle["reference"][name]) for name in feature_names],
    })
    return result.iloc[result["Change (pp)"].abs().argsort()[::-1]].reset_index(drop=True)


def response_profile(bundle, frame, feature):
    values = bundle["response_values"][feature]
    return pd.DataFrame({feature: values, "Mean model probability": [
        float(probabilities(bundle, frame.assign(**{feature: value})).mean()) for value in values
    ]})
