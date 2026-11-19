"""Reproducible train, validation, calibration and untouched test evaluation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.analytics import aggregate_summary
from src.config import AUDIT_ONLY, DEMO, METRICS, MODEL, PREDICTORS, PRIVATE_AUDIT, PROCESSED, SEED, TARGET, VERSION
from src.data_loader import load_dataset
from src.evaluation import bootstrap_intervals, evaluate, fairness_audit, threshold_for_f1
from src.explainability import global_importance, reference_values
from src.prediction import predict
from src.preprocessing import make_pipeline

MODEL_NAMES = ("Logistic Regression", "Random Forest")


def candidates(y, force_fallback=False):
    options = {
        MODEL_NAMES[0]: (make_pipeline(LogisticRegression(max_iter=1500, random_state=SEED), scale=True),
                         {"model__C": [0.03, 0.2, 1], "model__class_weight": [None, "balanced"]}),
        MODEL_NAMES[1]: (make_pipeline(RandomForestClassifier(n_estimators=160, random_state=SEED,
                                                               n_jobs=1, class_weight="balanced_subsample")),
                         {"model__max_depth": [5, 9, None], "model__min_samples_leaf": [3, 8]}),
    }
    if force_fallback:
        reason = "Histogram gradient boosting requested explicitly."
    else:
        try:
            from xgboost import XGBClassifier
            options["XGBoost"] = (make_pipeline(XGBClassifier(
                n_estimators=160, learning_rate=.05, n_jobs=1, tree_method="hist",
                eval_metric="logloss", subsample=.85, colsample_bytree=.85, random_state=SEED,
            )), {"model__max_depth": [2, 4, 6],
                 "model__scale_pos_weight": [1, float((y == 0).sum() / (y == 1).sum())]})
            return options, None
        except (ImportError, OSError) as error:
            reason = f"XGBoost unavailable ({type(error).__name__})."
        except Exception as error:
            if error.__class__.__module__ != "xgboost.core" or error.__class__.__name__ != "XGBoostError":
                raise
            reason = "XGBoost native runtime could not load."
    options["Histogram Gradient Boosting"] = (
        make_pipeline(HistGradientBoostingClassifier(max_iter=140, random_state=SEED, early_stopping=False)),
        {"model__max_leaf_nodes": [7, 15, 31], "model__class_weight": [None, "balanced"]},
    )
    return options, reason


def partition(frame):
    if frame[TARGET].value_counts().min() < 18:
        raise ValueError("Training requires at least 18 records in each target class.")
    training_pool, test = train_test_split(frame, test_size=.2, random_state=SEED, stratify=frame[TARGET])
    train, valid = train_test_split(training_pool, test_size=.25, random_state=SEED,
                                    stratify=training_pool[TARGET])
    return train, valid, test


def select_models(train, valid, features, force_fallback=False):
    X, y = train[features], train[TARGET]
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    options, fallback = candidates(y, force_fallback)
    fitted, results = {}, {}
    for name, (pipeline, grid) in options.items():
        print(f"Training {name} …", flush=True)
        search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, n_jobs=1,
                              error_score="raise").fit(X, y)
        raw = search.best_estimator_
        calibrated = CalibratedClassifierCV(raw, method="sigmoid", cv=cv, ensemble=False).fit(X, y)
        benchmark = {
            "Uncalibrated": (raw, raw.predict_proba(valid[features])[:, 1]),
            "Sigmoid": (calibrated, calibrated.predict_proba(valid[features])[:, 1]),
        }
        scores = {label: brier_score_loss(valid[TARGET], prob) for label, (_, prob) in benchmark.items()}
        separate_calibrator = min(scores, key=scores.get)
        fitted[name], prob = benchmark[separate_calibrator]
        results[name] = {
            "cv_average_precision": float(search.best_score_),
            "cv_average_precision_sd": float(search.cv_results_["std_test_score"][search.best_index_]),
            "params": search.best_params_, "calibration_method": separate_calibrator,
            "validation_brier_candidates": scores,
            "validation_ap": average_precision_score(valid[TARGET], prob),
            "threshold": threshold_for_f1(valid[TARGET], prob),
        }
    return fitted, results, fallback


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        json.dump(content, output, indent=2, ensure_ascii=False, allow_nan=False)
    Path(output.name).replace(path)


def train_project(path=None, sheet=None, force_fallback=False):
    run_id = uuid4().hex
    frame, quality = load_dataset(path, sheet)
    known = [c for c in PREDICTORS if c in frame]
    if not known or frame[known].duplicated().any():
        raise ValueError("The dataset needs distinct modeling profiles. Resolve matching predictor rows before splitting.")
    train, valid, test = partition(frame)
    features = [c for c in known if train[c].nunique() > 1]
    if not features:
        raise ValueError("No nonconstant supported predictors remain in the training partition.")
    if frame[features].duplicated().any():
        raise ValueError("Removing training-constant fields produced matching predictor profiles. Review duplicates before training.")
    with threadpool_limits(limits=2):
        fitted, results, fallback = select_models(train, valid, features, force_fallback)
        winner = max(results, key=lambda name: results[name]["cv_average_precision"])
        y_test = test[TARGET]
        for name, model in fitted.items():
            p = model.predict_proba(test[features])[:, 1]
            results[name].update(evaluate(y_test, p, results[name]["threshold"]))
        p = fitted[winner].predict_proba(test[features])[:, 1]
        importance = global_importance(fitted[winner], test[features], y_test, SEED)
        interval = bootstrap_intervals(y_test, p, SEED)
        reference, values = reference_values(train[features])
        bundle = {"model": fitted[winner], "features": features, "threshold": results[winner]["threshold"],
                  "reference": reference, "response_values": values, "name": winner,
                  "quality": quality, "version": VERSION, "run_id": run_id}
        scored = predict(bundle, frame)

    metadata = {
        "version": VERSION, "run_id": run_id, "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "sklearn_version": sklearn.__version__,
        "split": {"train": len(train), "validation": len(valid), "test": len(test),
                  "test_positive": int(y_test.sum())}, "selected": winner, "fallback": fallback,
        "selection": "Highest four-fold training cross-validation average precision.",
        "features": features, "dataset_sha256": quality["sha256"], "models": results,
        "bootstrap_intervals": interval,
        "prevalence_brier": brier_score_loss(y_test, np.full(len(test), train[TARGET].mean())),
        "importance": importance.to_dict(orient="records"),
    }
    audit = fairness_audit(test, p, bundle["threshold"])
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=MODEL.parent, delete=False) as model_file:
        joblib.dump(bundle, model_file.name, compress=3)
    Path(model_file.name).replace(MODEL)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    display = scored.drop(columns=list(AUDIT_ONLY), errors="ignore")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=PROCESSED, delete=False) as records_file:
        display.assign(_run=run_id).to_csv(records_file, index=False)
    Path(records_file.name).replace(PROCESSED / "scored_records.csv")
    write_json(PRIVATE_AUDIT, audit)
    write_json(METRICS, metadata)
    write_json(DEMO, {"run_id": run_id, **aggregate_summary(scored, quality)})
    print(f"Selected {winner}. Saved model, evaluation, scores and aggregate snapshot.", flush=True)
    return metadata
