"""Held-out evaluation, uncertainty and sample-aware group audit."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import SENSITIVE


def threshold_for_f1(y, p):
    precision, recall, t = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(t[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, p, threshold):
    pred = p >= threshold
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    # Uniform bins keep the calibration plot interpretable on the probability scale.
    observed, predicted = calibration_curve(y, p, n_bins=8)
    counts = np.bincount(np.searchsorted(np.linspace(0, 1, 9)[1:-1], p), minlength=8)
    populated_counts = counts[counts > 0]
    visible = populated_counts >= 5
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "threshold": threshold,
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
        "roc": {"x": fpr.tolist(), "y": tpr.tolist()},
        "pr": {"x": recall.tolist(), "y": precision.tolist()},
        "calibration": {"x": predicted[visible].tolist(), "y": observed[visible].tolist(), "n": populated_counts[visible].tolist()},
    }


def bootstrap_intervals(y, p, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    case, control = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = []
    for _ in range(500):
        indices = np.r_[rng.choice(case, len(case)), rng.choice(control, len(control))]
        observed, scores = y[indices], p[indices]
        samples.append([roc_auc_score(observed, scores), average_precision_score(observed, scores),
                        brier_score_loss(observed, scores)])
    limits = np.percentile(samples, [2.5, 97.5], axis=0).T.tolist()
    return dict(zip(("roc_auc", "average_precision", "brier"), limits))


def fairness_audit(test, p, threshold):
    audit = {}
    frame = test.assign(Flag=p >= threshold)
    for feature in set(SENSITIVE) & set(frame):
        results = []
        for group, rows in frame.dropna(subset=[feature]).groupby(feature):
            positives = int(rows.Attrition.sum())
            negatives = len(rows) - positives
            if len(rows) < 30 or min(positives, negatives) < 10:
                continue
            tn, fp, fn, tp = confusion_matrix(rows.Attrition, rows.Flag, labels=[0, 1]).ravel()
            results.append({"group": group, "n": len(rows), "positive": positives,
                            "false_positive_rate": fp / negatives, "false_negative_rate": fn / positives})
        audit[feature] = results
    return audit
