"""Exercise the training entry point on a small explicitly synthetic input."""
import json

import pandas as pd
from streamlit.testing.v1 import AppTest

import src.train as training
from src.config import ROOT
from src.data_loader import load_dataset
from src.prediction import load_model, probabilities


def test_train_evaluate_and_reload(tmp_path, monkeypatch, synthetic_frame):
    data = tmp_path / "synthetic.csv"
    synthetic_frame.to_csv(data, index=False)
    outputs = {
        "MODEL": tmp_path / "models" / "model.joblib", "METRICS": tmp_path / "evaluation.json",
        "DEMO": tmp_path / "demo.json", "PRIVATE_AUDIT": tmp_path / "private" / "fairness.json",
        "PROCESSED": tmp_path / "processed",
    }
    for key, path in outputs.items():
        monkeypatch.setattr(training, key, path)
    results = training.train_project(data, force_fallback=True)
    assert set(results["models"]) == {"Logistic Regression", "Random Forest", "Histogram Gradient Boosting"}
    assert results["selected"] == max(results["models"], key=lambda n: results["models"][n]["cv_average_precision"])
    assert results["split"]["test"] == 29
    assert all(len(row["confusion"]) == 2 for row in results["models"].values())
    assert results["bootstrap_intervals"]["roc_auc"][0] <= results["bootstrap_intervals"]["roc_auc"][1]
    assert outputs["METRICS"].exists() and outputs["PRIVATE_AUDIT"].exists()
    bundle = load_model(outputs["MODEL"])
    data, _ = load_dataset(data)
    assert probabilities(bundle, data).shape == (len(data),)
    exported = pd.read_csv(outputs["PROCESSED"] / "scored_records.csv")
    assert "Gender" not in exported and "MaritalStatus" not in exported
    demo = json.loads(outputs["DEMO"].read_text())
    demo_text = json.dumps(demo)
    assert not any(identifier in demo_text for identifier in data.Record)
    assert demo["run_id"] == results["run_id"] == bundle["run_id"]


def test_dashboard_starts_in_public_mode(monkeypatch):
    monkeypatch.delenv("ATTRITION_MODE", raising=False)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) >= 4
    assert not app.slider


def test_release_forces_aggregate_mode(tmp_path, monkeypatch):
    from src import dashboard_utils

    monkeypatch.setattr(dashboard_utils, "ROOT", tmp_path)
    monkeypatch.setenv("ATTRITION_MODE", "private")
    assert dashboard_utils.private_mode()
    (tmp_path / ".aggregate-only").touch()
    assert not dashboard_utils.private_mode()


def test_small_cohort_requires_complementary_suppression():
    from src.analytics import released_groups

    groups = pd.DataFrame([
        {"Group": "small", "Records": 22, "Left": 2, "Stayed": 20},
        {"Group": "first", "Records": 50, "Left": 12, "Stayed": 38},
        {"Group": "second", "Records": 90, "Left": 30, "Stayed": 60},
    ])
    assert released_groups(groups).Group.tolist() == ["second"]
