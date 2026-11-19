import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import PREDICTORS, SEED
from src.data_validation import validate
from src.explainability import local_factors, reference_values
from src.prediction import load_model, predict, probabilities, scenario
from src.preprocessing import make_pipeline
from src.train import partition


def test_reproducible_partition_and_pipeline_inference(tmp_path, synthetic_frame):
    frame, _ = validate(synthetic_frame)
    a = partition(frame)
    b = partition(frame)
    assert [list(x.Record) for x in a] == [list(x.Record) for x in b]
    assert len(set(a[0].Record) & set(a[2].Record)) == 0
    features = [c for c in PREDICTORS if c in frame]
    X = a[0][features]
    y = a[0].Attrition
    model = make_pipeline(LogisticRegression(random_state=SEED), scale=True).fit(X, y)
    reference, values = reference_values(X)
    bundle = {"model": model, "features": features, "reference": reference, "response_values": values}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = load_model(path)
    output = predict(loaded, a[2])
    np.testing.assert_array_equal(output.Probability, probabilities(bundle, a[2]))
    np.testing.assert_array_equal(output.RiskScore, output.Probability * 100)
    record = a[2].iloc[[0]].copy()
    copy = record.copy()
    new_income = record.MonthlyIncome.iat[0] + 1200
    original, edited = scenario(bundle, record, {"MonthlyIncome": new_income, "OverTime": "Yes"})
    assert original == pytest.approx(probabilities(bundle, record)[0])
    expected = probabilities(bundle, record.assign(MonthlyIncome=new_income, OverTime="Yes"))[0]
    assert edited == pytest.approx(expected)
    assert scenario(bundle, record, {}) == pytest.approx((original, original))
    pd.testing.assert_frame_equal(record, copy)
    local = local_factors(bundle, record).set_index("Feature")
    _, ref = scenario(bundle, record, {"OverTime": reference["OverTime"]})
    assert local.loc["OverTime", "Change (pp)"] == pytest.approx(100 * (original - ref))
    with pytest.raises(ValueError, match="Unsupported scenario"):
        scenario(bundle, record, {"Attrition": 1})
