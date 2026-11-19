import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import PREDICTORS, SEED
from src.data_validation import validate
from src.preprocessing import make_pipeline


def test_pipeline_imputation_and_unseen_categories(synthetic_frame):
    frame, _ = validate(synthetic_frame)
    X = frame[[c for c in PREDICTORS if c in frame]]
    y = frame.Attrition
    model = make_pipeline(LogisticRegression(random_state=SEED), scale=True).fit(X, y)
    query = X.iloc[:2].assign(Department=["Unseen", np.nan], MonthlyIncome=[np.nan, 3000],
                             JobSatisfaction=np.nan, OverTime=np.nan)
    p = model.predict_proba(query)[:, 1]
    assert np.isfinite(p).all()
    assert ((0 <= p) & (p <= 1)).all()
    assert "numeric__EngagementIndex" in model["prepare"].get_feature_names_out()
    assert not any("Gender" in name or "Attrition" in name
                   for name in model["prepare"].get_feature_names_out())
