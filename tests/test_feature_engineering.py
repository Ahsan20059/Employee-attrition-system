import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import WorkforceFeatures


def test_formulas_and_copy():
    data = pd.DataFrame({
        "MonthlyIncome": [1300, 6000], "TotalWorkingYears": [0, 12],
        "YearsAtCompany": [0, 9], "YearsInCurrentRole": [0, 2],
        "YearsSinceLastPromotion": [0, 5], "OverTime": ["No", "Yes"],
        "JobSatisfaction": [1, 4], "WorkLifeBalance": [2, 3],
        "EnvironmentSatisfaction": [np.nan, 4],
    })
    transformer = WorkforceFeatures().fit(data)
    result = transformer.transform(data)
    assert result.IncomePerExperienceYear.tolist() == [1300, 500]
    assert result.WorkloadProxy.tolist() == [1, 1]
    assert result.LongPromotionDelay.tolist() == [0, 1]
    assert result.EngagementIndex.tolist() == pytest.approx([1.5, 11 / 3])
    assert result.PriorCareerYears.tolist() == [0, 3]
    assert result.TenureBeforeCurrentRole.tolist() == [0, 7]
    assert result.columns.tolist() == transformer.get_feature_names_out().tolist()
    assert "EngagementIndex" not in data
