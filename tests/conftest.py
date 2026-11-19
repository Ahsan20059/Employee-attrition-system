"""Synthetic fixtures for tests only. Production uses the supplied CSV."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(271)
    n = 144
    p = rng.uniform(size=n)
    ot = rng.choice(["Yes", "No"], n)
    return pd.DataFrame({
        "Department": rng.choice(["Research", "Sales", "Support"], n),
        "JobRole": rng.choice(["Analyst", "Specialist", "Lead"], n),
        "MonthlyIncome": 1000 + 39 * np.arange(n),
        "TotalWorkingYears": rng.integers(0, 30, n),
        "OverTime": ot,
        "YearsSinceLastPromotion": rng.integers(0, 7, n),
        "JobSatisfaction": rng.choice([1, 2, 3, 4, np.nan], n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "Gender": rng.choice(["A", "B"], n),
        "MaritalStatus": rng.choice(["X", "Y"], n),
        "Attrition": np.where(p < np.where(ot == "Yes", .48, .25), "Yes", "No"),
    })
