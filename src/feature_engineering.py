"""Row-local transformations shared by fitting, inference and scenarios."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ENGAGEMENT = ("JobSatisfaction", "EnvironmentSatisfaction", "JobInvolvement",
              "RelationshipSatisfaction", "WorkLifeBalance")


class WorkforceFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        self.feature_names_out_ = self._transform(X).columns.to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._transform(X.loc[:, self.feature_names_in_])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    @staticmethod
    def _transform(X):
        out = X.copy()
        if {"MonthlyIncome", "TotalWorkingYears"} <= set(X):
            out["IncomePerExperienceYear"] = X.MonthlyIncome / X.TotalWorkingYears.clip(lower=1)
        if "YearsSinceLastPromotion" in X:
            delay = X.YearsSinceLastPromotion
            out["LongPromotionDelay"] = delay.ge(5).astype(float).where(delay.notna())
        engagement = [c for c in ENGAGEMENT if c in X]
        if engagement:
            out["EngagementIndex"] = X[engagement].mean(axis=1)
        if {"OverTime", "WorkLifeBalance"} <= set(X):
            overtime = X.OverTime.map({"Yes": 1.0, "No": 0.0})
            balance = X.WorkLifeBalance
            out["WorkloadProxy"] = overtime + balance.le(2).astype(float).where(balance.notna())
        if {"YearsAtCompany", "TotalWorkingYears"} <= set(X):
            out["PriorCareerYears"] = (X.TotalWorkingYears - X.YearsAtCompany).clip(lower=0)
        if {"YearsAtCompany", "YearsInCurrentRole"} <= set(X):
            out["TenureBeforeCurrentRole"] = (X.YearsAtCompany - X.YearsInCurrentRole).clip(lower=0)
        return out
