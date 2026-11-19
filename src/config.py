"""Paths, schema and constants shared by training and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODEL = ARTIFACTS / "models" / "attrition.joblib"
METRICS = ARTIFACTS / "metrics" / "evaluation.json"
DEMO = ARTIFACTS / "metrics" / "demo.json"
PRIVATE_AUDIT = METRICS.parent / "private" / "fairness.json"
SEED = 42
VERSION = "1.0.0"
TARGET = "Attrition"
SENSITIVE = ("Gender", "MaritalStatus")
AUDIT_ONLY = (*SENSITIVE, "Age")
ID_FIELDS = ("EmployeeID", "EmployeeNumber", "StaffID")
CATEGORIES = (
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
)
RATINGS = {
    "Education": (1, 5), "JobLevel": (1, 5), "StockOptionLevel": (0, 3),
    "PerformanceRating": (1, 4),
    **dict.fromkeys(("EnvironmentSatisfaction", "JobInvolvement",
                    "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"), (1, 4)),
}
NUMERICS = (
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager", *RATINGS,
)
EXPECTED = (*NUMERICS, *CATEGORIES, TARGET)
REQUIRED = ("Department", "JobRole", TARGET)
PREDICTORS = tuple(c for c in (*NUMERICS, *CATEGORIES) if c not in AUDIT_ONLY)
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
MIN_GROUP = 20
SOURCE_NOTE = "Internship case-study dataset · organization and collection provenance unverified"
