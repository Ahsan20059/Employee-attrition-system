"""Field labels and data dictionary for the supplied workforce schema."""
import re

DESCRIPTIONS = {
    "Record": "Stable derived key for this normalized source record.",
    "Age": "Age in years as recorded in the source.",
    "Attrition": "Historical target. 1 records a departure and 0 records a stayed label.",
    "BusinessTravel": "Categorical frequency of business travel.",
    "DailyRate": "Recorded daily rate. Units are unspecified.",
    "Department": "Organizational department.",
    "DistanceFromHome": "Recorded commuting distance. Units are unspecified.",
    "Education": "Ordinal education level, 1–5.",
    "EducationField": "Field of education.",
    "EnvironmentSatisfaction": "Ordinal environment satisfaction rating, 1–4.",
    "Gender": "Source-provided audit field excluded from dashboards and model inputs.",
    "HourlyRate": "Recorded hourly rate. Units are unspecified.",
    "JobInvolvement": "Ordinal job involvement rating, 1–4.",
    "JobLevel": "Ordinal level of the recorded position, 1–5.",
    "JobRole": "Recorded job role.",
    "JobSatisfaction": "Ordinal job satisfaction rating, 1–4.",
    "MaritalStatus": "Source-provided audit field excluded from dashboards and model inputs.",
    "MonthlyIncome": "Recorded monthly income. Currency is unspecified.",
    "MonthlyRate": "Recorded monthly rate field. Definition and units are unspecified.",
    "NumCompaniesWorked": "Number of employers reported in the source field.",
    "OverTime": "Source overtime flag, Yes or No.",
    "PercentSalaryHike": "Recorded salary increase percentage.",
    "PerformanceRating": "Ordinal performance rating.",
    "RelationshipSatisfaction": "Ordinal relationship satisfaction rating, 1–4.",
    "StockOptionLevel": "Ordinal stock option level, 0–3.",
    "TotalWorkingYears": "Recorded years of working experience.",
    "TrainingTimesLastYear": "Recorded training instances in the prior year.",
    "WorkLifeBalance": "Ordinal work–life balance rating, 1–4.",
    "YearsAtCompany": "Recorded company tenure in years.",
    "YearsInCurrentRole": "Recorded years in the current role.",
    "YearsSinceLastPromotion": "Recorded years since the last promotion.",
    "YearsWithCurrManager": "Recorded years with the current manager.",
    "Probability": "Selected model’s estimated attrition probability for the record.",
    "RiskScore": "Estimated probability multiplied by 100.",
    "RiskBand": "Display category derived from the selected risk boundaries.",
}
LABELS = {"OverTime": "Overtime", "WorkLifeBalance": "Work–life balance",
          "NumCompaniesWorked": "Companies worked at", "YearsWithCurrManager": "Years with current manager",
          "RiskBand": "Risk category", "RiskScore": "Risk score", "Attrition": "Historical attrition"}


def label(field):
    return LABELS.get(field, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", field).capitalize())
