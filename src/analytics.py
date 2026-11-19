"""Workforce cohort summaries shared by the dashboard and report generator."""
import pandas as pd

from src.config import MIN_GROUP, RISK_LABELS, TARGET

COHORTS = {
    "Department": "Department", "JobRole": "Job role", "OverTime": "Overtime",
    "MonthlyIncome": "Monthly income band", "JobSatisfaction": "Job satisfaction",
    "EnvironmentSatisfaction": "Environment satisfaction", "WorkLifeBalance": "Work–life balance",
    "YearsSinceLastPromotion": "Years since last promotion", "YearsAtCompany": "Company tenure",
    "BusinessTravel": "Business travel", "JobLevel": "Job level",
}


def cohorts(frame, column):
    values = frame[column]
    if column == "MonthlyIncome":
        values = pd.cut(values, [-1, 2999, 5999, 9999, float("inf")],
                        labels=["Below 3,000", "3,000–5,999", "6,000–9,999", "10,000+"])
    if column == "YearsAtCompany":
        values = pd.cut(values, [-1, 2, 5, 10, float("inf")],
                        labels=["0–2 years", "3–5 years", "6–10 years", "11+ years"])
    if column == "YearsSinceLastPromotion":
        values = pd.cut(values, [-1, 0, 2, 5, float("inf")],
                        labels=["0 years", "1–2 years", "3–5 years", "6+ years"])
    buckets = frame.groupby(values, observed=True, dropna=True)
    result = buckets[TARGET].agg(Records="size", Left="sum")
    result["Stayed"] = result.Records - result.Left
    result["Historical rate"] = result.Left / result.Records
    if "Probability" in frame:
        result["Mean risk"] = buckets.Probability.mean()
        for band in RISK_LABELS:
            result[band] = buckets.RiskBand.agg(lambda values: values.eq(band).sum())
    result.index = result.index.astype(str)
    return result.rename_axis("Group").reset_index()


def released_groups(groups):
    """Suppress small cells and a second group when a sole omission is inferable."""
    eligible = groups[(groups.Records >= MIN_GROUP) & (groups.Left >= 5) & (groups.Stayed >= 5)]
    if len(groups) - len(eligible) == 1 and not eligible.empty:
        eligible = eligible.drop(index=eligible.Records.idxmin())
    return eligible


def aggregate_summary(frame, quality):
    summary = {"records": len(frame), "left": int(frame[TARGET].sum()),
               "mean_risk": float(frame.Probability.mean()),
               "mean_satisfaction": (float(frame.JobSatisfaction.mean())
                                     if "JobSatisfaction" in frame and frame.JobSatisfaction.notna().any() else None),
               "bands": frame.RiskBand.value_counts(sort=False).to_dict(),
               "cohorts": {}, "quality": {
                   k: quality[k] for k in ("columns", "duplicates", "missing_values",
                                           "missing_expected", "numeric_fields", "categorical_fields",
                                           "tenure_inconsistencies")
               }}
    for col in COHORTS.keys() & set(frame):
        group = cohorts(frame, col)
        group = released_groups(group)
        # Release no predicted risk cell counts for subgroups in the public snapshot.
        group = group.drop(columns=RISK_LABELS)
        summary["cohorts"][col] = group.to_dict(orient="records")
    return summary
