"""Search, filters and source data quality with a public cohort-only mode."""
import pandas as pd
import streamlit as st

from src.analytics import COHORTS
from src.config import AUDIT_ONLY, RISK_LABELS
from src.dashboard_utils import card, comparison, comparison_table, download_csv, get_context, group_data, heading
from src.schema import DESCRIPTIONS, label

DISPLAY_PRIVATE = ["Record", "Department", "JobRole", "Attrition", "Probability", "RiskScore", "RiskBand"]


def quality_panels(quality):
    with st.expander("Data quality and column descriptions"):
        st.write(f"**{quality['columns']}** source columns · **{sum(quality['missing_values'].values())}** missing "
                 f"cells in validated fields · **{quality['duplicates']}** duplicate normalized records removed")
        missing_expected = quality["missing_expected"]
        if missing_expected:
            st.caption(f"Expected fields absent from input: {', '.join(missing_expected)}")
        st.dataframe(pd.DataFrame([
            {"Field": label(field), "Type": dtype, "Missing": quality["missing_values"].get(field),
             "Description": DESCRIPTIONS[field]}
            for dtype, group in (("Numeric", quality["numeric_fields"]), ("Categorical", quality["categorical_fields"]),
                                 ("Binary target", ["Attrition"]))
            for field in group if field not in AUDIT_ONLY
        ]), hide_index=True, use_container_width=True,
            column_config={"Description": st.column_config.TextColumn(width="large")})
        invalid = sum(quality["tenure_inconsistencies"].values())
        st.caption(f"{invalid} tenure consistency check violations across available checks.")


def render():
    summary, _, _, frame = get_context()
    heading("Data explorer", "Inspect source coverage and compare workforce cohorts.")
    if frame is None:
        available = [col for col in COHORTS if summary["cohorts"].get(col)]
        with card("public_explorer"):
            if available:
                field = st.selectbox("Group records by", available, format_func=COHORTS.get)
                table = group_data(summary, None, field)
                comparison(table, "public_explorer_chart")
                comparison_table(table, "public_explorer_table")
                download_csv(table, "Download cohort summary", "attrition-cohorts.csv")
            else:
                st.info("No cohorts are available above the release thresholds.")
        quality_panels(summary["quality"])
        return

    with card("private_explorer"):
        search, departments, roles, risk = st.columns([1.15, 1, 1, 1])
        with search:
            query = st.text_input("Search", placeholder="Record, job role or department").strip().casefold()
        with departments:
            selected_depts = st.multiselect("Department", sorted(frame.Department.dropna().unique()))
        with roles:
            selected_roles = st.multiselect("Job role", sorted(frame.JobRole.dropna().unique()))
        with risk:
            selected_risk = st.multiselect("Risk category", RISK_LABELS)
        fields = [c for c in ("YearsAtCompany", "MonthlyIncome", "JobSatisfaction")
                  if c in frame and frame[c].nunique() > 1]
        scoped = frame.copy()
        if query:
            haystack = scoped[["Record", "JobRole", "Department"]].fillna("").agg(" ".join, axis=1).str.casefold()
            scoped = scoped.loc[haystack.str.contains(query, regex=False)]
        for field, values in (("Department", selected_depts), ("JobRole", selected_roles), ("RiskBand", selected_risk)):
            if values:
                scoped = scoped.loc[scoped[field].isin(values)]
        with st.expander("Numeric filters and visible columns"):
            field = st.selectbox("Filter by a numeric field", [None, *fields],
                                 format_func=lambda value: "None" if value is None else label(value))
            if field:
                lower, upper = float(frame[field].min()), float(frame[field].max())
                lo, hi = st.slider(label(field), lower, upper, (lower, upper))
                scoped = scoped.loc[scoped[field].between(lo, hi)]
            extra = st.multiselect("Additional columns", [c for c in frame if c not in DISPLAY_PRIVATE],
                                   default=[c for c in fields[:2] if c in frame], format_func=label)
        st.subheader(f"{len(scoped):,} matching records")
        if scoped.empty:
            st.info("No matching records. Remove a filter or try a different search.")
        else:
            columns = DISPLAY_PRIVATE + extra
            table = scoped[columns].copy()
            table["Attrition"] = table.Attrition.map({0: "Stayed", 1: "Left"})
            comparison_table(table, "private_explorer_table")
            download_csv(table, "Download filtered records", "attrition-filtered-records.csv")
    quality_panels(summary["quality"])
