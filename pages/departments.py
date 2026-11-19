"""Department and role comparisons with cohort-size context."""
import plotly.graph_objects as go
import streamlit as st

from src.config import MIN_GROUP, RISK_LABELS
from src.dashboard_utils import (RISK_COLORS, bounds, card, category_totals, chart, comparison,
                                 comparison_table, donut, get_context, group_data, heading, kpi, legend)


def render():
    summary, _, _, frame = get_context()
    heading("Department analytics", "Compare workforce segments by predicted risk and recorded departures.")
    if frame is None:
        st.caption("Aggregate snapshot. Department and role breakdowns are shown independently. "
                   "Released groups contain at least 20 records and five records in each outcome class.")
        titles = {"Department": "Departments", "JobRole": "Job roles"}
        scope = st.radio("Compare by", list(titles), format_func=titles.get, horizontal=True)
        with card("aggregate_departments"):
            st.subheader(titles[scope])
            groups = group_data(summary, None, scope)
            comparison(groups, "aggregate_dept_plot")
            if not groups.empty:
                comparison_table(groups, "aggregate_dept_table")
        for column, source, name in zip(st.columns(2), ("OverTime", "JobSatisfaction"),
                                        ("Overtime", "Job satisfaction")):
            with column, card(f"aggregate_{source}"):
                st.subheader(name)
                comparison(group_data(summary, None, source), f"aggregate_{source}_plot", ("Historical rate",))
        return

    col_a, col_b = st.columns(2)
    with col_a:
        department = st.selectbox("Department", ["All departments", *sorted(frame.Department.dropna().unique())])
    scoped = frame if department == "All departments" else frame.loc[frame.Department == department]
    roles = ["All roles", *sorted(scoped.JobRole.dropna().unique())]
    with col_b:
        role = st.selectbox("Job role", roles)
    scoped = scoped if role == "All roles" else scoped.loc[scoped.JobRole == role]
    if scoped.empty:
        st.info("No records match the selected department and role.")
        return
    counts = category_totals(scoped, summary)
    for index, (container, metric) in enumerate(zip(st.columns(4), [
        ("Records", f"{len(scoped):,}", "Selected workforce cohort"),
        ("Mean predicted risk", f"{scoped.Probability.mean():.1%}", "Selected model"),
        ("High-risk records", f"{counts[-1]:,}", f"Probability above {bounds()[1]:.0%}"),
        ("Historical attrition", f"{scoped.Attrition.mean():.1%}", f"{int(scoped.Attrition.sum()):,} recorded departures"),
    ])):
        with container:
            kpi(f"dept_{index}", *metric)
    if len(scoped) < MIN_GROUP:
        st.warning(f"This selection contains fewer than {MIN_GROUP} records. Treat rates as unstable.")

    col_a, col_b = st.columns([1.5, 1])
    with col_a, card("dept_roles"):
        st.subheader("Attrition by job level")
        groups = group_data(summary, scoped, "JobLevel")
        comparison(groups, "dept_levels_plot")
        if not groups.empty:
            comparison_table(groups[["Group", "Records", "Historical rate", "Mean risk"]], "dept_levels")
    with col_b, card("dept_breakdown"):
        st.subheader("Risk categories")
        chart(donut(RISK_LABELS, counts, RISK_COLORS, f"{sum(counts):,}", "Selected records"), "dept_risks", 175)
        legend(RISK_LABELS, counts, RISK_COLORS)
    left, right = st.columns(2)
    for col, field, title in [(left, "OverTime", "Overtime"), (right, "JobSatisfaction", "Job satisfaction")]:
        with col, card(f"dept_{field}"):
            st.subheader(title)
            groups = group_data(summary, scoped, field)
            comparison(groups, f"dept_{field}_plot")
            if not groups.empty and groups.Records.min() < MIN_GROUP:
                st.caption(f"At least one displayed group contains fewer than {MIN_GROUP} records.")
    st.caption("These are cross-sectional comparisons. A recorded departure rate and an average model score answer different questions.")
