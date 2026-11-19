"""Executive dashboard with retrospective workforce summaries."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import RISK_LABELS, SOURCE_NOTE
from src.dashboard_utils import (
    PALETTE, RISK_COLORS, TEAL, VIOLET, bounds, card, category_totals, chart,
    comparison, comparison_table, donut, get_context, group_data, heading, kpi, legend,
)


def render():
    summary, evaluation, _, frame = get_context()
    heading("Workforce overview", "Review historical attrition, model-estimated risk and workforce composition.")
    totals = category_totals(frame, summary)
    low, high = bounds()
    total, left = summary["records"], summary["left"]
    mean_satisfaction = summary["mean_satisfaction"]
    kpis = [
        ("Total employee records", f"{total:,}", "Validated source records", "▦", ""),
        ("Average predicted risk", f"{summary['mean_risk']:.1%}", "Selected model · all records", "◈", ""),
        ("Historical attrition rate", f"{left / total:.1%}", f"{left:,} recorded departures", "↗", "amber"),
        ("Average job satisfaction", f"{mean_satisfaction:.2f} / 4" if mean_satisfaction is not None else "Unavailable",
         "Source rating scale", "✦", "teal"),
    ]
    with st.container(key="kpi_grid_overview"):
        for index, (column, metric) in enumerate(zip(st.columns(4), kpis)):
            with column:
                kpi(f"overview_{index}", *metric)

    department = group_data(summary, frame, "Department")
    left_col, right_col = st.columns([1.65, 1])
    with left_col, card("overview_departments"):
        st.subheader("Attrition by department")
        st.caption("Historical departure rate and mean model estimate for each department.")
        comparison(department, "overview_dept_chart")
    with right_col, card("overview_risk"):
        st.subheader("Predicted risk distribution")
        st.caption(f"Low < {low:.0%} · Medium {low:.0%}–{high:.0%} · High > {high:.0%}")
        chart(donut(RISK_LABELS, totals, RISK_COLORS, f"{totals[-1]:,}", "High risk"), "overview_risks", 165)
        legend(RISK_LABELS, totals, RISK_COLORS)

    left_col, right_col = st.columns([1.65, 1])
    with left_col, card("overview_priority"):
        if frame is not None:
            st.subheader("Highest-scoring records")
            st.caption("Includes both historical outcomes. Scores are decision-support estimates.")
            top = frame.loc[frame.RiskBand == "High Risk"].nlargest(6, "RiskScore")
            if top.empty:
                st.caption("No records meet the current high-risk boundary.")
            else:
                comparison_table(top[["Record", "Department", "JobRole", "RiskScore", "RiskBand"]], "top_risk")
        else:
            st.subheader("Department summary")
            st.caption("Cohort-level counts and average score for available departments.")
            if not department.empty:
                comparison_table(department.sort_values("Mean risk", ascending=False), "dept_summary")
        st.subheader("Model signals")
        drivers = [item["Feature"] for item in evaluation["importance"] if item["AP decrease"] > 0][:3]
        if drivers:
            from src.schema import label
            st.write(", ".join(label(feature) for feature in drivers) + " had the largest held-out permutation effects.")
        overtime = group_data(summary, frame, "OverTime")
        overtime = overtime.set_index("Group") if not overtime.empty else pd.DataFrame()
        if {"Yes", "No"} <= set(overtime.index):
            st.caption(f"Historical attrition was {overtime.loc['Yes', 'Historical rate']:.1%} among overtime records "
                       f"and {overtime.loc['No', 'Historical rate']:.1%} among other records.")
    with right_col, card("overview_mix"):
        st.subheader("Workforce composition")
        st.caption("Records by department.")
        if not department.empty:
            labels = department.Group.tolist()
            counts = department.Records.astype(int).tolist()
            chart(donut(labels, counts, PALETTE, f"{sum(counts):,}", "Records shown"), "workforce_mix", 165)
            legend(labels, counts, PALETTE)

    left_col, right_col = st.columns([1.65, 1])
    with left_col, card("overview_tenure"):
        st.subheader("Historical attrition by tenure")
        st.caption("Company tenure cohorts from a single snapshot.")
        comparison(group_data(summary, frame, "YearsAtCompany"), "overview_tenure_chart", ("Historical rate",))
    with right_col, card("overview_outcomes"):
        st.subheader("Historical outcomes")
        st.caption("Recorded stayed and left labels.")
        fig = go.Figure(go.Bar(x=["Stayed", "Left"], y=[total - left, left],
                               marker_color=[TEAL, VIOLET], hovertemplate="%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(barcornerradius=5)
        chart(fig, "actual_outcomes")
    st.caption("Historical labels describe the supplied snapshot. Model risk summarizes scores across that same "
               "snapshot, including training records. Held-out results are available under Model Performance.")
    st.caption(SOURCE_NOTE)
