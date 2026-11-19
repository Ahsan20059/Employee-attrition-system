"""Private profile, risk visualization and downloadable summary."""
import plotly.graph_objects as go
import streamlit as st

from pages.explain import local_chart
from src.dashboard_utils import (bounds, card, chart, download_csv, get_context,
                                 heading, kpi, record_picker, require_records)
from src.schema import label


def render():
    _, evaluation, bundle, frame = get_context()
    heading("Employee risk", "Review a workforce record and the inputs that influence its model estimate.")
    if not require_records(frame):
        return
    record = record_picker(frame, "profile_record")
    row = record.iloc[0]
    st.subheader(f"{row.Record} · {row.JobRole}")
    st.caption(f"{row.Department} · Derived record identifier")
    for i, (col, metric) in enumerate(zip(st.columns(3), [
        ("Predicted probability", f"{row.Probability:.1%}", "Selected model"),
        ("Risk score", f"{row.RiskScore:.1f} / 100", "Probability on a 100-point scale"),
        ("Risk category", row.RiskBand, "Current sidebar boundaries"),
    ])):
        with col:
            kpi(f"profile_{i}", *metric)
    first, second = st.columns([1.05, 1.3])
    with first, card("profile_gauge"):
        st.subheader("Probability scale")
        low, high = bounds()
        edges = [0, low, high, 1]
        fig = go.Figure(go.Indicator(mode="gauge+number", value=row.Probability * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 27}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#5242a4"},
                   "steps": [{"range": [100 * a, 100 * b], "color": color}
                             for a, b, color in zip(edges, edges[1:], ["#DFF2EC", "#F9EBD3", "#F6DDE5"])]}))
        chart(fig, "profile_gauge_plot", 186)
        actual = "Left" if row.Attrition else "Stayed"
        st.write(f"Historical label · **{actual}**")
        st.caption(f"Model operating threshold · {evaluation['models'][bundle['name']]['threshold']:.1%}. "
                   "Display risk categories use the adjustable boundaries.")
        fields = [c for c in ("OverTime", "JobLevel", "YearsAtCompany", "JobSatisfaction",
                              "WorkLifeBalance", "MonthlyIncome") if c in record]
        if fields:
            st.dataframe(record[fields].rename(columns=label), hide_index=True, use_container_width=True)
    with second, card("profile_signals"):
        st.subheader("Model contributions for this record")
        local_chart(bundle, record, "profile_signals_chart")
    export = record[["Record", "Department", "JobRole", "Attrition", "Probability", "RiskScore", "RiskBand"]]
    download_csv(export, "Export risk summary", f"{row.Record}-risk-summary.csv")
    st.caption("Use estimates as decision support for exploratory review. Employment decisions require independent human judgment and evidence.")
