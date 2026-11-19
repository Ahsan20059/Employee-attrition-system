"""Controlled edits with the same feature pipeline as baseline predictions."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import TEAL, VIOLET, bounds, card, chart, get_context, heading, record_picker, require_records
from src.prediction import scenario
from src.risk_scoring import risk_bands
from src.schema import label

EDITABLE = ("OverTime", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
            "MonthlyIncome", "YearsSinceLastPromotion")


def field_options(field, row, bundle):
    options = bundle["response_values"][field]
    baseline = row[field]
    if field == "YearsSinceLastPromotion" and "YearsAtCompany" in row and pd.notna(row.YearsAtCompany):
        options = [value for value in options if value <= row.YearsAtCompany]
    if pd.isna(baseline):
        return [None, *options]
    return [baseline, *[value for value in options if value != baseline]]


def display_value(value):
    if value is None:
        return "Keep missing source value"
    return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)


def render():
    _, _, bundle, frame = get_context()
    heading("Scenario analysis", "Compare a baseline estimate with a controlled change to selected workforce inputs.")
    if not require_records(frame):
        return
    record = record_picker(frame, "scenario_record")
    row = record.iloc[0]
    st.caption(f"{row.Department} · {row.JobRole}")
    left, right = st.columns([1, 1.25])
    with left, card("scenario_controls"):
        st.subheader("Scenario inputs")
        overrides = {}
        for field in EDITABLE:
            if field not in bundle["response_values"]:
                continue
            options = field_options(field, row, bundle)
            selected = st.selectbox(label(field), options, key=f"scenario_{row.Record}_{field}",
                                    format_func=display_value)
            if selected is not None and (pd.isna(row[field]) or selected != row[field]):
                overrides[field] = selected
        st.caption("Choices come from observed training values, with quantiles used to sample larger ranges. All other record fields stay fixed.")
    with right, card("scenario_result"):
        st.subheader("Model estimates")
        baseline, modified = scenario(bundle, record, overrides)
        original_band, modified_band = risk_bands([baseline, modified], *bounds())
        fig = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[baseline, modified],
                               marker_color=[VIOLET, TEAL], text=[f"{baseline:.1%}", f"{modified:.1%}"],
                               textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig.update_layout(barcornerradius=6)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.09])
        chart(fig, "scenario_comparison", 270)
        difference = (modified - baseline) * 100
        st.write(f"**{difference:+.2f} percentage points** · {original_band} → {modified_band}")
        if not overrides:
            st.caption("The scenario matches the baseline. Change an input to compare estimates.")
    st.info("This simulation reflects associations learned by the model. It does not estimate how an intervention "
            "would change an employee’s likelihood of leaving.")
