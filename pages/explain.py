"""Global model inspection and record-specific sensitivity."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import TEAL, VIOLET, card, chart, get_context, heading, record_picker
from src.explainability import local_factors, response_profile
from src.schema import label


def local_chart(bundle, record, key):
    factors = local_factors(bundle, record).head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        y=factors.Feature.map(label), x=factors["Change (pp)"], orientation="h",
        marker_color=[VIOLET if value >= 0 else TEAL for value in factors["Change (pp)"]],
        customdata=factors[["Observed", "Reference"]],
        hovertemplate="%{y}<br>Observed %{customdata[0]} · Reference %{customdata[1]}"
                      "<br>Observed minus reference %{x:+.2f} pp<extra></extra>",
    ))
    fig.update_xaxes(title="Probability change (percentage points)", zeroline=True,
                     zerolinecolor="#A5AAC0")
    chart(fig, key, 300)
    st.caption("Each bar compares this record with the model’s estimate after replacing one input with its "
               "training median or most common value. Bars can overlap in meaning and do not add to the score.")
    with st.expander("Observed values and comparison values"):
        st.dataframe(factors.iloc[::-1].assign(Feature=lambda x: x.Feature.map(label)), hide_index=True,
                     use_container_width=True,
                     column_config={"Change (pp)": st.column_config.NumberColumn(format="%+.2f")})


def render():
    _, evaluation, bundle, frame = get_context()
    heading("Explainability", "Inspect model reliance and understand why estimates vary across records.")
    with card("importance_global"):
        st.subheader("Global feature importance")
        features = pd.DataFrame(evaluation["importance"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(
            y=features.Feature.map(label), x=features["AP decrease"], orientation="h",
            marker_color=[VIOLET if value >= 0 else TEAL for value in features["AP decrease"]],
            error_x={"type": "data", "array": features["Shuffle SD"], "color": "#9586bd", "thickness": 1},
        ))
        fig.update_xaxes(title="Decrease in held-out average precision", zeroline=True,
                         zerolinecolor="#A5AAC0")
        chart(fig, "global_importance", 350)
        st.caption("Each raw feature is shuffled eight times on the test set and the full pipeline is rescored. "
                   "Error bars show variation across shuffles. Correlated inputs can share apparent importance.")
    if frame is None:
        st.caption("Individual explanations and model response profiles are available in an authorized private workspace.")
        return
    left, right = st.columns(2)
    with left, card("explain_local"):
        st.subheader("Record-level factors")
        record = record_picker(frame, "explain_record")
        local_chart(bundle, record, "explain_local_plot")
    with right, card("explain_response"):
        st.subheader("Model response profile")
        available = [row["Feature"] for row in evaluation["importance"] if row["Feature"] in bundle["response_values"]]
        if available:
            feature = st.selectbox("Input to vary", available, format_func=label)
            profile = response_profile(bundle, frame, feature)
            fig = go.Figure(go.Scatter(x=profile[feature], y=profile["Mean model probability"],
                                       mode="markers+lines", line_color=VIOLET))
            fig.update_xaxes(title=label(feature))
            fig.update_yaxes(title="Mean model estimate", tickformat=".1%", rangemode="tozero")
            chart(fig, "explain_response_plot", 300)
        st.caption("The selected value is assigned to each record and predictions are averaged. Some combinations "
                   "may be rare. These diagnostics describe model behavior and support no causal conclusions.")
