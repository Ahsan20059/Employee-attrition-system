"""Streamlit entry point. Aggregate mode is the default."""
import streamlit as st

from pages import departments, employee, explain, explorer, overview, performance, scenarios
from src.dashboard_utils import app_css, private_mode

st.set_page_config(page_title="Attrition Insight", page_icon=":material/monitoring:",
                   layout="wide", initial_sidebar_state="auto")
app_css()
pages = [
    st.Page(overview.render, title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page(employee.render, title="Employee Risk", icon=":material/person_search:", url_path="employee-risk"),
    st.Page(departments.render, title="Department Analytics", icon=":material/hub:", url_path="departments"),
    st.Page(performance.render, title="Model Performance", icon=":material/monitoring:", url_path="model"),
    st.Page(explain.render, title="Explainability", icon=":material/insights:", url_path="explainability"),
    st.Page(scenarios.render, title="Scenario Analysis", icon=":material/tune:", url_path="scenarios"),
    st.Page(explorer.render, title="Data Explorer", icon=":material/table_view:", url_path="data"),
]
current = st.navigation(pages, position="hidden")

with st.sidebar:
    st.html('<div class="brand"><span class="brand-mark" aria-hidden="true">A</span>'
            '<div><strong>Attrition Insight</strong><small>Workforce analytics</small></div></div>')
    st.html('<div class="nav-label">WORKSPACE</div>')
    for page in pages:
        with st.container(key="nav_active" if page == current else f"nav_{page.url_path}"):
            st.page_link(page)
    st.divider()
    if private_mode():
        st.slider("Risk category boundaries", min_value=0.0, max_value=1.0, value=(.30, .60),
                  step=.01, format="%.2f", key="risk_bounds",
                  help="Low is below the first boundary, medium includes both boundaries, and high is above the second.")
        if st.session_state.risk_bounds[0] >= st.session_state.risk_bounds[1]:
            st.warning("Choose two distinct boundaries in ascending order.")
            st.stop()
        st.caption("The display bands can change independently of the model’s classification threshold.")
    else:
        st.html('<div class="scope">Aggregate demo · individual records hidden</div>')
current.run()
