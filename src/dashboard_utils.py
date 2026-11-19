"""Presentation helpers and the public/private dashboard data boundary."""
import json
import os
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import cohorts
from src.config import DEMO, METRICS, MODEL, PROCESSED, RISK_LABELS, ROOT
from src.prediction import load_model
from src.risk_scoring import DEFAULT_BOUNDS, score_records

VIOLET = "#7658D6"
TEAL = "#459D94"
AMBER = "#D59435"
RED = "#C65373"
RISK_COLORS = [TEAL, AMBER, RED]
PALETTE = [VIOLET, "#55A9A4", "#BAA9EA", "#D99A47", "#7A89AA"]
ARCHIVE = PROCESSED / "scored_records.csv"


def private_mode():
    return not (ROOT / ".aggregate-only").exists() and os.getenv("ATTRITION_MODE", "").strip().casefold() == "private"


@st.cache_data(show_spinner=False)
def read_json(path, modified):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path):
    return read_json(path, path.stat().st_mtime_ns)


@st.cache_resource(show_spinner=False)
def private_bundle(modified):
    return load_model(MODEL)


@st.cache_data(show_spinner=False)
def private_records(modified):
    return pd.read_csv(ARCHIVE, float_precision="round_trip")


def get_context():
    try:
        summary, evaluation = load_json(DEMO), load_json(METRICS)
        if summary.get("run_id") != evaluation.get("run_id"):
            raise ValueError("The dashboard artifacts come from different training runs.")
        if not private_mode():
            return summary, evaluation, None, None
        model = private_bundle(MODEL.stat().st_mtime_ns)
        if model.get("run_id") != evaluation.get("run_id"):
            raise ValueError("The private model and dashboard results come from different training runs.")
        raw = private_records(ARCHIVE.stat().st_mtime_ns)
        if "_run" in raw and not raw["_run"].eq(evaluation.get("run_id")).all():
            raise ValueError("The scored records and model come from different training runs.")
        raw = raw.drop(columns="_run", errors="ignore")
    except FileNotFoundError:
        st.error("Required artifacts are missing. Run `python scripts/run_training.py` from the project folder.")
        st.stop()
    except (ValueError, OSError, ImportError) as error:
        st.error("The dashboard could not load its artifacts. Check data and dependency versions, then run training again.")
        if private_mode():
            st.caption(str(error))
        st.stop()
    return summary, evaluation, model, score_records(raw, raw.Probability.to_numpy(), bounds())


def bounds():
    return tuple(st.session_state.get("risk_bounds", DEFAULT_BOUNDS))


def heading(title, description):
    mode = "Private workspace" if private_mode() else "Aggregate demo"
    st.html(f'<div class="page-head"><div><h1>{escape(title)}</h1>'
            f'<p>{escape(description)}</p></div><span class="mode-label">{mode}</span></div>')


def card(key):
    return st.container(key=f"card_{key}")


def kpi(key, label, value, detail, mark="◇", tone=""):
    with card(key):
        st.html(f'<div class="kpi"><div><div class="kpi-title">{escape(label)}</div>'
                f'<div class="kpi-value">{escape(str(value))}</div>'
                f'<div class="kpi-detail">{escape(detail)}</div></div>'
                f'<div aria-hidden="true" class="kpi-mark {tone}">{escape(mark)}</div></div>')


def group_data(summary, frame, column):
    if frame is None:
        return pd.DataFrame(summary["cohorts"].get(column, []))
    return cohorts(frame, column) if column in frame else pd.DataFrame()


def chart(fig, key, height=260, **extra):
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=11, b=12), paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF", font=dict(family="Arial, sans-serif", size=12, color="#505873"),
        hoverlabel=dict(bgcolor="white", font_color="#22253a"),
        legend=dict(orientation="h", x=0, y=1.1), **extra,
    )
    fig.update_xaxes(gridcolor="#ECEEF5", automargin=True)
    fig.update_yaxes(gridcolor="#ECEEF5", automargin=True)
    st.plotly_chart(fig, key=key, use_container_width=True, theme=None,
                    config={"displayModeBar": False, "scrollZoom": False})


def donut(labels, values, colors, center, caption):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.73, sort=False, marker_colors=colors,
        textinfo="none", hovertemplate="%{label}<br>%{value:,} records · %{percent}<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(x=.5, y=.545, text=f"<b>{center}</b>", showarrow=False,
                       font=dict(size=26, color="#23263b"), xanchor="center")
    fig.add_annotation(x=.5, y=.42, text=caption, showarrow=False,
                       font=dict(size=12, color="#626a82"))
    return fig


def legend(labels, values, colors):
    lines = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;color:#505873;font-size:.78rem;'
        f'padding:4px 2px"><span style="width:8px;height:8px;border-radius:3px;background:{c}" '
        f'aria-hidden="true"></span><span style="flex:1">{escape(str(label))}</span>{value:,}</div>'
        for label, value, c in zip(labels, values, colors)
    )
    st.html(lines)


def comparison(groups, key, columns=("Historical rate", "Mean risk")):
    if groups.empty:
        st.caption("No available groups meet the current data or release criteria.")
        return
    fig = go.Figure()
    for i, column in enumerate(columns):
        fig.add_bar(x=groups.Group, y=groups[column], name=column,
                    marker_color=[RED, VIOLET][i] if len(columns) == 2 else VIOLET,
                    customdata=groups.Records,
                    hovertemplate="%{x}<br>%{y:.1%}<br>%{customdata} records<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="group", bargap=.3, barcornerradius=5)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    chart(fig, key)


def comparison_table(frame, key):
    configs = {column: st.column_config.NumberColumn(column, format="percent")
               for column in ("Historical rate", "Mean risk", "Probability") if column in frame}
    if "RiskScore" in frame:
        configs["RiskScore"] = st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f")
    if "RiskBand" in frame:
        configs["RiskBand"] = st.column_config.TextColumn("Risk category")
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config=configs, key=key)


def record_picker(frame, key):
    options = frame.Record.tolist()
    lookup = frame.set_index("Record").JobRole
    chosen = st.selectbox("Workforce record", options, key=key,
                          format_func=lambda identifier: f"{identifier} · {lookup[identifier]}")
    return frame.loc[frame.Record == chosen]


def require_records(frame):
    if frame is None:
        st.info("Individual profiles are available in an authorized private workspace. "
                "This demonstration uses cohort summaries and held-out evaluation.")
        return False
    return True


def category_totals(frame, summary):
    if frame is None:
        return [summary["bands"].get(band, 0) for band in RISK_LABELS]
    counts = frame.RiskBand.value_counts()
    return [int(counts.get(band, 0)) for band in RISK_LABELS]


def download_csv(frame, label, filename):
    # Escape leading spreadsheet formula characters in user-supplied text fields.
    sanitized = frame.copy()
    for col in sanitized.select_dtypes(include=["object", "string", "category"]):
        series = sanitized[col].astype("string")
        mask = series.str.match(r"^[\s]*[=+@-]|^[\t\r]", na=False)
        sanitized[col] = series.mask(mask, "'" + series)
    st.download_button(label, sanitized.to_csv(index=False).encode(), filename,
                       mime="text/csv", icon=":material/download:")


def app_css():
    st.html(ROOT / "src" / "dashboard.css")
