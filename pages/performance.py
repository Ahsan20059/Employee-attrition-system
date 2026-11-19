"""Held-out benchmark and probability diagnostics."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import PALETTE, VIOLET, card, chart, get_context, heading, kpi

METRICS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "roc_auc": "ROC-AUC", "average_precision": "Average precision", "brier": "Brier",
}


def render():
    _, evaluation, _, _ = get_context()
    selected = evaluation["selected"]
    all_models = evaluation["models"]
    split = evaluation["split"]
    heading("Model performance", "Evaluate classification, ranking and probability quality on the held-out test set.")
    st.caption(f"Selected model · {selected} · Test set · {split['test']:,} records, "
               f"{split['test_positive']} positive labels · Version {evaluation['version']}")
    if evaluation.get("fallback"):
        st.caption(evaluation["fallback"])
    model = st.selectbox("Model for detailed diagnostics", list(all_models), index=list(all_models).index(selected))
    row = all_models[model]
    for index, (col, metric) in enumerate(zip(st.columns(4), [
        ("ROC-AUC", f"{row['roc_auc']:.3f}", "Test ranking discrimination"),
        ("Average precision", f"{row['average_precision']:.3f}", "Test precision–recall summary"),
        ("Recall", f"{row['recall']:.1%}", "Departures flagged by the classifier"),
        ("Brier score", f"{row['brier']:.3f}", "Lower probability error is better"),
    ])):
        with col:
            kpi(f"performance_{index}", *metric)
    with card("model_compare"):
        st.subheader("Held-out model comparison")
        table = pd.DataFrame([
            {"Model": name + (" · Selected" if name == selected else ""),
             **{label: values[metric] for metric, label in METRICS.items()},
             "CV AP": values["cv_average_precision"]}
            for name, values in all_models.items()
        ])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in table if c != "Model"})
        st.caption("Model selection used four-fold training average precision (CV AP). The other columns use held-out test data.")
    a, b = st.columns(2)
    for col, key, title, x_label, y_label in [
        (a, "roc", "ROC curve", "False positive rate", "True positive rate"),
        (b, "pr", "Precision–recall curve", "Recall", "Precision"),
    ]:
        with col, card(f"model_{key}"):
            st.subheader(title)
            fig = go.Figure()
            for i, (name, other) in enumerate(all_models.items()):
                data = other[key]
                fig.add_scatter(x=data["x"], y=data["y"], mode="lines", name=name,
                                line={"color": PALETTE[i], "width": 2.8 if name == model else 1.6})
            baseline = split["test_positive"] / split["test"]
            line = [0, 1] if key == "roc" else [baseline, baseline]
            fig.add_scatter(x=[0, 1], y=line, mode="lines", name="Baseline",
                            line={"color": "#A3A7B8", "dash": "dot"})
            fig.update_xaxes(title=x_label, tickformat=".0%")
            fig.update_yaxes(title=y_label, tickformat=".0%", range=[0, 1.02])
            chart(fig, f"model_{key}_plot", 285)
    a, b = st.columns(2)
    with a, card("model_confusion"):
        st.subheader("Confusion matrix")
        truth = ["Stayed", "Left"]
        fig = go.Figure(go.Heatmap(z=row["confusion"], x=truth, y=truth, texttemplate="%{z}", showscale=False,
                                   colorscale=[[0, "#f2ebff"], [1, "#9b87d5"]]))
        fig.update_xaxes(title="Predicted label")
        fig.update_yaxes(title="Historical label", autorange="reversed")
        chart(fig, "model_confusion_plot", 240)
        _, fp = row["confusion"][0]
        fn, _ = row["confusion"][1]
        st.caption(f"At the validation-selected {row['threshold']:.1%} operating threshold, "
                   f"{fp} stayed records were flagged and {fn} departure records were missed. "
                   "A lower threshold typically flags more records and can reduce precision.")
    with b, card("model_calibration"):
        st.subheader("Probability calibration")
        data = row["calibration"]
        fig = go.Figure(go.Scatter(x=data["x"], y=data["y"], customdata=data["n"], mode="lines+markers",
                                  name=model, line_color=VIOLET,
                                  hovertemplate="Mean estimate %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata} records<extra></extra>"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal",
                        line={"color": "#A3A7B8", "dash": "dot"})
        fig.update_xaxes(title="Mean probability", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Observed attrition", tickformat=".0%", range=[0, 1])
        chart(fig, "model_calibration_plot", 240)
        st.caption(f"Probability method · {row['calibration_method']}. Constant-prevalence baseline Brier · "
                   f"{evaluation['prevalence_brier']:.3f}. Bins with fewer than five records are omitted from the plot.")
    with st.expander("Classification report and uncertainty"):
        report = pd.DataFrame(row["classification_report"]).T.loc[["0", "1", "macro avg", "weighted avg"]]
        report = report.rename(index={"0": "Stayed", "1": "Left"})
        st.dataframe(report.round(3), use_container_width=True)
        if model == selected:
            limits = evaluation["bootstrap_intervals"]
            st.caption("Selected model’s 95% stratified test bootstrap intervals, conditional on this fitted model.")
            st.dataframe(pd.DataFrame([{"Metric": METRICS[k], "Lower": f"{v[0]:.3f}", "Upper": f"{v[1]:.3f}"}
                                       for k, v in limits.items()]), hide_index=True, use_container_width=True)
