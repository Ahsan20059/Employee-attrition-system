"""Build static EDA figures and Markdown reports from saved experiment results."""
import json
from zoneinfo import ZoneInfo
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.analytics import COHORTS, cohorts, released_groups
from src.config import ARTIFACTS, DEMO, METRICS, NUMERICS, ROOT
from src.schema import DESCRIPTIONS, label

VIOLET = "#7053CB"
TEAL = "#42968F"
RED = "#BA5171"
MUTED = "#515970"
FIGURES = ARTIFACTS / "figures"
DOCS = ROOT / "docs"


def read_experiment(frame_quality):
    evaluation = json.loads(METRICS.read_text())
    summary = json.loads(DEMO.read_text())
    if (evaluation["dataset_sha256"] != frame_quality["sha256"]
            or evaluation["run_id"] != summary["run_id"]):
        raise ValueError("Reports require evaluation artifacts generated from the selected input file.")
    return evaluation, summary


def save(fig, filename):
    fig.savefig(FIGURES / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_axis(ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#dde1ed")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color="#edf0f8", linewidth=.7)
    ax.set_axisbelow(True)


def static_figures(frame, evaluation):
    FIGURES.mkdir(parents=True, exist_ok=True)
    grouped = [c for c in COHORTS if c in frame]
    fig, axes = plt.subplots((len(grouped) + 1) // 2, 2, figsize=(12, 3.1 * ((len(grouped) + 1) // 2)),
                             layout="constrained", squeeze=False)
    for ax, column in zip(axes.flat, grouped):
        table = cohorts(frame, column)
        ax.bar(np.arange(len(table)), table["Historical rate"], color=VIOLET)
        ax.set_xticks(np.arange(len(table)), [x.replace("_", " ") for x in table.Group], rotation=22, ha="right")
        for i, row in table.iterrows():
            ax.annotate(f"n={row.Records:,}", (i, row["Historical rate"]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.margins(y=.28)
        ax.set_title(COHORTS[column], loc="left", fontsize=10, color=MUTED)
        format_axis(ax)
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    fig.suptitle("Historical attrition across workforce cohorts", x=.02, ha="left", fontsize=16)
    save(fig, "attrition_cohorts.png")

    numeric = [c for c in NUMERICS if c in frame]
    if numeric:
        fig, axes = plt.subplots((len(numeric) + 3) // 4, 4, figsize=(13, 2.0 * ((len(numeric) + 3) // 4)),
                                 layout="constrained", squeeze=False)
        for ax, column in zip(axes.flat, numeric):
            series = frame[column].dropna()
            bins = (np.arange(series.min() - .5, series.max() + 1.5)
                    if series.nunique() <= 12 and not series.empty else 16)
            ax.hist(series, bins=bins, color=VIOLET)
            ax.set_title(label(column), loc="left", fontsize=9, color=MUTED)
            format_axis(ax)
        for ax in list(axes.flat)[len(numeric):]:
            ax.set_visible(False)
        fig.suptitle("Numeric feature distributions · absolute record counts", x=.02, ha="left", fontsize=15)
        save(fig, "numeric_distributions.png")

        correlations = frame[[*numeric, "Attrition"]].corr()
        order = correlations.Attrition.abs().sort_values(ascending=False).head(14).index
        matrix = correlations.loc[order, order]
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
        surface = ax.imshow(matrix, cmap="PuOr_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(order)), [label(c) for c in order], rotation=50, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)), [label(c) for c in order], fontsize=8)
        ax.set_title("Numeric relationships · Pearson correlation", loc="left", pad=14)
        fig.colorbar(surface, ax=ax, shrink=.8)
        save(fig, "numeric_correlations.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    split = evaluation["split"]
    prevalence = split["test_positive"] / split["test"]
    for ax, key, title, xl, yl, baseline in zip(
        axes, ["roc", "pr", "calibration"], ["ROC", "Precision–recall", "Calibration"],
        ["False positive rate", "Recall", "Mean probability"],
        ["True positive rate", "Precision", "Observed rate"],
        [[0, 1], [prevalence, prevalence], [0, 1]],
    ):
        ax.plot([0, 1], baseline, color="#9aa0b2", linestyle=":", linewidth=1.5)
        for (name, model), color in zip(evaluation["models"].items(), [VIOLET, TEAL, RED]):
            data = model[key]
            ax.plot(data["x"], data["y"], color=color, label=name, linewidth=1.6,
                    marker="o" if key == "calibration" else None, markersize=4)
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=xl, ylabel=yl, title=title)
        format_axis(ax)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    save(fig, "held_out_diagnostics.png")


def metrics_table(evaluation):
    keys = ["cv_average_precision", "accuracy", "precision", "recall", "f1",
            "roc_auc", "average_precision", "brier", "threshold"]
    lines = ["| Model | CV AP | Accuracy | Precision | Recall | F1 | ROC-AUC | Test AP | Brier | Cutoff |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in evaluation["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{values[key]:.3f}" for key in keys) + " |")
    return "\n".join(lines)


def findings(frame):
    observations = []
    for field in ("Department", "JobRole", "OverTime", "BusinessTravel", "JobSatisfaction",
                  "EnvironmentSatisfaction", "WorkLifeBalance", "YearsAtCompany",
                  "YearsSinceLastPromotion", "MonthlyIncome"):
        if field not in frame:
            continue
        groups = cohorts(frame, field)
        groups = released_groups(groups).sort_values("Historical rate", ascending=False)
        if len(groups) < 2:
            continue
        high, low = groups.iloc[0], groups.iloc[-1]
        observations.append(
            f"- **{COHORTS[field]}** · {high.Group.replace('_', ' ')} had {high['Historical rate']:.1%} "
            f"({high.Left:,}/{high.Records:,}) versus {low.Group.replace('_', ' ')} at "
            f"{low['Historical rate']:.1%} ({low.Left:,}/{low.Records:,})."
        )
    return "\n".join(observations)


def generate_reports(frame, quality):
    evaluation, summary = read_experiment(quality)
    static_figures(frame, evaluation)
    selected = evaluation["selected"]
    model = evaluation["models"][selected]
    rows, left = summary["records"], summary["left"]
    train, validation, test, positive = (evaluation["split"][key]
                                        for key in ("train", "validation", "test", "test_positive"))
    (_, fp), (fn, tp) = model["confusion"]
    dataset_findings = findings(frame)
    risks = ", ".join(f"{value:,} {name.split()[0].lower()}" for name, value in summary["bands"].items())
    imports = ", ".join(label(row["Feature"]) for row in evaluation["importance"][:3])
    intervals = evaluation["bootstrap_intervals"]
    date = datetime.fromisoformat(evaluation["evaluated_utc"]).astimezone(ZoneInfo("Asia/Kolkata")).date()
    latest = (
        f"{selected} ranked first by four-fold training CV average precision "
        f"({model['cv_average_precision']:.3f}). The F1-maximizing validation threshold was {model['threshold']:.1%}. "
        f"On the {test}-record test set it flagged {tp} of {positive} departures and missed {fn}. "
        f"It also flagged {fp} records with a stayed label. Test ROC-AUC was "
        f"{model['roc_auc']:.3f} and average precision was {model['average_precision']:.3f}."
    )
    overtime = next((item for item in dataset_findings.splitlines() if item.startswith("- **Overtime**")), "")
    DOCS.mkdir(exist_ok=True)
    (DOCS / "research_report.md").write_text(f"""# Workforce attrition modeling

Internship research report · {date:%d %B %Y} · Model version {evaluation['version']}

## 1. Abstract

This study evaluates methods for ranking historical attrition labels and estimating record-level probabilities. The supplied snapshot contains {rows:,} records and {left:,} positive labels ({left / rows:.1%}). {latest}

## 2. Introduction

Workforce analytics can help an HR team identify cohorts worth investigating. The assignment uses Palo Alto Networks as case-study context. The source file does not independently establish that the records describe staff of that company or how the data was collected.

## 3. Problem statement

Estimate the association between recorded workforce inputs and a binary historical attrition outcome. Review whether risk ranking and probability estimates outperform simple baselines on records reserved for testing.

## 4. Objectives

Audit source quality, compare three classifier families, evaluate threshold and calibration behavior, and create reproducible tools for cohort analysis, model inspection and controlled scenarios.

## 5. Dataset overview

The input was `{quality['file']}`. Worksheet selection was {"not applicable to this CSV" if quality['sheet'] is None else quality['sheet']}. The validated source had {quality['columns']} columns, {sum(quality['missing_values'].values())} missing cells in recognized fields and {quality['duplicates']} duplicate normalized records. Observed target counts were {quality['class_counts']['stayed']:,} stayed and {left:,} left. The source identifier field was {quality['identifier_field'] or 'absent'}, and the app uses derived record keys. Four available tenure consistency checks found {sum(quality['tenure_inconsistencies'].values())} violations.

## 6. Data preprocessing

The loader normalized binary targets, enforced rating ranges and normalized overtime and travel labels. Training folds fitted numeric medians, categorical modes and an unknown-tolerant one-hot encoder. Logistic Regression additionally used standard scaling. Gender, marital status, age, record identifiers and the target were excluded from inputs. The source was preserved byte for byte.

## 7. Exploratory data analysis

Comparisons show observed outcomes and denominators for released cohorts with at least 20 records and five labels of each outcome. Additional suppression can prevent inferring a small cell from source totals. Rates are descriptive and can reflect interdependent features and cohort composition.

{dataset_findings}

[Historical cohort figures](../artifacts/figures/attrition_cohorts.png) · [Numeric distributions](../artifacts/figures/numeric_distributions.png) · [Selected numeric correlations](../artifacts/figures/numeric_correlations.png)

## 8. Feature engineering

The pipeline adds income per experience year, a promotion delay flag, a satisfaction and involvement composite, an overtime and work–life balance proxy, prior career years and tenure before the current role. Formulas and missing-value behavior are recorded in [technical documentation](technical_documentation.md). The composite and proxy are descriptive constructions.

## 9. Model development

Training compared {', '.join(evaluation['models'])}. Each search tried six settings. Logistic Regression and boosting compared weighted and unweighted variants. Random Forest used per-bootstrap class weighting. Each family also compared cross-fitted sigmoid calibration with its original probabilities on validation Brier score.

## 10. Experimental setup

Seed {evaluation['seed']} allocated {train} training, {validation} validation and {test} test records in two stratified splits. Four-fold stratified CV on the training partition tuned candidates and selected the model family by average precision. The validation partition selected calibration and maximized positive-class F1 for the operating cutoff. Test records were used only for final evaluation and diagnostic interpretation.

## 11. Results and comparison

{metrics_table(evaluation)}

Selected model probability method · {model['calibration_method']}. The training-prevalence-only test Brier was {evaluation['prevalence_brier']:.3f}. The selected model's 95% test bootstrap intervals were {intervals['roc_auc'][0]:.3f}–{intervals['roc_auc'][1]:.3f} for ROC-AUC, {intervals['average_precision'][0]:.3f}–{intervals['average_precision'][1]:.3f} for average precision and {intervals['brier'][0]:.3f}–{intervals['brier'][1]:.3f} for Brier. These intervals hold the model and observed positive count fixed. Each model's confusion matrix, calibrated or original probability curves and classification report are saved in [evaluation.json](../artifacts/metrics/evaluation.json).

[Held-out ROC, precision–recall and calibration plots](../artifacts/figures/held_out_diagnostics.png)

## 12. Explainability

Held-out shuffling had the largest average precision effects for {imports}. Local diagnostics compare an individual estimate with one-input replacements using training medians or modes. These differences can overlap and do not add to the probability. The scenario page recomputes dependent features with the same fitted pipeline.

## 13. Key findings

{latest} Default display bands across the supplied snapshot contained {risks} risk records. This distribution includes training rows and historical departures. It describes the case-study snapshot.

## 14. Practical recommendations

Review workload distribution and overtime practices with teams showing elevated rates. Investigate role-specific satisfaction and tenure patterns with HR and employee feedback. Review promotion-process transparency and test any policy changes prospectively, using predefined outcomes and appropriate comparison groups.

## 15. Limitations

The cross-sectional input has no observation dates, verified provenance, future prediction window or external validation cohort. Feature collection timing, units for some rate fields and label definitions beyond the binary file are unspecified. The dataset may shift across organizations or over time. No prospective impact or causal factor is established.

## 16. Ethical considerations

The model excludes specified sensitive inputs and age, although correlated features may preserve group differences. A sample-size-gated audit compares held-out error rates for available source groups. Limited class counts leave substantial uncertainty. Refer to [limitations and ethics](limitations_and_ethics.md) for safeguards.

## 17. Conclusion

The selected model improved on the prevalence baseline for ranking and Brier score on this split. Its recall and precision show meaningful scope for errors. Future use would require documented data collection timing and independently validated calibration and operating thresholds.

## 18. References

- Supplied internship case-study CSV, preserved in the private project workspace.
- Scikit-learn. [Probability calibration](https://scikit-learn.org/stable/modules/calibration).
- Scikit-learn. [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance).
""", encoding="utf-8")
    (DOCS / "executive_summary.md").write_text(f"""# Employee attrition · executive summary

For HR and public-sector stakeholders reviewing the internship case study · {date:%d %B %Y}

The supplied case-study file contains {rows:,} workforce records, of which {left:,} ({left / rows:.1%}) carry a historical departure label. The organization, collection period and future prediction window are unverified.

The analysis compared three model families using separate training, validation and test partitions. {latest} At that threshold, test precision was {model['precision']:.1%}, recall was {model['recall']:.1%}, and Brier score was {model['brier']:.3f} compared with {evaluation['prevalence_brier']:.3f} for a prevalence-only estimate.

### Workforce signals

{overtime}
- {imports} had the largest model permutation effects on the test set.
- Applying the default display thresholds to all {rows:,} records produced {risks} risk records. Mean predicted risk was {summary['mean_risk']:.1%}. This includes historical departures and records used to train the model.

### Implications for review

Start with a cohort-level review of overtime, reported satisfaction, turnover by role and promotion practices. Use employee feedback and independently measured outcomes to evaluate interventions. Treat individual estimates as restricted analytical information. Decisions involving termination, promotion, compensation or discipline require independent processes and evidence. The source cannot establish how much a policy change would affect retention.
""", encoding="utf-8")
    fields = ["| Field | Inferred type | Role or definition |", "|---|---|---|"]
    fields.extend(f"| `{field}` | {dtype} | {DESCRIPTIONS[field]} |"
                  for field, dtype in quality["types"].items())
    fields.extend(f"| `{field}` | Application-derived | {DESCRIPTIONS[field]} |"
                  for field in ("Record", "Probability", "RiskScore", "RiskBand"))
    (DOCS / "dataset_schema.md").write_text("# Dataset schema\n\n" + "\n".join(fields) + "\n", encoding="utf-8")
    return evaluation
