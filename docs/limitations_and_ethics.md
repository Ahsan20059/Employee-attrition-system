# Limitations and responsible use

## Scope of the estimates

The case-study CSV has no observation dates, label window, verified employer provenance, or definition of the data collection process. The app displays comparisons by cohort. It cannot estimate time-to-departure or validate performance on a future workforce. Source fields such as monthly rate lack confirmed units or collection dates. External validity and potential post-outcome contamination therefore need prospective assessment. The same snapshot includes both training records and historical departures, so overview risk counts cannot be interpreted as the number of current employees likely to leave.

The model learns associations. Historical cohort differences, sensitivity charts and scenario changes do not establish intervention effects. Estimated probabilities and category boundaries can drift with the population, label definition, or HR practices. Re-evaluate calibration, utility, and group errors on a representative prospective cohort before considering workplace use.

## Group error audit

Age, Gender and MaritalStatus are excluded from the model and application exports. The local audit uses source-provided Gender and MaritalStatus, subject to authorized use. It does not infer missing values. It reports false-positive and false-negative rates on held-out groups with at least 30 records and at least 10 of each class. For the supplied data, both source Gender groups and two of the three MaritalStatus groups met those criteria. The third marital status group had too few positive labels for a reliable comparison. The available groups had different observed error rates, with small positive denominators. These estimates cannot establish parity, disparate treatment, or the statistical significance of differences. The audit is saved locally in `artifacts/metrics/private/fairness.json`.

## Use and privacy

The system supports an exploratory retention review with independent evidence, appropriately authorized users, and substantive human oversight. A false positive may draw unwarranted scrutiny or divert attention. A false negative may leave a cohort or record out of helpful follow-up. Avoid using risk estimates for termination, promotion, compensation, discipline, or adverse profiling. Consider team-level support, workload review and voluntary feedback.

Aggregate mode is the default, controlled at the server. It releases predefined single-dimension cohort summaries with at least 20 rows and five labels of each historical outcome. A second group can be withheld when a single omission could be inferred from totals. The aggregate release omits subgroup risk-band cells, individual row exports, or model file. The included aggregate data should still be reviewed under applicable data-sharing rules before public release. Private mode has no built-in user authentication. Run it on localhost for the capstone, or use separate authenticated infrastructure and organizational authorization. Derived record keys support reproducibility and are not an anonymization guarantee.
