# Deliverables & Timeline — ADSC x AIAA Fall 2026

> **How to read this file.** This is the PMs' best current estimate of what the project needs to ship and roughly when. It is a **living plan, not a contract** — deliverables will be added, dropped, split, or resequenced as the team learns more. The authoritative, up-to-the-minute picture always lives in the repo's **GitHub Issues and Project board**; this file is the high-level narrative that keeps everyone oriented.

## Project Summary

This is the ADSC × AIAA collaborative data science project for Fall 2026. The project uses real astronomical data (Solar System Planets Motions) to teach members practical machine learning and time-series modelling skills, with deliverables structured to build progressively toward a full prediction pipeline.

---

## Milestones at a glance

| # | Deliverable | Description | Owner (role) | Target |
|---|-------------|-------------|--------------|--------|
| 1 | **ML & Time-Series Fundamentals** | Train XGBoost, ARIMA, and an ensemble model on solar-system orbital data to predict Julian Date values. See [`adsc-deliverable1/`](adsc-deliverable1/). | Members | Week 2 |
| 2 | **EDA & Feature Engineering** | Full exploratory analysis of the planets-motions dataset; create lag features, rolling statistics, and encoded categoricals for downstream models. | Members | Week 3 |
| 3 | **Model Tuning & Hyperparameter Search** | Apply Optuna or grid search to tune XGBoost and ARIMA orders; document the best configuration and evaluation results. | Members + PM | Weeks 4–5 |
| 4 | **Advanced Forecasting** | Extend to SARIMA or Facebook Prophet to capture seasonal/periodic orbital patterns; compare against the Week 2 baseline. | Members | Weeks 5–6 |
| 5 | **Interpretability & Insights** | Use SHAP values to explain XGBoost predictions; write a short findings brief connecting model outputs to orbital mechanics intuition. | Members + PM | Week 6 |
| 6 | **Final Pipeline & Report** | End-to-end reproducible pipeline (data download → features → model → evaluation); slide deck or written report for the AIAA audience. | Members + PM | Weeks 7–8 |
| 7 | **Presentation & Handoff** | Live demo or recorded walkthrough; documentation, reproducibility check, lessons learned. | PM | Week 8 |

---

## Deliverable 1 Detail

**Location:** [`adsc-deliverable1/deliverable1_planetary_motion.ipynb`](adsc-deliverable1/deliverable1_planetary_motion.ipynb)

**Goal:** Introduce members to the full supervised-learning and time-series workflow on real data.

**Dataset:** [Solar System Planets Motions — Kaggle](https://www.kaggle.com/datasets/cpitrat/solar-system-planets-motions/data)  
**Target variable:** `julian` (Julian Date — the astronomical timestamp)

**What members implement (TODOs):**
- EDA: grouped box plots, correlation analysis, and written observations.
- Feature engineering: rolling-window statistics for orbital position and velocity columns.
- XGBoost: at least three hyperparameter configurations; residual analysis plot.
- ARIMA: select p, d, q from ACF/PACF plots; generate and plot forecasts with confidence intervals.
- Ensemble: weighted blend of XGBoost and ARIMA; alpha sweep and optional scipy optimisation.
- Evaluation: predicted-vs-actual scatter, time-series overlay plot, grouped RMSE/MAE/R² bar chart, written conclusion.

**Completion criteria:** All `raise NotImplementedError` lines removed; all plots render; evaluation table populated with all three models.

---

## Timeline (rough)

```
Week:   1     2     3     4     5     6     7     8
        |-----|-----|-----|-----|-----|-----|-----|
D1 ML   ████
EDA           ██
Tuning              ████
Forecast                  ████
SHAP                            ██
Pipeline                              ████
Present                                     ████
```

---

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the summary.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file. Don't let this file quietly go stale.
- **"Done" is defined per deliverable** via the completion criteria listed above — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
