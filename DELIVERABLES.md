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

**Goal:** Introduce members to the full supervised-learning and time-series workflow on real data, and to the judgment calls that decide whether a model is meaningful at all.

**Dataset:** [Solar System — Planets' Motions, by Colin Pitrat (Kaggle)](https://www.kaggle.com/datasets/cpitrat/solar-system-planets-motions/data)

- ~2000 years of orbital solutions for the Sun, the eight planets, Pluto, the Moon, and the Earth–Moon Barycenter.
- Coordinates `x, y, z` in **km**; velocities in **km/day**. ICRF frame, TDB time scale.
- Derived from the [IMCCE INPOP](https://www.imcce.fr/inpop) ephemeris.
- **~1.38 GB on disk** — the notebook teaches chunked loading, `usecols`, and `float32` downcasting rather than a naive `read_csv`.

**Framing:** given the configuration of the solar system, predict the **Julian Date** (`julian`) — an inverse problem, and a real technique in astronomy.

**The three models:**

| Model | Target | Role |
|---|---|---|
| XGBoost | `julian` | Non-linear map from orbital state to date (random split — interpolation) |
| ARIMA | one body's coordinate over time | Temporal structure in an oscillating signal |
| Hybrid ensemble | `julian` | ARIMA forecasts XGBoost's residuals as a correction (chronological split) |

**Two teaching points the notebook is built around:**

1. **Identifiability.** A single body's state does not determine the date — it repeats every orbital period. Members must reshape the data to wide format so each row holds several bodies at once. This reshape is what makes the problem solvable, not housekeeping.
2. **Interpolation vs. extrapolation.** Tree ensembles cannot predict outside their training target range. The random split works; the chronological split visibly fails. The notebook uses that failure deliberately — the residual drift it creates is what the ARIMA correction then exploits, which is also why the ensemble is a residual hybrid rather than a weighted average (averaging a date against a distance in km would be meaningless).

**What members implement — 12 numbered TODOs:**

| # | Task |
|---|---|
| 1 | Map the real column names and layout from the data preview |
| 2 | Complete the chunked, filtered loader |
| 3 | Reshape to wide format so `julian` becomes identifiable |
| 4 | Estimate each body's orbital period and explain the aliasing |
| 5 | Derive radius and speed features for every body |
| 6 | Tune XGBoost (≥4 configs), select the best, interpret importances |
| 7 | Predicted-vs-actual and residual diagnostic plots |
| 8 | Choose ARIMA `(p, d, q)` from ADF + ACF/PACF and fit |
| 9 | Forecast with confidence intervals; beat a naive baseline |
| 10 | Build the hybrid residual ensemble and judge whether it helped |
| 11 | Final model comparison chart (log scale) |
| 12 | Written conclusion citing own numbers |

**Completion criteria:**
- [ ] Every `raise NotImplementedError` removed (10 guards).
- [ ] *Restart kernel and run all* completes top to bottom without error.
- [ ] Every `YOUR ANSWER` comment filled in.
- [ ] Section 9 conclusion written in the member's own words, citing their numbers.
- [ ] `data/` not committed — it is git-ignored at any depth.

**Verified resources are embedded in the notebook**, section by section: StatQuest for trees and gradient boosting, ritvikmath's *Time Series Talk* for stationarity/ACF/PACF/ARIMA/SARIMA, and Hyndman & Athanasopoulos' free [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/).

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
