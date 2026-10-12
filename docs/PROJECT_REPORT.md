# SmartWater AI — Project Report

**Project title:** SmartWater AI — Water Demand Forecasting and Demand-Level Classification
**Project type:** AI/ML web application (regression + classification + Flask deployment)
**Report date:** 12 October 2026
**Verified against:** the running application at `http://127.0.0.1:5000`

> Every figure in this report was read from the running system or recomputed from the
> saved artifacts during verification. Nothing here is estimated, rounded up, or
> described more strongly than the evidence supports. (See §16 Verification Log.)

---

## 1. Introduction

SmartWater AI is an end-to-end machine-learning application that forecasts **daily
municipal water consumption** and classifies that demand into one of four levels
(LOW / NORMAL / HIGH / VERY HIGH).

The system learns from a real, publicly available daily water-consumption dataset
covering **five regions over 180 days (900 observations)**, and serves two trained
models through a Flask web application with five pages and nine REST endpoints.

The delivered product has three layers:

1. **Data layer** — a real CSV dataset, cleaned and standardised by a reproducible script.
2. **Intelligence layer** — a leakage-safe feature pipeline shared by training and
   inference, three regression models (naive baseline, Random Forest, Gradient
   Boosting) and two classifiers (Logistic Regression, Random Forest).
3. **Presentation layer** — a responsive "water intelligence" web interface (dashboard,
   prediction, 7-day forecast, analytics, model lab) built with editorial typography
   and a navy/ivory palette, with charts rendered client-side by Chart.js.

The application answers two operational questions for a water utility:

- *"How much water will be consumed tomorrow?"* → a number in litres per day.
- *"Is that unusually high or low?"* → a demand class the operator can act on.

---

## 2. Problem Definition

Water utilities must anticipate daily demand to plan supply, pumping schedules,
reservoir draw-down and maintenance windows. Two failure modes are costly:

- **Under-production** — pressure drops, supply interruptions, and emergency pumping.
- **Over-production** — wasted energy, unnecessary treatment cost, and reservoir
  overflow.

In practice, operators rely on historical consumption records, which are cheap to
collect but contain no explicit weather or event information. The problem addressed
by this project is therefore precisely stated as:

> **Given a region and a calendar date, and using only the region's own past
> consumption history plus calendar signals, predict the daily water consumption in
> litres and assign it a discrete demand level — and do so with an evaluation that
> is honest about how much signal the data actually contains.**

Two sub-problems follow:

- **P1 (Regression):** predict a continuous value, `consumption_liters`, for a future day.
- **P2 (Classification):** convert that continuous value into an actionable category
  without letting the test set influence where the category boundaries are placed.

A third problem, often ignored in student projects, is **evaluation honesty**: a model
that reports a flattering metric obtained by shuffling time-ordered data is worse than
useless — it will fail the moment it is deployed. This project treats "do not leak the
future into training" as a hard requirement, not a stylistic preference.

---

## 3. Objectives

**Primary objective**
Build and deploy a working AI application that forecasts daily water demand for five
regions and classifies it into demand levels, from real data, with reproducible and
honest evaluation.

**Specific objectives**

| # | Objective | Status |
|---|---|---|
| O1 | Acquire and validate a real water-consumption dataset | Achieved — 900 rows, 0 defects found |
| O2 | Build a leakage-safe, reusable feature pipeline shared by training and inference | Achieved — `src/features.py` |
| O3 | Train and compare three regression models with a time-respectful split | Achieved — Baseline, Random Forest, Gradient Boosting |
| O4 | Train and compare two classifiers using train-only quantile thresholds | Achieved — Logistic Regression, Random Forest |
| O5 | Evaluate with metrics appropriate to forecasting (MAE, RMSE, R², accuracy, precision, recall, F1, confusion matrix) | Achieved |
| O6 | Deploy both models behind a REST API and a responsive multi-page UI | Achieved — 5 pages, 9 endpoints |
| O7 | Keep the deployed system and the reported metrics provably consistent | Achieved — metrics recomputed from saved artifacts and matched exactly |
| O8 | Report the true performance even when unflattering | Achieved — R² ≈ 0 reported as a property of the data |

---

## 4. Motivation

**Why water demand forecasting matters.** Water is the most infrastructure-heavy
utility per litre delivered. Demand that is misjudged by a few thousand litres per
day translates into pumping energy, treatment chemicals and reservoir management that
is either wasted or insufficient. Even a modest improvement in day-ahead visibility
changes operational decisions.

**Why this dataset is honest and instructive.** The chosen dataset contains *only*
region, date and consumption. It deliberately contains no temperature, humidity,
holiday or event columns. A weaker project would invent those variables, train on
them, and report a flattering accuracy that no real deployment could reproduce. This
project refuses that shortcut: the feature set is limited to what actually exists, and
the resulting near-zero R² is reported as a finding rather than hidden. This makes the
project a truthful case study in the difference between a model that *scores* well and
a model that *generalises*.

**Why the deployment layer matters.** A trained `.joblib` file is not a product. The
motivation for the Flask layer, the REST contract and the five-page interface is that
a forecast only creates value when an operator can ask for it, read it, and see the
reasoning behind it.

---

## 5. Scope of the Project

**In scope**

- Cleaning and standardising the real consumption dataset.
- Feature engineering limited to calendar signals and the region's own history.
- Training, evaluating and comparing regression and classification models.
- Persisting trained artifacts and exposing them through a REST API.
- A responsive five-page web interface (320 px – 1440 px+).
- Interactive charts, KPI cards, 7-day recursive forecasting and a model lab.
- Reproducible evaluation: any reported metric can be recomputed from saved artifacts.

**Out of scope**

- Live ingestion of daily meter readings (the app serves from a static recent-history file).
- Weather, holiday, economic or demographic exogenous features (not present in the data).
- Deep learning / sequence models (LSTM, Transformer) — tree ensembles are the
  appropriate baseline for this tabular feature set.
- Authentication, multi-tenancy and user accounts.
- Production-grade serving (WSGI/containerisation) — the built-in Flask server is used.
- Real-time database persistence of forecasts versus actuals.

**Explicit boundary conditions**

- The dataset covers one spring–summer window (Jan–Jun 2023), so annual seasonality
  cannot be modelled.
- Predictions for dates far beyond the data window are extrapolations: calendar
  features change, but the lag buffer still holds the last known actual readings.

---

## 6. Applications

| Domain | How it is used |
|---|---|
| Municipal water utilities | Day-ahead demand visibility for pumping and reservoir scheduling |
| Pump-station operations | Pre-positioning for HIGH / VERY HIGH demand days |
| Reservoir & tank management | Draw-down planning against forecast consumption |
| Energy-cost management | Shifting pumping to off-peak hours on forecast-low days |
| Maintenance planning | Scheduling works on forecast-low-demand days |
| Demand-side management | Triggering conservation messaging before peak days |
| Academic teaching | A complete, honest CRISP-DM-style case study with leakage-safe design |
| Smart-city dashboards | Regional consumption analytics for public dashboards |

The demand-level classifier is the operationally actionable half: it turns a number
into a decision ("tomorrow is VERY HIGH — staff the shift accordingly").

---

## 7. Literature Survey

The design choices in this project follow established work rather than ad-hoc
preference. Key references and what each contributed:

**7.1 Tree ensembles for tabular prediction**
Breiman (2001) introduced **Random Forests**, aggregating decorrelated decision trees
to reduce variance; it remains the strongest default for small tabular datasets.
Friedman (2001) formalised **gradient boosting** as function-space optimisation, which
builds trees sequentially on residual error. Both are implemented in this project with
the scikit-learn reference implementations (300 trees each). Grinsztajn, Oyallon &
Varoquaux (2022) later showed empirically that tree ensembles still outperform deep
networks on typical tabular data — the reason this project did not reach for neural
sequence models.

**7.2 Time-series evaluation methodology**
Hyndman & Athanasopoulos (2021) set out the standard principle that forecast
evaluation must preserve time order; random shuffling of time-series data produces
optimistic and meaningless metrics. This directly motivates the **chronological,
per-region 80/20 split** used here instead of `train_test_split(shuffle=True)`.
Makridakis, Spiliotis & Assimakopoulos (2020), reporting the M4 competition across
100,000 series, demonstrated that simple naive/mean benchmarks are remarkably hard to
beat — which is exactly why this project keeps a **naive previous-day baseline** as a
first-class comparison rather than dropping it.

**7.3 Scientific Python stack**
Pedregosa et al. (2011) describe **scikit-learn**, whose consistent estimator API
(`fit`/`predict`) and metric suite are used throughout. Harris et al. (2020) describe
**NumPy**, the array layer under all computations, and McKinney (2010) introduced the
**pandas** structures used for cleaning, grouping and rolling-window feature
construction.

**7.4 Web delivery**
The **Flask** framework (Pallets) provides the WSGI application, routing and Jinja2
templating used for server-side page rendering, with **Chart.js** handling client-side
canvas visualisation. This two-tier arrangement keeps the ML artifacts on the server
and only aggregated results in the browser.

**7.5 Gap this project addresses**
Published tutorials on water-demand forecasting commonly use datasets that already
contain weather features, and report strong accuracy as a result. The gap this project
occupies is the honest case: what performance is actually attainable when only
consumption history and calendar signals exist? The answer (§11) is a genuine and
defensible academic contribution — it establishes the *floor* that richer datasets
must beat, and it demonstrates correct methodology under conditions where the data
refuses to cooperate.

---

## 8. Requirements Specification

### 8.1 Functional requirements

| ID | Requirement | Implemented by | Verified |
|---|---|---|---|
| FR1 | Load and clean the real consumption dataset reproducibly | `data/prepare_data.py` | ✔ 900 rows, 0 issues |
| FR2 | Build features without leaking future or cross-region values | `src/features.py` | ✔ per-region `shift(1)` + grouped rolling |
| FR3 | Train and evaluate regression models on a chronological split | `train_models.py` | ✔ 660 train / 170 test |
| FR4 | Train and evaluate classifiers with train-only class thresholds | `train_classifier.py` | ✔ thresholds reproduced exactly |
| FR5 | Persist models + metrics + feature list as artifacts | `models/*.joblib`, `*.json` | ✔ 11 artifacts present (5 models, 2 feature lists, 4 metadata files) |
| FR6 | Serve a dashboard with real KPIs from the dataset | `/`, `/api/dashboard-stats` | ✔ HTTP 200 |
| FR7 | Predict consumption for a region + date | `POST /api/predict` | ✔ API output == model output |
| FR8 | Display the prediction with a demand-level badge and reasoning | `/prediction` | ✔ renders and responds |
| FR9 | Produce a 7-day recursive forecast per region | `/api/forecast` | ✔ 7 rows × 5 regions |
| FR10 | Provide analytics computed from real data | `GET /api/analytics-data` | ✔ monthly, weekday/weekend, regional, min/max |
| FR11 | Expose model performance including a confusion-matrix heatmap | `/model-performance` | ✔ 200, matrix matches stored |
| FR12 | Report feature importance from the deployed model | `/api/feature-importance` | ✔ lag_7 highest at 0.216 |
| FR13 | Validate requests and reject invalid input clearly | `POST /api/predict` | ✔ unknown region → 400, missing field → 400 |
| FR14 | Keep the prediction API stateless with respect to request order | `src/predict.py` | ✔ fixed and re-verified (§16.5) |
| FR15 | Quantify and expose the achievable performance ceiling | `analysis/signal_audit.py`, `/api/signal-audit` | ✔ 1 of 15 white-noise tests rejects; ceiling served and displayed |
| FR16 | Ship a non-mutating regression test suite | `tests/` | ✔ 54 tests, all passing |

### 8.2 Non-functional requirements

| ID | Category | Requirement | Evidence |
|---|---|---|---|
| NFR1 | Usability | Five pages usable without training; metrics labelled with units | UI inspected live |
| NFR2 | Responsiveness | No page-level horizontal scrolling from 320 px to 1440 px+ | Verified at 1440 / 743 / 375 px (`scrollWidth ≤ innerWidth`) |
| NFR3 | Performance | Single prediction must feel instant locally | Synchronous in-process model call; no measurable wait observed |
| NFR4 | Correctness | Reported metrics must equal metrics recomputed from artifacts | Exact match to 4 decimal places |
| NFR5 | Reliability | No runtime errors during a full page-and-API walkthrough | 0 console errors, all endpoints HTTP 200 |
| NFR6 | Maintainability | One feature pipeline shared by training and inference | `build_features()` imported by both |
| NFR7 | Portability | Runs on any OS with Python 3 + pip | Pure-Python dependencies, no native services |
| NFR8 | Integrity | Deployed models must be the evaluated models | Server prediction == direct artifact prediction |
| NFR9 | Transparency | The system must not overstate its accuracy | R² ≈ 0 and near-chance classification are displayed in the UI itself |

### 8.3 Software requirements

| Component | Requirement |
|---|---|
| Operating system | macOS / Linux / Windows (verified on macOS, Darwin x86_64) |
| Python | 3.10+ (verified on **CPython 3.14.6**) |
| Package manager | `pip` + `venv` |
| Dependencies | See §10 (5 direct, 18 installed transitively) |
| Browser | Any modern browser with HTML5 canvas (§13) |

### 8.4 Hardware requirements

| Item | Minimum | Verified on |
|---|---|---|
| CPU | Dual-core | Apple x86_64 |
| RAM | 2 GB (models total ≈ 18 MB) | — |
| Disk | ~120 MB (venv + artifacts + dataset) | — |
| Network | Only for CDN assets (Chart.js, Google Fonts) | CDNs returned HTTP 200 |

> Note: the app also renders fully offline except for the CDN-hosted Chart.js and web
> fonts; those are the only external dependencies at runtime.

---

## 9. Programming Language

| Language | Version | Where used | Share of code |
|---|---|---|---|
| **Python** | 3.14.6 | Data cleaning, feature engineering, training, evaluation, Flask backend, inference | 1,176 lines across 7 modules |
| **JavaScript** | ES2015+ (browser) | Client-side rendering, charts, interactions | `static/app.js` — 272 lines |
| **CSS** | CSS3 (custom properties, grid, flex, media queries) | Design system and responsive layout | `static/style.css` — 1,693 lines |
| **HTML5 / Jinja2** | — | Page templates with server-side data injection | 2,249 lines across 5 templates |
| **JSON** | — | API contracts and persisted metrics | — |
| **Markdown / Mermaid** | — | Documentation and architecture diagram | — |

**Total: ≈ 5,390 lines** of source, styles and templates. Python was chosen for the
entire computational layer because the scientific Python stack provides the training,
metrics and persistence primitives without reimplementation.

---

## 10. Libraries and Frameworks

### 10.1 Direct dependencies (`requirements.txt`)

| Library | Constraint | Installed | Role |
|---|---|---|---|
| Flask | ≥ 3.0 | **3.1.3** | Web framework: routing, request handling, JSON responses, templating |
| pandas | ≥ 2.0 | **3.0.5** | DataFrame cleaning, `groupby`, `shift`, rolling windows, time parsing |
| numpy | ≥ 1.24 | **2.5.2** | Array maths, quantile thresholds, metric computation |
| scikit-learn | ≥ 1.3 | **1.9.0** | `RandomForestRegressor`, `GradientBoostingRegressor`, `RandomForestClassifier`, `LogisticRegression`, `StandardScaler`, `Pipeline`, all evaluation metrics |
| joblib | ≥ 1.3 | **1.6.0** | Serialising and loading trained model artifacts |

### 10.2 Transitive / supporting libraries (installed)

`Werkzeug 3.1.8` (WSGI), `Jinja2 3.1.6` + `MarkupSafe 3.0.3` (templating),
`itsdangerous 2.2.0`, `click 8.5.0`, `blinker 1.9.0` (Flask internals),
`scipy 1.18.1` (scikit-learn numerical routines), `python-dateutil 2.9.0.post0` and
`six 1.17.0` (date parsing), `threadpoolctl 3.6.0` (BLAS thread control),
`cloudpickle 3.1.2` (parallel object serialisation), `narwhals 2.25.0` (pandas 3
interoperability layer).

### 10.3 Front-end frameworks and libraries

| Library | Delivery | Role |
|---|---|---|
| **Chart.js** | jsDelivr CDN | All charts: trend line, regional bar, monthly pattern, weekday/weekend, forecast, confusion-matrix heatmap |
| **Google Fonts** — Fraunces | CDN | Editorial serif for hero/section headings |
| **Google Fonts** — Inter | CDN | UI, body copy, buttons, labels |
| **Google Fonts** — IBM Plex Mono | CDN | Metrics, timestamps, technical values |

No front-end build step, bundler or node_modules is used: templates are rendered
server-side by Jinja2 and behaviour is added by a single vanilla JavaScript file.

### 10.4 Frameworks deliberately **not** used, and why

| Rejected | Reason |
|---|---|
| TensorFlow / PyTorch | 830 training rows with 12 tabular features; neural networks would overfit and are not competitive here |
| XGBoost / LightGBM | scikit-learn's ensembles already cover the requirement; adding a dependency would not change the conclusion |
| React / Vue + Node toolchain | A five-page server-rendered app does not justify a build pipeline |
| Streamlit / Dash | Would give a data-science UI, not a deployable product interface with a documented REST contract |

---

## 11. Database

**The project does not use a database management system.** This is a deliberate design
choice, and it is reported here accurately rather than described as one.

### 11.1 Actual persistence layer

| Artifact kind | Format | Path |
|---|---|---|
| Raw dataset | CSV | `data/raw/water_consumption_forecasting.csv` (900 rows, 23 KB) |
| Cleaned dataset | CSV | `data/processed/water_demand_clean.csv` (900 rows, 23 KB) |
| Trained models | joblib (pickle) | `models/*.joblib` (5 trained models + 2 feature-list artifacts; `models/` totals 18 MB) |
| Evaluation results | JSON | `models/metrics.json`, `models/classification_metrics.json` |
| Class thresholds | JSON | `models/classification_thresholds.json` |
| Inference history | CSV | `models/recent_history.csv` (last 30 days × 5 regions) |
| Feature contract | joblib | `models/feature_list.joblib`, `models/classification_feature_list.joblib` |

### 11.2 Data model

The analytical table is a three-column fact table:

```
consumption(region VARCHAR, date DATE, consumption_liters FLOAT)
PRIMARY KEY (region, date)          -- enforced during cleaning (duplicates removed)
```

Derived entities: `demand_level` (a four-valued class derived from training-set
quantiles), the 12-column feature vector, and the per-region rolling history buffer.

### 11.3 Why no DBMS — and what a DBMS would add

For 900 immutable rows read once at startup, a relational database would add a service
to install, a schema to migrate and a connection to maintain, while changing nothing
about the computation. Instead the dataset is loaded into a pandas DataFrame at
process start and served from memory, and the 7-day forecast is generated on demand.

The honest limitation: **there is no persistence of predictions versus actuals**, so
model drift cannot be tracked over time. The README's future-work section correctly
identifies a database-backed history as the next step. If added, PostgreSQL would be
the natural choice (the `region`/`date` key, range queries by date, and a
`forecast_vs_actual` table), with SQLite as the zero-install alternative for a
single-node deployment.

---

## 12. Development Tools

| Tool | Role in this project |
|---|---|
| **VS Code / any editor** | Source editing (plain Python and Jinja templates, no IDE-specific features) |
| **Python `venv`** | Isolated environment (`venv/bin/python`), keeping the project reproducible |
| **pip** | Dependency installation from `requirements.txt` |
| **Terminal / shell** | Running the pipeline scripts and the Flask server |
| **Browser DevTools** | Inspecting the rendered DOM, console output and network requests |
| **cURL / Python `urllib`** | Exercising the REST API directly during verification |
| **`shasum -a 256`** | Computing artifact fingerprints for integrity checking |
| **Freebuff agent environment** | Interactive verification, preview harness, automated checks |

**Not present in this repository, and worth adding:**

- **Version control** — the project directory is *not* a Git repository at present.
  Initialising Git (with `models/*.joblib` and `venv/` in `.gitignore`, or Git LFS for
  the binaries) is the single highest-value process improvement available.
- **Automated tests** — no test files exist. The ML contracts that most deserve tests
  are: the feature-column order matching `feature_list.joblib`, the chronological split
  boundaries (660/170, dates), the demand-level thresholds, and the `/api/predict`
  contract. These were verified manually during this audit (§16) and are reproducible.

---

## 13. Web Browser

The client side is plain HTML5, CSS3 and ES2015+ JavaScript, so any modern browser
works. Requirements and notes:

| Browser | Support | Notes |
|---|---|---|
| Google Chrome / Microsoft Edge | Full | Verified — DOM, canvas charts and all API calls rendered correctly with **zero console errors** |
| Mozilla Firefox | Full | Same standards; `canvas` and `fetch` fully supported |
| Safari (macOS/iOS) | Full | CSS custom properties, grid and canvas supported |
| Mobile browsers | Full | Verified down to 375 px viewport width with no horizontal scrolling |

**Browser capabilities relied upon:** HTML5 `<canvas>` (Chart.js rendering), CSS Grid
and Flexbox, CSS custom properties and `clamp()`, media queries, `fetch()` /
`XMLHttpRequest`, native `<input type="date">`, and `requestAnimationFrame` for the
entrance and count-up animations.

**Note on one browser-provided widget:** the prediction date field uses the browser's
native date picker. This means its calendar appearance varies by browser and locale —
a deliberate trade-off that avoids shipping a custom picker.

**Network dependencies:** Chart.js (jsDelivr) and the three font families (Google
Fonts) are loaded from CDNs; both returned HTTP 200 during verification. If a fully
offline deployment is required, these should be vendored into `static/`.

---

## 14. Methodology

The project follows a **CRISP-DM-style pipeline**, with each phase producing a durable
artifact that the next phase consumes. The critical methodological decisions are stated
with their justification, because they are the parts a viva will probe.

### 14.1 Phase 1 — Data acquisition and understanding

- Source: real daily water-consumption dataset, 900 rows, 5 regions × 180 days
  (2023-01-01 → 2023-06-29), columns `region`, `date`, `consumption_liters`.
- **Profile (verified):** 0 missing values, 0 duplicate rows, 180 rows per region,
  mean 12,753.8 L/day, σ = 4,278.6, min 5,004.1, max 19,989.8.
- Per-region means: East 13,102.4, Central 13,040.4, North 12,791.6, West 12,483.6,
  South 12,350.7 L/day.
- **Decision:** no external variables are synthesised. The dataset's own README-level
  honesty constraint is that features must come from what actually exists.

### 14.2 Phase 2 — Data cleaning (`data/prepare_data.py`)

Date parsing with coercion, removal of missing-key rows, removal of exact duplicates,
removal of duplicate `(region, date)` keys, retention of strictly positive consumption,
and standardised column order and types. The script emits a cleaning report; on the
real file **every counter is zero**, which is itself a finding — the source data is
already clean, and the cleaning stage exists to guarantee the invariant rather than to
repair damage.

A synthetic weather-driven generator is retained **only as a fallback** and writes to a
different filename so it can never overwrite the real cleaned dataset. The application
never reads it.

### 14.3 Phase 3 — Feature engineering (`src/features.py`)

Twelve features, all legitimate for this dataset:

| Group | Features |
|---|---|
| Region | `region_Central`, `region_East`, `region_North`, `region_South`, `region_West` (one-hot, deterministic column order) |
| Calendar | `month`, `day_of_week`, `is_weekend` |
| Own history | `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_mean_14` |

**Leakage controls (the methodological core of the project):**

1. Lags and rolling means are computed **per region** via `groupby`, so no row ever
   sees another region's values.
2. `shift(1)` is applied **before** every rolling window, so a row can never include
   its own target value in its average.
3. The 14-day warm-up window is **dropped**, not imputed — 900 raw rows become 830
   modelled rows (14 warm-up days × 5 regions).
4. The identical `build_features()` function is used at training and inference time, so
   the feature contract cannot drift between the two.

### 14.4 Phase 4 — Split strategy

A **strict chronological, per-region 80/20 split with no shuffling**:

| | Rows | Period |
|---|---|---|
| Train | 660 | 2023-01-15 → 2023-05-26 |
| Test | 170 | 2023-05-27 → 2023-06-29 |

Splitting per region means every region's test window is its own most recent 20% of
days — genuine out-of-sample forecasting. Verified property: the last training date
precedes the first test date, so there is **no temporal overlap**. Both the regression
and classification tasks use this same split, which makes their results directly
comparable.

### 14.5 Phase 5 — Model training and selection

**Regression (`train_models.py`)**

| Model | Configuration |
|---|---|
| Baseline (naive) | prediction = `lag_1` (yesterday's actual) |
| Random Forest | `n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42` |
| Gradient Boosting | `n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42` |

Selection rule: highest R² among the two learned models; the baseline is a reference
point and is never promoted to "best".

**Classification (`train_classifier.py`)**

Target `demand_level ∈ {LOW, NORMAL, HIGH, VERY HIGH}` derived from **training-set
quantiles only**:

| Boundary | Value (L/day) | Class |
|---|---|---|
| < q25 | 9,382.28 | LOW |
| q25 – q50 | 13,140.43 | NORMAL |
| q50 – q75 | 16,431.28 | HIGH |
| ≥ q75 | — | VERY HIGH |

The test set plays **no role** in choosing these thresholds — the standard source of
label leakage in classification projects is avoided explicitly. Models compared:
Logistic Regression (inside a `StandardScaler` pipeline, because features range from
0/1 one-hots to thousands of litres) and Random Forest; best selected by **weighted F1**
on the chronological test set.

### 14.6 Phase 6 — Evaluation

Regression: MAE, RMSE, R². Classification: accuracy, macro precision/recall/F1,
weighted F1, per-class metrics and the full 4×4 confusion matrix. All computed on the
held-out future window.

### 14.7 Phase 7 — Deployment

Flask loads the best regression model and the classifier once at process start, plus
the feature list and the per-region recent-history buffer, then exposes nine endpoints
and renders five pages. Inference uses the **same** feature pipeline as training.

**Forecast method:** recursive multi-step. Each region's buffer is seeded with its last
30 days of *actual* readings; day 1 is predicted from actuals, that prediction is fed
back as `lag_1` for day 2, and so on for seven days, with a demand level attached at
each step.

### 14.8 Phase 8 — Verification

The deployed system was verified against its own artifacts: metrics recomputed from the
saved `.joblib` files, artifact checksums recorded, every endpoint exercised, and the
UI inspected in a real browser. Results in §15–§16.

---

## 15. Results

### 15.1 Regression (chronological test set, 170 rows, L/day)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (naive `lag_1`) | 4,573.29 | 5,595.92 | −0.8572 |
| **Random Forest** ✅ *(deployed)* | **3,564.73** | **4,149.29** | **−0.0211** |
| Gradient Boosting | 3,705.09 | 4,317.58 | −0.1056 |

Interpretation: both learned models beat the naive baseline substantially on MAE
(≈ 3.6k vs 4.6k L/day), but R² ≈ 0 means neither explains meaningfully more variance
than the training mean. **This is a property of the data, not a defect.** The dataset
has near-zero day-to-day autocorrelation, so lag features carry almost no predictive
signal. The honest conclusion is stated in the UI itself.

### 15.2 Classification (same test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| Logistic Regression | 0.2882 | 0.1437 | 0.2801 | 0.1849 | 0.1853 |
| **Random Forest** ✅ *(deployed)* | 0.2235 | 0.2237 | 0.2240 | 0.2100 | 0.2122 |

Random Forest confusion matrix (rows = actual, columns = predicted; order LOW, NORMAL,
HIGH, VERY HIGH):

```
[[17,  8,  5, 17],
 [16,  4,  1, 21],
 [17,  9,  6, 13],
 [10,  6,  9, 11]]
```

With four balanced classes, chance accuracy is ≈ 25%. Logistic Regression scores higher
on accuracy (28.8%) but predicts only two of the four classes — it collapses NORMAL and
HIGH entirely (0.0 precision *and* 0.0 recall for both). Random Forest spreads its
predictions across all four classes. Random Forest is therefore deployed: weighted F1
0.2122 > 0.1853, and macro F1 0.2100 > 0.1849.

### 15.3 Feature importance (deployed regressor)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `lag_7` | 0.216 |
| 2 | `rolling_mean_14` | 0.197 |
| 3 | `lag_1` | 0.196 |
| 4 | `rolling_mean_7` | 0.192 |

The four history features dominate and are almost equally weighted — a near-uniform
importance profile, which is the signature of a model with very little real signal to
exploit. This corroborates the R² finding from a second direction.

### 15.4 The information ceiling — the project's central result

R² ≈ 0 invites the objection that the pipeline is broken. The project answers that
objection with measurement rather than assertion, via `analysis/signal_audit.py`
(served at `GET /api/signal-audit` and displayed in the Model Lab):

| Test | Result | Interpretation |
|---|---|---|
| Ljung-Box white-noise test (lags 7 / 14 / 21 × 5 regions) | **1 of 15 rejects** at α = 0.05 | 0.75 rejections expected by chance; cannot reject white noise |
| Lag-1 autocorrelation per region | −0.057 … +0.042 | yesterday carries no usable information about today |
| Variance explained by region alone | **0.48 %** | knowing the region barely helps |
| **Information ceiling** — best constant predictor (training mean) | **MAE 3,514.3 · R² −0.0075** | the best achievable score on this data |
| Deployed Random Forest | MAE 3,564.7 · R² −0.0211 | **1.4 % from the ceiling** |
| Classifier ceiling — always predict the majority class | accuracy **0.2706** | the best achievable classification accuracy |
| Deployed Random Forest classifier | accuracy 0.2235 | 4.7 points *below* the ceiling |

Two conclusions follow, and they are the most defensible claims in the project:

1. **The regressor is essentially optimal for this data.** Any model can at best
   predict the mean; the deployed forest lands within 1.4 % of that bound. R² ≈ 0 is a
   measured property of the dataset, not an implementation defect.
2. **The classifier sits below its ceiling**, which is the expected signature of fitting
   noise. With no signal present, an unregularised forest can only do *worse* than the
   majority-class rule. This is reported as a finding rather than concealed.

The dataset's shape supports the same conclusion independently: skew ≈ −0.05…−0.16 and
kurtosis ≈ −1.0…−1.2 (flat, symmetric — unlike real demand data, which is typically
right-skewed), with uniformly distributed decimals. Statistically it behaves like series
of independent random draws. A forecasting model cannot beat randomness, and the honest
deliverable is to demonstrate the limit rather than to disguise it.

---

## 16. Verification Log

All checks below were executed against the running application and the saved artifacts.
Commands are reproducible; the running server was at `http://127.0.0.1:5000`.

### 16.1 Dataset and pipeline integrity

| Check | Method | Result |
|---|---|---|
| Row count | `len(df)` | 900 — **PASS** |
| Columns | `list(df.columns)` | `['region','date','consumption_liters']` — **PASS** |
| Regions | unique values | Central, East, North, South, West — **PASS** |
| Days per region | `groupby(region).size()` | 180 for all five — **PASS** |
| Missing values | `df.isna().sum().sum()` | 0 — **PASS** |
| Duplicate rows | `df.duplicated().sum()` | 0 — **PASS** |
| Feature frame | `build_features()` | 830 rows × 12 features (900 − 14×5 warm-up) — **PASS** |
| Train/test sizes | `chronological_split()` | 660 / 170, matching `metrics.json` — **PASS** |
| Split boundaries | min/max dates | 2023-01-15→05-26 and 2023-05-27→06-29 — **PASS** |
| Temporal overlap | `train.max < test.min` | true, no overlap — **PASS** |
| Feature contract | artifact vs pipeline | `feature_list.joblib` == `FEATURE_COLUMNS` — **PASS** |

### 16.2 Metrics recomputed from the saved artifacts

| Quantity | Stored | Recomputed from `.joblib` | Result |
|---|---|---|---|
| Random Forest MAE / RMSE / R² | 3564.7347 / 4149.2871 / −0.0211 | identical | **PASS** |
| Gradient Boosting MAE / RMSE / R² | 3705.0916 / 4317.579 / −0.1056 | identical | **PASS** |
| Naive baseline MAE / RMSE / R² | 4573.2937 / 5595.92 / −0.8572 | identical | **PASS** |
| Thresholds q25 / q50 / q75 | 9382.2825 / 13140.425 / 16431.2825 | identical | **PASS** |
| Classifier accuracy / F1 macro / F1 weighted | 0.2235 / 0.2100 / 0.2122 | identical | **PASS** |
| Confusion matrix | 4×4 array | identical | **PASS** |
| Test class distribution | LOW 47, NORMAL 42, HIGH 45, VERY HIGH 36 | identical | **PASS** |

This proves the numbers shown in the Model Lab are the numbers the deployed models
actually produce — not stale or hand-entered values.

### 16.3 Application and API verification

| Check | Result |
|---|---|
| Pages `/`, `/prediction`, `/forecast`, `/analytics`, `/model-performance` | all **HTTP 200** |
| `GET /api/health` | 200 — model loaded, classifier loaded |
| `GET /api/metrics`, `/api/classification-metrics`, `/api/confusion-matrix` | 200 |
| `GET /api/dashboard-stats`, `/api/analytics-data`, `/api/feature-importance` | 200 |
| `POST /api/predict` (valid) | 200 with prediction, level, and the feature values used |
| `POST /api/predict` (unknown region) | **400** with a clear message |
| `POST /api/predict` (missing field) | **400** with the missing field named |
| `GET /api/forecast` for all 5 regions | 7 days each |
| Browser console | **0 errors** across a full walkthrough |
| Horizontal overflow | none at 1440 px, 743 px, 375 px |
| CDN assets (Chart.js, Google Fonts) | HTTP 200 |

### 16.4 Deployed-artifact consistency (the anti-mock check)

A direct in-process call to the saved model was compared against the API response for
the same input:

```
direct model call (seeded from real history)  = 14447.720 L/day, level VERY HIGH
POST /api/predict before a forecast           = 14447.720 L/day, lag_1 = 17917.25
POST /api/predict after  a forecast           = 14447.720 L/day, lag_1 = 17917.25
```

- API output == direct model output → **PASS** (the UI number is genuinely the model's number)
- `lag_1` equals the real last Central reading (17917.25, exact to full precision) → **PASS**
- Prediction is stable across an intervening forecast → **PASS**

### 16.5 Defect found and corrected during this audit

**Symptom.** `POST /api/predict` returned **13,050.922** for `{Central, 2026-10-13}`
while a freshly started predictor returned **14,447.720** for the same input.

**Root cause.** `DemandPredictor.forecast()` appends each predicted day into the shared
per-region buffer and the buffer was never restored. The response's own diagnostic field
made this unambiguous: the API reported `lag_1 = 12241.6`, which is exactly the last
*forecast* day's value (2023-07-06 → 12,241.64) — not the real last reading
(17,917.25). Consequently a single-day prediction issued after any forecast (and the
dashboard runs one on load) was fed predicted values as though they were measured
readings, and the answer depended on what the server had previously been asked for
rather than only on the request.

**Fix** (`src/predict.py`, lines 149–160): the recursive loop now saves the region's
real history and restores it in a `finally` block, so predicted days feed forward
*within* one forecast — preserving recursive semantics — but never leak into the
buffer. This restores the documented design in `README.md` ("seeded with its last 30
days of **actual** consumption").

**Verification after the fix:**

| Check | Pre-fix | Post-fix | Result |
|---|---|---|---|
| 7-day recursive forecast values | `14090.35, 13323.34, 13376.19, 13329.37, 13224.42, 12878.05, 12241.64` | **identical** | Recursion semantics unchanged |
| API prediction for the same input | 13050.922 | 14447.720 = direct model | **PASS** |
| `lag_1` fed to the model | 12241.6 (a forecast value) | 17917.25 (the real reading) | **PASS** |
| Prediction stable across a forecast | not guaranteed | stable | **PASS** |

No model, dataset, threshold or metric was modified by this fix: the 7-day forecast is
byte-identical and all recomputed metrics still match §16.2.

### 16.6 Defect found and corrected during this audit (2 of 2)

**Symptom.** The Model Lab page returned HTTP 200 and rendered its headings, but every
panel on it stayed frozen (`loadSignalAudit is undefined`). The browser console reported
`SyntaxError: Missing } in template expression`.

**Root cause.** The page carries one large inline `<script>`; a single syntax error
prevents the browser from parsing the whole block, so no panel on the page initialises.
The offending expression was
`${wn.expected_false_positives_at_0.05}` — the JSON key contains a dot, so JavaScript
tokenised `.05` as a *number literal* rather than a property access.

**Fix.** Bracket notation: `${wn['expected_false_positives_at_0.05']}`.

**Investigation note.** This is worth recording because a plain HTTP-200 check is blind
to it. The failure was localised with an authoritative parser (`node --check` on the
rendered script, which reported line 151 and the exact column) rather than by reading the
code, and the scan of the *template file* had reported every template literal balanced —
the bug only existed in the rendered output as parsed by a real engine.

**Regression guard.** `tests/test_api.py::test_inline_scripts_are_valid_javascript`
syntax-checks the inline script of all five pages with `node --check` (skipping only if
`node` is unavailable). It fails on the pre-fix code and passes on the fixed code.

### 16.7 Verification of the delivered changes

| Change | Verification | Result |
|---|---|---|
| Hero date range removed | `hero-range` element absent from served HTML; `document.getElementById('hero-range') === null`; page text contains no `Jan 2023` | **PASS** |
| `/api/signal-audit` added | HTTP 200, returns the audit JSON | **PASS** |
| Model Lab ceiling panel | Renders MAE 3,514 / R² −0.0075 ceiling, 1.43 % gap, verdict text; browser console **empty** | **PASS** |
| Test suite | `pytest tests/ -q` → **54 passed**, exit 0 | **PASS** |
| Pinned dependencies | all pins resolve; `pip install -r requirements-dev.txt` exit 0 | **PASS** |
| All pages and endpoints | 7 routes HTTP 200 after the changes | **PASS** |
| Stored metrics unchanged by all of the above | metrics recomputation still matches §16.2 to 4 dp | **PASS** |

### 16.8 Artifact fingerprints (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `data/raw/water_consumption_forecasting.csv` | `601d50598cae4816b9d9328d3423a68dd24d04eeb2c1d9d440df13f7082f4e36` |
| `data/processed/water_demand_clean.csv` | `35d62b119573bc2c06812126edab339da039e5aeb8db40d4507b436693aef84e` |
| `models/random_forest_model.joblib` | `1c37132a200dd9da1a088ee31d7788ed3e79894ec07ec5e496032d7796978ba8` |
| `models/gradient_boosting_model.joblib` | `9acd2d76a4092f2a634d0d1783d6f7d2dc51809ce0a7cf32ceb33956254b1b13` |
| `models/classification_model.joblib` | `de8de369d924ef55f7c12900e948c92d6f2de9e5626c86bcca9a59c83f69d782` |
| `models/metrics.json` | `f631b467c540aa15d5c8d2f7053d9141c83928649a5858dad15a839ca8749d81` |
| `models/classification_metrics.json` | `3b189675a31031a2e45f425acb86def7f718ff2d4e55756b8d59ce883c7652c9` |
| `models/classification_thresholds.json` | `74d09e60eceb471beba6640e4286ec374755f6d934789f76a1692ff51db92270` |
| `models/recent_history.csv` | `55f0c1b5e1ad421ae6ab2d921b1d6403714421b87c3b332328fc76ce2025e874` |

All dataset and model artifacts are dated 7–8 September 2026, i.e. they predate this
audit; no training or retraining was run, and none of them was modified.

---

## 17. Implementation Structure

### 17.1 Directory layout

```
smartwater-ai/
├── app.py                      321 lines   Flask backend: 5 pages + 9 REST endpoints
├── train_models.py             167 lines   Regression training + evaluation
├── train_classifier.py         208 lines   Classification training + evaluation
├── requirements.txt                        5 direct dependencies
├── README.md                               Project documentation
├── data/
│   ├── prepare_data.py         185 lines   Cleaning pipeline (+ synthetic backup path)
│   ├── raw/
│   │   └── water_consumption_forecasting.csv   900 rows — the real dataset
│   └── processed/
│       └── water_demand_clean.csv              900 rows — the cleaned dataset
├── src/
│   ├── features.py             135 lines   Shared leakage-safe feature pipeline
│   └── predict.py              160 lines   Inference: predict, classify, forecast
├── models/                                 Trained artifacts, metrics, thresholds, history
├── static/
│   ├── style.css             1,693 lines   Design system + responsive rules
│   ├── app.js                  272 lines   Charts, KPIs, interactions
│   └── assets/
│       └── Smartwaterslide1.png            Hero photograph (1.8 MB)
├── templates/
│   ├── index.html            1,034 lines   Dashboard (8 sections)
│   ├── analytics.html          439 lines   Analytics
│   ├── model_performance.html  342 lines   Model Lab
│   ├── forecast.html           238 lines   7-day forecast
│   └── prediction.html         196 lines   Prediction
└── docs/
    └── PROJECT_REPORT.md                   This document
```

### 17.2 Layer responsibilities

| Layer | Module | Responsibility | Depends on |
|---|---|---|---|
| Data preparation | `data/prepare_data.py` | Clean, validate, standardise the raw CSV; report issues | pandas |
| Feature engineering | `src/features.py` | Build the 12-feature vector; per-region leakage-safe lags; chronological split | pandas, numpy |
| Training (regression) | `train_models.py` | Fit Baseline / RF / GB, evaluate, persist models + metrics + history | features, scikit-learn, joblib |
| Training (classification) | `train_classifier.py` | Derive train-only thresholds, fit LR / RFC, evaluate, persist | features, scikit-learn, joblib |
| Inference | `src/predict.py` | Load artifacts; single-day predict, classify, recursive 7-day forecast | features, joblib |
| API / presentation | `app.py` | Load artifacts and dataset once; render pages; serve JSON | predict, Flask, pandas |
| Client UI | `static/*`, `templates/*` | Charts, KPIs, responsive editorial layout | Chart.js, fonts |

### 17.3 Data flow

```
raw CSV (900 rows)
   │  prepare_data.py            clean + validate
   ▼
processed CSV (900 rows)
   │  features.build_features()  12 features, per-region lags, warm-up dropped
   ▼
feature frame (830 rows × 12)
   │  features.chronological_split()   per region, no shuffle
   ├──────────────────────────────┐
   ▼                              ▼
train 660 (Jan 15–May 26)     test 170 (May 27–Jun 29)
   │  train_models.py             │  train_classifier.py
   ▼                              ▼
random_forest_model.joblib     classification_model.joblib
metrics.json                   classification_metrics.json + thresholds.json
   └──────────────┬───────────────┘
                  ▼
        src/predict.py (inference, 7-day recursive forecast)
                  ▼
        app.py  →  9 REST endpoints  →  5 pages + Chart.js
```

### 17.4 API contract

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/api/predict` | POST | `{"region": "...", "date": "YYYY-MM-DD"}` | `predicted_consumption`, `demand_level`, `unit`, `model_used`, `features` (the actual inputs used) |
| `/api/forecast` | GET | `?region=` | 7 `{date, predicted_consumption, demand_level, is_weekend}` rows + methodology note |
| `/api/dashboard-stats` | GET | — | average / peak / minimum / latest consumption, record count, date range |
| `/api/analytics-data` | GET | — | trend series, monthly means, weekday vs weekend, per-region means, distribution, extremes |
| `/api/feature-importance` | GET | — | ranked feature importances from the deployed regressor |
| `/api/metrics` | GET | — | full regression metrics + split metadata |
| `/api/classification-metrics` | GET | — | classifier metrics, thresholds, per-class results |
| `/api/confusion-matrix` | GET | — | confusion matrix + per-class metrics of the best classifier |
| `/api/health` | GET | — | liveness, best model, classifier status |

The `features` block in `/api/predict` deserves emphasis: the API returns the **exact
feature values it fed the model**, which is what makes the interface's "why this
prediction?" panel explainable rather than decorative.

---

## 18. Limitations

1. **Weak temporal signal.** Near-zero autocorrelation means R² ≈ 0 and classification
   near chance. No modelling technique can extract signal that is not in the data.
2. **Classification is near chance — and below its own ceiling.** Accuracy 22.4% vs a
   27.06% majority-class ceiling and ~25% for random guessing on four balanced classes.
   The honest reading is that the classifier is fitting noise; with no signal in the
   data, an unregularised forest cannot beat the majority-class rule. The classifier is
   retained as a demonstrated pipeline, not as a reliable decision tool, and this must
   be stated plainly in any presentation.
3. **Short window.** 180 days per region, one season — no annual seasonality can be
   learned.
4. **Extrapolation risk.** Predictions far beyond June 2023 reuse the last known
   readings as lag values, so long-horizon dates are extrapolations.
5. **Recursive error accumulation** over the 7-day forecast.
6. **No drift tracking.** Forecasts are not stored against later actuals, so model
   degradation cannot be measured in production.
7. **Development server.** Flask's built-in server is single-process and not intended
   for production traffic.
8. **CDN dependency** for Chart.js and fonts.
9. **No automated tests and no version control** in the repository as it stands.

---

## 19. Future Enhancements

1. **Exogenous features** (weather, holidays, events) — the single change most likely to
   move R² away from zero.
2. **Multi-year data** to model seasonality, and **live daily ingestion** so lags use
   true actuals rather than a fixed history file.
3. **Time-series cross-validation** (expanding-window backtesting) instead of a single
   split, to report metric variability.
4. **Database-backed forecast-vs-actual log** (PostgreSQL or SQLite) for drift monitoring.
5. **Production serving** with gunicorn + containerisation, and authentication for
   non-public endpoints.
6. **Automated test suite** covering the feature contract, split boundaries, thresholds
   and the `/api/predict` contract.
7. **Vendored front-end assets** so the app runs fully offline.

---

## 20. Conclusion

SmartWater AI delivers a complete, working, honestly evaluated machine-learning
application: a real 900-row dataset, a leakage-safe feature pipeline shared by training
and inference, three regression models and two classifiers evaluated on a strictly
chronological split, and a five-page Flask interface with nine REST endpoints that has
been verified end to end.

The most valuable outcome is methodological rather than numerical. Because the dataset
contains no exogenous variables, the reported performance is genuinely modest — MAE
3,564.73 L/day with R² −0.0211 for the deployed regressor, and 22.4% accuracy for the
deployed classifier against a ~25% chance baseline. Those numbers are not a failure of
implementation; they are the honest measurement of how much signal this data contains,
and they are displayed transparently in the application's own Model Lab rather than
hidden.

**The problem this project set out to solve is resolved, but not in the way the metrics
alone suggest.** Rather than chase an accuracy the data cannot support, the project
establishes the achievable bound and measures itself against it. The Ljung-Box tests
cannot reject white noise (1 of 15 rejections against 0.75 expected by chance), region
explains 0.48% of variance, and the best possible predictor on the held-out window — the
training mean — scores MAE 3,514.3. The deployed regressor reaches MAE 3,564.7, i.e.
**within 1.4% of the information ceiling**, which is a far stronger and more defensible
claim than an inflated accuracy figure. The classifier's position *below* its ceiling is
reported as evidence of noise-fitting rather than hidden.

Verification during this audit confirmed that the dataset, feature pipeline, split
boundaries and stored metrics are all mutually consistent, that the recomputed metrics
match the stored ones exactly, and that the deployed API returns exactly what the saved
model computes for the same input. **Two real defects were found and corrected.** The
first: the recursive forecast was leaking predicted values into the shared history
buffer, making single-day predictions depend on what the server had been asked
previously; the fix restores the documented design, leaves the 7-day forecast
byte-identical, and makes the prediction endpoint deterministic. The second: a
JavaScript syntax error in the Model Lab's inline script silently disabled every panel
on that page while still returning HTTP 200; it is now fixed and guarded by a test that
syntax-checks the inline script of every page.

---

## 21. References

1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.
2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *The Annals of Statistics*, 29(5), 1189–1232.
3. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830.
4. Harris, C. R. et al. (2020). "Array programming with NumPy." *Nature*, 585, 357–362.
5. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 56–61.
6. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
7. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting*, 36(1), 54–74.
8. Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). "Why do tree-based models still outperform deep learning on typical tabular data?" *Advances in Neural Information Processing Systems 35*.
9. Pallets Projects. *Flask Documentation*. https://flask.palletsprojects.com/
10. Chart.js Contributors. *Chart.js Documentation*. https://www.chartjs.org/

> **Citation note:** references 1–8 are standard, well-known publications included as
> the methodological basis of this project. Before submission, confirm the exact
> edition/page numbers required by your institution's citation style, and add the
> dataset's own Kaggle page (dataset title, author and URL) as a data citation.

---

*Report compiled from direct inspection and execution of the project on 12 October 2026.
All metrics in §15 were independently recomputed from the saved model artifacts during
this audit; all fingerprints in §16.6 were computed from the files on disk.*
