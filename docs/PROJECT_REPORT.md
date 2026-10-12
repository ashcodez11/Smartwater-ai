# SmartWater AI — Project Report

**Project title:** SmartWater AI — Daily Water Consumption Forecasting and Demand-Level Classification  
**Domain:** Machine Learning / Data Science / Web Application  
**Target:** VTU undergraduate Computer Science and Engineering project  
**Dataset:** Real Kaggle dataset — `data/raw/water_consumption_forecasting.csv`  
**Report version:** Final, aligned with the committed code

---

## 1. INTRODUCTION

SmartWater AI is a machine-learning application that forecasts daily municipal
water consumption and classifies each predicted day into a demand level. It is
built on a real public dataset of daily water consumption covering five regions
over 180 days each, for a total of 900 observations.

The system performs two related machine-learning tasks on the same
leakage-safe feature pipeline:

1. **Regression** — predict the continuous daily consumption in liters per day.
2. **Classification** — predict the demand level, chosen from LOW, NORMAL,
   HIGH, and VERY HIGH.

Both tasks are exposed through a responsive Flask web application with a
dashboard, a dedicated prediction interface, a 7-day recursive forecast, an
analytics section, and a model-performance (“Model Lab”) page. The application
also exposes a set of REST API endpoints so that the same predictions can be
consumed programmatically.

The intended users are students, educators, and demonstration users who want to
understand how a real, honestly evaluated forecasting system is built end to
end. The project is explicitly educational: it shows not only how to build a
model, but also how to evaluate it truthfully, including the case where the
available data does not support a strong predictive result.

---

## 2. PROBLEM DEFINITION

The core problem is to forecast a region’s daily water consumption from its own
historical observations and from calendar information.

Formally, for a given region and a future date, the system must estimate
`consumption_liters` using only information that would have been available
before that date. This is a time-series forecasting problem with a small number
of explanatory variables.

The project also defines a derived classification problem: convert the predicted
consumption into one of four demand levels using thresholds computed from the
training data.

The problem is difficult for several honest reasons:

- The dataset is small: 900 rows total, 180 days per region.
- The dataset contains only three columns: `region`, `date`, and
  `consumption_liters`. There are no temperature, rainfall, humidity, holiday,
  population, or supply-schedule variables.
- The real series show very weak day-to-day autocorrelation, so yesterday’s
  consumption is only a weak guide to today’s.
- The available features are therefore mostly calendar signals and a region’s own
  recent consumption history.

An accurate, general-purpose forecaster would ideally use richer explanatory
variables and a longer history. This project works within the actual limits of
the data and reports the consequences honestly rather than masking them.

---

## 3. OBJECTIVES

The objectives of the project are:

1. To analyze a real water-consumption dataset and understand its structure,
   quality, and limitations.
2. To build a reproducible machine-learning pipeline for feature engineering,
   chronological splitting, training, and evaluation.
3. To predict daily consumption for the supported regions and future dates using
   a trained regression model.
4. To compare several candidate models using appropriate evaluation metrics.
5. To classify demand levels where supported by the implementation.
6. To provide a usable web-based interface for non-technical users.
7. To expose the prediction and forecast functionality through REST APIs.
8. To ensure that prediction behavior is reliable and deterministic.
9. To evaluate the system honestly, including its limitations.
10. To document the implementation so that the work can be defended in a viva.

---

## 4. MOTIVATION

Water is a limited and expensive resource. Utilities and local authorities must
plan supply, pumping schedules, storage, maintenance, and emergency response
around expected demand. If demand is underestimated, shortages and pressure
drops can occur. If demand is overestimated, water and energy may be wasted.

A system that can forecast tomorrow’s consumption and flag whether demand is
likely to be unusually low or high can support:

- better supply scheduling,
- better maintenance planning,
- better awareness of consumption patterns,
- evidence-based discussion of water usage.

SmartWater AI does not claim to be a production water-management control system.
Its motivation is to demonstrate a complete, honest, and reproducible pipeline
from raw data to trained models to a web interface, while making the real
limitations of the dataset visible. The educational value is in the process:
cleanup, leakage-safe features, chronological validation, comparison, evaluation,
and transparent reporting.

---

## 5. SCOPE OF THE PROJECT

### 5.1 Included in scope

- **Dataset:** 900 observations, 5 regions, 180 days each.
- **Date range:** 2023-01-01 to 2023-06-29.
- **Regions:** Central, East, North, South, West.
- **Regression:** Baseline, Random Forest, and Gradient Boosting models trained
  and evaluated on a chronological 80/20 split.
- **Classification:** Logistic Regression and Random Forest classifiers trained
  on the same split, using training-set quantile thresholds.
- **Forecasting interface:** 7-day recursive forecast per region.
- **Prediction interface:** single-day prediction for a chosen region and date.
- **Analytics:** real aggregations from the dataset, with no invented variables.
- **Model Lab:** regression metrics, classification metrics, confusion matrix,
  feature importance, thresholds, and training metadata.
- **APIs:** health, metrics, classification metrics, confusion matrix, predict,
  dashboard stats, analytics data, feature importance, forecast, and a signal
  audit endpoint.
- **Persistence (current):** CSV data, JSON metadata/metrics, and serialized
  model artifacts.
- **Reproducibility artifacts:** pinned dependencies, test suite, signal audit,
  and an independent model-comparison analysis.

### 5.2 Out of scope / future work

- Live ingestion of daily readings.
- Integration with real water infrastructure or live sensor feeds.
- Exogenous features such as weather, holidays, or population data, unless and
  until such data is actually available.
- Multi-year seasonality modeling, which requires a longer dataset than the one
  currently available.
- A conventional relational or NoSQL database, if and when persistence beyond
  CSV/JSON artifacts is required.
- Full production hardening such as authentication, containerization, and
  monitoring, which are recommended enhancements rather than delivered features.

---

## 6. APPLICATIONS

The techniques demonstrated by SmartWater AI can be applied in several practical
contexts, subject to the availability of suitable data:

1. **Municipal water planning** — understanding average and peak demand patterns
   across regions.
2. **Distribution scheduling** — anticipating daily demand to support pumping
   and storage decisions.
3. **Consumption monitoring** — comparing regions and time periods using real
   historical aggregations.
4. **Resource allocation** — identifying higher- or lower-demand periods to
   support operational planning.
5. **Anomaly investigation** — using demand-level flags as a starting point for
   investigating unusual consumption, if richer data becomes available.
6. **Demand awareness** — presenting consumption trends and patterns to
   non-technical stakeholders through a web interface.

These are potential applications of the underlying approach. The current project
does not integrate with a live water network, and it does not claim to do so.

---

## 7. LITERATURE SURVEY

### 7.1 Time-series forecasting foundations

A classic foundation for time-series modeling is Box, Jenkins, Reinsel, and
Ljung, *Time Series Analysis: Forecasting and Control*. This work establishes
the statistical view of time series, including the role of autocorrelation,
model identification, and diagnostic checking. One of its diagnostic ideas — the
Ljung-Box test — is used directly in this project’s signal audit to test whether
the consumption series behave like white noise.

### 7.2 Ensemble tree methods

Breiman, “Random Forests” (2001), introduced random forests as an ensemble of
decision trees trained with bagging and random feature selection. Random forests
are widely used for tabular regression and classification because they are
robust, require little preprocessing, and often perform well without extensive
tuning. This project uses a Random Forest regressor as the deployed model and a
Random Forest classifier for demand levels.

### 7.3 Gradient boosting

Friedman, “Greedy Function Approximation: A Gradient Boosting Machine” (2001),
described gradient boosting as a stage-wise additive model that fits new
predictors to the residual errors of earlier ones. Gradient boosting is another
strong candidate for tabular problems and is included in this project as a
comparison model.

### 7.4 Forecasting practice

Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, is a widely
used reference on practical forecasting. It emphasizes the importance of using
appropriate evaluation methods, respecting time order, and avoiding leakage
between training and test data. These principles guide the chronological split
and the past-only feature construction used in SmartWater AI.

### 7.5 Forecasting competitions

Makridakis, Spiliotis, and Assimakopoulos, “The M4 Competition: 100,000 Time
Series and 61 Forecasting Methods” (2020), reported on a large-scale comparison
of forecasting methods. A recurring lesson from such competitions is that no
single method dominates everywhere, and that the difficulty of the data often
matters more than the choice of model. This is consistent with the present
project’s finding that the available dataset does not provide strong temporal
signal, so several candidates perform similarly.

### 7.6 Relevance to this project

The works above are cited for context, not because the project reimplements them.
They support three decisions:

- use diagnostic checks such as autocorrelation and Ljung-Box tests,
- consider ensemble tree methods as strong tabular baselines,
- evaluate strictly in time order with no leakage.

---

## 8. REQUIREMENTS SPECIFICATION OF SOFTWARE

### 8.1 Functional requirements

1. The system shall clean and validate the real dataset.
2. The system shall build leakage-safe features from region, date, and past
   consumption.
3. The system shall split the data chronologically, per region, into training
   and test periods.
4. The system shall train and evaluate regression models and store the results.
5. The system shall train and evaluate classification models and store the
   results.
6. The system shall predict daily consumption for a supported region and date.
7. The system shall return a demand level alongside the prediction when the
   classifier is available.
8. The system shall provide a 7-day recursive forecast per region.
9. The system shall expose analytics computed from the real dataset.
10. The system shall expose model-performance information.
11. The system shall expose a health endpoint describing model and classifier
    status.
12. The system shall provide a web interface for the dashboard, prediction,
    forecast, analytics, and model lab.
13. The system shall serve pages and API responses that render in a standard
    web browser.

### 8.2 Non-functional requirements

1. **Determinism:** the same inputs shall produce the same prediction output
   across repeated calls.
2. **Reliability:** forecast results shall not contaminate the history used by
   later single-day predictions.
3. **Reproducibility:** the project shall pin the versions used to produce the
   reported metrics.
4. **Maintainability:** the code shall be organized so that training, inference,
   features, API, and templates have clear responsibilities.
5. **Testability:** the core contracts shall be covered by automated tests.
6. **Usability:** the interface shall be responsive enough for desktop and mobile
   viewing.
7. **Honesty:** the system shall report actual evaluation results, including weak
   results where they occur.

### 8.3 Verified versus desired enhancements

Verified requirements include the prediction API, forecast API, classification
support, the stored metrics, the chronological split, the leakage-safe features,
and the deterministic behavior of the predictor and forecast.

Desired enhancements include richer exogenous features, a database-backed
history, authentication, containerization, and production monitoring.

---

## 9. PROGRAMMING LANGUAGE

### 9.1 Python

Python is the primary language of the project. It is used for:

- data cleaning and validation,
- feature engineering,
- model training and evaluation,
- serialization of trained artifacts,
- the Flask backend and REST API,
- inference, including prediction, classification, and forecasting,
- the signal audit and model-comparison analysis,
- the automated test suite.

Python is a suitable choice because the ecosystem includes strong libraries for
data manipulation, machine learning, and web services.

### 9.2 Frontend languages

The frontend is built with standard web technologies that are present in the
repository:

- **HTML** for page structure, including the Jinja2 templates in `templates/`.
- **CSS** for the design system in `static/style.css`.
- **JavaScript** for shared UI helpers in `static/app.js` and for inline page
  behavior in the templates.

The frontend is not the main intellectual content of the project; it exists to
make the models usable through a browser.

---

## 10. LIBRARIES AND FRAMEWORKS

### 10.1 Verified runtime dependencies

The runtime dependencies are declared in `requirements.txt` and installed in the
development environment:

- **Flask** — web framework for the backend and API.
- **pandas** — data loading, cleaning, and manipulation.
- **numpy** — numerical operations used throughout the pipeline.
- **scikit-learn** — model training, evaluation metrics, and preprocessing.
- **joblib** — saving and loading trained models and feature lists.
- **scipy** — used by the signal audit for statistical tests such as the
  chi-square reference distribution for Ljung-Box p-values.
- **gunicorn** — production WSGI server for deployment.

### 10.2 Verified development dependencies

The development and test dependencies are declared in `requirements-dev.txt`:

- **pytest** — test runner for the regression suite.
- **httpx** — HTTP client used by some tests for external-style checks.

### 10.3 Role of each library

- **Flask** routes HTTP requests to page-rendering functions and API handlers.
- **pandas** reads the CSV dataset, computes aggregations, and supports the
  chronological split.
- **numpy** supports vectorized feature computation and metric calculation.
- **scikit-learn** provides the Random Forest, Gradient Boosting, Logistic
  Regression, and Random Forest classifiers, along with metrics such as MAE,
  RMSE, R², precision, recall, F1, and confusion matrix.
- **joblib** persists models and feature lists so the app can load them at
  startup.
- **scipy** supports the statistical reasoning in the signal audit.
- **gunicorn** allows the app to run as a production WSGI service instead of the
  single-threaded development server.

The report does not claim any library was used unless it is actually referenced
by the source code or dependency files.

---

## 11. DATABASE

### 11.1 Current persistence architecture

The project does not currently use a conventional SQL or NoSQL database.

Data and artifacts are stored as files:

- **CSV data**
  - `data/raw/water_consumption_forecasting.csv` — the original real dataset.
  - `data/processed/water_demand_clean.csv` — the cleaned dataset used by the
    pipeline.
- **JSON metadata/metrics**
  - `models/metrics.json` — regression metrics and split metadata.
  - `models/classification_metrics.json` — classification metrics, thresholds,
    and confusion matrix.
  - `models/classification_thresholds.json` — training-set quantile thresholds.
  - `analysis/signal_audit.json` — autocorrelation, white-noise tests, and the
    information ceiling.
  - `analysis/model_comparison.json` — independent model-comparison results.
- **Serialized model artifacts**
  - `models/random_forest_model.joblib`
  - `models/gradient_boosting_model.joblib`
  - `models/classification_model.joblib`
  - `models/feature_list.joblib`
  - `models/classification_feature_list.joblib`
  - `models/recent_history.csv` — per-region recent history used for inference
    features.

This file-based architecture is sufficient for a demonstration and research
project. It keeps the exact inputs, features, models, and metrics together so
that results are reproducible.

### 11.2 Database as future work

If the project later requires persistent user data, forecast logs, or an audit
trail of predictions versus actuals, a database would be appropriate. For
example, a relational database could store daily readings, forecasts, and actual
outcomes so that forecast accuracy can be tracked over time. Such integration is
a recommended future enhancement, not a delivered feature.

---

## 12. DEVELOPMENT TOOLS

The following development tools are used in the project:

- **Python** — main implementation language and environment.
- **venv** — isolated Python environment for dependencies.
- **pip** — package installation.
- **VS Code** — editor used for writing and reviewing code.
- **Git** — version control.
- **GitHub** — remote repository and collaboration.
- **Terminal / shell** — running the application, tests, and analysis scripts.
- **pytest** — automated testing.
- **Flask test client** — in-process testing of the web application.

These tools are used because they are directly supported by the project’s files
and workflow. Any claim about a development tool should be limited to what is
actually evident in the repository.

---

## 13. WEB BROWSER

### 13.1 How users access the application

Users access the application through a standard web browser by pointing it at
the Flask server, for example `http://127.0.0.1:5000/` during local development.

The browser loads HTML pages rendered by Flask/Jinja2 and then runs JavaScript
that calls the backend APIs. The main pages are:

- Dashboard (`/`)
- Prediction (`/prediction`)
- Forecast (`/forecast`)
- Analytics (`/analytics`)
- Model Lab (`/model-performance`)

### 13.2 Frontend–backend interaction

Each page may fetch data from the backend APIs, including:

- dashboard stats,
- analytics data,
- the forecast for a selected region,
- a single prediction for a chosen region and date,
- classification and regression metrics,
- feature importance,
- signal-audit evidence.

The frontend uses these responses to populate KPIs, charts, tables, and demand
badges. The same APIs can also be called directly by other clients.

### 13.3 Browser testing status

Some browser-oriented checks were completed:

- Pages render with HTTP 200 in tests and in direct HTTP checks.
- The inline JavaScript on each page is syntax-checked as part of the test
  suite when Node is available.
- Key UI text and structure is present in the rendered HTML.

The following browser verification remained incomplete at the time of writing:

- full visual, click-path, and layout verification across every page and screen
  size was not completed; the browser automation used for UI checks was
  unreliable in the final environment, so those checks were deprioritized in
  favor of the code, API, and test checks that are easier to verify
  reproducibly.

The project therefore emphasizes server-side and contract verification, while
being explicit that full manual or automated visual browser verification is a
remaining gap rather than a completed task.

---

## 14. METHODOLOGY

The project follows a complete, leakage-aware machine-learning pipeline.

### 14.1 Dataset acquisition and inspection

The real dataset is read from
`data/raw/water_consumption_forecasting.csv`. It is inspected for shape, columns,
regions, date range, missing values, duplicates, and invalid consumption values.

### 14.2 Data cleaning and validation

`data/prepare_data.py` cleans the dataset by:

- parsing dates,
- dropping rows with missing region, date, or consumption,
- removing exact duplicate rows,
- removing duplicate `(region, date)` keys,
- keeping only positive consumption values,
- standardizing column order and types.

The cleaning report on the real file found 0 issues, so 900 rows were written to
`data/processed/water_demand_clean.csv`.

### 14.3 Feature engineering

`src/features.py` builds the feature set identically for training and inference.
The final feature set includes:

- region one-hot columns,
- month,
- day_of_week,
- is_weekend,
- lag_1,
- lag_7,
- rolling_mean_7,
- rolling_mean_14.

Leakage safety is enforced by:

- grouping lag/rolling computations per region,
- shifting before rolling so that a row never uses its own target,
- dropping the initial warm-up rows instead of imputing them.

### 14.4 Chronological training/testing split

A strict chronological 80/20 split is applied per region. The training period is
2023-01-15 to 2023-05-26, and the test period is 2023-05-27 to 2023-06-29. No
random shuffling is used, because this is a time-series problem.

### 14.5 Model training

For regression, the project trains:

- a Baseline naive forecast using lag_1,
- a Random Forest regressor,
- a Gradient Boosting regressor.

For classification, the project trains:

- a Logistic Regression classifier,
- a Random Forest classifier,

using demand-level labels derived from training-set quantiles only.

### 14.6 Baseline comparison and model evaluation

Each model is evaluated on the held-out test window using MAE, RMSE, and R² for
regression, and accuracy, precision, recall, F1, and confusion matrix for
classification. The best regression model is selected by R², and the best
classifier is selected by weighted F1.

### 14.7 Saving and loading trained artifacts

Trained models, feature lists, metrics, thresholds, and recent per-region
history are saved to `models/`. The application loads these artifacts at startup
so that predictions are immediate.

### 14.8 Prediction and recursive multi-day forecasting

The inference module supports:

- single-day prediction for a region and date,
- classification of the demand level,
- a 7-day recursive forecast in which each predicted day is fed back into the
  feature buffer for the next day.

The forecast uses a private working copy of the region’s history so that later
single-day predictions are unaffected.

### 14.9 Demand-level classification

Classification uses the same leakage-safe features as regression and the same
chronological split. Demand levels are defined from training-set consumption
quantiles, so the test set never influences the class thresholds.

### 14.10 API and frontend integration

Flask routes connect the trained models to both HTML pages and JSON APIs. The
pages render the models’ results directly, and the APIs expose the same
functionality for other clients.

### 14.11 Regression testing and verification

A non-mutating test suite verifies:

- dataset shape and cleanliness,
- the warm-up drop,
- the per-region lag construction,
- the past-only rolling window,
- the chronological split boundaries,
- reproduction of stored regression and classification metrics,
- train-only thresholds,
- API endpoints and error paths,
- forecast isolation and determinism,
- and JavaScript syntax on the rendered pages.

---

## 15. IMPLEMENTATION STRUCTURE

### 15.1 Directory structure

The repository is organized as follows:

```text
smartwater-ai/
├── app.py                      # Flask backend and REST API
├── wsgi.py                     # production WSGI entrypoint
├── train_models.py             # regression training and evaluation
├── train_classifier.py         # classification training and evaluation
├── data/
│   ├── prepare_data.py         # real-data cleaning
│   ├── raw/
│   │   └── water_consumption_forecasting.csv   # REAL Kaggle dataset
│   └── processed/
│       └── water_demand_clean.csv              # cleaned dataset
├── src/
│   ├── __init__.py
│   ├── features.py             # leakage-safe feature engineering pipeline
│   └── predict.py              # inference: predict, classify, forecast
├── models/
│   ├── random_forest_model.joblib
│   ├── gradient_boosting_model.joblib
│   ├── classification_model.joblib
│   ├── feature_list.joblib
│   ├── classification_feature_list.joblib
│   ├── metrics.json
│   ├── classification_metrics.json
│   ├── classification_thresholds.json
│   └── recent_history.csv
├── static/
│   ├── style.css
│   └── app.js
├── templates/
│   ├── index.html
│   ├── prediction.html
│   ├── forecast.html
│   ├── analytics.html
│   └── model_performance.html
├── analysis/
│   ├── signal_audit.py
│   ├── signal_audit.json
│   ├── model_comparison.py
│   └── model_comparison.json
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_forecast_isolation.py
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── README.md
└── .gitignore
```

### 15.2 Responsibilities of key files

- **app.py** — creates the Flask application, loads the predictor and metrics at
  startup, and defines the page routes and API endpoints.
- **wsgi.py** — imports the Flask app for production deployment with gunicorn.
- **train_models.py** — builds features, splits the data, trains regression
  models, evaluates them, and saves the best model and artifacts.
- **train_classifier.py** — builds the same features, derives training-set
  quantile thresholds, trains classifiers, evaluates them, and saves the best
  classifier and artifacts.
- **data/prepare_data.py** — cleans the real dataset and writes the processed
  CSV.
- **src/features.py** — defines the leakage-safe feature pipeline and the
  chronological split function.
- **src/predict.py** — defines `DemandPredictor`, which loads the model and
  history and provides `predict`, `classify`, `predict_with_level`, and
  `forecast`.
- **templates/*.html** — the dashboard, prediction, forecast, analytics, and
  model-lab pages.
- **static/style.css** — the design system.
- **static/app.js** — shared UI helpers such as number formatting, badges, chart
  defaults, and micro-interactions.
- **analysis/signal_audit.py** — reproduces the white-noise tests and the
  information ceiling, and writes `analysis/signal_audit.json`.
- **analysis/model_comparison.py** — runs an independent, controlled comparison
  of candidate models and writes `analysis/model_comparison.json`.
- **tests/** — non-mutating regression tests for the pipeline, the API, and the
  forecast isolation behavior.

### 15.3 Data flow

The overall flow is:

1. Raw CSV → cleaned CSV.
2. Cleaned CSV → features + chronological train/test split.
3. Train split → trained models, metrics, thresholds, feature lists.
4. Trained artifacts + recent history → Flask predictor loaded at startup.
5. Flask predictor + metrics → pages and API responses.
6. Browser → pages and API calls → rendered results.

---

## 16. RESULTS AND PERFORMANCE EVALUATION

### 16.1 Regression results

The models are evaluated on the chronological test set (2023-05-27 to
2023-06-29, 170 rows).

| Model | MAE (L/day) | RMSE (L/day) | R² |
|---|---|---|---|
| Baseline (naive lag_1) | 4573.29 | 5595.92 | -0.8572 |
| Random Forest | 3564.73 | 4149.29 | -0.0211 |
| Gradient Boosting | 3705.09 | 4317.58 | -0.1056 |

The Random Forest is the best regression model by R² and is the deployed model.

### 16.2 Classification results

Classification is evaluated on the same test window using training-set quantile
thresholds.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| Logistic Regression | 0.2882 | 0.1437 | 0.2801 | 0.1849 | 0.1853 |
| Random Forest | 0.2235 | 0.2237 | 0.2240 | 0.2100 | 0.2122 |

The best classifier by weighted F1 is the Random Forest classifier.

### 16.3 Understanding the metrics

- **MAE (Mean Absolute Error)** is the average absolute difference between the
  predicted and actual consumption. It is easy to interpret in the original
  units: liters per day.
- **RMSE (Root Mean Squared Error)** is similar to MAE but penalizes larger
  errors more heavily.
- **R² (coefficient of determination)** compares the model’s error to the error
  of a constant predictor that always guesses the test-set mean. A positive R²
  means the model is better than that mean predictor. A negative R² means the
  model is worse than simply guessing the mean.

### 16.4 Why R² is negative here

A negative R² does not mean the code is broken. It means the model’s squared
error on the test set is larger than the squared error of the test-set mean
predictor. In this dataset, that happens because the test window’s consumption
varies around a mean, and the available features do not provide enough
signal to track that variation reliably.

This is not concealed. The project reports the actual negative R² and explains
that it reflects the limitations of the data rather than a software defect.

### 16.5 Constant-predictor reference

As a reference point, a constant predictor that always predicts the training-set
mean achieves approximately MAE 3514.3 on the test window. This is the
strongest simple constant baseline on this data. It is useful as a comparison
point, but it is not claimed to be a universal theoretical limit for all
possible models or datasets.

### 16.6 Independent model comparison

An independent comparison script (`analysis/model_comparison.py`) reproduced the
stored Random Forest metrics on the untouched test window and compared many
candidates using rolling-origin validation inside the training period.

Key findings from that analysis:

- The stored Random Forest metrics were reproduced exactly.
- No candidate’s MAE improvement over the deployed Random Forest was
  statistically significant, because every 95% bootstrap confidence interval of
  the MAE difference included zero.
- The best constant-style candidate was better only by a small margin and would
  be operationally unsuitable as the API forecaster.
- The conclusion was not to force a model replacement.

This supports the deployed model choice and prevents over-interpreting tiny
differences on a small, weak dataset.

### 16.7 Why a weak score must not be disguised

A common temptation in student projects is to inflate or select metrics so that
the result looks better. This project takes the opposite approach: it reports
what the models actually achieved and explains why. That is more useful in a
viva, because it shows understanding of both machine learning and evaluation
discipline.

### 16.8 Why the data limits prediction

The signal audit found that the series are statistically indistinguishable from
white noise:

- Across 15 Ljung-Box tests at lags 7, 14, and 21 for the five regions, only 1
  test rejected white noise at the 0.05 level.
- The expected number of false rejections at that level is 0.75.
- Lag-1 autocorrelations are small across regions.
- Region alone explains only a small share of total variance.

When the data are effectively white noise, no model can consistently beat the
mean by much. The honest conclusion is that the available dataset does not
contain strong temporal signal for the features used.

---

## 17. VERIFICATION AND TESTING

### 17.1 Command used

The test suite is run with:

```bash
python -m pytest tests/ -q
```

### 17.2 Results

The suite passed in the verified environment, covering the pipeline contracts,
API endpoints, forecast isolation, and page script syntax.

### 17.3 API checks

The main endpoints returned successful responses during verification:

- `GET /api/health`
- `GET /api/metrics`
- `GET /api/classification-metrics`
- `GET /api/confusion-matrix`
- `GET /api/dashboard-stats`
- `GET /api/analytics-data`
- `GET /api/feature-importance`
- `GET /api/signal-audit`
- `GET /api/forecast`
- `POST /api/predict`

The prediction endpoint returns deterministic results for the same inputs when
called repeatedly.

### 17.4 Bug-fix verification

A defect in `src/predict.py` was found and fixed: the forecast method was
mutating the shared per-region buffer, which could make later predictions
depend on earlier forecasts. The fix makes the forecast use a private working
copy and restores clean, deterministic behavior. The forecast-isolation tests
cover this behavior.

### 17.5 Known gaps

- Full visual and interactive browser verification was not completed, because
  the browser-automation environment was unreliable for that purpose.
- The application was verified through direct HTTP checks and tests, but not
  through an exhaustive manual walkthrough of every screen.

---

## 18. LIMITATIONS AND FUTURE ENHANCEMENTS

### 18.1 Limitations

1. **Small dataset:** only 900 rows and 180 days per region.
2. **Limited regions:** only five regions are covered.
3. **Weak temporal signal:** the series behave close to white noise, so
   forecasting from lags and calendar features alone is inherently limited.
4. **Missing explanatory variables:** there is no weather, holiday, population,
   or supply-schedule information in the dataset.
5. **Short history:** one spring–summer window is not enough to model multi-year
   seasonality.
6. **Recursive forecast error:** multi-step forecasts can accumulate error over
   the 7-day horizon.
7. **No conventional database:** persistence is file-based.
8. **Incomplete browser verification:** full visual confirmation across all pages
   and screen sizes was not completed.

### 18.2 Future enhancements

1. Collect richer data such as rainfall, temperature, holidays, population, or
   supply-schedule information if available.
2. Gather more observations over longer periods to support seasonal modeling
   where justified.
3. Evaluate seasonal or other time-series models only when the data justify them.
4. Add a database backend for persistent storage of readings, forecasts, and
   actual outcomes.
5. Add production deployment hardening such as containerization, logging,
   monitoring, and authentication if the system is used beyond demonstration.
6. Continue to report results honestly, even when the model is not strong.

These are recommendations, not existing capabilities.

---

## 19. CONCLUSION

SmartWater AI demonstrates a complete and honest machine-learning pipeline for
daily water-consumption forecasting and demand-level classification. It cleans a
real dataset, builds leakage-safe features, splits the data chronologically, trains
and compares regression and classification models, and exposes the results
through both a web interface and REST APIs.

The experiments show that the deployed Random Forest regressor is the best model
among those tested, but that the dataset provides only weak temporal signal. The
result is a negative R² on the test window and classification accuracy near
chance. Rather than hide this, the project documents it, reproduces it, and
explains it through autocorrelation checks, a white-noise analysis, and an
independent model comparison.

The main educational value is therefore not a dramatic accuracy number, but the
process: disciplined data handling, leakage prevention, time-aware evaluation,
reproducibility, and truthful reporting. The project also establishes that the
forecast logic no longer contaminates shared history, and it provides a test
suite that protects the key contractual behaviors.

Future work should focus on richer data, longer histories, and production
hardening where needed. Until then, the project stands as a transparent example
of how to build and evaluate a forecasting system when the data are limited.

---

## 20. REFERENCES

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M., *Time
   Series Analysis: Forecasting and Control*, 5th ed., Wiley, 2015.

2. Breiman, L., “Random Forests”, *Machine Learning*, vol. 45, no. 1, pp.
   5–32, 2001.

3. Friedman, J. H., “Greedy Function Approximation: A Gradient Boosting
   Machine”, *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

4. Hyndman, R. J., and Athanasopoulos, G., *Forecasting: Principles and
   Practice*, 3rd ed., OTexts, 2021.

5. Makridakis, S., Spiliotis, E., and Assimakopoulos, V., “The M4 Competition:
   100,000 Time Series and 61 Forecasting Methods”, *International Journal of
   Forecasting*, vol. 36, no. 1, pp. 54–74, 2020.

6. Scikit-learn developers, *Scikit-learn: Machine Learning in Python*,
   https://scikit-learn.org/.

7. pandas development team, *pandas — powerful Python data analysis toolkit*,
   https://pandas.pydata.org/.

8. Flask authors and contributors, *Flask — documentation*,
   https://flask.palletsprojects.com/.

9. Joblib developers, *joblib — lightweight pipelining with Python*,
   https://joblib.readthedocs.io/.

10. SciPy developers, *SciPy — Open source scientific tools for Python*,
    https://scipy.org/.

---

*End of report.*
