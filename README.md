# Mathematical Modeling of Tridosha Homeostasis Using Dynamical Systems and Feedback Control Theory

A locally runnable research project: coupled ordinary differential equations, fixed compositional equilibrium, bounded negative feedback, Python simulation and a light-mode interactive website. The ODE model is illustrative and is not clinically calibrated.

## Run the website

Tested with Python 3.14.2 on Windows. Install Python 3.14, then open a terminal in this extracted folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m lab.server
```

Open http://127.0.0.1:8000 in a browser. Keep the terminal running. Stop with Ctrl+C. No Node build, external fonts, API key, database or cloud account is required. Do not open index.html directly: the website calls the Python server.

On macOS/Linux, substitute `.venv/bin/python` for `.venv\Scripts\python`. The pinned versions were exercised on Windows/Python 3.14; other operating systems have not been tested here. An internet connection is needed for the initial dependency installation only.

## Reproduce

```powershell
.venv\Scripts\python -m pytest tests -q --junitxml=results/tests.xml
.venv\Scripts\python -m lab.experiments
```

Experiments overwrite derived results in `results/`. Fixed seeds make calibration and cohort folds reproducible. Minor floating point differences across platforms are possible. API documentation is at http://127.0.0.1:8000/docs.

## Contents

- `lab/model.py`: centered ODEs, bounded controller, simplex mapping, integration, stability and recovery metrics.
- `lab/server.py`: validated local FastAPI endpoints and static website serving.
- `lab/experiments.py`: analytic comparison, controller sweep, synthetic parameter recovery, cohort ML benchmark.
- `web/`: HTML, CSS and JavaScript. All scientific calculations run in Python.
- `tests/`: deterministic model and API regression tests.
- `results/`: measured numerical results, trajectories, test evidence and environment record.
- `report/`: detailed project report and supporting figures.
- `baseline_v2/`: supplied project preserved for traceability, including its original data attribution and legacy code. It is not imported by the new web service. Its original application remains separate and is not certified as production-ready.

## Scope and interpretation

The active website is an ODE research laboratory aligned to the project title. The original questionnaire, daily logs, weather, seasonal and herb modules remain in the baseline archive, not as new web product features. No new health tracking, food analysis, NLP, treatment recommender, account system or medical claims were added.

The new model improves mathematical structure and experimental reproducibility. It does not demonstrate higher clinical predictive accuracy. The ML code benchmarks extreme Prakriti classification separately; cohort data cannot calibrate recovery dynamics. See the report for exact gains and tradeoffs, including the lower accuracy of one log-loss-selected classifier.

The server binds only to localhost. It stores no personal profiles. Export downloads the last completed experiment, including its actual settings. Changed settings require Run simulation. There is no authentication or production hosting configuration; do not expose the development server publicly.

## Data and attribution

Original dataset notices are retained under `baseline_v2/data/`. Mendeley cohort attribution: Tiwari et al., doi:10.17632/dyx5cgxdrf.1, CC BY 4.0 as documented in the supplied archive. Feature names are not treated as a verified clinical questionnaire dictionary. No new patient data was collected. The original code's license was not specified; this package does not assert a new license over supplied material.
