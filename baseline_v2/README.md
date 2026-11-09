# Dosha Tracker v2: Prakriti · Vikriti · Ritu (built to the 5 Nov plan)

    pip install -r requirements.txt
    pytest -q tests                      # 13 tests
    python scripts/sign_tests.py         # 5 classical sign tests
    python scripts/validate_mendeley.py  # every data result in the report
    uvicorn dosha.api:app                # service;  open web/dosha-tracker.html for the browser version

## Layout
| Path | Role |
| --- | --- |
| `dosha/simplex.py` | Aitchison geometry of (Vata, Pitta, Kapha): clr, ilr, perturbation, distance, triangle coordinates |
| `dosha/prakriti.py` | `ExpertPrakriti` (transparent Dirichlet scoring), `DataPrakriti` (softmax on Mendeley), cross-cohort and out-of-sample checks |
| `dosha/drivers.py` | daily log -> pushes: sleep band 7-9 h, mood, stress, dosha-clock energy, Agni, MET-based activity, screen/study, herbs, weather anomaly, ritu blend |
| `dosha/vikriti.py` | state update `x_t = center(c tanh((rho x_{t-1} + f)/c))`, Vikriti = closure(p exp(x)), Monte-Carlo 80% region |
| `dosha/herbs.py` | herb effects read from the Amidha database (360 herbs, CC BY 4.0) |
| `dosha/engine.py`, `api.py` | profile (Prakriti fixed), logging with gap handling, forecast, FastAPI |
| `data/` | Mendeley cohorts, herb database, Compendium 2024 subset, attributions, inferred feature dictionary |
| `web/dosha-tracker.html` | working browser app using the same equations |
| `legacy_v1/` | first design (food model, Kalman filter); superseded by the 5 Nov plan, kept for history |

## Results (reproduce with the scripts)
* Softmax trained on one cohort, tested on the other: West->North 97.7 % (Vata 100, Pitta 92, Kapha 100), North->West 88.5 %.
* Transparent equal-weight vote counting with rules learned on one cohort: 90.7 % and 86.3 % on the other.
* 5/5 classical sign tests; 13/13 unit tests.

## Honest limits
* Mendeley data = 217 physician-labelled people, **extreme types only**; labels come from the same questionnaire family, so models learn the physicians' rule. Shares for mixed types are not validated.
* Mendeley columns are anonymous (F1...). `feature_dictionary_INFERRED.csv` is my reading of the answer sets; confirm against the paper's S1 Table.
* All Vikriti coefficients (guna matrix, rho, input effects) are drafts for the practitioner panel; Vikriti has no ground truth until a pilot is run.
* Weather: Open-Meteo with seasonal fallback (live call not exercised in the build sandbox).
* Not a medical device; health-flag users get herb effects disabled and a physician notice.
