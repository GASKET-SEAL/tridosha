# Dosha ML – Prakriti · Vikriti · Dosha Triangle

```
dosha_ml/
  config.py            all Ayurvedic constants (guna signatures, rasa table, ritu, inertia) – expert-editable
  compositional.py     simplex geometry: clr / ilr / perturbation / Aitchison distance / triangle xy
  questionnaire.py     21-question Prakriti form, encoder, synthetic generator
  prakriti_model.py    MODEL 1  soft-label softmax regression  (+ benchmark vs RF / GBM / rules)
  food_guna_model.py   MODEL 2  food text -> 10 gunas + 6 rasas (char-ngram retrieval + LLM fallback)
  drivers.py           daily log -> causes (drivers) and symptoms (evidence); weather & ritu
  vikriti_model.py     MODEL 3  adaptive Kalman filter on the simplex  (the mathematical model)
  engine.py / api.py   profile store, daily logging, forecast, plots / FastAPI service
  simulate.py          virtual-user simulator that validates the filter
scripts/  train_prakriti.py  demo.py        tests/test_all.py (18 tests)
```
Quick start: `pip install -r requirements.txt && python scripts/demo.py && pytest -q tests`
API: `uvicorn dosha_ml.api:app` → `/questionnaire /profile /log /state/{id} /forecast/{id} /triangle/{id}.png`

## 1. The mathematics

**Doshas are compositions.** (V,P,K) is positive and sums to 1 – only ratios matter – so we work in Aitchison
geometry: `clr(p)=log p − mean(log p)`, `ilr(p)=Vᵀ-projection to R²` (isometry), perturbation
`p ⊕ x = closure(p·eˣ)`, distance `d_A(p,q)=‖clr p − clr q‖`.

**Prakriti** `p0 ∈ simplex` – fixed reference, written once, never updated by daily logs.

**Vikriti** is the Prakriti perturbed by a log-excess `x_t` (2 free numbers `s_t`):

    Vikriti_t = p0 ⊕ x_t = closure( p0 · exp(x_t) ),   x_t = Vᵀ s_t
    s_t = 0  ⇔  Vikriti = Prakriti;      deviation_t = ‖x_t‖ = d_A(Vikriti_t, p0)

**Dynamics** (homeostasis + daily causes):

    s_t = Φ_t s_{t−1} + Σ_j θ_j C_{j,t} + w_t ,     w_t ~ N(0,Q)
    Φ_t = V diag(ρ(1−damping_t)) Vᵀ ,   ρ = (0.55, 0.65, 0.80)      Vata fast, Kapha slow (chala vs sthira/guru)
    C_{j,t} = V[(1−ρ) ⊙ S ⊙ c_{j,t}] ,   S_d = clip(1 + γ(3p0_d − 1), .5, 1.5)   "like increases like"

`c_j` is the raw dosha push (log-units) of module j ∈ {food, meal timing, sleep, exercise, stress, screen/study,
self-care, weather, ritu, cycle}. Triphala/brahmi/happy mood raise the return speed (damping).

**Guna alignment** (food, weather) – one principle: `effect_d = A·⟨Ĝ_d , f⟩/√10` where `f∈[−1,1]¹⁰` are the
guna values of the item and `Ĝ_d` the unit guna signature of dosha d (config.py). Food adds the classical rasa table.
**Ritu**: circular interpolation of ritu vectors between mid-dates. **Weather**: temperature→sheeta/ushna,
humidity→snigdha/ruksha, wind→chala, rain→picchila, then the same alignment.

**Measurements** – symptoms (agni, bowel, mood, energy, sleep quality), each possibly missing:
`y_k = V e_k = s_t + v_k`, `v_k ~ N(0, σ_k² I)`.

**Learning (ML).** Module gains θ_j (prior 1, sd 0.3) are appended to the state and follow a slow random walk:
`z=[s,θ]`, `F=[[Φ, C],[0, I]]`. The Kalman filter then personalises each user's sensitivities online, O(1) per day,
handles blank fields natively (no measurement ⇒ no update), and its covariance gives the 95 % ellipse on the triangle.
Unlogged days run with implicit drivers only (weather, ritu) and decay.

## 2. Why these models (efficiency)
| Need | Model | Why |
|---|---|---|
| Prakriti as a point in the triangle | soft-label softmax regression, L-BFGS | outputs a simplex natively, convex, 66×3 weights, 0.03 s training, 0.0005 ms/prediction, handles skipped questions, per-answer explanations. On the synthetic benchmark it is best on share error and ~200× faster to train than Random Forest. |
| Food name → gunas | char-n-gram TF-IDF k-NN regression + modifiers (+ optional LLM) | ~1 ms, offline, typo/synonym tolerant (90 % top-1 under single-typo noise), no GPU; LLM only for unknown foods, cached. |
| Vikriti | linear-Gaussian adaptive Kalman filter on ilr coordinates | exact Bayesian inference, 12×12 matrices, missing data, uncertainty, personalisation. In simulation: error 0.19 (open-loop) → 0.12 (fixed-gain KF) → 0.10 (adaptive). |

## 3. Honest limitations – read before presenting results
1. **No real labelled data was supplied.** Prakriti ships trained on *synthetic* data generated from the questionnaire's
   structure; the benchmark only shows the code is sound. Retrain on a real dataset: `python scripts/train_prakriti.py data.csv label`
   (labels V,P,K,VP,PK,VK,VPK are converted to soft targets) and re-run `benchmark()` – do not quote synthetic accuracies.
2. **Guna/rasa values, effect sizes, ρ, noise levels and severity cut-offs are expert-style priors**, not values taken
   from a specific paper. They all sit in `config.py`, `drivers.py` and `food_guna_model.py` with the intent that you
   replace them with the equations you finalise from your literature review; have an Ayurvedic practitioner review the food lexicon.
3. The simulator validates the **inference machinery** (that the filter recovers a hidden state), not that the
   Ayurvedic mapping is physiologically correct. Clinical validation needs real users + a vikriti ground truth.
4. Symptom evidence is assumed relative to the person's *usual* (the form must phrase it "compared with your normal").
5. Open-Meteo fetch has a climatology fallback; it was not exercised against the live API in the build sandbox.
6. This is a wellness/educational tool; the medical fields (conditions, medications, allergies) are stored but not used
   for recommendations – add a safety-screening layer before giving diet advice.
