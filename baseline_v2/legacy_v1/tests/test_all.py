import sys, datetime as dt
sys.path.insert(0, ".")
import numpy as np
import pytest
from dosha_ml import compositional as C, questionnaire as Q
from dosha_ml.prakriti_model import PrakritiModel, SoftmaxCompositionModel
from dosha_ml.food_guna_model import FoodGunaModel
from dosha_ml.vikriti_model import VikritiFilter
from dosha_ml.drivers import compute, weather_effect, food_effect, ritu_effect
from dosha_ml.logs import DailyLog, from_preset
from dosha_ml.weather import Weather
from dosha_ml.engine import DoshaEngine
from dosha_ml.simulate import monte_carlo


@pytest.fixture(scope="module")
def food(): return FoodGunaModel(use_llm=False)

@pytest.fixture(scope="module")
def pm(): return PrakritiModel().fit()


# ---- geometry
def test_simplex_roundtrip():
    p = np.array([.5, .3, .2])
    assert np.allclose(C.clr_inv(C.clr(p)), p) and np.allclose(C.ilr_inv(C.ilr(p)), p)
    assert abs(C.perturb(p, [1, -.3, 2]).sum() - 1) < 1e-12

def test_ilr_is_isometry():
    a, b = np.array([.6, .3, .1]), np.array([.2, .2, .6])
    assert np.isclose(np.linalg.norm(C.ilr(a) - C.ilr(b)), C.aitchison_dist(a, b))

# ---- prakriti
def test_prakriti_outputs_simplex_and_tracks_answers(pm):
    allv = pm.predict({k: 0 for k in Q.KEYS if k != "bmi"})["composition"]
    allk = pm.predict({k: 2 for k in Q.KEYS if k != "bmi"})["composition"]
    assert abs(allv.sum() - 1) < 1e-9 and allv.argmax() == 0 and allk.argmax() == 2 and allv.min() >= 0.05 - 1e-9

def test_prakriti_handles_missing_answers(pm):
    r = pm.predict({"frame": 0})
    assert r["answered"] == 1 and abs(r["composition"].sum() - 1) < 1e-9

def test_softmax_beats_count_baseline():
    X, Y = Q.generate_synthetic(3000, 1); Xt, Yt = Q.generate_synthetic(1000, 2)
    m = SoftmaxCompositionModel(100).fit(X, Y)
    base = Xt.reshape(len(Xt), Q.N_Q, 3).sum(1) + .5; base /= base.sum(1, keepdims=True)
    assert np.abs(m.predict(Xt) - Yt).mean() < np.abs(base - Yt).mean()

# ---- food
def test_food_typos_and_compound(food):
    assert food.predict("panner")["items"][0]["match"] == "paneer"
    assert [i["match"] for i in food.predict("dal chawal, curd and salad")["items"]][1:] == ["curd", "cucumber"]
    assert food.predict("xyzzy qwerty") is None

def test_food_direction(food):
    eff = lambda t: food_effect(**{k: food.predict(t)[k] for k in ("gunas", "rasas")})
    assert np.argmax(eff("ice cream")) == 2                 # cold heavy sweet -> Kapha
    assert np.argmax(eff("green chilli")) == 1              # hot sharp -> Pitta
    assert np.argmax(eff("coffee")) == 0                    # dry stimulating -> Vata

# ---- drivers / weather
def test_weather_direction():
    assert np.argmax(weather_effect(Weather(8, 25, 30))) == 0      # cold dry windy -> Vata
    assert np.argmax(weather_effect(Weather(40, 25, 10))) == 1     # hot -> Pitta
    assert np.argmax(weather_effect(Weather(12, 85, 3, 3))) == 2   # cool damp -> Kapha

def test_blank_log_only_implicit(food):
    inp = compute(DailyLog(date=dt.date(2026, 11, 5)), food)
    assert set(inp.drivers) == {"weather", "ritu"} and inp.evidence == {}

# ---- vikriti
def test_no_input_stays_at_prakriti():
    f = VikritiFilter([.5, .3, .2])
    for _ in range(20): st = f.step({}, {})
    assert np.allclose(st["vikriti"], [.5, .3, .2]) and st["deviation"] < 1e-9

def test_returns_to_prakriti_after_shock():
    f = VikritiFilter([.4, .3, .3], adaptive=False)
    for _ in range(5): f.step({"stress": np.array([.6, 0, 0])}, {})
    peak = f.state()["deviation"]
    for _ in range(25): f.step({}, {})
    assert peak > .2 and f.state()["deviation"] < .05

def test_prakriti_is_never_overwritten():
    f = VikritiFilter([.4, .3, .3]); p = f.p0.copy()
    for _ in range(30): f.step({"food": np.array([0, .5, 0])}, {"agni": np.array([0, .5, 0])})
    assert np.array_equal(f.p0, p)

def test_symptoms_pull_estimate_and_shrink_uncertainty():
    f = VikritiFilter([1/3, 1/3, 1/3]); u0 = f.state()["uncertainty_sd"]
    for _ in range(5): f.step({}, {"agni": np.array([.5, 0, 0]), "bowel": np.array([.5, 0, 0])})
    st = f.state()
    assert st["aggravated"] == "vata" and st["uncertainty_sd"] < u0 + 1e-9

def test_susceptibility_like_increases_like():
    v = VikritiFilter([.7, .2, .1], adaptive=False); k = VikritiFilter([.1, .2, .7], adaptive=False)
    for f in (v, k): f.step({"stress": np.array([.5, 0, 0])}, {})
    assert v.state()["deviation"] > k.state()["deviation"]

def test_filter_serialisation_roundtrip():
    f = VikritiFilter([.5, .3, .2]); f.step({"food": np.array([.2, 0, 0])}, {"mood": np.array([.5, 0, 0])})
    g = VikritiFilter.from_dict(f.to_dict())
    assert np.allclose(f.z, g.z) and np.allclose(f.P, g.P)

def test_simulation_adaptive_beats_baselines():
    r = monte_carlo(8, 80)
    assert r["KF adaptive (ours)"] < r["KF fixed gains"] < r["open-loop"]

# ---- engine
def test_engine_end_to_end(pm, food, tmp_path):
    eng = DoshaEngine(pm, food, store_path=tmp_path / "u.json")
    r = eng.create_profile("u", {k: 1 for k in Q.KEYS if k != "bmi"}, sex="female", height_cm=165, weight_kg=60)
    p0 = r["prakriti"]
    w = Weather(20, 50, 8)
    day = dt.date(2026, 11, 1)
    s1 = eng.log_day("u", from_preset("exam_day", date=day, weather=w, meals=[(9, "chai"), (14, "pizza")]))
    s2 = eng.log_day("u", DailyLog(date=day + dt.timedelta(3), weather=w))      # 2 unlogged days between
    assert s2["previous_vikriti"] == s1["vikriti"] and eng.users["u"]["prakriti"] == p0
    with pytest.raises(ValueError):
        eng.log_day("u", DailyLog(date=day))
    eng.plot("u", str(tmp_path / "t.png")); assert (tmp_path / "t.png").exists()
    eng2 = DoshaEngine(pm, food, store_path=tmp_path / "u.json")                   # persistence
    assert np.allclose(eng2.state("u")["vikriti"], s2["vikriti"])


def test_api_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient
    from dosha_ml import api
    c = TestClient(api.app)
    assert len(c.get("/questionnaire").json()) == Q.N_Q - 1
    r = c.post("/profile", json={"user_id": "a", "answers": {"frame": 0, "skin": 1}, "sex": "male", "height_cm": 170, "weight_kg": 60})
    assert r.status_code == 200 and abs(sum(r.json()["prakriti"]) - 1) < 1e-9
    r = c.post("/log", json={"user_id": "a", "date": "2026-11-05", "preset": "exam_day", "meals": [[9, "chai"]]})
    assert r.status_code == 200 and "vikriti" in r.json()
    assert c.post("/log", json={"user_id": "a", "date": "2026-11-05"}).status_code == 409
    assert c.get("/triangle/a.png").status_code == 200
