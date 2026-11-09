import sys, datetime as dt
sys.path.insert(0, ".")
import numpy as np, pytest
from dosha import Engine, DailyLog, from_preset
from dosha.simplex import closure, clr, clr_inv, aitchison_dist
from dosha.prakriti import ExpertPrakriti, cross_cohort, transparent_oos, ptype, QUESTIONS
from dosha.vikriti import run, DayRecord
from dosha.drivers import ritu_vector, life_push, env_push
from dosha.weather import Weather
from dosha import herbs
import scripts.sign_tests as st


def test_simplex():
    p = np.array([.5, .3, .2]); assert np.allclose(clr_inv(clr(p)), p)

def test_sign_tests_all_pass():
    assert all(ok for _, ok in st.tests())

def test_expert_prakriti_follows_answers():
    s = ExpertPrakriti(); k = [q[0] for q in QUESTIONS]
    assert s.predict({x: 0 for x in k})["composition"].argmax() == 0 and s.predict({x: 2 for x in k})["composition"].argmax() == 2
    assert abs(s.predict({})["composition"].sum() - 1) < 1e-9

def test_cross_cohort_matches_paper_level():
    r = cross_cohort(); assert r["West->North"]["accuracy"] > .93 and r["North->West"]["accuracy"] > .83

def test_transparent_scoring_close_to_ml():
    r = transparent_oos(); assert all(v["accuracy"] > .82 for v in r.values())

def test_blank_log_only_implicit():
    lp = life_push(DailyLog()); assert lp["n"] == 0 and not lp["fl"].any()

def test_sleep_band_is_7_to_9():
    assert not life_push(DailyLog(sleep_hours=8))["fl"].any() and life_push(DailyLog(sleep_hours=5))["fl"].any()

def test_ritu_blend_is_continuous():
    a, b = ritu_vector(dt.date(2026, 5, 12)), ritu_vector(dt.date(2026, 5, 16)); assert np.abs(a - b).max() < .5

def test_herb_lookup_matches_database():
    v, ok = herbs.effect("Ashwagandha"); assert ok and v[0] < 0 and v[2] < 0 and v[1] > 0     # pacifies V,K; aggravates P
    assert herbs.is_tridoshic("Amla")

def test_prakriti_not_changed_by_logs(tmp_path):
    e = Engine(tmp_path / "u.json"); r = e.create_profile("u", {"frame": 0, "sleep": 0}, age=21)
    for i in range(3): e.log_day("u", from_preset("exam", date=dt.date(2026, 11, 1 + i), weather=Weather(20, 50, 8, 0)))
    assert e.users["u"]["p"] == r["prakriti"]

def test_engine_gap_and_persistence(tmp_path):
    e = Engine(tmp_path / "u.json"); e.create_profile("u", {"frame": 1}, flags=["diabetes"])
    w = Weather(20, 50, 8, 0)
    s1 = e.log_day("u", from_preset("normal", date=dt.date(2026, 11, 1), weather=w, self_care=["Ashwagandha"]))
    s2 = e.log_day("u", DailyLog(date=dt.date(2026, 11, 5), weather=w))
    assert s2["previous"] == s1["vikriti"] and "region80" in s2
    assert Engine(tmp_path / "u.json").state("u")["vikriti"] == s2["vikriti"]

def test_api_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient
    from dosha import api
    c = TestClient(api.app)
    assert len(c.get("/questionnaire").json()) == len(QUESTIONS)
    assert c.post("/profile", json={"user_id": "a", "answers": {"frame": 0}}).status_code == 200
    r = c.post("/log", json={"user_id": "a", "date": "2026-11-05", "preset": "exam"}); assert r.status_code == 200 and "vikriti" in r.json()


def test_forecast_runs_and_widens(tmp_path):
    e = Engine(); e.create_profile("u", {"frame": 0, "skin": 1}, age=21)
    e.log_day("u", from_preset("normal", date=dt.date(2026, 11, 8), weather=Weather(22, 50, 8, 0)))
    f = e.forecast("u", 5, mc=60)
    assert len(f) == 5 and all(abs(sum(x["vikriti"]) - 1) < 1e-9 for x in f)
    assert f[-1]["region80"]["rx"] * f[-1]["region80"]["ry"] >= f[0]["region80"]["rx"] * f[0]["region80"]["ry"]
