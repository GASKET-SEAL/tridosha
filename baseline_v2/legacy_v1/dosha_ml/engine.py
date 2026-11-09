"""Orchestration: profile creation (Prakriti, fixed) -> daily logs -> Vikriti, forecast, plots."""
import datetime as dt, json
from pathlib import Path
import numpy as np
from .prakriti_model import PrakritiModel
from .food_guna_model import FoodGunaModel
from .vikriti_model import VikritiFilter
from .drivers import compute, ritu_effect, weather_effect, ritu_name
from .logs import DailyLog
from .questionnaire import bmi_bucket
from .compositional import to_xy
from .weather import get_weather, fetch_forecast, climatology
from .visualize import plot_triangle
from .config import MODULES, DOSHAS


class DoshaEngine:
    def __init__(self, prakriti_model: PrakritiModel = None, food_model: FoodGunaModel = None, store_path=None):
        self.prakriti_model = prakriti_model or PrakritiModel().fit()
        self.food = food_model or FoodGunaModel(cache_path="food_llm_cache.json" if store_path else None)
        self.store_path = Path(store_path) if store_path else None
        self.users = json.loads(self.store_path.read_text()) if self.store_path and self.store_path.exists() else {}
        self._filters = {}

    # ---------------------------------------------------------------- profile (one-time)
    def create_profile(self, uid, answers: dict, sex="male", age=None, height_cm=None, weight_kg=None,
                       lat=29.85, lon=77.89, medical=None, adaptive=True):
        bb = bmi_bucket(height_cm, weight_kg) if height_cm and weight_kg else None
        res = self.prakriti_model.predict(answers, bb)
        p0 = [float(x) for x in res["composition"]]
        self.users[uid] = {"prakriti": p0, "prakriti_type": res["type"], "sex": sex, "age": age, "lat": lat, "lon": lon,
                           "medical": medical or {}, "filter": VikritiFilter(p0, adaptive=adaptive).to_dict(),
                           "history": [], "last_date": None}
        self._save()
        return {"prakriti": p0, "type": res["type"], "xy": to_xy(p0).tolist(),
                "explanation": self.prakriti_model.explain(answers, bb)}

    def _filter(self, uid):
        if uid not in self._filters:
            self._filters[uid] = VikritiFilter.from_dict(self.users[uid]["filter"])
        return self._filters[uid]

    # ---------------------------------------------------------------- daily
    def log_day(self, uid, log: DailyLog):
        u = self.users[uid]; f = self._filter(uid)
        last = dt.date.fromisoformat(u["last_date"]) if u["last_date"] else None
        if last and log.date <= last:
            raise ValueError(f"log for {log.date} is not after last logged day {last}")
        if last:                                           # unlogged days: only implicit drivers + decay
            for k in range(1, (log.date - last).days):
                d = last + dt.timedelta(k)
                inp = compute(DailyLog(date=d, weather=climatology(u["lat"], d)), self.food, u["sex"], u["lat"])
                f.step(inp.drivers, {}, 0.0)
        if log.weather is None:
            log.weather = get_weather(u["lat"], u["lon"], log.date)
        inp = compute(log, self.food, u["sex"], u["lat"])
        prev = u["history"][-1]["vikriti"] if u["history"] else None
        st = f.step(inp.drivers, inp.evidence, inp.damping)
        u["history"].append({"date": log.date.isoformat(), "vikriti": st["vikriti"], "deviation": st["deviation"]})
        u["last_date"] = log.date.isoformat()
        u["filter"] = f.to_dict(); self._save()
        st["previous_vikriti"] = prev
        st["drivers_today"] = {m: dict(zip(("v", "p", "k"), np.round(v, 3).tolist())) for m, v in inp.drivers.items()}
        st["notes"] = inp.notes
        return st

    def state(self, uid):
        return self._filter(uid).state()

    # ---------------------------------------------------------------- "dosha weather forecast"
    def forecast(self, uid, days=7):
        u = self.users[uid]; f = self._filter(uid)
        import copy; g = copy.deepcopy(f); out = []
        for d, w in fetch_forecast(u["lat"], u["lon"], days):
            env = weather_effect(w) + 0 * ritu_effect(d)
            inp = compute(DailyLog(date=d, weather=w), self.food, u["sex"], u["lat"])
            st = g.step(inp.drivers, {}, 0.0)
            drive = inp.drivers["weather"] + inp.drivers["ritu"]
            out.append({"date": d.isoformat(), "ritu": ritu_name(d), "weather": f"{w.temp_c:.0f}C/{w.humidity:.0f}%/{w.wind_kmh:.0f}kmh",
                        "pushes_towards": DOSHAS[int(np.argmax(drive - drive.mean()))], "projected_vikriti": st["vikriti"]})
        return out

    def plot(self, uid, path="triangle.png"):
        u = self.users[uid]; f = self._filter(uid); h = u["history"]
        return plot_triangle(f.p0, f.composition(), previous=h[-2]["vikriti"] if len(h) > 1 else None,
                             trail=[x["vikriti"] for x in h[-14:]], ellipse=f.ellipse(), path=path,
                             title=f"Prakriti {u['prakriti_type']}")

    def _save(self):
        if self.store_path:
            self.store_path.write_text(json.dumps(self.users))
