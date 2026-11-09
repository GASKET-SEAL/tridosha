"""Profile (fixed Prakriti) -> daily logs -> Vikriti, 80% region, forecast. JSON persistence."""
import datetime as dt, json
from pathlib import Path
import numpy as np
from .prakriti import ExpertPrakriti, ptype
from .logs import DailyLog
from .drivers import life_push, env_push, ritu_index
from .vikriti import DayRecord, run, monte_carlo, ellipse80, severity
from .simplex import to_xy
from .weather import get_weather, climatology, fetch_forecast
from . import config as C

SAFETY = ["pregnant_or_breastfeeding", "diabetes", "heart_or_bp", "kidney_or_liver", "regular_medication", "mental_health_diagnosis"]


class Engine:
    def __init__(self, store=None):
        self.scorer = ExpertPrakriti(); self.store = Path(store) if store else None
        self.users = json.loads(self.store.read_text()) if self.store and self.store.exists() else {}

    def create_profile(self, uid, answers, age=None, sex=None, flags=(), lat=29.85, lon=77.89):
        r = self.scorer.predict(answers, age)
        self.users[uid] = dict(alpha=r["alpha"].tolist(), p=r["composition"].tolist(), comfort=r["comfort_c"], flags=list(flags),
                               sex=sex, lat=lat, lon=lon, hist=[], emean=0.0)
        self._save()
        return dict(prakriti=r["composition"].tolist(), type=ptype(r["composition"]), answered=r["answered"],
                    safety_notice=bool(flags))

    def _recs(self, u):
        return [DayRecord(dt.date.fromisoformat(h["date"]), np.array(h["fl"]), np.array(h["fe"]), h["sStar"], h["chi"], h["damping"], h["n"]) for h in u["hist"]]

    def log_day(self, uid, log: DailyLog):
        u = self.users[uid]
        if log.weather is None: log.weather = get_weather(u["lat"], u["lon"], log.date)
        lp = life_push(log, u["emean"], herbs_allowed=not u["flags"])
        fe = env_push(log.weather, u["comfort"], log.date)
        u["hist"] = [h for h in u["hist"] if h["date"] != log.date.isoformat()]
        u["hist"].append(dict(date=log.date.isoformat(), fl=lp["fl"].tolist(), fe=fe.tolist(), sStar=lp["sStar"], chi=lp["chi"],
                              damping=lp["damping"], n=lp["n"]))
        em = [v for v in (log.energy_morning, log.energy_midday, log.energy_late) if v is not None]
        if em: u["emean"] = .8 * u["emean"] + .2 * float(np.mean(em))
        self._save()
        return self.state(uid) | dict(logged_fields=lp["n"], unknown_herbs=lp["unknown_herbs"])

    def state(self, uid, mc=150):
        u = self.users[uid]; recs = self._recs(u); p = np.array(u["p"])
        out = run(recs, p); d, x, v = out[-1]; prev = out[-2][2] if len(out) > 1 else None
        dev = float(np.linalg.norm(x)); vs = monte_carlo(recs, np.array(u["alpha"]), mc)
        conf = recs[-1].n / 15 if recs else 0
        return dict(date=d.isoformat(), prakriti=p.tolist(), vikriti=v.tolist(), previous=None if prev is None else prev.tolist(),
                    xy=to_xy(v).tolist(), deviation=dev, severity=severity(dev),
                    leaning=C.DOSHAS[int(np.argmax(x))] if dev >= .15 else None, region80=ellipse80(vs),
                    confidence="high" if conf >= .6 else "medium" if conf >= .3 else "low")

    def forecast(self, uid, days=7, mc=100, seed=0):
        """Project the next days with the user's typical routine (mean of last 7 life pushes) and forecast weather.
        Weather noise grows with sqrt(lead time); parameters are resampled as in state()."""
        from .weather import Weather
        u = self.users[uid]; recs = self._recs(u); p = np.array(u["p"]); rng = np.random.default_rng(seed)
        last = recs[-1].date if recs else dt.date.today()
        typical = np.mean([r.fl for r in recs[-7:]], 0) if recs else np.zeros(3)
        wx = fetch_forecast(u["lat"], u["lon"], days)

        def future(k, noise):
            out = []
            for i in range(k + 1):
                d, w = wx[i]
                if noise:
                    s = np.sqrt(i + 1)
                    w = Weather(w.temp_c + rng.normal() * 1.5 * s, float(np.clip(w.humidity + rng.normal() * 6 * s, 0, 100)),
                                max(0.0, w.wind_kmh + rng.normal() * 3), w.rain_mm, w.source)
                out.append(DayRecord(last + dt.timedelta(i + 1), typical, env_push(w, u["comfort"], d)))
            return recs + out

        res = []
        for k, (d, _) in enumerate(wx):
            v = run(future(k, False), p)[-1][2]; pts = []
            for _ in range(mc):
                rho = np.clip(C.RHO * (1 + .15 * rng.normal(size=3)), .3, .95)
                pts.append(run(future(k, True), rng.dirichlet(u["alpha"]), rho, C.BETA * (1 + .3 * rng.normal()))[-1][2])
            res.append(dict(date=d.isoformat(), ritu=C.RITU_NAMES[ritu_index(d)], vikriti=v.tolist(), region80=ellipse80(np.array(pts))))
        return res

    def _save(self):
        if self.store: self.store.write_text(json.dumps(self.users))
