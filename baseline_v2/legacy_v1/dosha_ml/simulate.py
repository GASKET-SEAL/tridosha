"""
Synthetic-user simulator, used to VALIDATE the filter mechanics (not Ayurveda itself):
a virtual user has hidden per-module sensitivities theta_true != 1 and a hidden true dosha state; the
logs/symptoms are generated from it. We then check that the filter recovers the hidden state and
learns theta better than (a) open-loop prediction and (b) a non-adaptive filter.
"""
import datetime as dt
import numpy as np
from .compositional import V_BASIS as V, aitchison_dist
from .config import MODULES
from .drivers import compute
from .logs import DailyLog, MOODS, BOWEL, AGNI
from .vikriti_model import VikritiFilter, susceptibility
from .food_guna_model import FoodGunaModel
from .weather import Weather, climatology

FOODS = ["dal chawal", "paneer", "samosa", "ice cream", "khichdi", "salad", "pizza", "fruit", "roti sabzi",
         "spicy chicken curry", "curd rice", "coffee", "poha", "sweets", "moong dal", "maggi", "idli", "banana"]
TYPES = ["strength", "cardio", "sports", "yoga", "walking", "hiit", "pranayama"]


def random_log(rng, day, p_blank=0.15):
    def maybe(v): return None if rng.random() < p_blank else v
    meals = [(8.5, rng.choice(FOODS)), (13.0, rng.choice(FOODS)), (20 + 2 * rng.random(), rng.choice(FOODS))]
    return DailyLog(
        date=day, energy=maybe(rng.random()), sleep_quality=maybe(rng.random()), stress=maybe(rng.random()),
        sleep_hours=maybe(4.5 + 5 * rng.random()), bedtime_hour=maybe(21.5 + 4 * rng.random()),
        mood=maybe(str(rng.choice(MOODS))), bowel=maybe(str(rng.choice(BOWEL))), agni=maybe(str(rng.choice(AGNI))),
        meals=maybe(meals), activities=maybe([(str(rng.choice(TYPES)), float(rng.integers(15, 90)), float(rng.random()))]),
        screen_hours=maybe(12 * rng.random()), study_hours=maybe(10 * rng.random()),
        weather=Weather(10 + 28 * rng.random(), 30 + 65 * rng.random(), 20 * rng.random(), float(rng.random() < .2) * 8))


def run(seed=0, days=90, p0=(.5, .3, .2), food=None):
    rng = np.random.default_rng(seed)
    food = food or FoodGunaModel(use_llm=False)
    theta_true = np.clip(np.exp(rng.normal(0, 0.45, len(MODULES))), 0.3, 2.5)
    base = VikritiFilter(p0)
    fa = VikritiFilter(p0, adaptive=True)              # adaptive + symptoms
    fn = VikritiFilter(p0, adaptive=False)             # fixed gains + symptoms
    fo = VikritiFilter(p0, adaptive=False)             # open loop (drivers only)
    s_true = np.zeros(2)
    err = {"open-loop": [], "KF fixed gains": [], "KF adaptive (ours)": []}
    day0 = dt.date(2026, 1, 1)
    for t in range(days):
        log = random_log(rng, day0 + dt.timedelta(t))
        inp = compute(log, food)
        C = base._C(inp.drivers)
        Phi = base._phi(inp.damping)
        s_true = Phi @ s_true + C @ theta_true + rng.normal(0, 0.05, 2)
        ev = {ch: (s_true + rng.normal(0, base_sigma(ch), 2)) @ V for ch in inp.evidence}
        fa.step(inp.drivers, ev, inp.damping); fn.step(inp.drivers, ev, inp.damping); fo.step(inp.drivers, {}, inp.damping)
        if t >= 15:
            truth = base.composition(s_true)
            for k, f in (("open-loop", fo), ("KF fixed gains", fn), ("KF adaptive (ours)", fa)):
                err[k].append(float(aitchison_dist(f.composition(), truth)))
    return {k: float(np.mean(v)) for k, v in err.items()}, theta_true, fa


def base_sigma(ch):
    from .config import EVIDENCE_SIGMA
    return EVIDENCE_SIGMA[ch]


def monte_carlo(n_users=20, days=90):
    food = FoodGunaModel(use_llm=False)
    rows = [run(s, days, food=food)[0] for s in range(n_users)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
