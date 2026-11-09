"""
From a DailyLog to the two quantities the state-space model needs:

  DRIVERS  c_j in R^3 (log-units, [V,P,K]) : what today's *causes* push each dosha towards.
  EVIDENCE e_k in R^3                      : what today's *symptoms* say about the current dosha state.

Causes: food (gunas/rasas), meal timing, sleep, exercise, stress, screen/study, self-care,
        weather, ritu (season), menstrual cycle.
Symptoms: agni, bowel, mood, energy, sleep quality.

Missing fields contribute nothing (no driver, no evidence) - the filter simply gets less information.
Food, weather and ritu all use ONE principle, guna alignment:
        effect_d = A * <G_d , f> / sqrt(10)        (f = guna vector of the thing, G_d = unit signature of dosha d)
i.e. a thing aggravates the dosha it resembles and pacifies the one it opposes (samanya-vishesha).
"""
import datetime as dt
from dataclasses import dataclass, field
import numpy as np
from .config import G_UNIT, R_EFFECT, RITU_TABLE, GUNA_AXES
from .logs import DailyLog
from .weather import Weather, climatology

SQ10 = np.sqrt(10.0)
A_GUNA, A_RASA, A_WEATHER = 0.80, 0.25, 1.20


def ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def vec(v=0.0, p=0.0, k=0.0):
    return np.array([v, p, k], float)


# ---------------------------------------------------------------- individual modules ----------
def food_effect(gunas, rasas):
    r = rasas / max(1.0, rasas.sum())
    return A_GUNA * (G_UNIT @ gunas) / SQ10 + A_RASA * (R_EFFECT @ r)

def weather_effect(w: Weather):
    f = np.zeros(len(GUNA_AXES))
    f[1] = np.clip((22 - w.temp_c) / 12, -1, 1)                 # cold (+sheeta) / hot (-> ushna)
    f[2] = np.clip((w.humidity - 50) / 40, -1, 1)               # humid (snigdha) / dry (ruksha)
    f[4] = 0.3 - np.clip(w.wind_kmh / 25, 0, 1)                 # still (sthira) / windy (chala)
    f[6] = np.clip((w.humidity - 70) / 30, 0, 1) * (1.0 if w.rain_mm > 1 else 0.5)   # damp / sticky (picchila)
    return A_WEATHER * (G_UNIT @ f) / SQ10

def ritu_effect(day: dt.date):
    """Circular linear interpolation between ritu mid-dates."""
    doy = day.timetuple().tm_yday
    pts = sorted(RITU_TABLE, key=lambda r: r[1])
    mids = [p[1] for p in pts]
    for i, m in enumerate(mids):                                  # find bracketing ritus
        nxt = mids[(i + 1) % len(mids)] + (365 if i == len(mids) - 1 else 0)
        d = doy + (365 if (i == len(mids) - 1 and doy < mids[0]) else 0)
        if m <= d < nxt:
            t = (d - m) / (nxt - m)
            return (1 - t) * pts[i][2] + t * pts[(i + 1) % len(pts)][2]
    return pts[0][2]

def ritu_name(day: dt.date):
    doy = day.timetuple().tm_yday
    return min(RITU_TABLE, key=lambda r: min(abs(doy - r[1]), 365 - abs(doy - r[1])))[0]

def sleep_effect(hours, bedtime):
    v = p = k = 0.0
    if hours is not None:
        v += 0.35 * ramp(7.0 - hours, 0, 3)
        k += 0.30 * ramp(hours - 8.5, 0, 3)
    if bedtime is not None:
        b = bedtime + 24 if bedtime < 12 else bedtime            # 1 am -> 25
        late = ramp(b - 22.5, 0, 3)
        p += 0.20 * late; v += 0.15 * ramp(b - 24.5, 0, 2)       # Pitta time 10pm-2am; very late -> Vata
    return vec(v, p, k)

ACTIVITY_DOSE = {                     # effect per hour of full-intensity activity [V,P,K]
    "strength": vec(0.05, 0.10, -0.35), "cardio": vec(0.10, 0.15, -0.40), "hiit": vec(0.25, 0.30, -0.40),
    "sports": vec(0.10, 0.20, -0.35), "yoga": vec(-0.20, -0.05, -0.15),
    "pranayama": vec(-0.25, -0.05, -0.10), "walking": vec(-0.05, 0.0, -0.20),
}
def exercise_effect(acts):
    e, load = np.zeros(3), 0.0
    for typ, minutes, inten in acts:
        if typ in ACTIVITY_DOSE:
            e += ACTIVITY_DOSE[typ] * (minutes / 60.0) * (0.4 + 0.6 * inten)
            load += minutes * (0.4 + 0.6 * inten) if typ not in ("yoga", "pranayama") else 0.7 * minutes
    e += vec(0.30 * ramp(load, 60, 150), 0.20 * ramp(load, 60, 150), 0.0)       # over-exertion
    e += vec(0, 0, 0.20 * (1 - ramp(load, 0, 20)))                               # sedentary
    return e

def meal_timing_effect(meals):
    hours = sorted(h for h, _ in meals)
    if len(hours) < 2:
        return np.zeros(3)
    gaps = np.diff(hours)
    v = 0.25 * ramp(gaps.max(), 5, 9)                                            # irregular / long gaps
    p = 0.20 if not any(11.5 <= h <= 15.0 for h in hours) else 0.0              # skipped midday meal
    k = 0.15 * ramp(hours[-1], 20.5, 23.0)                                       # late heavy dinner
    return vec(v, p, k)

SELF_CARE_EFFECT = {
    "abhyanga": vec(-0.25, -0.05, 0.03), "tongue_scraping": vec(0, -0.02, -0.10),
    "oil_pulling": vec(-0.10, -0.10, -0.05), "ashwagandha": vec(-0.20, 0.05, -0.10),
    "brahmi": vec(-0.15, -0.10, 0.0), "triphala": vec(0, 0, 0),
}
DAMPING = {"triphala": 0.10, "brahmi": 0.03}                                     # tridoshic: faster return to prakriti

CYCLE_EFFECT = {"menstrual": vec(0.25, 0.05, 0), "follicular": vec(0, 0, 0.05),
                "ovulatory": vec(0, 0.10, 0), "luteal": vec(0, 0.10, 0.10)}

EVID_MOOD = {"anxious": vec(.5, 0, 0), "angry": vec(0, .5, 0), "sad": vec(.15, 0, .4),
             "neutral": vec(), "happy": vec()}
EVID_BOWEL = {"constipated": vec(.5, 0, 0), "loose": vec(0, .5, 0), "sluggish": vec(0, 0, .5), "normal": vec()}
EVID_AGNI = {"vishama": vec(.5, 0, 0), "tikshna": vec(0, .5, 0), "manda": vec(0, 0, .5), "sama": vec()}


@dataclass
class DayInputs:
    drivers: dict = field(default_factory=dict)       # module -> (3,)
    evidence: dict = field(default_factory=dict)      # channel -> (3,)
    damping: float = 0.0
    notes: dict = field(default_factory=dict)         # human-readable trace


def compute(log: DailyLog, food_model=None, sex="male", lat=29.85) -> DayInputs:
    out, D, E, N = DayInputs(), None, None, None
    D, E, N = out.drivers, out.evidence, out.notes

    # ---- implicit (always available) ----
    w = log.weather or climatology(lat, log.date)
    D["weather"] = weather_effect(w); N["weather"] = f"{w.temp_c:.0f}C {w.humidity:.0f}% wind {w.wind_kmh:.0f} ({w.source})"
    D["ritu"] = ritu_effect(log.date); N["ritu"] = ritu_name(log.date)

    # ---- causes ----
    if log.sleep_hours is not None or log.bedtime_hour is not None:
        D["sleep"] = sleep_effect(log.sleep_hours, log.bedtime_hour)
    if log.meals:
        if food_model is not None:
            effs, unknown = [], []
            for _, text in log.meals:
                r = food_model.predict(text)
                (effs.append(food_effect(r["gunas"], r["rasas"])) if r else unknown.append(text))
            if effs: D["food"] = np.mean(effs, 0)
            if unknown: N["unrecognised_food"] = unknown
        D["meal_timing"] = meal_timing_effect(log.meals)
    if log.activities is not None:
        D["exercise"] = exercise_effect(log.activities)
    if log.stress is not None:
        D["stress"] = vec(0.30, 0.20, 0.0) * ramp(log.stress, 0.3, 1.0)
    if log.screen_hours is not None or log.study_hours is not None or log.classes is not None:
        s = log.screen_hours or 0.0; st = log.study_hours or 0.0; c = log.classes or 0
        D["screen"] = vec(0.12 * ramp(s, 2, 10) + 0.10 * ramp(st, 3, 10) + 0.05 * ramp(c, 3, 8),
                          0.08 * ramp(s, 2, 10) + 0.05 * ramp(c, 3, 8),
                          0.08 * ramp(st, 4, 10))
    if log.self_care is not None:
        D["self_care"] = sum((SELF_CARE_EFFECT.get(h, vec()) for h in log.self_care), vec())
        out.damping += sum(DAMPING.get(h, 0.0) for h in log.self_care)
    if log.cycle_phase and sex == "female":
        D["cycle"] = CYCLE_EFFECT.get(log.cycle_phase, vec())
    if log.mood == "happy":
        out.damping += 0.05

    # ---- symptoms (evidence) ----
    if log.agni in EVID_AGNI: E["agni"] = EVID_AGNI[log.agni]
    if log.bowel in EVID_BOWEL: E["bowel"] = EVID_BOWEL[log.bowel]
    if log.mood in EVID_MOOD: E["mood"] = EVID_MOOD[log.mood]
    if log.energy is not None:
        E["energy"] = vec(0, 0, 0.4 * ramp(0.4 - log.energy, 0, 0.4))
    if log.sleep_quality is not None:
        E["sleep_quality"] = vec(0.4 * ramp(0.6 - log.sleep_quality, 0, 0.6), 0, 0)
    out.damping = min(out.damping, 0.3)
    return out
