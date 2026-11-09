"""Daily log -> pushes f_t in R^3 (log-units for [Vata, Pitta, Kapha]). Mirrors the website (dosha-tracker.html)."""
import datetime as dt
import numpy as np
from . import config as C, herbs
from .logs import DailyLog, FIELDS
from .weather import Weather, climatology


def clamp(x, a=-1.0, b=1.0):
    return float(np.clip(x, a, b))


def gd(g):
    return C.G @ np.asarray(g, float) / C.G_NORM


def _has(v):
    return v is not None and v != "" and not (isinstance(v, list) and not v)


def ritu_index(day):
    n = day.timetuple().tm_yday; i = 5
    for k, b in enumerate(C.RB):
        if n >= b: i = k
    return i


def ritu_vector(day):
    """Blend across a two-week sandhi at each boundary (no hard switch)."""
    n = day.timetuple().tm_yday; i = ritu_index(day); pv, nx = (i + 5) % 6, (i + 1) % 6
    ds, de = (n - C.RB[i]) % 365, (C.RB[nx] - n) % 365
    v = C.RT[i].copy()
    if ds < 7:
        t = clamp((ds + 7) / 14, 0, 1); v = (1 - t) * C.RT[pv] + t * v
    if de < 7:
        t = (7 - de) / 14; v = (1 - t) * v + t * C.RT[nx]
    return v


def env_push(w: Weather, comfort_c: float, day):
    """Weather as anomaly from the user's comfort temperature + humidity/wind/rain, plus the slow ritu term."""
    h = clamp((w.humidity - 60) / 30); wd = clamp(w.wind_kmh / 20, 0, 1); r = 1.0 if w.rain_mm > 1 else 0.0
    heat = clamp((w.temp_c - comfort_c) / 8) - .4 * r
    g = [heat, .6 * h - .5 * wd, .5 * h + .6 * r, wd, .5 * heat if heat > 0 else 0]
    return C.ENV_SCALE * gd(g) + C.RITU_SCALE * ritu_vector(day)


def life_push(log: DailyLog, energy_mean=0.0, herbs_allowed=True):
    """Returns dict(fl, sStar, chi, n, damping, unknown_herbs)."""
    g = np.zeros(5); d = np.zeros(3); sev = []; chi = -1; damping = 0.0; unknown = []
    if _has(log.sleep_hours):
        lo, hi = C.SLEEP_BAND
        z = clamp((lo - log.sleep_hours) / 2) if log.sleep_hours < lo else (-clamp((log.sleep_hours - hi) / 2) if log.sleep_hours > hi else 0.0)
        g += abs(z) * np.array([0, -.5, -.5, 1, 0] if z > 0 else [0, .5, 1, -.5, 0])
    if _has(log.sleep_restful):
        g += clamp((.6 - log.sleep_restful) / .6, 0, 1) * np.array([0, -.4, -.3, .8, 0])
    if _has(log.mood): d += .4 * np.array(C.MOOD[log.mood])
    if _has(log.stress): d += log.stress * np.array([.3, .2, 0])
    for key, val in (("em", log.energy_morning), ("ea", log.energy_midday), ("ee", log.energy_late)):
        if _has(val): d[C.ENERGY_OWNER[key]] -= .3 * (val - energy_mean)
    if _has(log.appetite):
        sev.append(0 if log.appetite == "bal" else 1)
        if log.appetite != "bal": chi = {"irr": 0, "sharp": 1, "slow": 2}[log.appetite]
    if _has(log.bowel):
        if log.bowel != "norm":
            j = {"dry": 0, "loose": 1, "slow": 2}[log.bowel]; d[j] += .15; sev.append(1)
            if chi < 0: chi = j
        else: sev.append(0)
    if _has(log.meal_irregularity):
        d[0] += .25 * log.meal_irregularity; sev.append(log.meal_irregularity)
        if log.meal_irregularity > 0 and chi < 0: chi = 0
    if _has(log.activity):
        if log.activity == "none": d[2] += .2
        else:
            lo, hi = C.MET[log.activity]; met = lo + (hi - lo) * log.intensity
            g += .8 * (1 - np.exp(-(met - 1) * log.minutes / C.MET_SCALE)) * np.array(C.ACT_GUNA[log.activity])
    s, late, st = log.screen_hours or 0, log.late_screen_hours or 0, log.study_hours or 0
    d += np.array([.03 * s + .08 * late, .02 * s + .06 * late, .025 * st])
    for h in (log.self_care or []):
        e, found = herbs.effect(h)
        if not found: unknown.append(h)
        elif herbs_allowed or h in herbs.OILS: d += e
        if herbs.is_tridoshic(h) and herbs_allowed: damping += .05
    if log.mood == "happy": damping += .05
    n = sum(_has(getattr(log, f)) for f in FIELDS)
    return dict(fl=gd(g) + d, sStar=float(np.mean(sev)) if sev else 0.0, chi=chi, n=n, damping=min(damping, .3), unknown_herbs=unknown)
