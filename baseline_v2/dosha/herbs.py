"""Self-care herb effects from the Amidha Ayurveda Herb Database v2.0 (CC BY 4.0, 360 herbs). No invented herb values."""
import csv
from pathlib import Path
import numpy as np
_P = Path(__file__).resolve().parent.parent / "data" / "herb_dosha_lookup.csv"
_DB = None
IDX = {"vata": 0, "pitta": 1, "kapha": 2}
PAC, AGG = -0.12, 0.06
OILS = {"abhyanga": (-.2, 0, 0), "gandusha": (-.1, -.05, 0), "tongue": (0, 0, -.12)}   # practices, not in the herb DB (draft)


def db():
    global _DB
    if _DB is None:
        _DB = {r["name"].lower(): r for r in csv.DictReader(open(_P, encoding="utf-8"))}
    return _DB


def effect(name):
    """Returns (push[3], found). Tridoshic herbs give no directional push (they shorten recovery instead, see damping)."""
    n = name.lower()
    if n in OILS:
        return np.array(OILS[n], float), True
    r = db().get(n)
    if r is None:
        return np.zeros(3), False
    v = np.zeros(3)
    if r["tridosha"] == "True":
        return v, True
    for d in filter(None, (x.strip().lower() for x in r["pacify"].split(";"))):
        if d in IDX: v[IDX[d]] += PAC
    for d in filter(None, (x.strip().lower() for x in r["aggravate"].split(";"))):
        if d in IDX: v[IDX[d]] += AGG
    return v, True


def is_tridoshic(name):
    r = db().get(name.lower())
    return bool(r) and r["tridosha"] == "True"
