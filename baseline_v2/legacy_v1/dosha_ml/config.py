"""
Central constants for the dosha model.

Everything that encodes *Ayurvedic knowledge* (as opposed to code) lives here, so a domain
expert can review / replace the numbers without touching any algorithm.
All coefficients are PRIORS drawn from classical principles (Charaka / Ashtanga Hridaya
samanya-vishesha: "like increases like, opposites decrease"). Replace with values from the
research you finalise; the adaptive filter in vikriti_model.py then personalises them per user.
"""
import numpy as np

DOSHAS = ("vata", "pitta", "kapha")

# 10 guna pairs. A food/environment is a signed vector in [-1, 1]^10.
# positive = first name, negative = its opposite (second row below)
GUNA_AXES = ("guru", "sheeta", "snigdha", "manda", "sthira",
             "mridu", "picchila", "slakshna", "sukshma", "sandra")
GUNA_NEG = ("laghu", "ushna", "ruksha", "tikshna", "chala",
            "kathina", "vishada", "khara", "sthula", "drava")
RASAS = ("madhura", "amla", "lavana", "katu", "tikta", "kashaya")

# Guna signature of each dosha (rows: V, P, K). +1 means "this dosha IS that guna".
#                  guru  sheeta snigdha manda sthira mridu picchila slakshna sukshma sandra
G_SIGNATURE = np.array([
    [-1.0,  1.0, -1.0,  0.0, -1.0,  0.0, -0.6, -1.0,  1.0, -0.3],   # Vata: laghu sheeta ruksha chala khara sukshma vishada
    [-0.4, -1.0,  0.3, -1.0, -0.4,  0.2,  0.0,  0.0,  0.2, -1.0],   # Pitta: ushna tikshna drava (slightly snigdha/laghu/sara)
    [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0,  1.0],   # Kapha: guru sheeta snigdha manda sthira mridu picchila slakshna sthula sandra
])
G_UNIT = G_SIGNATURE / np.linalg.norm(G_SIGNATURE, axis=1, keepdims=True)

# Effect of each rasa (taste) on each dosha (+1 increases, -1 decreases). Classical table.
#                 madhura amla lavana katu tikta kashaya
R_EFFECT = np.array([
    [-1, -1, -1,  1,  1,  1],   # Vata
    [-1,  1,  1,  1, -1, -1],   # Pitta
    [ 1,  1,  1, -1, -1, -1],   # Kapha
], dtype=float)

# Persistence rho per dosha for the daily recursion  x_t = rho*x_{t-1} + (1-rho)*u_t.
# Guna logic: Kapha is guru/sthira (slow to move, slow to clear), Vata is chala/laghu (fast).
RHO = np.array([0.55, 0.65, 0.80])      # half-lives ~1.2, 1.6, 3.1 days

# Susceptibility: "like increases like". A Vata-prakriti person reacts more to Vata-raising inputs.
SUSCEPTIBILITY_GAMMA = 0.25

# Driver modules whose gain (theta) is learned per user.
MODULES = ("food", "meal_timing", "sleep", "exercise", "stress",
           "screen", "self_care", "weather", "ritu", "cycle")

# Symptom channels used as measurements (std in log-units; larger = less trusted).
EVIDENCE_SIGMA = {"agni": 0.30, "bowel": 0.30, "mood": 0.40, "energy": 0.50, "sleep_quality": 0.45}

# Ritu (season) vectors [V, P, K], log-units, placed at the mid-date of each ritu (day-of-year).
RITU_TABLE = [  # (name, mid day-of-year, vector)
    ("hemanta",  350, np.array([0.05, -0.10, 0.10])),
    ("shishira",  45, np.array([0.15, -0.05, 0.15])),
    ("vasanta",  105, np.array([0.00,  0.00, 0.25])),
    ("grishma",  166, np.array([0.15,  0.15, -0.15])),
    ("varsha",   227, np.array([0.25,  0.10, 0.00])),
    ("sharad",   288, np.array([0.00,  0.25, -0.05])),
]
