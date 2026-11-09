"""One-time Prakriti questionnaire: spec, encoder, and a synthetic data generator."""
import numpy as np
from .compositional import prakriti_type

# (key, question, [Vata option, Pitta option, Kapha option], informativeness q in (0,1])
# q = how reliably an answer reflects the true constitution (used by the synthetic generator only)
QUESTIONS = [
    ("frame", "Body frame", ["Thin / light", "Medium / athletic", "Heavy / solid"], 0.75),
    ("skin", "Skin", ["Dry / rough", "Warm / sensitive / freckled", "Thick / cool / moist"], 0.75),
    ("hair", "Hair", ["Dry / coarse / frizzy", "Fine / thinning / early greying", "Thick / lustrous / oily"], 0.70),
    ("eyes", "Eyes", ["Small / dry / restless", "Sharp / light-sensitive / reddish", "Large / calm / moist"], 0.65),
    ("nails", "Nails", ["Brittle / rough", "Soft / pinkish", "Strong / thick / smooth"], 0.60),
    ("joints", "Joints", ["Very flexible / crackle", "Moderate", "Stable / well padded"], 0.60),
    ("weight", "Weight tendency", ["Hard to gain", "Stable", "Gains easily, hard to lose"], 0.70),
    ("appetite", "Appetite", ["Irregular", "Sharp, irritable when hungry", "Slow and steady, can skip meals"], 0.75),
    ("digestion", "Digestion tendency", ["Gas / bloating / constipation", "Acidity / loose stools", "Sluggish / heavy after meals"], 0.80),
    ("sweat", "Sweating", ["Scanty", "Profuse / warm", "Moderate / cool"], 0.55),
    ("thermal", "Climate preference", ["Dislike cold, love warmth", "Dislike heat, prefer cool", "Dislike cold + damp"], 0.75),
    ("extremities", "Hands & feet", ["Usually cold / dry", "Usually warm", "Cool and moist"], 0.65),
    ("stress", "Reaction to stress", ["Anxiety / overthinking", "Irritation / anger", "Withdrawal / calm"], 0.80),
    ("learning", "Learning & memory", ["Learn fast, forget fast", "Moderate, sharp focus", "Slow, long-term memory"], 0.70),
    ("sleep", "Sleep", ["Light / interrupted", "Moderate / sound", "Deep / heavy"], 0.75),
    ("speech", "Speech", ["Fast / variable", "Sharp / precise", "Slow / melodious"], 0.55),
    ("gait", "Walking pace", ["Quick / light", "Purposeful", "Slow / steady"], 0.50),
    ("decisions", "Decision style", ["Changeable / indecisive", "Decisive / determined", "Slow / stubborn"], 0.55),
    ("mood", "Mood pattern", ["Changes quickly", "Intense / competitive", "Stable / content"], 0.65),
    ("energy", "Energy pattern", ["Bursts then crashes", "Strong and steady", "Slow start, lasting stamina"], 0.65),
    ("bowel", "Usual bowel habit", ["Dry, irregular", "Loose / frequent", "Heavy, once a day"], 0.70),
    ("bmi", "BMI bucket (auto from height & weight)", ["<18.5", "18.5-25", ">25"], 0.35),
]
KEYS = [q[0] for q in QUESTIONS]
N_Q = len(QUESTIONS)
N_FEATURES = N_Q * 3


def bmi_bucket(height_cm, weight_kg):
    bmi = weight_kg / (height_cm / 100) ** 2
    return 0 if bmi < 18.5 else (1 if bmi <= 25 else 2)


def encode(answers: dict, bmi=None) -> np.ndarray:
    """answers: {key: 0|1|2 or None}. Missing/None -> all-zero block (no evidence). -> (66,) one-hot."""
    a = dict(answers)
    if bmi is not None:
        a.setdefault("bmi", bmi)
    x = np.zeros(N_FEATURES)
    for i, k in enumerate(KEYS):
        v = a.get(k)
        if v is not None:
            x[i * 3 + int(v)] = 1.0
    return x


def encode_many(rows) -> np.ndarray:
    return np.vstack([encode(r) for r in rows])


# ---- synthetic data (only for bootstrapping when no labelled dataset is available) -------------

def sample_compositions(n, rng):
    kind = rng.choice(3, n, p=[0.45, 0.40, 0.15])
    out = np.empty((n, 3))
    for i, k in enumerate(kind):
        if k == 0:
            a = np.full(3, 1.2); a[rng.integers(3)] = 7.0
        elif k == 1:
            a = np.full(3, 1.2); a[rng.choice(3, 2, replace=False)] = 5.0
        else:
            a = np.full(3, 8.0)
        out[i] = rng.dirichlet(a)
    return out


def generate_synthetic(n=4000, seed=0, p_missing=0.05):
    """Latent composition theta -> answer to question i ~ Cat(q_i*theta + (1-q_i)/3)."""
    rng = np.random.default_rng(seed)
    theta = sample_compositions(n, rng)
    X = np.zeros((n, N_FEATURES))
    for i, (_, _, _, q) in enumerate(QUESTIONS):
        probs = q * theta + (1 - q) / 3
        cum = probs.cumsum(1)
        ans = (rng.random((n, 1)) > cum).sum(1).clip(0, 2)
        X[np.arange(n), i * 3 + ans] = 1.0
    miss = rng.random((n, N_Q)) < p_missing
    X = X * np.repeat(~miss, 3, axis=1)
    return X, theta


# Hard 7-class labels (as in public datasets such as DoshaMitra's) -> soft composition targets
LABEL_TO_COMPOSITION = {
    "V": (0.75, 0.15, 0.10), "P": (0.15, 0.75, 0.10), "K": (0.10, 0.15, 0.75),
    "VP": (0.45, 0.45, 0.10), "PK": (0.10, 0.45, 0.45), "VK": (0.45, 0.10, 0.45),
    "VPK": (1/3, 1/3, 1/3),
}
