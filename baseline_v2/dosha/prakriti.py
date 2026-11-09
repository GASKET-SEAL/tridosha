"""
Prakriti, two ways (5 Nov plan: transparent scoring first, ML only for calibration/validation).

ExpertPrakriti : alpha = 1 + sum of expert weights of the chosen options (+ age prior);  p = alpha / sum(alpha);  uncertainty = Dirichlet(alpha).
DataPrakriti   : softmax regression trained on the physician-labelled Mendeley cohorts (Vata/Pitta/Kapha, 217 people).
                 NOTE: only extreme types are in that data, so shares for mixed constitutions are not validated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .simplex import closure
from .softmax import SoftmaxCompositionModel

DATA = Path(__file__).resolve().parent.parent / "data" / "mendeley"
# key, question, [Vata, Pitta, Kapha] options, expert weight  (DRAFT weights; directions agree with the Mendeley data)
QUESTIONS = [
    ("frame", "Body frame", ["Thin, light", "Medium, athletic", "Broad, heavy"], 1.5),
    ("skin", "Skin", ["Dry, rough, cool", "Warm, sensitive, oily", "Thick, smooth, moist"], 1.2),
    ("hair", "Hair", ["Dry, coarse", "Fine, early greying", "Thick, oily"], 1.0),
    ("eyes", "Eyes", ["Small, dry, restless", "Sharp, light-sensitive", "Large, calm"], 1.0),
    ("nails", "Nails", ["Brittle", "Soft, pink", "Thick, strong"], .8),
    ("joints", "Joints", ["Flexible, crackle", "Moderate", "Stable, padded"], .8),
    ("weight", "Weight pattern", ["Difficulty gaining", "Stable", "Gains easily, loses with difficulty"], 1.5),
    ("appetite", "Appetite", ["Irregular", "Sharp, irritable if late", "Slow but steady"], 1.5),
    ("gut", "Usual digestive trouble", ["Gas, dry stool, constipation", "Acidity, loose stool", "Heaviness, sluggish"], 1.5),
    ("thermal", "Climate preference", ["Warm, dislikes cold", "Cool, dislikes heat", "Dry and warm, dislikes damp cold"], 1.2),
    ("stress", "Reaction to stress", ["Worry, anxiety", "Irritation, anger", "Withdrawal"], 1.5),
    ("decide", "Speech and decisions", ["Unsteady, wavering", "Sharp, accurate", "Firm, steady, well thought out"], 1.2),
    ("sleep", "Sleep", ["Shallow, light", "Sound", "Deep, heavy"], 1.2),
]
COMFORT_C = {0: 27.0, 1: 20.0, 2: 24.0}


class ExpertPrakriti:
    def __init__(self, questions=QUESTIONS, base=1.0):
        self.q = questions; self.base = base

    def alpha(self, answers: dict, age=None):
        a = np.full(3, self.base)
        for key, _, _, w in self.q:
            c = answers.get(key)
            if c is not None: a[int(c)] += w
        if age: a[2 if age < 16 else 1 if age <= 50 else 0] += .5
        return a

    def predict(self, answers: dict, age=None):
        a = self.alpha(answers, age); p = a / a.sum()
        return dict(composition=p, alpha=a, comfort_c=COMFORT_C.get(answers.get("thermal"), 24.0),
                    answered=sum(answers.get(k) is not None for k, *_ in self.q))

    @staticmethod
    def sample(alpha, n=200, seed=0):
        return np.random.default_rng(seed).dirichlet(alpha, n)


def ptype(p, gap1=.08, gap2=.05):
    o = np.argsort(-p); n = "VPK"
    if p[o[0]] - p[o[1]] > gap1: return n[o[0]]
    return "".join(sorted(n[i] for i in o[:2])) if p[o[1]] - p[o[2]] > gap2 else "VPK"


# ---------------------------------------------------------------- data-trained model -----------
CLASSES = ("Vata", "Pitta", "Kapha")


def load_cohorts():
    return pd.read_csv(DATA / "Western_India.csv"), pd.read_csv(DATA / "Northern_India.csv")


def shared_features(W, N):
    F = sorted(set(W.columns) & set(N.columns) - {"SampleID", "Class"}, key=lambda c: int(c[1:]))
    return [f for f in F if set(W[f].dropna()) & set(N[f].dropna())]


def _vocab(frames, feats):
    return [(f, v) for f in feats for v in sorted(set().union(*[set(d[f].dropna()) for d in frames]))]


def _encode(df, vocab):
    return np.column_stack([(df[f] == v).values.astype(float) for f, v in vocab])


def _y(df):
    return np.eye(3)[df["Class"].map({c: i for i, c in enumerate(CLASSES)}).values]


class DataPrakriti:
    def __init__(self, l2=10.0): self.l2 = l2

    def fit(self, frames=None):
        W, N = load_cohorts(); frames = frames or [W, N]
        self.feats = shared_features(W, N); self.vocab = _vocab([W, N], self.feats)
        X = np.vstack([_encode(d, self.vocab) for d in frames]); Y = np.vstack([_y(d) for d in frames])
        self.core = SoftmaxCompositionModel(self.l2).fit(X, Y); return self

    def predict(self, df):
        """df: rows with the shared feature columns (text answers). Missing answers contribute nothing."""
        return self.core.predict(_encode(df, self.vocab))


def cross_cohort(l2=10.0):
    """Train on one cohort, test on the other (the paper's replication design). Returns dict of accuracy and recall."""
    W, N = load_cohorts(); feats = shared_features(W, N); vocab = _vocab([W, N], feats); res = {}
    for name, tr, te in (("West->North", W, N), ("North->West", N, W)):
        m = SoftmaxCompositionModel(l2).fit(_encode(tr, vocab), _y(tr))
        pred = m.predict(_encode(te, vocab)).argmax(1); y = _y(te).argmax(1)
        res[name] = dict(n_train=len(tr), n_test=len(te), accuracy=float((pred == y).mean()),
                         recall={c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)})
    return res


# Features whose meaning is clear from their answers (inferred; confirm with the paper's S1 Table / authors).
# answer -> expected dosha, used to check the draft expert weights against the physician labels.
MAPPED = {
    "F1": {"Weaklydeveloped": 0, "Moderatelydeveloped": 1, "Welldeveloped": 2},                    # body development
    "F2": {"Thin/Narrow": 0, "Medium": 1, "Broad": 2},                                              # frame breadth
    "F5": {"Thin/Narrow": 0, "Medium": 1, "Broad": 2},
    "F59": {"Difficultyingaining": 0, "Stable": 1, "Gainandloseeasily": 1, "Gaineasilyandlosewithdifficulty": 2},  # weight pattern
    "F55": {"Shallow": 0, "Sound": 1, "Deep": 2},                                                   # sleep
    "F29": {"Dry": 0, "Seasonal/Variable": 0, "Oily": 1, "Normal": 2},                              # skin
    "F43": {"Constipation": 0, "Loosemotion": 1},                                                   # bowel tendency
    "F74": {"Unsteady": 0, "Sharp/Accurate": 1, "Firm/Steady": 2},                                  # steadiness
    "F75": {"Wavering/Easilydeviated": 0, "Sharp/Accurate/Spontaneous": 1, "Wellthoughtof": 2},     # thought process
    "F77": {"Unsteady": 0, "Sharp/Accurate": 1, "Firm/Steady": 2},
}


def check_expert_scoring(weight=1.0):
    """How well does transparent vote-counting over MAPPED features reproduce the physician labels?"""
    W, N = load_cohorts(); P = pd.concat([W, N], ignore_index=True); y = _y(P).argmax(1)
    A = np.ones((len(P), 3)); agree = []
    for f, m in MAPPED.items():
        for val, d in m.items():
            hit = (P[f] == val).values; A[hit, d] += weight
            if hit.sum() >= 8: agree.append((f, val, CLASSES[d], int(hit.sum()), float((y[hit] == d).mean())))
    pred = A.argmax(1)
    return dict(accuracy=float((pred == y).mean()), n=len(P), per_answer=pd.DataFrame(agree, columns=["feature", "answer", "expected", "n", "share_in_expected_class"]))


def transparent_oos(min_v=0.5, min_n=5):
    """Honest out-of-sample test of transparent scoring: derive answer->dosha rules from one cohort only
    (features with Cramer's V >= min_v), then vote-count equal weights on the OTHER cohort."""
    from scipy.stats import chi2_contingency
    W, N = load_cohorts(); feats = shared_features(W, N); res = {}
    for name, tr, te in (("rules from West -> scored on North", W, N), ("rules from North -> scored on West", N, W)):
        rules = {}
        for f in feats:
            ct = pd.crosstab(tr[f], tr["Class"]).reindex(columns=list(CLASSES), fill_value=0)
            if ct.shape[0] < 2: continue
            v = np.sqrt(chi2_contingency(ct)[0] / (len(tr) * (min(ct.shape) - 1)))
            if v >= min_v:
                rules[f] = {a: int(np.argmax(ct.loc[a].values)) for a in ct.index if ct.loc[a].sum() >= min_n}
        A = np.ones((len(te), 3))
        for f, m in rules.items():
            for a, d in m.items(): A[(te[f] == a).values, d] += 1
        y = _y(te).argmax(1); pred = A.argmax(1)
        res[name] = dict(n_rules_features=len(rules), accuracy=float((pred == y).mean()),
                         recall={c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)})
    return res
