"""
MODEL 1 - Prakriti as a point on the simplex.

Why this model: the output must be a COMPOSITION (Vata, Pitta, Kapha shares) so it can be drawn in
the triangle - not just a class label. A multinomial-logit (softmax) regression trained with SOFT
targets does exactly that: p = softmax(W^T x + b), minimising KL(target || p) + L2. It is convex
(L-BFGS finds the global optimum in milliseconds), tiny (66x3 weights), handles missing answers
natively (a skipped question is an all-zero block), and is fully explainable per question.
benchmark() compares it with expert-count rules, Random Forest and gradient boosting.
"""
import time, joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from . import questionnaire as Q
from .compositional import aitchison_dist, closure, ilr, ilr_inv, prakriti_type


class SoftmaxCompositionModel:
    def __init__(self, l2=10.0):
        self.l2 = l2
        self.W = None
        self.b = None

    def fit(self, X, Y):
        X = np.asarray(X, float); Y = closure(Y)
        n, d = X.shape
        l2 = self.l2

        def f(w):
            W = w[: d * 3].reshape(d, 3); b = w[d * 3:]
            Z = X @ W + b
            logp = Z - logsumexp(Z, axis=1, keepdims=True)
            loss = -(Y * logp).sum() / n + 0.5 * l2 * (W ** 2).sum() / n
            Gz = (np.exp(logp) - Y) / n
            return loss, np.concatenate([(X.T @ Gz + l2 * W / n).ravel(), Gz.sum(0)])

        res = minimize(f, np.zeros(d * 3 + 3), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.W = res.x[: d * 3].reshape(d, 3); self.b = res.x[d * 3:]
        return self

    def predict(self, X):
        Z = np.atleast_2d(X) @ self.W + self.b
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


class PrakritiModel:
    """Production wrapper: encode answers -> composition, plus explanation and persistence.
    `floor`: every person carries all three doshas, so each share is kept >= floor (also keeps the
    log-ratio geometry of the Vikriti model well conditioned)."""

    def __init__(self, l2=10.0, floor=0.05):
        self.core = SoftmaxCompositionModel(l2)
        self.floor = floor

    def fit(self, X=None, Y=None, tune=True, seed=0):
        """X: (n,66) encoded answers, Y: (n,3) soft targets (or one-hot / LABEL_TO_COMPOSITION rows).
        With no data given, bootstraps on synthetic data - REPLACE with a real labelled dataset."""
        if X is None:
            X, Y = Q.generate_synthetic(4000, seed)
        if tune:
            best = (1e9, self.core.l2)
            for l2 in (1, 10, 100, 300):
                errs = []
                for tr, te in KFold(3, shuffle=True, random_state=seed).split(X):
                    m = SoftmaxCompositionModel(l2).fit(X[tr], Y[tr])
                    errs.append(aitchison_dist(m.predict(X[te]), Y[te]).mean())
                if np.mean(errs) < best[0]:
                    best = (np.mean(errs), l2)
            self.core.l2 = best[1]
        self.core.fit(X, Y)
        return self

    def predict(self, answers: dict, bmi_bucket=None):
        x = Q.encode(answers, bmi_bucket)
        p = self.core.predict(x)[0]
        p = (1 - 3 * self.floor) * p + self.floor
        return {"composition": p, "type": prakriti_type(p),
                "answered": int(x.sum()), "of": Q.N_Q}

    def explain(self, answers: dict, bmi_bucket=None, top=5):
        """Which answers pushed the person towards each dosha (logit contribution)."""
        x = Q.encode(answers, bmi_bucket)
        contrib = x[:, None] * self.core.W
        out = {}
        for j, d in enumerate(("vata", "pitta", "kapha")):
            rel = contrib[:, j] - contrib.mean(1)            # relative to the other doshas
            idx = np.argsort(-rel)[:top]
            out[d] = [(Q.KEYS[i // 3], Q.QUESTIONS[i // 3][2][i % 3], float(rel[i])) for i in idx if x[i] > 0 and rel[i] > 0]
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


# ---------------------------------------------------------------- benchmark ---------------------
class _CountBaseline:           # what rule-based expert systems do: share of answers per dosha
    def fit(self, X, Y): return self
    def predict(self, X):
        c = X.reshape(len(X), Q.N_Q, 3).sum(1)
        return closure(c + 0.5)

class _SkRegressor:             # tree models regress the 2 ilr coordinates, then map back to simplex
    def __init__(self, make): self.make = make
    def fit(self, X, Y):
        T = ilr(closure(Y)); self.ms = [self.make().fit(X, T[:, k]) for k in range(2)]; return self
    def predict(self, X): return ilr_inv(np.column_stack([m.predict(X) for m in self.ms]))

class _Soft:
    def __init__(self, l2): self.m = SoftmaxCompositionModel(l2)
    def fit(self, X, Y): self.m.fit(X, Y); return self
    def predict(self, X): return self.m.predict(X)


def _floor(p, f=0.03):
    """Floor tiny shares so Aitchison distance is not dominated by near-zero components."""
    return closure(np.clip(p, f, None))


def benchmark(n=4000, seed=0, folds=5):
    """K-fold CV. Metrics: mean share error (MAE), floored Aitchison distance, 7-type accuracy.
    NOTE: on synthetic data the ranking only shows the models are sane; re-run on a real labelled
    dataset (load it, call Q.encode_many, then this function's loop) before trusting the numbers."""
    X, Y = Q.generate_synthetic(n, seed)
    models = {
        "expert-count rules": lambda: _CountBaseline(),
        "Softmax regression (ours)": lambda: _Soft(100.0),
        "Random Forest": lambda: _SkRegressor(lambda: RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
        "HistGradientBoosting": lambda: _SkRegressor(lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)),
    }
    rows = []
    for name, mk in models.items():
        mae, ad, acc, t_fit, t_pred = [], [], [], 0.0, 0.0
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
            t0 = time.perf_counter(); m = mk().fit(X[tr], Y[tr]); t_fit += time.perf_counter() - t0
            t0 = time.perf_counter(); P = m.predict(X[te]); t_pred += time.perf_counter() - t0
            mae.append(np.abs(P - Y[te]).mean())
            ad.append(aitchison_dist(_floor(P), _floor(Y[te])).mean())
            acc.append(np.mean([prakriti_type(a) == prakriti_type(b) for a, b in zip(P, Y[te])]))
        rows.append(dict(model=name, share_mae=np.mean(mae), aitchison=np.mean(ad), type_acc=np.mean(acc),
                         fit_s=t_fit / folds, ms_per_pred=1000 * t_pred / n))
    return rows
