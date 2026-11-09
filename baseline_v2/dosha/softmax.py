"""Soft-label softmax regression (convex, L-BFGS). Reused from v1 because it won the model benchmark."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .simplex import closure


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
