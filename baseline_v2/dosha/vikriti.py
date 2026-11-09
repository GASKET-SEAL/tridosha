"""
Vikriti state model (5 Nov plan).  v_t = p (+) exp(x_t) = closure(p * exp(x_t)),  sum(x_t) = 0,  deviation = ||x_t||.
    x_t = center( c * tanh( (rho . x_{t-1}  +  f_life + f_agni + (1 + beta p) . f_env) / c ) )
Blank days decay (x <- center(c tanh(rho x / c))). Uncertainty: Monte-Carlo over Dirichlet(alpha) draws of p and jitter of rho, beta.
"""
from dataclasses import dataclass
import datetime as dt
import numpy as np
from . import config as C
from .simplex import closure, aitchison_dist, to_xy


def center(z): return z - z.mean()
def squash(z): return center(C.TANH_C * np.tanh(z / C.TANH_C))


@dataclass
class DayRecord:
    date: dt.date
    fl: np.ndarray        # life push
    fe: np.ndarray        # env push (before susceptibility)
    sStar: float = 0.0    # agni disturbance 0..1
    chi: int = -1         # dosha of the agni type
    damping: float = 0.0
    n: int = 0


def run(hist, p, rho=C.RHO, beta=C.BETA):
    """Replay the history; returns list of (date, x, v)."""
    p = np.asarray(p, float); x = np.zeros(3); s = 0.0; prev = None; out = []
    for h in sorted(hist, key=lambda r: r.date):
        if prev is not None:
            for _ in range((h.date - prev).days - 1):          # unlogged days: decay only
                x = squash(rho * x); s *= C.AGNI_SMOOTH
        s = (1 - C.AGNI_SMOOTH) * s + C.AGNI_SMOOTH * h.sStar
        f = h.fl + (1 + beta * p) * h.fe
        if h.chi >= 0: f = f.copy(); f[h.chi] += C.AGNI_GAIN * s
        x = squash(rho * (1 - h.damping) * x + f)
        out.append((h.date, x.copy(), closure(p * np.exp(x)))); prev = h.date
    return out


def ellipse80(points, k=3.22):
    """80% region (chi2, 2 dof) of simplex points in triangle xy-coordinates -> (cx, cy, rx, ry, angle_deg)."""
    P = np.array([to_xy(v) for v in points]); m = P.mean(0); C_ = np.cov(P.T)
    w, V = np.linalg.eigh(C_); a = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    return dict(cx=m[0], cy=m[1], rx=float(np.sqrt(k * w[1])), ry=float(np.sqrt(max(k * w[0], 1e-12))), angle=float(a))


def monte_carlo(hist, alpha, n=150, seed=0):
    rng = np.random.default_rng(seed); vs = []
    for _ in range(n):
        p = rng.dirichlet(alpha); rho = np.clip(C.RHO * (1 + .15 * rng.normal(size=3)), .3, .95)
        vs.append(run(hist, p, rho, C.BETA * (1 + .3 * rng.normal()))[-1][2])
    return np.array(vs)


def severity(dev):
    return "balanced" if dev < .15 else "mild" if dev < .4 else "moderate" if dev < .8 else "marked"
