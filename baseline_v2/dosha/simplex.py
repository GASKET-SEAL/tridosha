"""
Compositional-data (Aitchison) geometry of the dosha simplex.

A (Vata, Pitta, Kapha) triple is a COMPOSITION: positive, sums to 1, and only ratios matter.
Ordinary Euclidean arithmetic on it is invalid (e.g. you can leave the triangle). We therefore do
all arithmetic in log-ratio coordinates and map back:

    clr(p)  = log p - mean(log p)               (3 numbers summing to 0)
    ilr(p)  = V @ clr(p)                        (2 free numbers, an isometry of the simplex)
    p (+) x = closure(p * exp(x))               (perturbation: the simplex's "addition")
    d_A(p,q)= || clr(p) - clr(q) ||             (Aitchison distance)
"""
import numpy as np

EPS = 1e-9
# Orthonormal Helmert-type basis of the sum-zero plane in R^3 (rows are orthonormal, sum to 0)
V_BASIS = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0.0],
                    [1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)]])

# Triangle corners for plotting: Vata bottom-left, Pitta bottom-right, Kapha top.
TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])


def closure(x):
    x = np.clip(np.asarray(x, float), EPS, None)
    return x / x.sum(-1, keepdims=True)

def clr(p):
    l = np.log(closure(p))
    return l - l.mean(-1, keepdims=True)

def clr_inv(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def ilr(p):
    return clr(p) @ V_BASIS.T

def ilr_inv(y):
    return clr_inv(np.asarray(y, float) @ V_BASIS)

def perturb(p, x_clr):
    """p (+) exp(x): shift composition p by log-ratio vector x."""
    return clr_inv(clr(p) + np.asarray(x_clr, float))

def aitchison_dist(p, q):
    return np.linalg.norm(clr(p) - clr(q), axis=-1)

def to_xy(p):
    """Composition -> 2-D point inside the equilateral triangle."""
    return np.asarray(p, float) @ TRI_VERTS

def prakriti_type(p, single_gap=0.15, dual_gap=0.10):
    """7-way classical label (V, P, K, VP, PK, VK, VPK) from a composition. Thresholds are conventions."""
    p = np.asarray(p, float)
    order = np.argsort(-p)
    s = p[order]
    names = "VPK"
    if s[0] - s[1] > single_gap:
        return names[order[0]]
    if s[1] - s[2] > dual_gap:
        return "".join(names[i] for i in sorted(order[:2]))
    return "VPK"
