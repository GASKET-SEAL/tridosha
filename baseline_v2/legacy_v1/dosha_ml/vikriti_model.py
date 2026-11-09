"""
MODEL 3 - Mathematical model of Vikriti (the current dosha state): an adaptive Kalman filter on the
simplex, with Prakriti as a FIXED reference.

State (2 numbers, ilr coordinates of the log-excess of each dosha over the person's Prakriti):
        s_t in R^2 ,   x_t = V^T s_t in R^3 (sum-zero),
        Vikriti_t = Prakriti (+) exp(x_t) = closure( p0 * exp(x_t) )          s_t = 0  <=>  Vikriti = Prakriti
        deviation_t = ||x_t||  (Aitchison distance Vikriti <-> Prakriti)

Dynamics (daily step; homeostasis pulls s -> 0, today's causes push it):
        s_t = Phi_t s_{t-1} + sum_j theta_j * C_j,t + w_t ,         w_t ~ N(0, Q)
        Phi_t = V diag(rho * (1 - damping_t)) V^T                   per-dosha inertia (Kapha slow, Vata fast)
        C_j,t = V ( (1 - rho) * S * c_j,t )                         module-j driver, scaled by susceptibility
        S_d   = clip(1 + gamma (3 p0_d - 1), .5, 1.5)               "like increases like"

Measurements (symptom channels k = agni, bowel, mood, energy, sleep quality; any may be missing):
        y_k,t = V e_k,t = s_t + v_k ,                               v_k ~ N(0, sigma_k^2 I)

ML part: the module gains theta_j (prior 1) are appended to the state and learned online per user
(random-walk, augmented Kalman filter), so each person's own sensitivities are personalised from their
logs. Cost is O(1) per day (12x12 matrices). The posterior covariance gives a confidence ellipse.
"""
from dataclasses import dataclass
import numpy as np
from .compositional import V_BASIS as V, perturb, aitchison_dist, clr, to_xy, clr_inv
from .config import RHO, SUSCEPTIBILITY_GAMMA, MODULES, EVIDENCE_SIGMA

J = len(MODULES)
CHI2_95_2DOF = 5.991


def susceptibility(p0, gamma=SUSCEPTIBILITY_GAMMA):
    return np.clip(1 + gamma * (3 * np.asarray(p0) - 1), 0.5, 1.5)


class VikritiFilter:
    def __init__(self, prakriti, rho=RHO, adaptive=True, process_sd=0.12, theta_prior_sd=0.30,
                 theta_drift_sd=0.01, init_sd=0.20, gamma=SUSCEPTIBILITY_GAMMA):
        self.p0 = np.asarray(prakriti, float) / np.sum(prakriti)       # FIXED reference, never modified
        self.rho = np.asarray(rho, float)
        self.S = susceptibility(self.p0, gamma)
        self.adaptive = adaptive
        n = 2 + J
        self.z = np.concatenate([np.zeros(2), np.ones(J)])             # [s, theta]
        self.P = np.diag(np.concatenate([np.full(2, init_sd ** 2), np.full(J, theta_prior_sd ** 2)]))
        self.Q = np.diag(np.concatenate([np.full(2, process_sd ** 2), np.full(J, theta_drift_sd ** 2)]))
        self.theta_prior = np.ones(J)
        self.n_updates = 0

    # ---- helpers
    def _C(self, drivers: dict):
        C = np.zeros((2, J))
        for j, m in enumerate(MODULES):
            if m in drivers:
                C[:, j] = V @ ((1 - self.rho) * self.S * np.asarray(drivers[m], float))
        return C

    def _phi(self, damping=0.0):
        return V @ np.diag(self.rho * (1 - damping)) @ V.T

    # ---- one day
    def predict(self, drivers: dict, damping: float = 0.0):
        C, Phi = self._C(drivers), self._phi(damping)
        F = np.eye(2 + J); F[:2, :2] = Phi; F[:2, 2:] = C
        theta = self.z[2:] if self.adaptive else self.theta_prior
        z = self.z.copy(); z[:2] = Phi @ self.z[:2] + C @ theta
        self.z = z
        Q = self.Q.copy()
        if not self.adaptive:
            F[:2, 2:] = 0; Q[2:, 2:] = 0
        self.P = F @ self.P @ F.T + Q

    def update(self, evidence: dict):
        n = 2 + J
        H = np.zeros((2, n)); H[:, :2] = np.eye(2)
        for ch, e in evidence.items():
            y = V @ np.asarray(e, float)
            R = np.eye(2) * EVIDENCE_SIGMA.get(ch, 0.4) ** 2
            S_ = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S_)
            self.z = self.z + K @ (y - H @ self.z)
            I_KH = np.eye(n) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T               # Joseph form (stays PSD)
            self.n_updates += 1
        if self.adaptive:
            self.z[2:] = np.clip(self.z[2:], 0.0, 3.0)

    def step(self, drivers: dict, evidence: dict = None, damping: float = 0.0):
        self.predict(drivers, damping)
        if evidence:
            self.update(evidence)
        return self.state()

    # ---- readout
    @property
    def s(self): return self.z[:2]

    def composition(self, s=None):
        s = self.s if s is None else s
        return perturb(self.p0, s @ V)

    def ellipse(self, n=60):
        """95% credible region of Vikriti, as polygon in triangle xy-coordinates."""
        P = self.P[:2, :2]
        L = np.linalg.cholesky(P + 1e-12 * np.eye(2))
        a = np.linspace(0, 2 * np.pi, n)
        pts = self.s[None, :] + np.sqrt(CHI2_95_2DOF) * (L @ np.vstack([np.cos(a), np.sin(a)])).T
        return np.array([to_xy(self.composition(p)) for p in pts])

    def state(self):
        v = self.composition()
        x = self.s @ V
        dev = float(np.linalg.norm(x))
        elev = int(np.argmax(x))
        label = "balanced" if dev < 0.15 else "mild" if dev < 0.4 else "moderate" if dev < 0.8 else "marked"   # conventions - calibrate
        return {
            "prakriti": self.p0.tolist(), "vikriti": v.tolist(),
            "xy_prakriti": to_xy(self.p0).tolist(), "xy_vikriti": to_xy(v).tolist(),
            "deviation": dev, "severity": label,
            "aggravated": ("vata", "pitta", "kapha")[elev] if dev >= 0.15 else None,
            "log_shift": x.tolist(),
            "theta": dict(zip(MODULES, self.z[2:].round(3).tolist())),
            "uncertainty_sd": float(np.sqrt(np.trace(self.P[:2, :2]) / 2)),
        }

    # ---- (de)serialise for the profile store
    def to_dict(self):
        return {"p0": self.p0.tolist(), "z": self.z.tolist(), "P": self.P.tolist(), "adaptive": self.adaptive,
                "n_updates": self.n_updates}

    @classmethod
    def from_dict(cls, d):
        f = cls(d["p0"], adaptive=d["adaptive"])
        f.z, f.P, f.n_updates = np.array(d["z"]), np.array(d["P"]), d["n_updates"]
        return f
