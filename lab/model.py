"""Continuous, compositional model. Time is in illustrative days, not fitted clinical time."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax

P = np.eye(3) - np.ones((3, 3)) / 3
Q = np.array([[1/np.sqrt(2), 1/np.sqrt(6)], [-1/np.sqrt(2), 1/np.sqrt(6)], [0, -2/np.sqrt(6)]])
L = np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])

@dataclass(frozen=True)
class Parameters:
    decay: tuple = (0.30, 0.22, 0.14)
    coupling: float = 0.06
    gain: float = 0.45
    saturation: float = 0.8

def composition(p, x):
    return softmax(np.log(p) + np.asarray(x), axis=-1)

def rhs(x, forcing, par):
    x = P @ np.asarray(x)
    return P @ (-np.asarray(par.decay)*x - par.coupling*L@x
                - par.gain*par.saturation*np.tanh(x/par.saturation) + forcing)

def stability(par):
    a = -Q.T @ (np.diag(par.decay) + par.coupling*L + par.gain*np.eye(3)) @ Q
    eig = np.linalg.eigvalsh(a)
    return dict(eigenvalues=eig.tolist(), stable=bool(np.all(eig < 0)),
                natural_decay_bound=float(min(par.decay)+3*par.coupling),
                local_time_constant=float(-1/eig.max()))

def simulate(p=(.34,.33,.33), initial=(.85,-.3,-.55), days=21.,
             par=None, amplitude=.6, target=0, start=3., duration=2., step=.1):
    par = par or Parameters()
    p = np.asarray(p, float); initial = np.asarray(initial, float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(),1):
        raise ValueError('Baseline must contain three positive shares summing to one.')
    if initial.shape != (3,) or not np.isfinite(initial).all() or np.max(np.abs(initial)) > 4:
        raise ValueError('Initial deviation must be three finite values between -4 and 4.')
    vals = [*par.decay, par.coupling, par.gain, par.saturation, days, amplitude, start, duration, step]
    if not np.isfinite(vals).all() or len(par.decay)!=3 or min(par.decay)<=0 or par.coupling<0 or par.gain<0 or par.saturation<=0:
        raise ValueError('Invalid model parameters.')
    if not 1 <= days <= 90 or not .02 <= step <= 1 or not 0 <= amplitude <= 3 or target not in (0,1,2) or start<0 or duration<0 or start+duration>days:
        raise ValueError('Invalid simulation window or perturbation.')
    # Split at forcing discontinuities: no solver step may smear a pulse boundary.
    times = np.unique(np.r_[np.arange(0,days,step), days, start, start+duration])
    cuts = sorted(set([0.,start,start+duration,days]))
    states = np.empty((len(times),3)); x = P@initial
    for a,b in zip(cuts[:-1], cuts[1:]):
        if b <= a: continue
        force = np.zeros(3)
        if start <= (a+b)/2 < start+duration: force[target] = amplitude
        sol = solve_ivp(lambda t,y: rhs(y,force,par), (a,b), x,
                        rtol=1e-9, atol=1e-11, dense_output=True, max_step=.25)
        if not sol.success: raise RuntimeError(sol.message)
        mask = (times>=a)&(times<=b); states[mask] = sol.sol(times[mask]).T
        x = sol.y[:,-1]
    states = states @ P
    distance = np.linalg.norm(states,axis=1)
    controls = -par.gain*par.saturation*np.tanh(states/par.saturation) @ P
    stop = start+duration
    eligible = np.flatnonzero(times>=stop)
    settle = next((float(times[i]-stop) for i in eligible if np.all(distance[i:]<=.05)), None)
    return dict(time=times.tolist(), deviation=states.tolist(), shares=composition(p,states).tolist(),
                distance=distance.tolist(), control=controls.tolist(), baseline=p.tolist(),
                stability=stability(par), parameters=asdict(par),
                metrics=dict(final_distance=float(distance[-1]), peak_distance=float(distance.max()),
                             settling_days=settle, integrated_error=float(np.trapezoid(distance,times)),
                             control_effort=float(np.trapezoid(np.sum(controls**2,axis=1),times))))

def compare(**kwargs):
    par = kwargs.pop('par', Parameters())
    controlled = simulate(par=par, **kwargs)
    natural = simulate(par=Parameters(par.decay,par.coupling,0,par.saturation), **kwargs)
    return dict(controlled=controlled, natural=natural)
