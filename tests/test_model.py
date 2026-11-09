import numpy as np
import pytest
from scipy.linalg import expm
from fastapi.testclient import TestClient
from lab.model import *
from lab.server import app

def test_equilibrium():
    r=simulate(initial=(0,0,0),amplitude=0)
    assert np.max(np.abs(r['deviation']))==0
    assert np.allclose(r['shares'],r['baseline'])

def test_simplex_and_centering():
    r=simulate(initial=(4,-4,1),amplitude=3)
    assert np.all(np.array(r['shares'])>0)
    assert np.allclose(np.sum(r['shares'],axis=1),1)
    assert np.allclose(np.sum(r['deviation'],axis=1),0,atol=1e-12)

def test_analytic_solution():
    par=Parameters(gain=0); r=simulate(par=par,amplitude=0)
    a=-Q.T@(np.diag(par.decay)+par.coupling*L)@Q
    expected=np.array([Q@expm(a*t)@Q.T@np.array([.85,-.3,-.55]) for t in r['time']])
    assert np.max(np.abs(expected-r['deviation']))<1e-7

def test_lyapunov_derivative():
    rng=np.random.default_rng(42)
    for x in rng.normal(size=(200,3))@P:
        assert x@rhs(x,np.zeros(3),Parameters())<0

def test_feedback_improves_default_recovery():
    r=compare()
    assert r['controlled']['metrics']['settling_days']<r['natural']['metrics']['settling_days']
    assert r['controlled']['metrics']['integrated_error']<r['natural']['metrics']['integrated_error']

@pytest.mark.parametrize('target',[0,1,2])
def test_perturbation_direction(target):
    r=simulate(initial=(0,0,0),target=target,start=0,duration=1)
    assert np.argmax(r['deviation'][1])==target

def test_baseline_is_fixed():
    p=[.5,.3,.2]; simulate(p=p); assert p==[.5,.3,.2]

def test_aitchison_distance():
    r=simulate(p=(.5,.3,.2)); v=np.array(r['shares']); p=np.array(r['baseline'])
    clr=lambda z:np.log(z)-np.log(z).mean(axis=-1,keepdims=True)
    assert np.allclose(np.linalg.norm(clr(v)-clr(p),axis=1),r['distance'])

def test_zero_gain_matches_natural():
    r=compare(par=Parameters(gain=0)); assert r['controlled']==r['natural']

def test_unsettled_is_null():
    assert simulate(days=5,start=3,duration=2)['metrics']['settling_days'] is None

@pytest.mark.parametrize('kwargs',[{'p':[0,.5,.5]},{'p':[.5,.5,.5]},{'initial':[float('nan'),0,0]},{'duration':50},{'par':Parameters(decay=(-1,1,1))},{'amplitude':-1}])
def test_invalid_input(kwargs):
    with pytest.raises(ValueError):simulate(**kwargs)

def test_time_resolution():
    a=simulate(step=.1);b=simulate(step=.05)
    assert np.allclose(a['shares'][-1],b['shares'][-1],atol=1e-9)

def test_api_and_static():
    c=TestClient(app)
    assert c.get('/').status_code==200
    assert c.get('/api/health').json()['status']=='ok'
    assert c.post('/api/simulate',json={}).status_code==200
    for payload in [{'gain':-1},{'baseline':[0,.5,.5]},{'decay':[0,.2,.3]},{'days':1000},{'duration':30},{'unknown':1}]:
        assert c.post('/api/simulate',json=payload).status_code==422

def test_forcing_switch_is_present_in_output():
    r=simulate(start=3.17,duration=1.21)
    assert 3.17 in r['time'] and 4.38 in r['time']

def test_feedback_dissipates_and_is_bounded():
    r=simulate(initial=(4,-4,0))
    x=np.array(r['deviation']);u=np.array(r['control'])
    assert np.max(np.sum(x*u,axis=1))<=1e-12
    # Projection preserves a 2*k*s component bound, not the raw k*s bound.
    assert np.max(np.abs(u))<=2*.45*.8

def test_stability_on_tangent_plane():
    s=stability(Parameters())
    assert len(s['eigenvalues'])==2 and max(s['eigenvalues'])<0
    assert np.allclose(Q.T@Q,np.eye(2)) and np.allclose(Q@Q.T,P)

def test_shift_of_initial_clr_is_irrelevant():
    a=simulate(initial=(.7,-.2,-.5));b=simulate(initial=(1.7,.8,.5))
    assert np.allclose(a['shares'],b['shares'],atol=1e-10)

def test_extreme_api_parameters_remain_finite():
    c=TestClient(app)
    r=c.post('/api/simulate',json={'gain':3,'coupling':1,'decay':[2,2,2],'days':1,'start':0,'duration':1,'amplitude':3})
    assert r.status_code==200
    assert np.isfinite(r.json()['controlled']['shares']).all()
