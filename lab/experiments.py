"""Reproducible numerical experiments and training-only cohort benchmark."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix, f1_score
from .model import Parameters, simulate, compare, P, Q, L, rhs

ROOT=Path(__file__).resolve().parent.parent

def benchmark():
    frames=[pd.read_csv(ROOT/'baseline_v2/data/mendeley'/name) for name in ['Western_India.csv','Northern_India.csv']]
    labels=['Vata','Pitta','Kapha']; results={}
    for name,tr,te in [('West to North',frames[0],frames[1]),('North to West',frames[1],frames[0])]:
        # Column intersection is schema alignment only; no target values/category vocabulary are inspected.
        features=sorted(set(tr.columns)&set(te.columns)-{'SampleID','Class'})
        x=tr[features].fillna('__missing__').astype(str); xt=te[features].fillna('__missing__').astype(str)
        pipe=Pipeline([('encode',OneHotEncoder(handle_unknown='ignore')),('model',LogisticRegression(max_iter=3000))])
        search=GridSearchCV(pipe,{'model__C':[.01,.1,1,10]},cv=StratifiedKFold(4,shuffle=True,random_state=42),scoring='neg_log_loss',n_jobs=1)
        search.fit(x,tr.Class)
        model=search.best_estimator_.named_steps['model']
        encoder=search.best_estimator_.named_steps['encode']
        stem=name.lower().replace(' ','_')
        np.savez(ROOT/'results'/f'{stem}_model.npz',coef=model.coef_,intercept=model.intercept_)
        (ROOT/'results'/f'{stem}_schema.json').write_text(json.dumps(dict(features=features,
            categories=[c.tolist() for c in encoder.categories_],classes=search.classes_.tolist(),
            selected_C=search.best_params_['model__C'],unknown_categories='all-zero encoding'),indent=2))
        pred=search.predict(xt); prob=search.predict_proba(xt)
        fixed=Pipeline([('encode',OneHotEncoder(handle_unknown='ignore')),('model',LogisticRegression(C=.1,max_iter=3000))]).fit(x,tr.Class)
        majority=tr.Class.value_counts().idxmax()
        rng=np.random.default_rng(42); correct=np.asarray(pred==te.Class)
        boot=[correct[rng.integers(len(te),size=len(te))].mean() for _ in range(2000)]
        results[name]=dict(n_train=len(tr),n_test=len(te),features=len(features),selected_C=search.best_params_['model__C'],
            accuracy=float(accuracy_score(te.Class,pred)),balanced_accuracy=float(balanced_accuracy_score(te.Class,pred)),
            macro_f1=float(f1_score(te.Class,pred,average='macro')),log_loss=float(log_loss(te.Class,prob,labels=search.classes_)),
            accuracy_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            confusion_labels=labels,confusion_matrix=confusion_matrix(te.Class,pred,labels=labels).tolist(),
            majority_accuracy=float(np.mean(te.Class==majority)),fixed_C_accuracy=float(accuracy_score(te.Class,fixed.predict(xt))),
            fixed_C_log_loss=float(log_loss(te.Class,fixed.predict_proba(xt),labels=fixed.classes_)))
    return results

def calibrate():
    # Synthetic recovery check only: known forcing and state observations, no clinical measurements.
    truth=Parameters((.30,.22,.14),.06,0)
    rng=np.random.default_rng(91)
    initials=[(.8,-.2,-.6),(-.5,.9,-.4),(-.3,-.4,.7)]
    data=[]
    for i,x in enumerate(initials):
        r=simulate(initial=x,days=10,par=truth,target=i,amplitude=.5,start=2,duration=2,step=.5)
        data.append(np.asarray(r['deviation'])+rng.normal(0,.004,(len(r['time']),3))@P)
    # Fit three decay rates; coupling fixed because complete-graph coupling and isotropic decay are confounded.
    def residual(rates):
        par=Parameters(tuple(rates),.06,0)
        return np.concatenate([(np.asarray(simulate(initial=x,days=10,par=par,target=i,amplitude=.5,start=2,duration=2,step=.5)['deviation'])-y).ravel() for i,(x,y) in enumerate(zip(initials,data))])
    fit=least_squares(residual,[.5,.5,.5],bounds=(.03,1),xtol=1e-9,ftol=1e-9,gtol=1e-9)
    hold=dict(initial=(.6,.2,-.8),days=14,target=1,amplitude=.8,start=4,duration=3)
    target=np.asarray(simulate(par=truth,**hold)['deviation'])
    before=np.asarray(simulate(par=Parameters((.5,.5,.5),.06,0),**hold)['deviation'])
    after=np.asarray(simulate(par=Parameters(tuple(fit.x),.06,0),**hold)['deviation'])
    return dict(kind='synthetic parameter recovery only',seed=91,true_decay=list(truth.decay),fitted_decay=fit.x.tolist(),
                success=bool(fit.success),jacobian_rank=int(np.linalg.matrix_rank(fit.jac)),
                heldout_rmse_before=float(np.sqrt(np.mean((before-target)**2))),heldout_rmse_after=float(np.sqrt(np.mean((after-target)**2))))

def main():
    r=compare(); par=Parameters(gain=0)
    closed=simulate(par=par,amplitude=0)
    A=-Q.T@(np.diag(par.decay)+par.coupling*L)@Q
    exact=np.array([Q@expm(A*t)@Q.T@np.array([.85,-.3,-.55]) for t in closed['time']])
    sensitivity=[]
    for scale in [.8,1.,1.2]:
        for coupling in [.03,.06,.09]:
            par_s=Parameters(tuple(scale*np.array([.30,.22,.14])),coupling,.45)
            s=compare(par=par_s)
            sensitivity.append(dict(decay_scale=scale,coupling=coupling,
                controlled_settling=s['controlled']['metrics']['settling_days'],
                natural_settling=s['natural']['metrics']['settling_days'],
                error_reduction=1-s['controlled']['metrics']['integrated_error']/s['natural']['metrics']['integrated_error']))
    results=dict(default_experiment={k:v['metrics']|{'stability':v['stability']} for k,v in r.items()},
        analytic_max_error=float(np.max(np.abs(exact-np.asarray(closed['deviation'])))),
        sensitivity=sensitivity,
        gain_sweep=[dict(gain=g,**simulate(par=Parameters(gain=g))['metrics']) for g in [0,.15,.45,.9,1.5]],
        calibration=calibrate(),cohort_benchmark=benchmark())
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/experiments.json').write_text(json.dumps(results,indent=2))
    (ROOT/'results/default_trajectory.json').write_text(json.dumps(r))
    pd.DataFrame({'day':r['controlled']['time'],**{f'{n}_{d}':np.array(r[n]['shares'])[:,i] for n in r for i,d in enumerate(['V','P','K'])}}).to_csv(ROOT/'results/default_trajectory.csv',index=False)
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
