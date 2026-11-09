"""Local research API and website: python -m lab.server"""
from pathlib import Path
from typing import Annotated
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .model import compare, Parameters

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title='Tridosha Homeostasis Laboratory', version='3.0')
class Simulation(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    baseline: list[float] = Field(default=[.34,.33,.33], min_length=3,max_length=3)
    initial: list[float] = Field(default=[.85,-.3,-.55],min_length=3,max_length=3)
    days: float = Field(default=21,ge=1,le=90)
    decay: list[float] = Field(default=[.30,.22,.14],min_length=3,max_length=3)
    coupling: float = Field(default=.06,ge=0,le=1)
    gain: float = Field(default=.45,ge=0,le=3)
    amplitude: float = Field(default=.6,ge=0,le=3)
    target: int = Field(default=0,ge=0,le=2)
    start: float = Field(default=3,ge=0)
    duration: float = Field(default=2,ge=0)

@app.get('/api/health')
def health(): return {'status':'ok','model':'coupled ODE v3'}

@app.post('/api/simulate')
def run(q: Simulation):
    if any(r<=0 or r>2 for r in q.decay): raise HTTPException(422,'Decay rates must be in (0,2].')
    try:
        return compare(p=q.baseline,initial=q.initial,days=q.days,
                       par=Parameters(tuple(q.decay),q.coupling,q.gain),
                       amplitude=q.amplitude,target=q.target,start=q.start,duration=q.duration)
    except ValueError as e: raise HTTPException(422,str(e)) from e

@app.get('/api/evidence')
def evidence():
    path=ROOT/'results'/'experiments.json'
    if not path.exists(): raise HTTPException(404,'Run python -m lab.experiments first.')
    return json.loads(path.read_text())

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='website')
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
