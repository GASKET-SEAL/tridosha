"""FastAPI service.  uvicorn dosha.api:app --reload"""
import datetime as dt
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .engine import Engine, SAFETY
from .logs import DailyLog, from_preset, FIELDS
from .prakriti import QUESTIONS

app = FastAPI(title="Dosha Tracker v2")
eng = Engine(store="users_v2.json")


class ProfileIn(BaseModel):
    user_id: str
    answers: Dict[str, Optional[int]]
    age: Optional[int] = None
    sex: Optional[str] = None
    safety_flags: List[str] = []
    lat: float = 29.85
    lon: float = 77.89


class LogIn(BaseModel):
    user_id: str
    date: Optional[dt.date] = None
    preset: Optional[str] = None
    energy_morning: Optional[float] = None
    energy_midday: Optional[float] = None
    energy_late: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_restful: Optional[float] = None
    mood: Optional[str] = None
    stress: Optional[float] = None
    meal_irregularity: Optional[float] = None
    appetite: Optional[str] = None
    bowel: Optional[str] = None
    activity: Optional[str] = None
    minutes: Optional[float] = None
    intensity: Optional[float] = None
    screen_hours: Optional[float] = None
    late_screen_hours: Optional[float] = None
    study_hours: Optional[float] = None
    self_care: Optional[List[str]] = None


@app.get("/questionnaire")
def questionnaire():
    return [dict(key=k, question=q, options=o) for k, q, o, _ in QUESTIONS]


@app.get("/safety-flags")
def flags():
    return SAFETY


@app.post("/profile")
def profile(p: ProfileIn):
    return eng.create_profile(p.user_id, p.answers, p.age, p.sex, p.safety_flags, p.lat, p.lon)


@app.post("/log")
def log(l: LogIn):
    if l.user_id not in eng.users: raise HTTPException(404, "unknown user")
    data = l.model_dump(exclude={"user_id", "preset", "date"}, exclude_none=True)
    day = l.date or dt.date.today()
    dl = from_preset(l.preset, date=day, **data) if l.preset else DailyLog(date=day, **data)
    return eng.log_day(l.user_id, dl)


@app.get("/state/{uid}")
def state(uid: str): return eng.state(uid)


@app.get("/forecast/{uid}")
def forecast(uid: str, days: int = 7): return eng.forecast(uid, days)
