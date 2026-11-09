"""FastAPI service.  Run:  uvicorn dosha_ml.api:app --reload"""
import datetime as dt
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .engine import DoshaEngine
from .logs import DailyLog, from_preset
from .questionnaire import QUESTIONS

app = FastAPI(title="Dosha ML")
engine = DoshaEngine(store_path="users.json")


class ProfileIn(BaseModel):
    user_id: str
    answers: Dict[str, Optional[int]]
    sex: str = "male"
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    lat: float = 29.85
    lon: float = 77.89


class LogIn(BaseModel):
    user_id: str
    date: Optional[dt.date] = None
    preset: Optional[str] = None
    energy: Optional[float] = None
    sleep_quality: Optional[float] = None
    stress: Optional[float] = None
    sleep_hours: Optional[float] = None
    bedtime_hour: Optional[float] = None
    mood: Optional[str] = None
    bowel: Optional[str] = None
    agni: Optional[str] = None
    cycle_phase: Optional[str] = None
    meals: Optional[List[Tuple[float, str]]] = None
    activities: Optional[List[Tuple[str, float, float]]] = None
    self_care: Optional[List[str]] = None
    screen_hours: Optional[float] = None
    study_hours: Optional[float] = None
    classes: Optional[int] = None


@app.get("/questionnaire")
def questionnaire():
    return [{"key": k, "question": q, "options": o} for k, q, o, _ in QUESTIONS if k != "bmi"]


@app.post("/profile")
def profile(p: ProfileIn):
    return engine.create_profile(p.user_id, p.answers, p.sex, p.age, p.height_cm, p.weight_kg, p.lat, p.lon)


@app.post("/log")
def log(l: LogIn):
    if l.user_id not in engine.users:
        raise HTTPException(404, "unknown user")
    data = l.model_dump(exclude={"user_id", "preset", "date"}, exclude_none=True)
    day = l.date or dt.date.today()
    dl = from_preset(l.preset, date=day, **data) if l.preset else DailyLog(date=day, **data)
    try:
        return engine.log_day(l.user_id, dl)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/state/{uid}")
def state(uid: str):
    return engine.state(uid)


@app.get("/forecast/{uid}")
def forecast(uid: str, days: int = 7):
    return engine.forecast(uid, days)


@app.get("/triangle/{uid}.png")
def triangle(uid: str):
    engine.plot(uid, f"/tmp/{uid}.png")
    return FileResponse(f"/tmp/{uid}.png")
