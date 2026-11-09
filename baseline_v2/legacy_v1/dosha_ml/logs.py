"""Daily log schema. EVERY field is optional - blank = 'no information', never 'zero'."""
import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Optional, List, Tuple
from .weather import Weather

MOODS = ("anxious", "angry", "sad", "neutral", "happy")
BOWEL = ("constipated", "normal", "loose", "sluggish")
AGNI = ("vishama", "tikshna", "manda", "sama")          # variable / sharp / slow / balanced
CYCLE = ("menstrual", "follicular", "ovulatory", "luteal")
ACTIVITIES = ("strength", "cardio", "hiit", "sports", "yoga", "pranayama", "walking")
SELF_CARE = ("abhyanga", "tongue_scraping", "oil_pulling", "ashwagandha", "triphala", "brahmi")


@dataclass
class DailyLog:
    date: dt.date = field(default_factory=dt.date.today)
    # sliders (0..1) -------------------------------------------------------
    energy: Optional[float] = None
    sleep_quality: Optional[float] = None
    stress: Optional[float] = None
    # sleep ---------------------------------------------------------------
    sleep_hours: Optional[float] = None
    bedtime_hour: Optional[float] = None               # 23.5 = 11:30 pm ; 1.0 = 1 am
    # emoji / categorical ----------------------------------------------------
    mood: Optional[str] = None
    bowel: Optional[str] = None
    agni: Optional[str] = None
    cycle_phase: Optional[str] = None                  # only asked if profile sex == female
    # meals: [(hour_of_day, "dal chawal and curd")] ---------------------------
    meals: Optional[List[Tuple[float, str]]] = None
    # activities: [(type, minutes, intensity 0..1)] ---------------------------
    activities: Optional[List[Tuple[str, float, float]]] = None
    self_care: Optional[List[str]] = None
    # student mode --------------------------------------------------------
    screen_hours: Optional[float] = None
    study_hours: Optional[float] = None
    classes: Optional[int] = None
    # implicit -------------------------------------------------------------
    weather: Optional[Weather] = None


PRESETS = {  # predefined day-types: fill the blanks the user does not want to type
    "normal_day": dict(energy=.6, sleep_hours=7.5, bedtime_hour=23.0, stress=.3, mood="neutral", bowel="normal", agni="sama"),
    "rest_day": dict(energy=.5, sleep_hours=8.5, stress=.1, mood="happy", activities=[("walking", 30, .3)]),
    "exam_day": dict(energy=.5, sleep_hours=5.5, bedtime_hour=1.0, stress=.8, mood="anxious", screen_hours=7, study_hours=9),
    "travel_day": dict(energy=.4, sleep_hours=6, stress=.6, mood="neutral", bowel="constipated"),
    "sick_day": dict(energy=.2, sleep_hours=9.5, stress=.4, mood="sad", agni="manda"),
}


def from_preset(name: str, **overrides) -> DailyLog:
    d = dict(PRESETS[name]); d.update({k: v for k, v in overrides.items() if v is not None})
    return DailyLog(**d)


def slider_to_hours(v: float, lo=3.0, hi=11.0) -> float:
    """Gradient slider 0..1 -> hours (sleep: 3..11 h, screen: 0..12 h via lo/hi)."""
    return lo + max(0.0, min(1.0, v)) * (hi - lo)
