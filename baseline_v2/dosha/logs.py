"""Daily log. Every field optional: blank = no information = zero push (never an imputed value)."""
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, List
from .weather import Weather


@dataclass
class DailyLog:
    date: dt.date = field(default_factory=dt.date.today)
    energy_morning: Optional[float] = None      # -1..1
    energy_midday: Optional[float] = None
    energy_late: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_restful: Optional[float] = None       # 0..1
    mood: Optional[str] = None                  # anxious irritable withdrawn tired calm happy
    stress: Optional[float] = None              # 0..1
    meal_irregularity: Optional[float] = None   # 0 regular, .5 somewhat, 1 irregular
    appetite: Optional[str] = None              # bal irr sharp slow
    bowel: Optional[str] = None                 # norm dry loose slow
    activity: Optional[str] = None              # none gym sports yoga pran walk hiit
    minutes: float = 30
    intensity: float = 0.5
    screen_hours: Optional[float] = None
    late_screen_hours: Optional[float] = None
    study_hours: Optional[float] = None
    self_care: Optional[List[str]] = None       # herb names (looked up in the herb database) or abhyanga etc.
    weather: Optional[Weather] = None


FIELDS = ["sleep_hours", "sleep_restful", "mood", "stress", "energy_morning", "energy_midday", "energy_late",
          "meal_irregularity", "appetite", "bowel", "activity", "screen_hours", "late_screen_hours", "study_hours", "self_care"]

PRESETS = {
    "normal": dict(sleep_hours=7.5, sleep_restful=.7, mood="calm", stress=.3, energy_morning=.2, energy_midday=.3, energy_late=0,
                   meal_irregularity=0, appetite="bal", bowel="norm", activity="yoga", minutes=30, intensity=.5),
    "exam": dict(sleep_hours=5, sleep_restful=.3, mood="anxious", stress=.9, energy_morning=-.3, energy_late=-.5, screen_hours=7,
                 late_screen_hours=2, study_hours=9, meal_irregularity=1, appetite="irr", bowel="dry", activity="none"),
    "rest": dict(sleep_hours=9, sleep_restful=.8, mood="happy", stress=.1, energy_morning=.5, energy_midday=.5, energy_late=.3, activity="yoga", minutes=40, intensity=.3),
    "sick": dict(sleep_hours=9.5, sleep_restful=.4, mood="tired", stress=.4, energy_morning=-.7, energy_midday=-.6, energy_late=-.6, appetite="slow", bowel="slow", activity="none"),
}


def from_preset(name, **kw):
    d = dict(PRESETS[name]); d.update({k: v for k, v in kw.items() if v is not None})
    return DailyLog(**d)
