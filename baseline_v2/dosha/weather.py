"""Implicit input: weather (Open-Meteo, free, no API key) with an offline climatology fallback."""
import json, math, urllib.request, datetime as dt
from dataclasses import dataclass


@dataclass
class Weather:
    temp_c: float          # daily mean
    humidity: float        # %
    wind_kmh: float
    rain_mm: float = 0.0
    source: str = "api"


def climatology(lat: float, day: dt.date) -> Weather:
    """Rough sinusoidal climate for the Indo-Gangetic plain (used when offline). Replace per region."""
    doy = day.timetuple().tm_yday
    t = 24 + 9 * math.sin(2 * math.pi * (doy - 105) / 365)
    h = 55 + 20 * math.sin(2 * math.pi * (doy - 170) / 365)
    return Weather(t, h, 8.0, 0.0, "climatology")


def fetch_forecast(lat: float, lon: float, days: int = 7, timeout: float = 6.0):
    """Returns list[(date, Weather)] for today + days-1. Falls back to climatology on any failure."""
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&timezone=auto&forecast_days=%d"
           "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
           % (lat, lon, days))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)["daily"]
        out = []
        for i, day in enumerate(d["time"]):
            out.append((dt.date.fromisoformat(day), Weather(
                (d["temperature_2m_max"][i] + d["temperature_2m_min"][i]) / 2,
                d["relative_humidity_2m_mean"][i], d["wind_speed_10m_max"][i], d["precipitation_sum"][i] or 0.0)))
        return out
    except Exception:
        today = dt.date.today()
        return [(today + dt.timedelta(i), climatology(lat, today + dt.timedelta(i))) for i in range(days)]


def get_weather(lat: float, lon: float, day: dt.date) -> Weather:
    for d, w in fetch_forecast(lat, lon, 1):
        if d == day:
            return w
    return climatology(lat, day)
