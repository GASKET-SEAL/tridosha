"""Classical sign tests (plan step 'validation by classical rules')."""
import sys, datetime as dt; sys.path.insert(0, ".")
import numpy as np
from dosha.vikriti import DayRecord, run
from dosha.drivers import env_push, life_push
from dosha.weather import Weather
from dosha.logs import DailyLog
P = np.ones(3) / 3; D0 = dt.date(2026, 1, 1)


def hist(w, log=None, n=2):
    lp = life_push(log or DailyLog()); fe = env_push(w, 24, D0)
    return [DayRecord(D0 + dt.timedelta(i), lp["fl"], fe - 0.15 * np.array([0, 0, 0.5]) * 0) for i in range(n)]


def lead(h): x = run(h, P)[-1][1]; return int(np.argmax(x))


def tests():
    out = []
    out.append(("cold, dry, windy day raises Vata", lead(hist(Weather(5, 30, 25, 0, "t"))) == 0))
    out.append(("hot day raises Pitta", lead(hist(Weather(42, 30, 4, 0, "t"))) == 1))
    out.append(("damp, heavy, sedentary day raises Kapha", lead(hist(Weather(14, 92, 2, 6, "t"), DailyLog(activity="none"))) == 2))
    shock = run([DayRecord(D0, np.array([1., 0, 0]), np.zeros(3))], P)[0][1]
    rest = run([DayRecord(D0, np.array([1., 0, 0]), np.zeros(3)), DayRecord(D0 + dt.timedelta(8), np.zeros(3), np.zeros(3))], P)[-1][1]
    out.append(("blank days drift back toward Prakriti", np.linalg.norm(rest) < .3 * np.linalg.norm(shock)))
    big = run([DayRecord(D0 + dt.timedelta(i), np.array([5., 0, 0]), np.zeros(3)) for i in range(30)], P)[-1][2]
    out.append(("extreme push never reaches a corner", big.max() < .97))
    return out


if __name__ == "__main__":
    for n, ok in tests(): print(("PASS " if ok else "FAIL ") + n)
