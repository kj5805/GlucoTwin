import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
ehr = pd.read_csv("data/ehr.csv")

DAYS = 14
PER_DAY = 288            # one reading every 5 minutes
T = DAYS * PER_DAY
times = pd.date_range("2026-01-01 00:00", periods=T, freq="5min")
idx = np.arange(T)


def ar_noise(n, phi, sd):
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def simulate(p):
    # ---------- Sleep (about 11 PM each night) ----------
    nights = np.clip(rng.normal(6.8, 1.0, DAYS + 1), 4, 9)   # hours slept
    asleep = np.zeros(T, bool)
    stage = np.zeros(T, int)          # 0 awake, 1 light, 2 deep, 3 REM
    dip = np.zeros(T)                 # night-time glucose drop from medication
    for k in range(DAYS + 1):
        bed = (k - 1) * 1440 + int(rng.normal(1380, 30))
        wake = bed + int(nights[k] * 60)
        a, b = max(bed // 5, 0), min(wake // 5, T)
        if b <= a:
            continue
        asleep[a:b] = True
        mins = np.arange(a, b) * 5 - bed
        ph = mins % 90
        stage[a:b] = np.where(ph < 20, 1, np.where(ph < 50, 2, np.where(ph < 75, 1, 3)))
        depth = (20 * p.on_sulfonylurea + 25 * p.on_insulin) * rng.uniform(0.5, 2.0)
        dip[a:b] = depth * np.sin(np.pi * (np.arange(a, b) - bed / 5) / ((wake - bed) / 5))

    # ---------- Insulin resistance (EHR + last night's sleep) ----------
    ir0 = (1 + 0.06 * (p.bmi - 28) + 0.10 * (p.hba1c - 7.5)
           + 0.01 * p.years_diabetes + 0.15 * p.tcf7l2_risk_allele)
    ir_day = np.clip(ir0 * (1 + 0.08 * (7 - nights[:DAYS])), 0.5, 2.5)

    # ---------- Steps ----------
    steps = np.zeros(T)
    tod = (idx % PER_DAY) * 5
    awake_day = (~asleep) & (tod >= 7 * 60) & (tod < 22 * 60)
    steps[awake_day] = rng.poisson(6, awake_day.sum())

    def add_walk(s, L):
        e = min(s + L, T)
        n = e - s
        if n > 0:
            steps[s:e] += rng.normal(350, 80, n).clip(100)

    meals = []
    for d in range(DAYS):
        for hour, size in ((8, 0.9), (13, 1.1), (20, 1.2)):
            m = d * PER_DAY + int((hour * 60 + rng.normal(0, 30)) // 5)
            meals.append((m, d, size * rng.uniform(0.7, 1.3)))
            if rng.random() < 0.4:                       # walk after the meal
                add_walk(m + int(rng.integers(4, 7)), 3)
        for _ in range(int(rng.integers(1, 3))):         # random walks
            add_walk(d * PER_DAY + int(rng.integers(7 * 12, 21 * 12)), int(rng.integers(2, 6)))

    # ---------- Glucose ----------
    base = 80 + (p.hba1c - 5.5) * 18 - 5 * p.on_metformin
    g = np.full(T, base) - dip
    for m, d, size in meals:
        post = steps[m + 4:m + 18].sum()                 # walking 20-90 min after meal
        reduction = 1 - 0.5 * min(post / 1500, 1)
        amp = 55 * ir_day[d] * size * reduction * (1 - 0.15 * p.on_metformin)
        e = min(m + 48, T)
        tau = (idx[m:e] - m) * 5.0
        g[m:e] += amp * (tau / 45) * np.exp(1 - tau / 45)
    g = g + ar_noise(T, 0.98, 1.5) + rng.normal(0, 2, T)
    g = np.clip(g, 40, 400)

    # ---------- Heart rate ----------
    rest = 58 + 0.15 * (p.age - 40) + 0.3 * (p.bmi - 25) + 4 * p.hypertension
    hr = (rest + 0.11 * steps - 8 * asleep + 0.04 * np.clip(g - 180, 0, None)
          + ar_noise(T, 0.9, 1.0) + rng.normal(0, 1.5, T))
    hr = np.clip(hr, 40, 190)

    # ---------- HRV ----------
    sleep_day = np.repeat(nights[:DAYS], PER_DAY)
    hrv = (60 - 0.5 * (p.age - 30) - 8 * p.hypertension - 0.3 * (p.bmi - 25)) \
          * (1 - 0.05 * (7 - sleep_day))
    hrv = hrv * (1 + 0.15 * asleep) - 0.02 * steps + rng.normal(0, 4, T)
    hrv = np.clip(hrv, 8, 120)

    # ---------- Sensor problems: dropout gaps + random missing ----------
    gap = np.zeros(T, bool)
    for d in range(DAYS):
        if rng.random() < 0.5:
            s = d * PER_DAY + int(rng.integers(0, PER_DAY - 12))
            gap[s:s + int(rng.integers(6, 13))] = True    # 30-60 minute gap
    glucose = np.where(gap | (rng.random(T) < 0.02), np.nan, g)
    hr = np.where(gap | (rng.random(T) < 0.02), np.nan, hr)
    hrv = np.where(gap | (rng.random(T) < 0.02), np.nan, hrv)
    steps = np.where(gap, np.nan, steps)

    return pd.DataFrame({
        "patient_id": p.patient_id, "timestamp": times,
        "glucose": glucose.round(1), "hr": hr.round(1), "hrv": hrv.round(1),
        "steps": steps.round(), "sleep_stage": stage,
    })


frames = []
for p in ehr.itertuples():
    frames.append(simulate(p))
sensors = pd.concat(frames, ignore_index=True)
sensors.to_csv("data/sensors.csv.gz", index=False)

print(sensors.shape)
print(sensors[["glucose", "hr", "hrv", "steps"]].describe().round(1))
print("Missing share:", sensors[["glucose", "hr", "hrv"]].isna().mean().round(3).to_dict())
print("Glucose > 180:", round((sensors.glucose > 180).mean() * 100, 1), "%")
print("Glucose < 70 :", round((sensors.glucose < 70).mean() * 100, 1), "%")