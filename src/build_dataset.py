import numpy as np
import pandas as pd

ehr = pd.read_csv("data/ehr.csv")
ehr["sex"] = (ehr["sex"] == "M").astype(int)
sensors = pd.read_csv("data/sensors.csv.gz", parse_dates=["timestamp"])

HIST = 288   # need 24 hours of past data
FUT = 24     # predict the next 2 hours (24 readings)
STEP = 6     # one example every 30 minutes

parts = []
for pid, d in sensors.groupby("patient_id"):
    d = d.sort_values("timestamp").reset_index(drop=True)
    T = len(d)
    g_raw = d["glucose"]
    g = g_raw.ffill(limit=12)            # fill short gaps using PAST values only
    asleep = (d["sleep_stage"] > 0).astype(int)

    # ---------- Features: only data up to "now" ----------
    f = pd.DataFrame(index=d.index)
    f["glucose_now"] = g
    f["g_delta_30"] = g - g.shift(6)
    f["g_delta_60"] = g - g.shift(12)
    f["g_mean_6h"] = g.rolling(72, min_periods=1).mean()
    f["g_std_6h"] = g.rolling(72, min_periods=2).std()
    f["hr_mean_1h"] = d["hr"].rolling(12, min_periods=1).mean()
    f["hr_mean_6h"] = d["hr"].rolling(72, min_periods=1).mean()
    f["hrv_mean_1h"] = d["hrv"].rolling(12, min_periods=1).mean()
    f["hrv_mean_6h"] = d["hrv"].rolling(72, min_periods=1).mean()
    f["steps_1h"] = d["steps"].rolling(12, min_periods=1).sum()
    f["steps_6h"] = d["steps"].rolling(72, min_periods=1).sum()
    f["sleep_hours_24h"] = asleep.rolling(288).sum() * 5 / 60
    f["glucose_missing_1h"] = g_raw.isna().rolling(12).mean()
    f["hour"] = d["timestamp"].dt.hour

    # ---------- Label: uses the FUTURE only ----------
    bad = ((g_raw > 180) | (g_raw < 70)).to_numpy().astype(int)
    r = np.zeros(T, dtype=int)
    r[:T - 2] = bad[:-2] & bad[1:-1] & bad[2:]       # 3 bad readings in a row (15 min)
    cs = np.concatenate([[0], np.cumsum(r)])
    ii = np.arange(HIST, T - FUT, STEP)
    label = (cs[ii + 23] - cs[ii + 1]) > 0           # a run starting in the next 2 hours

    f = f.iloc[ii].copy()
    f["label"] = label.astype(int)
    f["patient_id"] = pid
    f["timestamp"] = d["timestamp"].iloc[ii].values
    parts.append(f)

df = pd.concat(parts, ignore_index=True)

# Keep only real forecasts: glucose is known and currently in range
df = df.dropna(subset=["glucose_now"])
df = df[(df.glucose_now >= 70) & (df.glucose_now <= 180)]

df = df.merge(ehr, on="patient_id", how="left")
df.to_csv("data/dataset.csv.gz", index=False)

print(df.shape)
print("Patients:", df.patient_id.nunique())
print("Positive rate:", round(df.label.mean() * 100, 1), "%")
