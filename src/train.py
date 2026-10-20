import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

df = pd.read_csv("data/dataset.csv.gz")

SENSOR = ["glucose_now", "g_delta_30", "g_delta_60", "g_mean_6h", "g_std_6h",
          "hr_mean_1h", "hr_mean_6h", "hrv_mean_1h", "hrv_mean_6h",
          "steps_1h", "steps_6h", "sleep_hours_24h", "glucose_missing_1h", "hour"]
EHR = ["age", "sex", "bmi", "hba1c", "years_diabetes", "on_metformin",
       "on_sulfonylurea", "on_insulin", "hypertension", "egfr", "tcf7l2_risk_allele"]

# ---------- Split by PATIENT (no patient appears in two sets) ----------
rng = np.random.default_rng(42)
pids = df["patient_id"].unique()
rng.shuffle(pids)
n = len(pids)
train_p = set(pids[: int(0.70 * n)])
val_p = set(pids[int(0.70 * n): int(0.85 * n)])
test_p = set(pids[int(0.85 * n):])
assert not (train_p & val_p) and not (train_p & test_p) and not (val_p & test_p)

tr = df[df.patient_id.isin(train_p)]
va = df[df.patient_id.isin(val_p)]
te = df[df.patient_id.isin(test_p)]
print("Patients  train/val/test:", len(train_p), len(val_p), len(test_p))
print("Rows      train/val/test:", len(tr), len(va), len(te))
print("Positive  train/val/test:", round(tr.label.mean(), 3),
      round(va.label.mean(), 3), round(te.label.mean(), 3))


def lgbm():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=42, verbose=-1)


def fit(kind, feats):
    if kind == "logreg":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000))
        m.fit(tr[feats], tr.label)
    else:
        m = lgbm()
        m.fit(tr[feats], tr.label, eval_set=[(va[feats], va.label)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def evaluate(m, feats):
    # alert threshold is chosen on VALIDATION data only (10% false alarm rate)
    pv = m.predict_proba(va[feats])[:, 1]
    thr = float(np.quantile(pv[va.label.values == 0], 0.90))
    pt = m.predict_proba(te[feats])[:, 1]
    y = te.label.values
    fpr, tpr, _ = roc_curve(y, pt)
    alert = pt >= thr
    return {
        "auroc": round(roc_auc_score(y, pt), 4),
        "auprc": round(average_precision_score(y, pt), 4),
        "sens_at_10pct_fpr": round(float(np.interp(0.10, fpr, tpr)), 4),
        "alert_threshold": round(thr, 4),
        "test_sensitivity_at_threshold": round(float(alert[y == 1].mean()), 4),
        "test_false_alarm_rate_at_threshold": round(float(alert[y == 0].mean()), 4),
    }


experiments = {
    "history_only_logreg": ("logreg", EHR),
    "wearable_only_lgbm": ("lgbm", SENSOR),
    "fusion_lgbm": ("lgbm", SENSOR + EHR),
}

results, models = {}, {}
for name, (kind, feats) in experiments.items():
    m = fit(kind, feats)
    results[name] = evaluate(m, feats)
    models[name] = (m, feats)
    print(f"\n{name}\n", results[name])

results["test_base_rate"] = round(float(te.label.mean()), 4)
results["note"] = ("Synthetic data only. Results show the pipeline works, "
                   "not clinical accuracy.")

fusion, fusion_feats = models["fusion_lgbm"]
joblib.dump({"model": fusion, "features": fusion_feats,
             "threshold": results["fusion_lgbm"]["alert_threshold"]},
            "models/fusion_lgbm.joblib")
with open("reports/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
pd.Series(sorted(test_p)).to_csv("data/test_patients.csv", index=False, header=["patient_id"])

imp = pd.Series(fusion.feature_importances_, index=fusion_feats).sort_values(ascending=False)
print("\nTop fusion features:\n", imp.head(10))
print("\nSaved models/fusion_lgbm.joblib and reports/evaluation.json")
