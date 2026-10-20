import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="GlucoTwin", layout="wide")

LABELS = {
    "glucose_now": "current glucose", "g_delta_30": "30-min glucose change",
    "g_delta_60": "1-hour glucose change", "g_mean_6h": "6-hour average glucose",
    "g_std_6h": "6-hour glucose variability", "hr_mean_1h": "heart rate (1h)",
    "hr_mean_6h": "heart rate (6h)", "hrv_mean_1h": "HRV (1h)",
    "hrv_mean_6h": "HRV (6h)", "steps_1h": "steps in last hour",
    "steps_6h": "steps in last 6 hours", "sleep_hours_24h": "sleep in last 24h",
    "glucose_missing_1h": "missing glucose data", "hour": "time of day",
    "age": "age", "sex": "sex", "bmi": "BMI", "hba1c": "HbA1c",
    "years_diabetes": "years with diabetes", "on_metformin": "metformin",
    "on_sulfonylurea": "sulfonylurea", "on_insulin": "insulin",
    "hypertension": "hypertension", "egfr": "eGFR",
    "tcf7l2_risk_allele": "TCF7L2 risk allele (synthetic)",
}


@st.cache_resource
def load_model():
    return joblib.load("models/fusion_lgbm.joblib")


@st.cache_data
def load_data():
    test = pd.read_csv("data/test_patients.csv")["patient_id"].tolist()
    ds = pd.read_csv("data/dataset.csv.gz", parse_dates=["timestamp"])
    ds = ds[ds.patient_id.isin(test)]
    sn = pd.read_csv("data/sensors.csv.gz", parse_dates=["timestamp"])
    sn = sn[sn.patient_id.isin(test)]
    return ds, sn


bundle = load_model()
model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
ds, sn = load_data()

st.title("GlucoTwin: Digital Twin for Type 2 Diabetes")
st.caption("Research prototype on synthetic data. Not for clinical use.")

# ---------- Sidebar ----------
pid = st.sidebar.selectbox("Patient (test set)", sorted(ds.patient_id.unique()))
rows = ds[ds.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
i = st.sidebar.slider("Time step (30 min each)", 0, len(rows) - 1, len(rows) // 2)
r = rows.iloc[i]
t = r["timestamp"]
st.sidebar.write("Current time:", str(t))

scenario = st.sidebar.selectbox(
    "What-if scenario",
    ["None", "Walk 30 minutes now", "Sleep 8 hours last night", "Stop sulfonylurea"])

# ---------- Patient summary ----------
st.subheader(f"Patient {pid}")
c = st.columns(6)
c[0].metric("Age", int(r["age"]))
c[1].metric("BMI", r["bmi"])
c[2].metric("HbA1c", r["hba1c"])
c[3].metric("Years with diabetes", r["years_diabetes"])
c[4].metric("Hypertension", "Yes" if r["hypertension"] else "No")
meds = [n for n, k in (("Metformin", "on_metformin"),
                       ("Sulfonylurea", "on_sulfonylurea"),
                       ("Insulin", "on_insulin")) if r[k]]
c[5].metric("Medications", ", ".join(meds) if meds else "None")

# ---------- Prediction (with optional what-if) ----------
X = rows.loc[[i], feats]
base_prob = float(model.predict_proba(X)[0, 1])

if scenario == "Walk 30 minutes now":
    X = X.assign(steps_1h=X["steps_1h"] + 2100, steps_6h=X["steps_6h"] + 2100)
elif scenario == "Sleep 8 hours last night":
    X = X.assign(sleep_hours_24h=8.0)
elif scenario == "Stop sulfonylurea":
    X = X.assign(on_sulfonylurea=0)

prob = float(model.predict_proba(X)[0, 1])
contrib = model.predict(X, pred_contrib=True)[0][:-1]
order = np.argsort(contrib)[::-1]
reasons = []
for j in order[:3]:
    if contrib[j] > 0:
        val = X.iloc[0, j]
        reasons.append(f"{LABELS.get(feats[j], feats[j])} ({val:.1f})")

left, right = st.columns([2, 1])

with right:
    st.subheader("2-hour risk")
    if scenario == "None":
        st.metric("Probability of glucose excursion", f"{prob*100:.1f}%")
    else:
        st.metric(f"Risk with: {scenario}", f"{prob*100:.1f}%",
                  delta=f"{(prob - base_prob)*100:+.1f} pts vs now",
                  delta_color="inverse")
        st.caption("What-if is a model sensitivity analysis, not a physiological simulation.")
    if prob >= thr:
        st.error("ALERT: high risk of glucose excursion in the next 2 hours")
    else:
        st.success("Below alert threshold")
    st.caption(f"Alert threshold: {thr*100:.1f}% (chosen on validation data)")
    st.markdown("**Why:**")
    if prob >= thr and reasons:
        st.write("Main risk drivers: " + ", ".join(reasons))
    elif prob >= thr:
        st.write("Alert is active, but no single feature dominates.")
    else:
        st.write("Risk is low. No major risk drivers right now.")
    if r["glucose_missing_1h"] > 0.2:
        st.warning("Data quality: more than 20% of glucose readings missing in the last hour.")
    actual = bool(r["label"])
    st.markdown("**What actually happened (simulated):** "
                + ("excursion occurred" if actual else "no excursion"))

with left:
    s = sn[(sn.patient_id == pid)
           & (sn.timestamp >= t - pd.Timedelta(hours=24))
           & (sn.timestamp <= t + pd.Timedelta(hours=2))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=s.timestamp, y=s.glucose, name="Glucose (mg/dL)",
                             line=dict(color="#d62728")), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.timestamp, y=s.hr, name="Heart rate",
                             line=dict(color="#1f77b4")), row=2, col=1)
    fig.add_hline(y=180, line_dash="dash", row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", row=1, col=1)
    fig.add_vrect(x0=t, x1=t + pd.Timedelta(hours=2), fillcolor="grey",
                  opacity=0.15, line_width=0, row=1, col=1)
    fig.add_vline(x=t, line_color="black")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grey area = the next 2 hours, which the model has NOT seen.")