
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="GlucoTwin", layout="wide")

LABELS = {
    "glucose_now": "current glucose",
    "g_delta_30": "30-min glucose change",
    "g_delta_60": "1-hour glucose change",
    "g_mean_6h": "6-hour average glucose",
    "g_std_6h": "6-hour glucose variability",
    "hr_mean_1h": "heart rate (1h)",
    "hr_mean_6h": "heart rate (6h)",
    "hrv_mean_1h": "HRV (1h)",
    "hrv_mean_6h": "HRV (6h)",
    "steps_1h": "steps in last hour",
    "steps_6h": "steps in last 6 hours",
    "sleep_hours_24h": "sleep in last 24h",
    "glucose_missing_1h": "missing glucose data",
    "hour": "time of day",
    "age": "age",
    "sex": "sex",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "years_diabetes": "years with diabetes",
    "on_metformin": "metformin",
    "on_sulfonylurea": "sulfonylurea",
    "on_insulin": "insulin",
    "hypertension": "hypertension",
    "egfr": "eGFR",
    "tcf7l2_risk_allele": "TCF7L2 risk allele (synthetic)",
}


@st.cache_resource
def load_model():
    return joblib.load("models/fusion_lgbm.joblib")


@st.cache_data
def load_data():
    test = pd.read_csv("data/test_patients.csv")["patient_id"].tolist()

    ds = pd.read_csv(
        "data/dataset.csv.gz",
        parse_dates=["timestamp"],
    )
    ds = ds[ds.patient_id.isin(test)]

    sn = pd.read_csv(
        "data/sensors.csv.gz",
        parse_dates=["timestamp"],
    )
    sn = sn[sn.patient_id.isin(test)]

    return ds, sn


bundle = load_model()
model = bundle["model"]
feats = bundle["features"]
thr = bundle["threshold"]

ds, sn = load_data()

st.title("GlucoTwin: Digital Twin for Type 2 Diabetes")
st.caption(
    "Research prototype using synthetic data only. "
    "Not validated for clinical use or medical decisions."
)

# ---------- Sidebar ----------
pid = st.sidebar.selectbox(
    "Patient (test set)",
    sorted(ds.patient_id.unique()),
)

rows = (
    ds[ds.patient_id == pid]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

i = st.sidebar.slider(
    "Time step (30 min each)",
    0,
    len(rows) - 1,
    len(rows) // 2,
)

r = rows.iloc[i]
t = r["timestamp"]

st.sidebar.write("Current time:", str(t))

scenario = st.sidebar.selectbox(
    "What-if scenario",
    [
        "None",
        "Walk 30 minutes now",
        "Sleep 8 hours last night",
        "Medication-indicator sensitivity test",
    ],
)

reveal_outcome = st.sidebar.checkbox(
    "Reveal simulated outcome (retrospective)",
    value=False,
    help=(
        "Reveal future simulated sensor readings and the known outcome "
        "for retrospective evaluation. These are not available at "
        "forecast time."
    ),
)

# ---------- Patient summary ----------
st.subheader(f"Patient {pid}")

c = st.columns(6)

c[0].metric("Age", int(r["age"]))
c[1].metric("BMI", r["bmi"])
c[2].metric("HbA1c", r["hba1c"])
c[3].metric("Years with diabetes", r["years_diabetes"])
c[4].metric(
    "Hypertension",
    "Yes" if r["hypertension"] else "No",
)

meds = [
    name
    for name, key in [
        ("Metformin", "on_metformin"),
        ("Sulfonylurea", "on_sulfonylurea"),
        ("Insulin", "on_insulin"),
    ]
    if r[key]
]

c[5].metric(
    "Medications",
    ", ".join(meds) if meds else "None",
)

# ---------- Prediction and what-if analysis ----------
X = rows.loc[[i], feats].copy()

base_prob = float(model.predict_proba(X)[0, 1])

if scenario == "Walk 30 minutes now":
    X = X.assign(
        steps_1h=X["steps_1h"] + 2100,
        steps_6h=X["steps_6h"] + 2100,
    )

elif scenario == "Sleep 8 hours last night":
    X = X.assign(sleep_hours_24h=8.0)

elif scenario == "Medication-indicator sensitivity test":
    X = X.assign(on_sulfonylurea=0)

prob = float(model.predict_proba(X)[0, 1])

# Model feature contributions for the selected input.
contrib = model.predict(X, pred_contrib=True)[0][:-1]
order = np.argsort(contrib)[::-1]

reasons = []

for j in order:
    if len(reasons) == 3:
        break

    if contrib[j] > 0:
        val = X.iloc[0, j]
        reasons.append(
            f"{LABELS.get(feats[j], feats[j])} ({val:.1f})"
        )

# ---------- Prediction dashboard ----------
left, right = st.columns([2, 1])

with right:
    st.subheader("2-hour risk")

    if scenario == "None":
        st.metric(
            "Probability of glucose excursion",
            f"{prob * 100:.1f}%",
        )
    else:
        st.metric(
            f"Risk with: {scenario}",
            f"{prob * 100:.1f}%",
            delta=f"{(prob - base_prob) * 100:+.1f} pts vs baseline",
            delta_color="inverse",
        )

        st.caption(
            "What-if results show model sensitivity to changed input "
            "features. They do not simulate physiological effects "
            "or provide treatment advice."
        )

    if prob >= thr:
        st.error(
            "Model alert: predicted excursion risk exceeds "
            "the alert threshold."
        )
    else:
        st.success(
            "Prediction is below the model's alert threshold."
        )

    st.caption(
        f"Alert threshold: {thr * 100:.1f}% "
        "(selected using validation data)."
    )

    st.markdown("**Why this prediction?**")

    if prob >= thr and reasons:
        st.write("Main positive model contributors: " + ", ".join(reasons))

    elif prob >= thr:
        st.write(
            "The alert is active, but no individual feature "
            "has a positive contribution among the leading features."
        )

    else:
        st.write(
            "The prediction is below the alert threshold. "
            "This is not a clinical assessment or a guarantee "
            "that no excursion will occur."
        )

    if r["glucose_missing_1h"] > 0.2:
        st.warning(
            "Data quality warning: more than 20% of glucose readings "
            "were missing in the last hour."
        )

    # Reveal the known simulated outcome only on request.
    if reveal_outcome:
        actual = bool(r["label"])

        st.markdown("**Retrospective outcome (simulated):**")

        if actual:
            st.write(
                "A qualifying glucose excursion occurred "
                "in the forecast window."
            )
        else:
            st.write(
                "No qualifying glucose excursion occurred "
                "in the forecast window."
            )
    else:
        st.caption(
            "Actual simulated outcome is hidden. Enable the retrospective "
            "checkbox in the sidebar to reveal it."
        )

# ---------- Sensor chart ----------
with left:
    # Default: show only sensor readings up to the prediction time.
    # Retrospective mode optionally reveals the next two hours.
    chart_end = (
        t + pd.Timedelta(hours=2)
        if reveal_outcome
        else t
    )

    s = sn[
        (sn.patient_id == pid)
        & (sn.timestamp >= t - pd.Timedelta(hours=24))
        & (sn.timestamp <= chart_end)
    ].sort_values("timestamp")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=s.timestamp,
            y=s.glucose,
            name="Glucose (mg/dL)",
            line=dict(color="#d62728"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=s.timestamp,
            y=s.hr,
            name="Heart rate",
            line=dict(color="#1f77b4"),
        ),
        row=2,
        col=1,
    )

    # Glucose reference lines.
    fig.add_hline(
        y=180,
        line_dash="dash",
        row=1,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        row=1,
        col=1,
    )

    # Shade the forecast window even when future readings are hidden.
    fig.add_vrect(
        x0=t,
        x1=t + pd.Timedelta(hours=2),
        fillcolor="grey",
        opacity=0.15,
        line_width=0,
        row=1,
        col=1,
    )

    # Mark the prediction time.
    fig.add_vline(
        x=t,
        line_color="black",
    )

    # Keep the complete forecast window visible on the x-axis.
    fig.update_xaxes(
        range=[
            t - pd.Timedelta(hours=24),
            t + pd.Timedelta(hours=2),
        ]
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h"),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    if reveal_outcome:
        st.caption(
            "Retrospective mode: future simulated sensor readings are "
            "shown for comparison. They were not available to the model "
            "at forecast time."
        )
    else:
        st.caption(
            "Grey area = the next 2 hours. Future sensor readings are "
            "hidden to show only the information available at forecast "
            "time."
        )
