import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 300

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

age = rng.normal(55, 10, N).clip(30, 85).round()
sex = rng.choice(["M", "F"], N)
bmi = rng.normal(28, 5, N).clip(18, 45).round(1)
years_diabetes = rng.gamma(2, 3, N).clip(0, 30).round(1)

hba1c = (6.0 + 0.04 * (bmi - 28) + 0.05 * years_diabetes
         + rng.normal(0, 1.0, N)).clip(5.5, 12).round(1)

on_metformin = rng.binomial(1, 0.85, N)
on_sulfonylurea = rng.binomial(1, sigmoid(-1.5 + 0.08 * years_diabetes))
on_insulin = rng.binomial(1, sigmoid(-3 + 0.12 * years_diabetes + 0.3 * (hba1c - 8)))
hypertension = rng.binomial(1, sigmoid(-4 + 0.06 * age + 0.05 * (bmi - 25)))
egfr = (110 - 0.8 * (age - 30) + rng.normal(0, 10, N)).clip(20, 120).round()
tcf7l2_risk_allele = rng.binomial(1, 0.30, N)

ehr = pd.DataFrame({
    "patient_id": [f"P{i:03d}" for i in range(N)],
    "age": age, "sex": sex, "bmi": bmi,
    "hba1c": hba1c, "years_diabetes": years_diabetes,
    "on_metformin": on_metformin, "on_sulfonylurea": on_sulfonylurea,
    "on_insulin": on_insulin, "hypertension": hypertension,
    "egfr": egfr, "tcf7l2_risk_allele": tcf7l2_risk_allele,
})

ehr.to_csv("data/ehr.csv", index=False)
print(ehr.shape)
print(ehr.describe().round(1))