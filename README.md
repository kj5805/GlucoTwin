GlucoTwin: A Digital Twin for 2-Hour Early Warning of Glucose Excursions

Happiest Health Digital Twin Challenge 2026**  
Team Lead: Kashish Jain  
Institution: SRM Institute of Science and Technology, Kattankulathur

 Overview

GlucoTwin is a research prototype that combines synthetic electronic health records (EHR) with simulated glucose-sensor and wearable data to estimate the risk of a glucose excursion over the next two hours.

It explores whether combining medical history with recent glucose trends, heart rate, heart-rate variability (HRV), sleep, and physical activity can improve early-warning predictions.

Important:All data used in this prototype is synthetic. GlucoTwin has not been clinically validated and must not be used to make medical decisions.

 Key Features

-Multimodal data fusion:** Combines medical-record and wearable features.
- Two-hour early warning:** Predicts simulated future glucose excursions.
-Patient-level evaluation:** Keeps patients separate across training, validation, and test sets.
- Model comparison:Compares Logistic Regression and LightGBM approaches.
- Interactive dashboard:Displays patient summaries, trends, risk scores, and alert status.
- Model explanations: Uses feature contributions to inspect individual predictions.
-What-if analysis: Explores how changing selected inputs affects model predictions.

Dataset

The prototype uses simulated records for 300 synthetic patients, with 14 days of five-minute sensor observations per patient. No real patient data is used.

The simulated data includes glucose, heart rate, HRV, steps, sleep, and selected medical-history features.

Evaluation Results

| Model | AUROC | AUPRC | Sensitivity at 10% FPR |
| History-only Logistic Regression | 0.6657 | 0.0885 | 0.2418 |
| Wearable-only LightGBM | 0.9087 | 0.3576 | 0.7171 |
| Fusion LightGBM | 0.9302 | 0.4215 | 0.7763 |

Test-set positive rate: 4.83%.

These results come from a synthetic-data experiment. The test set contains 45 patients from a single patient-level split. Clinical effectiveness has not been established.

 Technology Stack

- Python
- NumPy and pandas
- scikit-learn
- LightGBM
- Joblib
- Streamlit
- Plotly

 Setup and Run

Clone the repository:

```bash
git clone https://github.com/kj5805/GlucoTwin.git
cd GlucoTwin
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Generate the synthetic data and train the models:

```bash
python src/make_ehr.py
python src/simulate_sensors.py
python src/build_dataset.py
python src/train.py
```

Launch the dashboard:

```bash
streamlit run app/app.py
```

Run the commands from the project root, in the order shown.

 Limitations and Responsible Use

- The dataset is synthetic and may not represent real-world patient populations.
- Relationships built into the simulator may inflate predictive performance.
- Evaluation uses one patient-level split with only 45 test patients.
- Confidence intervals, external validation, and clinical evaluation have not been performed.
- Feature contributions do not establish causation.
- What-if predictions are model sensitivity analyses, not physiological simulations or treatment recommendations.

GlucoTwin is an educational research prototype, not a medical device or diagnostic system. It must not be used to change medication or replace professional medical advice.

 Repository

[GitHub: kj5805/GlucoTwin](https://github.com/kj5805/GlucoTwin)

## Acknowledgements

Developed for the Happiest Health Digital Twin Challenge 2026.
