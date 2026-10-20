##### **GlucoTwin: A Digital Twin for 2-Hour Early Warning of Glucose Excursions**

##### 

##### **Happiest Health Digital Twin Challenge 2026**



\- Team Lead:Kashish Jain

\- Institution:SRM Institute of Science and Technology, Kattankulathur

\- Project Name:GlucoTwin

\- GitHub Repository:https://github.com/kj5805/GlucoTwin

\- License:MIT License

1\. Project Overview

GlucoTwin is a research prototype that explores how a digital twin-inspired system can estimate the risk of a glucose excursion over the next two hours by combining simulated glucose sensor readings, wearable signals, and synthetic electronic health record (EHR) features.



The system compares machine-learning models trained on medical-history features, wearable and sensor features, and a combination of both. An interactive dashboard presents risk estimates, historical trends, model information, and what-if sensitivity analyses.

**Objective:**Explore whether combining multiple data sources can improve early awareness of potential glucose excursions.

2\. Problem Statement



Glucose excursions can be influenced by several factors, including current glucose levels, recent glucose trends, sleep, physical activity, and individual health characteristics. A single measurement may not capture these changing patterns.



GlucoTwin explores a data-driven approach that combines multiple simulated data sources to estimate the risk of a future glucose excursion within a two-hour prediction window.

3\. Proposed Solution



GlucoTwin integrates a synthetic data pipeline, machine-learning model comparison, and an interactive dashboard.



The prototype:

\- Generates and processes synthetic patient and sensor data.

\- Extracts features from glucose trends and wearable signals.

\- Compares EHR-only, wearable-only, and multimodal fusion models.

\- Estimates the probability of a glucose excursion in the next two hours.

\- Displays historical trends, risk estimates, and influential model features.

\- Supports controlled what-if sensitivity analysis.

&#x20;4. Key Features



\- Two-hour risk estimation: Estimates the probability of a future glucose excursion.

\- Multimodal data fusion: Combines sensor, wearable, and EHR-derived features.

\- Model comparison:Evaluates three machine-learning approaches.

\- Interactive dashboard:Displays risk estimates, thresholds, and historical trends.

\- Explainability:Highlights features influencing model predictions.

\- What-if analysis:Explores how selected input changes affect predictions.

\- Synthetic-data prototyping:Supports experimentation without using real patient records.

&#x20;5. Dataset and Methodology



The prototype uses synthetic data representing 300 simulated patients, with 14 days of simulated sensor observations per patient at five-minute intervals.

&#x20;Data features



\- Glucose readings and recent glucose trends

\- Heart rate and heart-rate variability (HRV)

\- Physical activity and sleep features

\- Selected medical-history and laboratory features, including BMI and HbA1c

Prediction target



A future glucose excursion is labelled when three consecutive future readings are above 180 mg/dL or below 70 mg/dL. Current out-of-range readings are excluded from the target definition.

&#x20;Evaluation methodology



\- Patients are separated into training, validation, and test sets.

\- Splitting is performed at the patient level to reduce patient-level data leakage.

\- Reported results are based on a single train/validation/test split.

\- Models are compared using AUROC, AUPRC, and sensitivity at a target false-positive rate.

6\. Model Performance

Model Performance Results



1\. EHR-only Logistic Regression

&#x20;  AUROC: 0.6657

&#x20;  AUPRC: 0.0885

&#x20;  Sensitivity at 10% FPR: 0.2418



2\. Wearable-only LightGBM

&#x20;  AUROC: 0.9087

&#x20;  AUPRC: 0.3576

&#x20;  Sensitivity at 10% FPR: 0.7171



3\. Fusion LightGBM

&#x20;  AUROC: 0.9302

&#x20;  AUPRC: 0.4215

&#x20;  Sensitivity at 10% FPR: 0.7763







&#x20;Key observations



\- The fusion model achieved the highest AUROC and AUPRC among the three evaluated models.

\- Wearable and sensor-derived features showed stronger predictive performance than EHR-only features in this experiment.

\- Combining data sources showed promise for estimating future glucose excursion risk.



These findings are preliminary and apply only to the current synthetic dataset and evaluation split. They do not establish clinical effectiveness.



&#x20;7. System Architecture



The prototype follows this pipeline:



1\. Synthetic EHR Data:Demographics, medical history, and laboratory features.

2\. Simulated Wearable and Sensor Data: Glucose, heart rate, HRV, sleep, and activity.

3\. Data Preprocessing: Cleaning, alignment, and feature engineering.

4\. Combined Dataset:Integration of relevant features.

5\. Patient-Level Splitting:Training, validation, and testing.

6\. Model Training and Comparison: Logistic Regression and LightGBM.

7\. Two-Hour Risk Estimation:Model probability and threshold-based risk indication.

8\. Interactive Dashboard:Historical trends, risk estimates, explanations, and what-if analysis.



Architecture Diagram:\[View GlucoTwin System Architecture](https://drive.google.com/file/d/1TFCd8xj5eyBpmDCU\_atQSL53OGzRIYEJ/view?usp=drive\_link)



&#x20;8. Technology Stack



Programming Language: Python

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, LightGBM

Dashboard: Streamlit

Visualization: Plotly

Version Control: Git, GitHub



&#x20;9. Installation and Usage



Prerequisites



\- Python installed

\- Git installed

\- A terminal such as PowerShell



&#x20;Step 1: Clone the repository



```bash

git clone https://github.com/kj5805/GlucoTwin.git

cd GlucoTwin

```



\### Step 2: Install dependencies



```bash

pip install -r requirements.txt

```



Step 3: Run the dashboard



```bash

streamlit run app/app.py

```



&#x20;Optional: Train and evaluate the models



```bash

python src/train.py

```



The training command may regenerate model artifacts and evaluation outputs.



10\. Project Deliverables



\- Public GitHub Repository:https://github.com/kj5805/GlucoTwin

\- Project Presentation: \[View GlucoTwin Presentation](https://docs.google.com/presentation/d/1SdbzV9eVr1XYxxbdd\_ROSW8cvZXYEDJD/edit?usp=sharing)

\- System Architecture:\[View Architecture Diagram](https://drive.google.com/file/d/1TFCd8xj5eyBpmDCU\_atQSL53OGzRIYEJ/view?usp=drive\_link)

\- Prototype Demonstration Video:\[VIDEO LINK TO BE ADDED]

Video Demonstration



The demonstration video will cover the problem statement, proposed solution, system architecture, dataset and methodology, model evaluation, interactive dashboard, and project limitations.



Video URL: 



PASTE YOUR VIDEO LINK HERE AFTER UPLOADING



11\. Limitations and Responsible Use



\- This prototype uses synthetic data and does not establish performance on real patients.

\- Simulated data may not represent real-world patient populations or clinical conditions.

\- The evaluation is based on a single patient-level data split and requires further testing.

\- What-if scenarios demonstrate model sensitivity, not proven physiological effects or treatment outcomes.

\- Risk estimates are not medical diagnoses and must not be used to make treatment decisions.

\- Clinical use would require appropriate real-world validation, clinical review, privacy safeguards, and regulatory assessment.



GlucoTwin is a research prototype, not a clinically validated medical system. It is not intended to replace professional medical advice or guide treatment 12. Future Scope



Potential future improvements include:



\- Validation on appropriately governed real-world datasets.

\- Repeated patient-level evaluation and independent external testing.

\- Probability calibration and assessment across different operating thresholds.

\- Improved handling of missing or noisy sensor readings.

\- Further explainability and fairness analysis.

\- Evaluation of alert usefulness and false-alarm burden.



13\. License



This project is released under the \*\*MIT License\*\*. See \[`LICENSE.txt`](LICENSE.txt) for the complete license text.



14\. Acknowledgements



Developed for the Happiest Health Digital Twin Challenge 2026 as a research prototype exploring digital-twin-inspired approaches to glucose excursion risk estimation.

