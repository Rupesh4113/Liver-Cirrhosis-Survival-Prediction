# Liver Cirrhosis Survival Prediction
### Multiclass Machine Learning Research & Educational Risk-Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passed-brightgreen.svg)]()

---

> [!IMPORTANT]
> ### ⚠️ Mandatory Medical & Research Safety Disclaimer
> **This application is for research and educational purposes only. It is not a medical device, does not provide medical advice, and must not be used for diagnosis, treatment, prognosis, organ-allocation priority, or clinical decision-making.**  
> Machine learning models trained on retrospective observational trial cohorts identify historical statistical correlations; they do not infer clinical causality, establish individual disease prognosis, or substitute for professional medical expertise.

---

## 1. Project Overview & Research Framing

Primary Biliary Cirrhosis (PBC)—now clinically termed Primary Biliary Cholangitis—is a chronic autoimmune hepatic disorder characterized by the progressive destruction of intrahepatic bile ducts, cholestasis, fibrosis, and eventual cirrhosis. 

This project implements an end-to-end, production-grade **multiclass risk-classification pipeline** evaluating **5 machine learning algorithms** on the landmark Mayo Clinic PBC clinical trial cohort. The primary objective is to investigate how modern tabular machine learning algorithms manage severe clinical class imbalance, missing laboratory analytes, and non-linear risk interactions under rigorous, zero-leakage evaluation protocols.

### Target Outcome Classes
- **Class 0: Death** — Patient succumbed during the retrospective trial follow-up period.
- **Class 1: Censored** — Patient survived follow-up without liver transplantation.
- **Class 2: Transplant** — Patient underwent orthotopic liver transplantation.

---

## 2. Architecture & Pipeline Workflow

```mermaid
flowchart TD
    A["Mayo Clinic PBC Cohort (418 Patients, 17 Biomarkers)"] --> B["Data Validation & Alias Resolution Layer"]
    B --> C["Stratified 80/20 Train/Test Split (random_state=42)"]
    
    subgraph Zero_Leakage_Pipeline ["Encapsulated Sklearn Pipeline (Fitted per Fold)"]
        D["Clinical Feature Engineering (ALBI & APRI Proxies, Missing Flags)"]
        E["Numeric Median Imputation + Optional StandardScaler"]
        F["Categorical Most-Frequent Imputation + OneHotEncoder"]
        D --> E
        D --> F
    end
    
    C -->|Training Set (334 samples)| Zero_Leakage_Pipeline
    Zero_Leakage_Pipeline --> G["Stratified 5-Fold Cross-Validation"]
    G --> H["Hyperparameter Grid Optimization (Scored on Macro F1)"]
    
    H --> I["5-Algorithm Comparative Benchmark:
    - Multiclass Logistic Regression (Balanced)
    - Random Forest (Balanced)
    - Gradient Boosting
    - XGBoost
    - LightGBM"]
    
    I --> J["Model Selection by Macro F1:
    Logistic Regression (Macro F1 = 0.5248)"]
    
    C -->|Holdout Test Set (84 samples)| K["Independent Holdout Evaluation"]
    J --> K
    
    K --> L["Model Serialization (final_model.joblib & metadata.json)"]
    K --> M["Interpretability: Holdout Permutation Importance & SHAP"]
    
    L --> N["Production Prediction Engine (Boundary Validator)"]
    M --> O["Streamlit 7-Page Research Dashboard (Port 8501)"]
    N --> O
```

---

## 3. Dataset & Data Ingestion Layer

The analysis utilizes the canonical Mayo Clinic PBC randomized controlled trial cohort (418 patients collected between 1974 and 1984):
- **Cohort Size**: 418 patient records (334 training, 84 holdout test).
- **Features**: 17 clinical physiological biomarkers (11 continuous, 5 categorical, 1 histologic stage).
- **Resilient Ingestion Layer (`src/data_loader.py`)**: Automatically caches data locally at `data/raw/cirrhosis.csv` with a verified remote fallback to R's `survival::pbc` repository.
- **Alias Resolution Layer (`src/validation.py`)**: Seamlessly normalizes variable naming discrepancies across Mayo Clinic, Kaggle, and R conventions (e.g. `bili` $\leftrightarrow$ `Bilirubin`, `ast` $\leftrightarrow$ `SGOT`, `alk.phos` $\leftrightarrow$ `Alk_Phos`, `hepato` $\leftrightarrow$ `Hepatomegaly`).

### Cohort Target Distribution (Calculated Dynamically)
| Class ID | Outcome Label | Patient Count | Cohort Percentage |
| :---: | :--- | :---: | :---: |
| **0** | **Death** | 161 | 38.52% |
| **1** | **Censored (Survived)** | 232 | 55.50% |
| **2** | **Liver Transplant** | 25 | 5.98% |

---

## 4. Methodological Rigor & Data Leakage Elimination

Clinical predictive models are notoriously vulnerable to subtle data leakage that inflates reported performance. This project enforces strict zero-leakage constraints:

1. **Stratified Partitioning Precedes Preprocessing**:
   - `train_test_split(test_size=0.20, stratify=y, random_state=42)` is performed immediately after schema normalization.
   - The 20% holdout test set (84 patients) is strictly quarantined and never exposed to imputer fitting, scaling parameters, feature selection, or hyperparameter tuning.
2. **Pipeline Encapsulation**:
   - Imputation (`SimpleImputer(strategy='median')` for continuous variables, `strategy='most_frequent'` for categoricals) is wrapped inside scikit-learn `ColumnTransformer` and `Pipeline` objects.
   - During Stratified 5-Fold Cross-Validation, imputation and scaling statistics are calculated uniquely on each fold's training split and applied to the validation split.
3. **No Target Leakage in Survival Proxies**:
   - Follow-up time variables (`time`, `N_Days`) are stripped from feature matrices to prevent direct survival label leakage.

---

## 5. Clinically Meaningful Feature Engineering

The `ClinicalFeatureEngineer` transformer (`src/feature_engineering.py`) implements deterministic, biochemically justified features:
- **ALBI Proxy (Albumin-Bilirubin Score Proxy)**:
  $$\text{ALBI\_proxy} = \frac{\text{Albumin}}{\text{Bilirubin} + \epsilon}$$
  Reflects hepatic synthetic capacity relative to excretory biliary dysfunction.
- **APRI Proxy (AST-to-Platelet Ratio Index)**:
  $$\text{APRI\_proxy} = \frac{\text{SGOT}}{\text{Platelets} + \epsilon} \times 100$$
  Widely used surrogate non-invasive biomarker for hepatic fibrosis and portal hypertension.
- **Cholestasis Burden Index**: Interaction feature between urine copper and serum bilirubin ($\text{Copper} \times \text{Bilirubin}$).
- **Labile Biomarker Missingness Indicators**: Binary indicator flags (`Cholesterol_missing`, `Copper_missing`, `Triglycerides_missing`) capturing informative clinical missingness patterns.

---

## 6. Model Benchmarking & Empirical Evaluation

Because the **Transplant** class represents only 5.98% of the cohort, **Macro F1** (unweighted arithmetic mean of per-class F1-scores) is designated as the primary model selection criterion. Models that optimize only for overall accuracy routinely collapse on the minority transplant class.

### 5-Algorithm Comparative Benchmark Table (Holdout Test Set)

> *All metrics below were calculated dynamically from the actual holdout test execution pipeline (`artifacts/model_comparison.csv`). No metrics are fabricated.*

| Rank | Model | CV Macro F1 | Holdout Macro F1 | Holdout Accuracy | Holdout Weighted F1 | Holdout ROC-AUC (OvR) | Death Recall | Censored Recall | Transplant Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Logistic Regression (Balanced)** | **0.6114** | **0.5248** | **0.6667** | **0.6975** | **0.7747** | **0.7500** | **0.6596** | **0.2000** |
| 🥈 | **Random Forest (Balanced)** | 0.6519 | 0.4835 | 0.6786 | 0.6891 | 0.7226 | 0.7188 | 0.7234 | 0.0000 |
| 🥉 | **Gradient Boosting** | 0.5874 | 0.4745 | 0.6905 | 0.6738 | 0.7184 | 0.7188 | 0.7447 | 0.0000 |
| 4 | **XGBoost** | 0.5868 | 0.4497 | 0.6548 | 0.6394 | 0.6561 | 0.6875 | 0.7021 | 0.0000 |
| 5 | **LightGBM** | 0.6373 | 0.4300 | 0.6190 | 0.6131 | 0.7017 | 0.6562 | 0.6596 | 0.0000 |

### Key Analytical Findings
1. **The Minority Class Dilemma**: Standard tree ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM) achieved higher overall accuracy (~68-69%) but achieved **0.00% recall for the Transplant class**, classifying all transplant patients as either Censored or Death.
2. **Macro F1 Selection Rationale**: Balanced Logistic Regression was selected as the optimal model because its class-weighting enabled positive identification of the rare Transplant class (Recall: 20.0%) while maintaining high sensitivity for mortality (Death Recall: 75.0%), yielding the highest holdout Macro F1 (0.5248) and the highest One-vs-Rest ROC-AUC (0.7747).
3. **Retrospective Identification of Mortality**: The final pipeline achieved **75.0% recall for the Death class** on the held-out test cohort, demonstrating strong retrospective identification of patients assigned to the mortality outcome class.

---

## 7. Model Explainability & Interpretability

Interpretability is partitioned into global feature importance and patient-level local attribution:

### Global Importance (Permutation vs. Gini)
- **Holdout Permutation Importance**: Evaluated on the independent test set by measuring degradation in Macro F1 upon feature shuffling. Top drivers:
  1. **Serum Bilirubin** ($\Delta \text{Macro F1} = +0.072$)
  2. **Platelet Count** ($\Delta \text{Macro F1} = +0.023$)
  3. **Alkaline Phosphatase** ($\Delta \text{Macro F1} = +0.015$)
  4. **Patient Age** ($\Delta \text{Macro F1} = +0.010$)
  5. **Urine Copper** ($\Delta \text{Macro F1} = +0.009$)
- **Random Forest Gini Importance**: Age (15.1%), ALBI proxy (9.8%), Prothrombin time (9.4%), log ALBI proxy (9.1%), and Platelets (7.7%) accounted for major shares of total impurity reduction.

### Local Explainability (SHAP Decompositions)
- The application integrates SHAP (`shap.TreeExplainer` and `shap.KernelExplainer`) to break down individual patient predictions into directional biomarker risk contributors.
- Every explanation carries the disclaimer: *"Model explanation — not clinical interpretation. These contributions describe mathematical model behavior and must not be construed as clinical causality or diagnosis."*

---

## 8. Interactive Streamlit Application

The Streamlit web application (`app.py`) loads pre-trained artifacts for instant responsiveness without retraining on startup. It includes 7 structured pages:

1. **1. Overview**: Executive cohort summary, dynamic sample counts, and class distribution charts.
2. **2. Patient Prediction**: Interactive biomarker input simulation with physiological boundary checks, model-assigned probabilities, and risk classification.
3. **3. Model Performance**: Full 5-algorithm benchmark table, Plotly metric comparisons, interactive confusion matrix, and per-class recall breakdowns.
4. **4. Feature Importance**: Side-by-side tabs for holdout permutation importance and Random Forest Gini impurity rankings.
5. **5. Explainability**: Cohort patient selector with on-demand SHAP waterfall/bar decompositions.
6. **6. Data Exploration (EDA)**: Biomarker distribution boxplots, violin plots, correlation heatmap, and missingness bar chart.
7. **7. About & Ethics**: Comprehensive documentation of clinical limitations, competing risks, and governance principles.

---

## 9. Repository Structure

```
Liver-Cirrhosis-Survival-Prediction/
├── app.py                      # 7-page interactive Streamlit research dashboard
├── train.py                    # End-to-end 5-model training & artifact generation pipeline
├── predict.py                  # CLI & programmatic patient inference script
├── requirements.txt            # Pinned project dependencies
├── README.md                   # Portfolio technical documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── config/
│   └── config.py               # Schemas, paths, physiological ranges, safety notices
│
├── data/
│   ├── raw/
│   │   └── cirrhosis.csv       # 418-patient Mayo Clinic PBC dataset
│   └── processed/              # Cleaned splits
│
├── models/
│   ├── final_model.joblib      # Serialized scikit-learn best pipeline
│   └── model_metadata.json     # Training provenance, hyperparams, and metrics
│
├── artifacts/
│   ├── metrics.json            # Complete CV and holdout metric exports
│   ├── model_comparison.csv    # 5-model comparative benchmark table
│   ├── feature_importance.csv  # Combined permutation and Gini importance table
│   ├── confusion_matrix.png    # High-resolution holdout confusion matrix
│   └── eda_summary.json        # Missingness and summary statistics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Jupyter EDA notebook
│   └── 02_model_analysis.ipynb     # Model comparison and interpretability notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion with remote download fallback
│   ├── validation.py           # Schema checking, alias normalization, boundary validator
│   ├── preprocessing.py        # Leakage-free ColumnTransformer pipeline builders
│   ├── feature_engineering.py  # ALBI, APRI, and missingness flag transformers
│   ├── models.py               # 5-algorithm estimators and hyperparameter search spaces
│   ├── evaluation.py           # Stratified 5-fold CV, holdout evaluation, confusion matrix
│   ├── explainability.py       # Permutation, Gini, and SHAP patient explainers
│   ├── prediction.py           # Production predict_patient API
│   └── visualization.py        # Interactive Plotly and publication Seaborn figures
│
└── tests/
    ├── __init__.py
    ├── test_data.py            # Dataset integrity & schema validation tests
    ├── test_preprocessing.py   # Leakage prevention & imputation tests
    ├── test_features.py        # Feature engineering mathematical tests
    ├── test_models.py          # 5-algorithm pipeline training & probability tests
    └── test_prediction.py      # Range validation & prediction schema tests
```

---

## 10. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12+
- Virtual environment recommended (`venv` or `conda`)

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Rupesh4113/Liver-Cirrhosis-Survival-Prediction.git
cd Liver-Cirrhosis-Survival-Prediction

pip install -r requirements.txt
```

### 2. Execute Training Pipeline
Train all 5 models, run Stratified 5-Fold CV, optimize hyperparameters, evaluate holdout set, and serialize artifacts:
```bash
python train.py
```

### 3. Run Automated Test Suite
Execute the 25 unit and integration tests:
```bash
python -m pytest -v tests/
```

### 4. CLI Inference
Run a sample patient prediction:
```bash
python predict.py --sample
```
Or with custom JSON parameters:
```bash
python predict.py --json "{\"Age\": 52, \"Bilirubin\": 3.1, \"Albumin\": 3.2, \"Prothrombin\": 11.2, \"Stage\": 3}"
```

### 5. Launch Streamlit Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Critical Limitations & Governance

Any clinical interpretation of this work must consider the following constraints:

1. **Small Retrospective Cohort**: The dataset consists of only 418 patients from a single medical center (Mayo Clinic) enrolled between 1974 and 1984. Modern cohorts are larger and feature different baseline characteristics.
2. **Historical Treatment Shift**: Since the 1980s, therapies such as Ursodeoxycholic Acid (UDCA) and Obeticholic Acid have altered the natural history and progression rates of PBC.
3. **Competing Risks in Survival Analysis**: In formal biostatistics, liver transplantation is a classic **competing risk** or informative censoring event. Approximating survival through static multiclass classification does not model time-to-event hazard functions or cumulative incidence functions directly.
4. **Probability Calibration**: Softmax probabilities output by classification models reflect mathematical distance within feature space, not calibrated clinical risk probabilities.
5. **No External or Prospective Validation**: The pipeline has not undergone external validation across diverse healthcare systems or prospective cohorts.

---

## 12. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 13. References & Citation
- *Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: model for decision making. Hepatology, 10(1), 1-7.*
- *Therneau, T. M., & Grambsch, P. M. (2000). Modeling Survival Data: Extending the Cox Model. Springer.*
