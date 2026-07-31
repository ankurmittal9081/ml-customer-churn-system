# 📊 Universal Customer Churn & Retention Prediction System
> **An End-to-End Production-Grade Machine Learning Solution for Universal Customer Retention**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
Customer Churn is one of the most critical challenges facing modern subscription-based businesses (Telecom, SaaS, Banking, E-Commerce). Acquiring a new customer can cost up to **5x to 25x more** than retaining an existing one. 

This repository presents an **End-to-End Machine Learning Pipeline** designed to detect high-risk customer churn before it occurs. By performing automated preprocessing, feature engineering, probability calibration, multi-model evaluation (Logistic Regression, Decision Trees, Random Forest Ensembles), and **custom business decision thresholding (0.30)**, the system boosts **Recall from 63% to 83.56%**, helping business stakeholders intercept high-risk customers with proactive retention campaigns.

---

## 🎯 Key Highlights & Business Results

- **Data Imputation**: Outlier-robust median imputation for monetary features (`Total_Charges_INR`).
- **Categorical Encoding**: Dummy variable trap prevention using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
- **Feature Scaling**: Z-Score standardization via `StandardScaler` for distance/gradient models.
- **Multi-Model Evaluation**:
  - Baseline Logistic Regression (72.00% Accuracy)
  - Decision Tree Classifier (71.00% Accuracy)
  - **Random Forest Classifier (100 Trees Ensemble - 72.50% Accuracy, 61.84% Recall)**
- **Probability Calibration**: `predict_proba` risk scoring for every test customer.
- **Custom Business Threshold Tuning**: Adjusted decision boundary from $0.50$ to $0.30$, boosting **Recall to 83.56%** (catching 83.5% of churners before they leave!).

---

## 📈 Model Comparison & Performance Table

| Model Algorithm | Accuracy | Recall (Churn) | Key Insight / Advantage |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (0.50)** | 72.00% | 63.15% | Linear baseline model |
| **Logistic Regression (0.30 Custom)** | 70.00% | **83.56%** 🚀 | **Maximum Churn Capture (Proactive Alert)** |
| **Decision Tree (Single Tree)** | 71.00% | 48.68% | Human-readable If-Else rules |
| **Random Forest (100 Trees Ensemble)** | **72.50%** | **61.84%** 🌲 | **Higher Stability & Generalization** |

---

## 📊 Exploratory Data Analysis (EDA) Highlights

The system includes automated visual data exploratory pipelines generated under `data/plots/`:

1. **Target Distribution**: Visualizes Retained (63.2%) vs Churned (36.8%) ratio.
2. **Contract Type vs Churn**: Highlights that Month-to-Month contracts have the highest churn risk, while 2-Year contracts exhibit near-zero churn.
3. **Correlation Heatmap**: Highlights strong negative correlation between `Tenure_Months` ($r = -0.41$) and `Churn`.

---

## 📁 Clean Repository Structure

```text
ml-customer-churn-system/
├── data/                       # Customer Datasets & Visualization Plots
│   ├── customer_churn_data.csv
│   └── plots/
│       ├── 1_target_distribution.png
│       ├── 2_contract_vs_churn.png
│       └── 3_correlation_heatmap.png
├── src/                        # Modular Source Code Scripts
│   ├── create_dataset.py       # Dataset Generation Script
│   └── eda_visualizations.py   # Automated Plot Generation Script
├── app.py                      # Interactive Streamlit Web Dashboard
├── index.py                    # Master End-to-End Pipeline Script
├── .gitignore                  # Git Exclusion Rules
└── README.md                   # Project Documentation
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/ankurmittal9081/ml-customer-churn-system.git
cd ml-customer-churn-system
```

### 2. Install Required Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 3. Run Preprocessing & Multi-Model Training Pipeline
```bash
python index.py
```

### 4. Generate EDA Visualization Charts
```bash
python src/eda_visualizations.py
```

### 5. Launch Interactive Streamlit Web Dashboard
```bash
python -m streamlit run app.py
```

---

## 👤 Author & Acknowledgements
- **Author**: Ankur Mittal
- **Domain**: Machine Learning & Predictive Analytics
- **License**: MIT License
