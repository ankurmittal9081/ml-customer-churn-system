# ==============================================================================
# MACHINE LEARNING PREPROCESSING MASTER REFERENCE
# Author: Ankur Mittal
#
# Topics Covered:
# 1. Import Libraries
# 2. Load Dataset
# 3. Dataset Inspection
# 4. Missing Value Handling
# 5. Categorical Columns (Manual + Automatic Encoding)
# 6. Numerical Columns (Manual + Automatic Extraction)
# 7. Feature Scaling
# 8. Features & Target Isolation
# 9. Train-Test Split
# 10. Model Training (Logistic Regression)
# 11. Prediction
# 12. Evaluation Metrics
# 13. Training vs Testing Accuracy (Overfitting Check)
# 14. Probability Prediction (predict_proba)
# 15. Custom Thresholding (0.30 Risk Threshold)
# 16. Syntax Cheat Sheet
# ==============================================================================

# ==============================================================================
# 1. IMPORT LIBRARIES
# ==============================================================================

import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score
)

# ==============================================================================
# UTF-8 Support for Windows Console
# ==============================================================================

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 2. LOAD DATASET
# ==============================================================================

df = pd.read_csv("data/customer_churn_data.csv")


# ==============================================================================
# 3. BASIC INSPECTION
# ==============================================================================

print("=== 1. HEAD ===")
print(df.head())

print("\n=== 2. SHAPE ===")
print(df.shape)

print("\n=== 3. INFO ===")
print(df.info())

print("\n=== 4. DESCRIBE ===")
print(df.describe())

print("\n=== 5. NULL VALUES CHECK ===")
print(df.isnull().sum())


# ==============================================================================
# 4. MISSING VALUE IMPUTATION
# ==============================================================================

median_value = df["Total_Charges_INR"].median()
df["Total_Charges_INR"] = df["Total_Charges_INR"].fillna(median_value)


# ==============================================================================
# 5. CATEGORICAL COLUMNS
# ==============================================================================

# Manual List Approach
cat_cols = [
    "Gender",
    "Contract_Type",
    "Payment_Method",
    "Tech_Support",
    "Paperless_Billing"
]

df_encoded = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

# Automatic Encoding Approach
categorical_columns = df.select_dtypes(include="object").columns.tolist()
categorical_columns.remove("CustomerID")

df_auto = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)


# ==============================================================================
# 6. NUMERICAL COLUMNS
# ==============================================================================

# Manual List Approach
num_cols = [
    "Age",
    "Tenure_Months",
    "Monthly_Charges_INR",
    "Total_Charges_INR"
]

# Automatic Numerical Columns Approach
auto_num_cols = df.select_dtypes(include="number").columns.tolist()
auto_num_cols.remove("Churn")


# ==============================================================================
# 7. FEATURE SCALING
# ==============================================================================

scaler = StandardScaler()

df_encoded[num_cols] = scaler.fit_transform(
    df_encoded[num_cols]
)

# Automatic Scaling
df_auto[auto_num_cols] = scaler.fit_transform(
    df_auto[auto_num_cols]
)


# ==============================================================================
# 8. FEATURES & TARGET
# ==============================================================================

X = df_encoded.drop(
    columns=["CustomerID", "Churn"]
)

y = df_encoded["Churn"]


# ==============================================================================
# 9. TRAIN TEST SPLIT
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==============================================================================
# 10. MODEL TRAINING
# ==============================================================================

model = LogisticRegression()
model.fit(X_train, y_train)


# ==============================================================================
# 11. PREDICTION
# ==============================================================================

y_pred = model.predict(X_test)


# ==============================================================================
# 12. EVALUATION
# ==============================================================================

print("\n=== ACCURACY SCORE ===")
print(accuracy_score(y_test, y_pred))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))


# ==============================================================================
# 13. TRAIN VS TEST ACCURACY
# ==============================================================================

train_accuracy = accuracy_score(
    y_train,
    model.predict(X_train)
)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nTraining Accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")


# ==============================================================================
# 14. PROBABILITY PREDICTION
# ==============================================================================

y_probability = model.predict_proba(X_test)

print("\n=== FIRST 5 CUSTOMERS RISK PROBABILITIES ===")
for i in range(5):
    stay_probability = y_probability[i][0] * 100
    churn_probability = y_probability[i][1] * 100
    print(f"Customer {i+1}")
    print(f"  Stay  : {stay_probability:.2f}%")
    print(f"  Churn : {churn_probability:.2f}%")


# ==============================================================================
# 15. CUSTOM THRESHOLD (0.30 RISK ALERT)
# ==============================================================================

custom_prediction = (
    model.predict_proba(X_test)[:, 1] >= 0.30
).astype(int)

print("\n=== CUSTOM THRESHOLD (0.30) CONFUSION MATRIX ===")
print(confusion_matrix(y_test, custom_prediction))

print(
    f"\nBoosted Recall Score : {recall_score(y_test, custom_prediction) * 100:.2f}%"
)

# ==============================================================================
# 16. IMPORTANT SYNTAX CHEAT SHEET
# ==============================================================================
# Read CSV            -> pd.read_csv()
# Shape               -> df.shape
# Head                -> df.head()
# Info                -> df.info()
# Missing Values      -> df.isnull().sum()
# Median              -> df["column"].median()
# Fill Missing        -> df["column"].fillna(value)
# Object Columns      -> df.select_dtypes(include="object")
# Numerical Columns   -> df.select_dtypes(include="number")
# One Hot Encoding    -> pd.get_dummies()
# Feature Scaling     -> StandardScaler()
# Fit Transform       -> scaler.fit_transform()
# Split               -> train_test_split()
# Train               -> model.fit()
# Predict             -> model.predict()
# Probability         -> model.predict_proba()
# Accuracy            -> accuracy_score()
# Confusion Matrix    -> confusion_matrix()
# Classification Report -> classification_report()
# Recall              -> recall_score()
