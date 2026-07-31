# ==============================================================================
# END-TO-END MACHINE LEARNING PROJECT: CUSTOMER CHURN PREDICTION
# Master Pipeline: Preprocessing -> Logistic Regression -> Decision Tree -> Random Forest -> Gradient Boosting
# ==============================================================================

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load Data & Clean Missing Values
print("=== 1. LOADING & CLEANING DATASET ===")
df = pd.read_csv('data/customer_churn_data.csv')
median_val = df['Total_Charges_INR'].median()
df['Total_Charges_INR'] = df['Total_Charges_INR'].fillna(median_val)
print(f"Dataset Shape: {df.shape} | Missing Values Fixed!\n")

# 2. Categorical Encoding (One-Hot Encoding)
cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Feature Scaling (StandardScaler)
scaler = StandardScaler()
num_cols = ['Age', 'Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 4. Train-Test Split
X = df_encoded.drop(columns=['CustomerID', 'Churn'])
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model 1: Logistic Regression Baseline
print("=== 2. MODEL 1: LOGISTIC REGRESSION ===")
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_custom_pred = (lr_model.predict_proba(X_test)[:, 1] >= 0.30).astype(int)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%")
print(f"Logistic Regression Recall (At 0.30 Threshold): {recall_score(y_test, lr_custom_pred) * 100:.2f}%\n")

# 6. Model 2: Decision Tree Classifier
print("=== 3. MODEL 2: DECISION TREE CLASSIFIER ===")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred) * 100:.2f}%")
print(f"Decision Tree Recall:   {recall_score(y_test, dt_pred) * 100:.2f}%\n")

# 7. Model 3: Random Forest Classifier (100 Trees Ensemble)
print("=== 4. MODEL 3: RANDOM FOREST CLASSIFIER (ENSEMBLE) ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print(f"Random Forest Recall:   {recall_score(y_test, rf_pred) * 100:.2f}%\n")

# 8. Model 4: Gradient Boosting Classifier
print("=== 5. MODEL 4: GRADIENT BOOSTING CLASSIFIER ===")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_pred) * 100:.2f}%")
print(f"Gradient Boosting Recall:   {recall_score(y_test, gb_pred) * 100:.2f}%\n")

print("=== MULTI-MODEL PIPELINE EXECUTION COMPLETE ===")