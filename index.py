# ==============================================================================
# END-TO-END MACHINE LEARNING PROJECT: CUSTOMER CHURN PREDICTION SYSTEM
# Master Pipeline: Preprocessing -> ColumnTransformer -> Multi-Model -> Model Persistence
# ==============================================================================

import sys
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load Raw Data
print("=== 1. LOADING RAW DATASET ===")
df = pd.read_csv('data/customer_churn_data.csv')
print(f"Dataset Loaded Successfully! Shape: {df.shape}\n")

# 2. Features & Target Isolation
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

num_cols = ['Age', 'Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']
cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']

# 3. Train-Test Split (Raw Data Split - No Data Leakage!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Production Scikit-Learn Pipelines Setup
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 5. Master Production Pipeline with Random Forest
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit Pipeline on Training Data
print("=== 2. TRAINING PRODUCTION PIPELINE ===")
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

print(f"Master Pipeline Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Master Pipeline Test Recall:   {recall_score(y_test, y_pred) * 100:.2f}%\n")

# 6. Model Persistence (Save Trained Pipeline to Disk)
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'churn_pipeline.pkl')
joblib.dump(full_pipeline, model_path)
print(f"=== 3. MODEL PERSISTENCE ===")
print(f"Trained Pipeline saved to: {model_path}\n")

print("=== PIPELINE EXECUTION COMPLETE & READY FOR DEPLOYMENT ===")