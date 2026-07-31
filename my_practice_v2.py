# ==============================================================================
# 🎯 REAL-WORLD PRACTICE CHALLENGE 2.0
# File: my_practice_v2.py
# ==============================================================================

# TASK 1: Complete Baseline Pipeline
# Write your code for loading, cleaning, encoding, scaling, splitting & fitting model here...
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df=pd.read_csv('data/customer_churn_data.csv')
median=df['Total_Charges_INR'].median()
df['Total_Charges_INR']=df['Total_Charges_INR'].fillna(median)
print(df.isnull().sum())

# apply one hot encoding
cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded=pd.get_dummies(df,columns=cat_cols,drop_first=True)
print(df_encoded.head())

#aply sacaling
scale=StandardScaler()
num_cols=['Age', 'Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']
df_encoded[num_cols]=scale.fit_transform(df_encoded[num_cols])
print(df_encoded.head())

#split data 
# model=train_test_split()
x=df_encoded.drop(columns=['CustomerID', 'Churn'])
y=df_encoded['Churn']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)

#model train 
test=LogisticRegression()
test.fit(X_train,y_train)
y_pred=test.predict(X_test)

train_acc=accuracy_score(y_train,test.predict(X_train))
test_acc=accuracy_score(y_test,y_pred)
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

# First 5 Test Customers ke exact probabilities print karo
y_proba = test.predict_proba(X_test)

print("\n=== FIRST 5 CUSTOMERS RISK PROBABILITIES ===")
for i in range(5):
    stay_prob = y_proba[i][0] * 100
    churn_prob = y_proba[i][1] * 100
    print(f"Customer {i+1}: Stay Risk = {stay_prob:.1f}% | Churn Risk = {churn_prob:.1f}%")

# Default threshold (0.50) ki jagah custom threshold (0.30) lagao
y_pred_custom = (test.predict_proba(X_test)[:, 1] >= 0.30).astype(int)

print("\n=== CUSTOM THRESHOLD (0.30) PERFORMANCE ===")
print("New Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

from sklearn.metrics import recall_score
print(f"New Recall Score: {recall_score(y_test, y_pred_custom) * 100:.2f}%")