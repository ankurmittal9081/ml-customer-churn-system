# ==============================================================================
# 🎯 MASTER PROJECT SCRATCH CHALLENGE (NO HINTS!)
# Build the complete Machine Learning Pipeline from Zero to Model Evaluation
# File: my_master_project_scratch.py
# ==============================================================================

# STEP 1: Import Libraries
# Write imports here...
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# STEP 2: Load Dataset ('data/customer_churn_data.csv') & Check Shape
# Write loading code here...
df = pd.read_csv('data/customer_churn_data.csv')
print(df.shape)
# STEP 3: Handle Missing Values (Impute Total_Charges_INR with Median)
# Write cleaning code here...

df_median=df['Total_Charges_INR'].median()
df['Total_Charges_INR']=df['Total_Charges_INR'].fillna(df_median)



# STEP 4: One-Hot Encoding for Categorical Columns
# Write encoding code here...

cat_cols=['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded=pd.get_dummies(df,columns=cat_cols,drop_first=True)
# STE 5: Feature Scaling (StandardScaler on Numerical Columns
# Write scaling code here...

num_cols=['Age', 'Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']

scalar=StandardScaler()
df_encoded[num_cols]=scalar.fit_transform(df_encoded[num_cols])


# STEP 6: Train-Test Split (80% Train, 20% Test, random_state=42)
# Write split code here...
x=df_encoded.drop(columns=['CustomerID','Churn'])
y=df_encoded['Churn']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
                  


# STEP 7: Train Model (Logistic Regression)
# Write model training code here...
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# STEP 8: Model Predictions & Evaluation Metrics
# Write evaluation code here...

acc=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cfc=classification_report(y_test,y_pred)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cfc)
