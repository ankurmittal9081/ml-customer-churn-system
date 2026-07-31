import pandas as pd
df=pd.read_csv('data/customer_churn_data.csv')

print(df.isnull().sum())
median=df['Total_Charges_INR'].median()
df['Total_Charges_INR']=df['Total_Charges_INR'].fillna(median)

cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded=pd.get_dummies(df,columns=cat_cols,drop_first=True)

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
num_cols = ['Age', 'Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']
df_encoded[num_cols]=scale.fit_transform(df_encoded[num_cols])

# PART 3: Split X and y
from sklearn.model_selection import train_test_split
X = df_encoded.drop(columns=['CustomerID', 'Churn']) # Use columns=[...]
y = df_encoded['Churn']                               # Use [] brackets, not ()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=== FIRST 10 PREDICTIONS ===")
print(y_pred[:10])

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,recall_score
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))

y_proba=model.predict_proba(X_test)
print("\nFirst Customer Churn Risk %:", y_proba[0][1] * 100)

# Custom threshold check
y_pred_custom=(model.predict_proba(X_test)[:,1]>=0.30).astype(int)
print("Boosted Recall Score:", recall_score(y_test, y_pred_custom) * 100)
