import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Data & Clean Missing Values
df = pd.read_csv('data/customer_churn_data.csv')
total_charges_median = df['Total_Charges_INR'].median()
df['Total_Charges_INR'] = df['Total_Charges_INR'].fillna(total_charges_median)

# 2. Automatic Categorical Encoding (Text)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('CustomerID')
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 3. Automatic Feature Scaling (Numbers)
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['number']).columns.tolist()
num_cols.remove('Churn')
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 4. Train-Test Split
x = df_encoded.drop(columns=['CustomerID', 'Churn'])
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Fit & Predict
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))