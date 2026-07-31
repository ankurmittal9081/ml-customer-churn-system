# my_practice.py
# Write your practice code here step-by-step!

import pandas as pd
df = pd.read_csv('data/customer_churn_data.csv')
# print(df.shape)

# print(df.isnull().sum())
df_median=df['Total_Charges_INR'].median()
# print(df_median)    
df['Total_Charges_INR']=df['Total_Charges_INR'].fillna(df_median)
# print(df.isnull().sum());

cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']

df_encoded= pd.get_dummies(df,columns=cat_cols,drop_first=True)
# print(df_encoded.head())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num_cols=['Age','Tenure_Months', 'Monthly_Charges_INR', 'Total_Charges_INR']
df_encoded[num_cols]=scaler.fit_transform(df_encoded[num_cols])
# print(df_encoded.head())

from sklearn.model_selection import train_test_split
x=df_encoded.drop(columns=['CustomerID', 'Churn'])
y=df_encoded['Churn']

# print(x)
# print(y)                           

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)
print(X_test)
print(y_test)

