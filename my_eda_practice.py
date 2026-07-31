import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df=pd.read_csv('data/customer_churn_data.csv')
cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

corr = df_encoded.drop(columns=['CustomerID']).corr()
sb.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()



# df['Churn'].value_counts().plot(kind='bar',color=['green','red'])
# df.groupby(['Contract_Type', 'Churn']).size().unstack().plot(kind='bar',color=['green','red'])

# # plt.title('Customer Churn Count (Green=Stayed,Red=left)')
# plt.title('Churn Rate by Contract Type')
# plt.xlabel('Contract Type')
# plt.ylabel('Number of Customers')

# plt.show()
