# ==============================================================================
# LESSON 14: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS
# File: src/eda_visualizations.py
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory to save plots
os.makedirs('data/plots', exist_ok=True)

# 1. Load Data
df = pd.read_csv('data/customer_churn_data.csv')

# ------------------------------------------------------------------------------
# CHART 1: Target Distribution
# ------------------------------------------------------------------------------
df['Churn'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('1. Customer Churn Count (Green=Stayed, Red=Left)')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')
plt.savefig('data/plots/1_target_distribution.png')
plt.close()

# ------------------------------------------------------------------------------
# CHART 2: Churn Breakdown by Contract Type
# ------------------------------------------------------------------------------
df.groupby(['Contract_Type', 'Churn']).size().unstack().plot(kind='bar', color=['green', 'red'])
plt.title('2. Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.savefig('data/plots/2_contract_vs_churn.png')
plt.close()

# ------------------------------------------------------------------------------
# CHART 3: Feature Correlation Heatmap
# ------------------------------------------------------------------------------
cat_cols = ['Gender', 'Contract_Type', 'Payment_Method', 'Tech_Support', 'Paperless_Billing']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
corr = df_encoded.drop(columns=['CustomerID']).corr()

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('3. Feature Correlation Heatmap')
plt.savefig('data/plots/3_correlation_heatmap.png')
plt.close()

print("All 3 Plots saved successfully in data/plots/ folder!")
