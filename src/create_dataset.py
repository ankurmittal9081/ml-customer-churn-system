import pandas as pd
import numpy as np
import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set seed for reproducible synthetic dataset
np.random.seed(42)

n_samples = 1000

# 1. Generate realistic features
customer_ids = [f'CUST-{1000 + i}' for i in range(n_samples)]
gender = np.random.choice(['Male', 'Female'], size=n_samples)
age = np.random.randint(18, 70, size=n_samples)
tenure = np.random.randint(1, 72, size=n_samples) # 1 to 72 months

contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(['UPI / Electronic', 'Credit Card', 'Bank Transfer', 'Mailed Check'], size=n_samples)

monthly_charges = np.round(np.random.uniform(299.0, 2499.0, size=n_samples), 2)
total_charges = np.round(monthly_charges * tenure + np.random.uniform(-50, 50, size=n_samples), 2)

# Introduce 20 missing values in Total_Charges to practice Data Cleaning!
missing_indices = np.random.choice(n_samples, size=20, replace=False)
total_charges[missing_indices] = np.nan

# Tech support / Addon services
tech_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.35, 0.65])
paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.60, 0.40])

# 2. Mathematical churn probability formula to create realistic patterns
churn_prob = (
    0.35 * (contract == 'Month-to-month') +
    0.25 * (tenure < 12) +
    0.20 * (monthly_charges > 1499.0) +
    0.15 * (tech_support == 'No') -
    0.20 * (contract == 'Two year')
)
churn_prob = np.clip(churn_prob, 0.05, 0.85)
churn = np.random.binomial(1, churn_prob)

# 3. Assemble into Pandas DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': gender,
    'Age': age,
    'Tenure_Months': tenure,
    'Contract_Type': contract,
    'Payment_Method': payment_method,
    'Monthly_Charges_INR': monthly_charges,
    'Total_Charges_INR': total_charges,
    'Tech_Support': tech_support,
    'Paperless_Billing': paperless_billing,
    'Churn': churn
})

# Path directly inside project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

output_path = os.path.join(data_dir, 'customer_churn_data.csv')
df.to_csv(output_path, index=False)

print(f"[SUCCESS] Dataset successfully generated at: {output_path}")
print(f"[INFO] Total Rows (Customers): {df.shape[0]}")
print(f"[INFO] Total Columns (Features + Target): {df.shape[1]}")
