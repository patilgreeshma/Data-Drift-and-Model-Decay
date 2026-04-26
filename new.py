import pandas as pd
import numpy as np

target_col = 'default.payment.next.month' # Define target column

# Load your original dataset
df = pd.read_csv("UCI_Credit_Card.csv")

# Copy to simulate "future" data
future_df = df.copy()

np.random.seed(42)

# Apply realistic drift
future_df['LIMIT_BAL'] = future_df['LIMIT_BAL'] * 1.5
future_df['BILL_AMT1'] = future_df['BILL_AMT1'] * 1.3
future_df['PAY_AMT1'] = future_df['PAY_AMT1'] * 0.7
future_df['AGE'] = future_df['AGE'] + 2

# Add mild noise (only numeric columns EXCEPT the target)
num_cols = future_df.select_dtypes(include=np.number).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col) # Don't add noise to labels!

future_df[num_cols] = future_df[num_cols] + np.random.normal(0, 0.05, future_df[num_cols].shape)

# Save
future_df.to_csv("new_data.csv", index=False)

print("✅ new_data.csv created with clean targets")
