import pandas as pd
import numpy as np

def generate_drifted_adult_data(input_path='/Users/greeshmapatil/Desktop/ads/adult.csv', output_path='drifted_adult_data.csv'):
    print(f"Loading original data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Create a copy to modify
    drifted_df = df.copy()
    
    # 1. Numerical Drift: Aging population (Shift mean age up)
    drifted_df['age'] = drifted_df['age'] + 8
    
    # 2. Numerical Drift: Economy shift (Increase capital gain by 20%)
    drifted_df['capital.gain'] = drifted_df['capital.gain'] * 1.5
    
    # 3. Numerical Drift: Work culture shift (Decrease hours per week)
    drifted_df['hours.per.week'] = drifted_df['hours.per.week'] - 5
    
    # 4. Categorical Drift: Workforce shift (Change 20% of 'Private' to 'Self-emp-not-inc')
    mask = (drifted_df['workclass'] == 'Private')
    n_to_change = int(0.2 * mask.sum())
    indices_to_change = drifted_df[mask].sample(n_to_change, random_state=42).index
    drifted_df.loc[indices_to_change, 'workclass'] = 'Self-emp-not-inc'
    
    # 5. Add Noise to all numeric columns
    num_cols = drifted_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = np.random.normal(0, drifted_df[col].std() * 0.1, size=len(drifted_df))
        drifted_df[col] = drifted_df[col] + noise
    
    # Clip numerical values to stay realistic
    drifted_df['age'] = drifted_df['age'].clip(17, 100)
    drifted_df['hours.per.week'] = drifted_df['hours.per.week'].clip(1, 99)
    
    print(f"Saving drifted data to {output_path}...")
    drifted_df.to_csv(output_path, index=False)
    print("✅ Drifted data generated successfully!")

if __name__ == '__main__':
    generate_drifted_adult_data()
