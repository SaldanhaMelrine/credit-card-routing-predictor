import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Preprocess Data
def preprocess_data(df):

    # Handling missing values
    if df.isnull().any().any(): 
        print("Missing values detected. Handling missing values...")
        print(df.isnull().sum())  # Display count of missing values per column
        df['amount'] = df['amount'].fillna(df['amount'].median())
        df['PSP'] = df['PSP'].fillna('Unknown')
        df['success'] = df['success'].fillna(0)
        df.fillna(df.mode().iloc[0], inplace=True)
        print("Missing values handled.")
    else:
        print("No Missing values detected.")
    
    # Ensure the timestamp column is in datetime format
    df['tmsp'] = pd.to_datetime(df['tmsp'])

    # Sort the data by country, amount, and timestamp for logical grouping
    df = df.sort_values(by=['country', 'amount', 'tmsp'])

    # Create the attempt_group column
    df['attempt_group'] = (
        (df['country'] != df['country'].shift(1)) |
        (df['amount'] != df['amount'].shift(1)) |
        ((df['tmsp'] - df['tmsp'].shift(1)).dt.total_seconds() > 60)
    ).cumsum()

    fee_structure = {
        'Moneycard': {'success': 5, 'fail': 2},
        'Goldcard': {'success': 10, 'fail': 5},
        'UK_Card': {'success': 3, 'fail': 1},
        'Simplecard': {'success': 1, 'fail': 0.5}
    }

    # Fee calculation for each transaction based on PSP and success
    df['fee'] = df.apply(
        lambda row: fee_structure[row['PSP']]['success'] if row['success'] == 1
        else fee_structure[row['PSP']]['fail'], axis=1
    )

    # Cumulative fees and attempt numbers calculation
    df['cumulative_fee'] = df.groupby('attempt_group')['fee'].cumsum()  # Cumulative fee
    df['attempt_number'] = df.groupby('attempt_group').cumcount() + 1  # Attempt number within group

    # PSP success rates 
    psp_success_rate = df.groupby('PSP')['success'].mean().rename('psp_success_rate')
    psp_country_success_rate = df.groupby(['PSP', 'country'])['success'].mean().rename('psp_country_success_rate')
    df = df.merge(psp_success_rate, on='PSP', how='left')
    df = df.merge(psp_country_success_rate, on=['PSP', 'country'], how='left')

    # PSP costs
    df['psp_success_cost'] = df['PSP'].map(lambda x: fee_structure[x]['success'])
    df['psp_failure_cost'] = df['PSP'].map(lambda x: fee_structure[x]['fail'])

    # Calculate avg_cumulative_fee per PSP
    df['avg_cumulative_fee'] = df.groupby('PSP')['cumulative_fee'].transform('mean')

    # Calculate avg_psp_reattempts per PSP
    # Reattempts: Failed transaction followed by a successful transaction for the same PSP
    df['reattempt'] = (df['success'] == 0) & (df['success'].shift(-1) == 1)
    df['avg_psp_reattempts'] = df.groupby('PSP')['reattempt'].transform('mean')

    #Create amount bins
    bins = [0, 100, 500, float('inf')]  # Low, Mid, High ranges
    labels = ['Low', 'Mid', 'High']
    df['amount_range'] = pd.cut(df['amount'], bins=bins, labels=labels, right=False)

    # Label Encoding for 'PSP' using sklearn's LabelEncoder
    le_psp = LabelEncoder()
    df['PSP'] = le_psp.fit_transform(df['PSP'])  # Encode PSP column

    # Print the mapping of PSP to codes
    print("PSP Encoding Mapping (Labels -> Codes):")
    for i, name in enumerate(le_psp.classes_):
        print(f"Code {i}: {name}")

    # Label Encoding for all categorical columns
    le = LabelEncoder()
    categorical_columns = ['country', 'card']  
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])

        # Print the mapping of each category to the encoded label
        print(f"\n{column} Encoding Mapping (Labels -> Codes):")
        for i, name in enumerate(le.classes_):
            print(f"Code {i}: {name}")

    # Calculate Pre-Prediction Success Rate
    pre_success_rate = df[df['success'] == 1].shape[0] / df.shape[0]  

    # Calculation for fee Optimization
    pre_model_fees = df['psp_success_cost'].mean()

    return df, le_psp, pre_success_rate, pre_model_fees
