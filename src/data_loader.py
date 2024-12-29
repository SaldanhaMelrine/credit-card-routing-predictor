import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    # Drop 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df
