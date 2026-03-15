import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    for col in ['customerID', 'customer_id', 'CustomerID']:
        if col in df.columns:
            df = df.drop(columns=[col])

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    return df