import pandas as pd

def _map_binary(s:pd.Series) -> pd.Series:
    # df['gender'] = df['gender'].replace({
    #     'Male': 1, 
    #     'Female': 0
    # })

    # df = pd.get_dummies(df, columns = ['country'], drop_first = True)

    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {'Yes', 'No'}:
        return s.map({'Yes': 1, 'No': 0}).astype(int)
    
    if valset == {'Male', 'Female'}:
        return s.map({'Male': 1, 'Female': 0}).astype(int)
    
    if len(vals) == 2:
        return s.map({vals[0]: 1, vals[1]: 0}).astype(int)
    
    return s # not binary

def build_features(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:

    df = df.copy()
    print("Starting feature engineering...")

    # get the categorical and numeric columns
    obj_cols = [c for c in df.columns if df[c].dtype == 'object' and c != target_col]
    numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()

    print(f"Found {len(obj_cols)} categorical columns and {len(numeric_cols)} numeric columns.")
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    if binary_cols:
        print("Binary:", binary_cols)
    if multi_cols:
        print("Multi:", multi_cols)

    # apply the binary encoding
    for c in binary_cols: 
        original_dtype = df[c].dtype
        df[c] = _map_binary(df[c])

        print(f" {c}: {original_dtype} -> {df[c].dtype}")

    # apply one hot encoding
    bool_cols  = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    if multi_cols:
        df = pd.get_dummies(df, columns = multi_cols, drop_first = True)
        print(f"Applied one-hot encoding to {len(multi_cols)} columns.")
    
    return df

    
    