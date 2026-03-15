import os 
import pandas as pd

import sys
sys.path.append(os.path.abspath('src'))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

DATA_PATH = r'C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\data\raw\Bank Customer Churn Prediction.csv'
TARGET_COL = 'churn'

def main():

    print("TEST PHASE 1: Loading , preprocess and build features:")

    # Load data
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Data shape: {df.shape}")
    print(df.head(5))

    # Preprocess data
    print("Preprocessing data...")
    df_clean = preprocess_data(df)
    print("Data preprocess shape: ", df_clean.shape)
    print(df_clean.head(5))

    # Build features
    print("Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Features shape: {df_features.shape}")
    print(df_features.head(5))

if __name__ == "__main__":
    main()