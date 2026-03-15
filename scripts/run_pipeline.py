import sys
import os
import time

import mlflow
import mlflow.sklearn
from posthog import project_root

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score

from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_data

def main(args):

    # mlflow setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    mlruns_path = args.mlflow_uri or "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlruns_path)

    if args.experiment:
        mlflow.set_experiment(args.experiment)
    else:
        mlflow.set_experiment("ChurnEndToEndPipeline")

    # start logging for this experiment
    with mlflow.start_run():
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)


        # 1. Load data
        print("Loading data...")
        df = load_data(args.input)

        # 2. Data Quality Validation
        print("Validating data...")
        print("Columns found in DF:", df.columns.tolist())
        print("Unique values in 'churn':", df['churn'].unique() if 'churn' in df.columns else "Column missing")

        is_valid, failed = validate_data(df)

        mlflow.log_metric("data_quality_passed", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent = 2),artifact_file="failed_expectations.json")
            raise ValueError(f"Data validation failed. Failed expectations: {failed}")
        else: 
            print("Data validation passed. Proceeding with preprocessing and modeling.")

        
        # Data preprocessing
        print("Preprocessing data...")
        df = preprocess_data(df)

        processed_path = os.path.join(project_root, 'data', 'processed', 'Processed_Churn_DS.csv')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")


        # Feature engineering
        print("Building features...")
        target = args.target

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        
        df_enc = build_features(df, target_col=target)

        # for c in df_enc.columns:

        #     if c is not None:
        #         print(f"Column: {c}, dtype: {df_enc[c].dtype}")
        
        # raise ValueError("Stopping after feature engineering to check column types. Comment out this line to proceed with training.")

        for c in df_enc.select_dtypes(include = ['bool']).columns:
            print(c)
            df_enc[c] = df_enc[c].astype(int)

        # Save feature metadata
        import json, joblib
        artifacts_dir = os.path.join(project_root, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns = [target]).columns)

        with open(os.path.join(artifacts_dir, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)

        # log MLflow for production
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # save artifacts
        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target_column": target
        }

        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, 'preprocessing_artifact.pkl'))
        mlflow.log_artifact(os.path.join(artifacts_dir, 'preprocessing_artifact.pkl'))
        print("Saved feature columns for Serving consitency")

        # Train-test split
        print("Splitting data into train and test sets...")

        y = df_enc[target]
        X = df_enc.drop(columns=[target])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print("Scale pos weight: ", scale_pos_weight)

        # Train XGBoost model
        print("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight = scale_pos_weight,
            eval_metric='logloss'
        )


        # Train model and Track time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        mlflow.log_metric("train_time_seconds", train_time)
        print(f"Model training completed in {train_time:.2f} seconds.")

        # Evaluate model
        print("Evaluating model...")
        eval_start_time = time.time()
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba >= args.threshold).astype(int)
        eval_time = time.time() - eval_start_time
        mlflow.log_metric("eval_time_seconds", eval_time)
        print(f"Model evaluation completed in {eval_time:.2f} seconds.")


        # log evaluated metrics to MLflow
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)


        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC AUC Score: ", roc_auc)

        # MODEL SERIALIZATION
        print("Serializing model...")
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model logged to MLflow.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run the end-to-end ML pipeline for customer churn prediction.")
    p.add_argument('--input',
                    type=str,
                    default=r'C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\data\raw\Bank Customer Churn Prediction.csv', 
                    help="Path to input raw data CSV file.")
    p.add_argument('--target', type = str, default='churn', help="Target column name.")
    p.add_argument('--threshold', type = float, default=0.3, help="Classification threshold for positive class.")
    p.add_argument('--test_size', type = float, default=0.2, help="Test set size as a fraction.")
    p.add_argument('--mlflow_uri', type = str, default=None, help="MLflow tracking URI. If not provided, defaults to local mlruns directory.")
    p.add_argument('--experiment', type = str, default="Customer_Churn_Prediction", help="MLflow experiment name.")

    args = p.parse_args()
    main(args)

# """
# python scripts/run_pipeline.py\
#  --input "C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\data\raw\Bank Customer Churn Prediction.csv"\
#     --target "churn"\
#      --threshold 0.3 \
#     --test_size 0.2 \
#     --mlflow_uri "file://C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\mlruns" \
#     --experiment "Customer_Churn_Prediction"

# python scripts/run_pipeline.py --input r"C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\data\raw\Bank Customer Churn Prediction.csv" --target churn
#  """