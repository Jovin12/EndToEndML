import mlflow
import pandas as pd
import mlflow.xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 

def train_model(df: pd.DataFrame, target_col: str = "churn"):

    y = df[target_col]
    X = df.drop(columns = [target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    scale_pos_weight = (y_train == 0).sum() / (y_train ==1).sum()

    params = {'n_estimators': 507, 'learning_rate': 0.013512420752260309, 'max_depth': 3, 'subsample': 0.9914249477695516, 'min_child_weight': 2, 'gamma': 4, 'reg_alpha': 0.0012193664946194005}
    params.update({
        "random_state" : 42, 
        "n_jobs" : -1, 
        "scale_pos_weight" : scale_pos_weight, })
    
    xgb = XGBClassifier(**params)

    with mlflow.start_run("xgb-churn"):
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1_score", f1)
        mlflow.xgboost.log_model(xgb, "model")

        train_ds = mlflow.data.from_pandas(df, source = "training_data")
        mlflow.log_input(train_ds, context = "training")

        print(f"Model trained with accuracy: {acc:.4f}, recall: {rec:.4f}, precision: {prec:.4f}, f1_score: {f1:.4f}")