import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import optuna

print("TEST PHASE 2: Modeling")

df = pd.read_csv(r'C:\Users\jovin\Desktop\TF\MLOPS\EndToEndMLPipeline\data\processed\ProcessedChurnDS.csv')

if df['churn'].dtype == 'object':
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

assert df['churn'].isna().sum() == 0, "Target column contains missing values."
assert set(df['churn'].unique()) <= {0, 1}, "Target column contains values other than 0 and 1."

y = df['churn']
X = df.drop(columns=['churn'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

THRESHOLD = 0.3


scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]


def objective(trial):
    # every parameter in XGBOOST, and start-end values for each parameters for optuna to try
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300,800),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth":         trial.suggest_int("max_depth", 3,10),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0), 
        "colsample_bytree" : trial.suggest_float("subsample", 0.5, 1.0), 
        "min_child_weight":  trial.suggest_int("min_child_weight", 1,10),
        "gamma":             trial.suggest_int("gamma",  0,5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0,5),
        "reg_lambda":        trial.suggest_float("reg_alpha", 0,5),
        "random_state" : 42, 
        "n_jobs" : -1, 
        "scale_pos_weight" : scale_pos_weight, 
        # "eval_metrics" :'logloss'
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    y_pred = (proba >= THRESHOLD).astype(int)

    return recall_score(y_test, y_pred, pos_label = 1)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params: ", study.best_params)
print("Best Recall: ", study.best_value)