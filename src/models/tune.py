import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X,y):
    # fine tuning
    # scale_pos_weight = (y_train == 0).sum() / (y_train ==1).sum()


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
            # "scale_pos_weight" : scale_pos_weight, 
            # "eval_metrics" :'logloss'
        }

        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv = 3, scoring = "recall")
        return scores.mean()
    
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = 30)

    print("BestParams:", study.best_params)
    print("BestRecall:", study.best_value)

    return study.best_params