"""
bayesian_opt.py

Bayesian Hyperparameter Optimization using Optuna and XGBoost.
"""

import pandas as pd
import optuna

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


# -----------------------------
# Load Dataset
# -----------------------------

RANDOM_STATE = 42

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# Objective Function
# -----------------------------

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# -----------------------------
# Bayesian Optimization
# -----------------------------

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=20
)

# -----------------------------
# Best Model
# -----------------------------

best_model = XGBClassifier(
    **study.best_params,
    random_state=RANDOM_STATE,
    eval_metric="logloss"
)

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# -----------------------------
# Results
# -----------------------------

print("Bayesian Optimization Results")
print("-----------------------------")

print("Best Parameters:")
print(study.best_params)

print(f"\nTest Accuracy: {accuracy:.4f}")
