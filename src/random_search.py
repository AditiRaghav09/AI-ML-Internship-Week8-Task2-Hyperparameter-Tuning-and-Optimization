"""
random_search.py

Random Search Hyperparameter Optimization using XGBoost.
"""

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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
# Hyperparameter Search Space
# -----------------------------

param_distributions = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# -----------------------------
# Random Search
# -----------------------------

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    ),
    param_distributions=param_distributions,
    n_iter=10,
    scoring="accuracy",
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------

best_model = random_search.best_estimator_

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Random Search Results")
print("----------------------")

print("Best Parameters:")
print(random_search.best_params_)

print(f"\nTest Accuracy: {accuracy:.4f}")
