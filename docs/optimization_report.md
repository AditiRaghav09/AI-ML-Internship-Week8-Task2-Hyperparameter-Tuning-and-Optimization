# Hyperparameter Optimization Report

# 1. Introduction

This project explores hyperparameter optimization techniques for improving the performance of an XGBoost classification model.

Three approaches were evaluated:

* Baseline XGBoost Model
* Random Search Hyperparameter Optimization
* Bayesian Optimization using Optuna

The objective was to compare their effectiveness in terms of classification accuracy, execution time, and search efficiency.

---

# 2. Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

**Source:** Scikit-learn Built-in Dataset

**Problem Type:** Binary Classification

The dataset contains diagnostic measurements used to classify tumors as malignant or benign.

---

# 3. Baseline Model

A baseline XGBoost classifier was trained using its default hyperparameters.

The baseline model established a reference point for evaluating the impact of hyperparameter optimization.

The following metrics were recorded:

* Classification Accuracy
* Training Execution Time

---

# 4. Random Search

Random Search was implemented using `RandomizedSearchCV`.

A predefined hyperparameter search space was explored by randomly sampling different parameter combinations.

The following hyperparameters were optimized:

* Number of estimators
* Maximum tree depth
* Learning rate
* Subsample ratio
* Column sampling ratio

The best-performing configuration was selected based on cross-validation accuracy.

---

# 5. Bayesian Optimization

Bayesian Optimization was implemented using the Optuna framework.

Unlike Random Search, Bayesian Optimization uses information from previous trials to intelligently guide future searches.

The optimization process focused on maximizing classification accuracy while efficiently exploring the hyperparameter space.

---

# 6. Performance Comparison

The optimization methods were compared using:

* Classification Accuracy
* Training Execution Time

### Baseline Model

* Default XGBoost configuration
* Fastest training time
* Reference performance

### Random Search

* Improved model performance by exploring random parameter combinations.
* Required additional computation compared to the baseline model.

### Bayesian Optimization

* Achieved strong model performance using intelligent hyperparameter selection.
* Reduced unnecessary exploration by learning from previous optimization trials.

---

# 7. Optimal Hyperparameters

The best hyperparameter configuration obtained through Bayesian Optimization produced the highest-performing XGBoost model.

These optimized parameters improved the model compared with the baseline configuration and demonstrated the effectiveness of intelligent search strategies.

---

# 8. Conclusion

This project demonstrated the importance of hyperparameter optimization in machine learning.

The comparison showed that:

* The baseline model provides a useful performance benchmark.
* Random Search is simple and effective for exploring hyperparameter combinations.
* Bayesian Optimization is more efficient because it uses previous search results to guide future trials.

For this classification problem, Bayesian Optimization proved to be the most suitable optimization technique because it effectively balanced search efficiency and model performance.

Overall, this project highlights how systematic hyperparameter optimization can significantly improve machine learning models while reducing unnecessary computational effort.
