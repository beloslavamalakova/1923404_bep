import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train = pd.read_csv("train_features.csv")
test = pd.read_csv("test_features.csv")

# --------------------------------------------------
# DEFINE FEATURES
# --------------------------------------------------

feature_cols = [col for col in train.columns
                if col.startswith("glucose_lag_")
                or col in ["basal", "bolus", "bolus_30min", "glucose_slope"]]

X_train = train[feature_cols]
y_train = train["target"]

X_test = test[feature_cols]
y_test = test["target"]


# --------------------------------------------------
# SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# RIDGE WITH CROSS-VALIDATION
# --------------------------------------------------

param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

ridge = Ridge()

grid = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("Best alpha:", grid.best_params_["alpha"])


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

y_pred = best_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nTest Performance")
print("------------------")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

results = test[["patient", "timestamp"]].copy()
results["true_glucose_30min"] = y_test
results["predicted_glucose_30min"] = y_pred

results.to_csv("ridge_predictions.csv", index=False)

print("\nSaved ridge_predictions.csv")

naive_pred = test["glucose"]

mae_naive = mean_absolute_error(y_test, naive_pred)
rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))
r2_naive = r2_score(y_test, naive_pred)

print("\nNaive Baseline")
print("------------------")
print(f"MAE  : {mae_naive:.3f}")
print(f"RMSE : {rmse_naive:.3f}")
print(f"R²   : {r2_naive:.3f}")
