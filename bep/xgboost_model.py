import pandas as pd
import numpy as np

from xgboost import XGBRegressor
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

feature_cols = [col for col in train.columns
                if col.startswith("glucose_lag_")
                or col in ["basal", "bolus", "bolus_30min", "glucose_slope"]]

X_train = train[feature_cols]
y_train = train["target"]

X_test = test[feature_cols]
y_test = test["target"]


# --------------------------------------------------
# XGBOOST MODEL
# --------------------------------------------------

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Test Performance")
print("------------------")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

results = test[["patient", "timestamp"]].copy()
results["true_glucose_30min"] = y_test
results["xgb_predicted_glucose_30min"] = y_pred

results.to_csv("xgb_predictions.csv", index=False)

print("\nSaved xgb_predictions.csv")
