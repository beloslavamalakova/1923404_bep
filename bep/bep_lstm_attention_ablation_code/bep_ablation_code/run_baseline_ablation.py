"""
Baseline ablation study for OhioT1DM.

Runs three baseline models on the same feature sets used in the LSTM/Attention/Transformer ablation:
  1. glucose_only
  2. glucose_insulin
  3. glucose_meal

Models:
  1. Ridge Regression
  2. XGBoost
  3. LSTM

Input files, expected in the same folder:
  train_features_ablation.csv
  test_features_ablation.csv

Output:
  baseline_ablation_outputs/baseline_ablation_results.csv
  baseline_ablation_outputs/predictions_*.csv
"""

import os
import random
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_feature_sets(columns):
    lag_cols = [f"glucose_lag_{i}" for i in range(12, 0, -1)]  # lag_12 ... lag_1

    feature_sets = {
        "glucose_only": [],
        "glucose_insulin": [
            "basal",
            "bolus",
            "bolus_30min",
            "time_since_last_bolus_min",
        ],
        "glucose_meal": [
            "carbs",
            "carbs_30min",
            "time_since_last_meal_min",
        ],
    }

    cleaned = {}
    for name, static_cols in feature_sets.items():
        missing_lags = [c for c in lag_cols if c not in columns]
        if missing_lags:
            raise ValueError(f"Missing glucose lag columns: {missing_lags}")

        available_static = [c for c in static_cols if c in columns]
        missing_static = [c for c in static_cols if c not in columns]
        if missing_static:
            print(f"Warning for {name}: missing static columns ignored: {missing_static}")

        cleaned[name] = {
            "lag_cols": lag_cols,
            "static_cols": available_static,
            "all_cols": lag_cols + available_static,
        }

    return cleaned


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_ridge(feature_set_name, train, test, feature_cols, target_col, output_dir):
    needed = ["patient", "timestamp"] + feature_cols + [target_col]
    train_sub = train[needed].dropna().reset_index(drop=True)
    test_sub = test[needed].dropna().reset_index(drop=True)

    X_train = train_sub[feature_cols].values
    y_train = train_sub[target_col].values
    X_test = test_sub[feature_cols].values
    y_test = test_sub[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge()
    grid = GridSearchCV(
        ridge,
        param_grid={"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test_scaled)
    metrics = evaluate(y_test, y_pred)

    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose"] = y_test
    pred_df["predicted_glucose"] = y_pred
    pred_df["model"] = "ridge"
    pred_df["feature_set"] = feature_set_name
    pred_df.to_csv(os.path.join(output_dir, f"predictions_ridge_{feature_set_name}.csv"), index=False)

    return {
        "model": "ridge",
        "feature_set": feature_set_name,
        "features": ", ".join(feature_cols),
        "best_params": str(grid.best_params_),
        "n_train": len(train_sub),
        "n_test": len(test_sub),
        **metrics,
    }


def run_xgboost(feature_set_name, train, test, feature_cols, target_col, output_dir):
    if XGBRegressor is None:
        print("Skipping XGBoost because xgboost is not installed.")
        return None

    needed = ["patient", "timestamp"] + feature_cols + [target_col]
    train_sub = train[needed].dropna().reset_index(drop=True)
    test_sub = test[needed].dropna().reset_index(drop=True)

    X_train = train_sub[feature_cols].values
    y_train = train_sub[target_col].values
    X_test = test_sub[feature_cols].values
    y_test = test_sub[target_col].values

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    grid = GridSearchCV(
        xgb,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=0,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose"] = y_test
    pred_df["predicted_glucose"] = y_pred
    pred_df["model"] = "xgboost"
    pred_df["feature_set"] = feature_set_name
    pred_df.to_csv(os.path.join(output_dir, f"predictions_xgboost_{feature_set_name}.csv"), index=False)

    return {
        "model": "xgboost",
        "feature_set": feature_set_name,
        "features": ", ".join(feature_cols),
        "best_params": str(grid.best_params_),
        "n_train": len(train_sub),
        "n_test": len(test_sub),
        **metrics,
    }


class GlucoseDataset(Dataset):
    def __init__(self, seq_x, static_x, y):
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, seq_input_size=1, hidden_size=64, num_layers=1, static_size=0, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq_x, static_x):
        _, (h_n, _) = self.lstm(seq_x)
        seq_repr = h_n[-1]
        combined = torch.cat([seq_repr, static_x], dim=1)
        return self.fc(combined)


def run_lstm(feature_set_name, train, test, lag_cols, static_cols, args, device):
    set_seed(args.seed)

    needed = ["patient", "timestamp"] + lag_cols + static_cols + [args.target_col]
    train_sub = train[needed].dropna().reset_index(drop=True)
    test_sub = test[needed].dropna().reset_index(drop=True)

    X_train_seq = train_sub[lag_cols].values.astype(np.float32)
    X_test_seq = test_sub[lag_cols].values.astype(np.float32)

    seq_scaler = StandardScaler()
    n_train, seq_len = X_train_seq.shape
    n_test = X_test_seq.shape[0]

    # Same style as the newer ablation code: global scaling over all sequence values.
    X_train_seq = seq_scaler.fit_transform(X_train_seq.reshape(-1, 1)).reshape(n_train, seq_len, 1)
    X_test_seq = seq_scaler.transform(X_test_seq.reshape(-1, 1)).reshape(n_test, seq_len, 1)

    if static_cols:
        X_train_static = train_sub[static_cols].values.astype(np.float32)
        X_test_static = test_sub[static_cols].values.astype(np.float32)
        static_scaler = StandardScaler()
        X_train_static = static_scaler.fit_transform(X_train_static)
        X_test_static = static_scaler.transform(X_test_static)
    else:
        X_train_static = np.zeros((len(train_sub), 0), dtype=np.float32)
        X_test_static = np.zeros((len(test_sub), 0), dtype=np.float32)

    y_train = train_sub[args.target_col].values.astype(np.float32).reshape(-1, 1)
    y_test = test_sub[args.target_col].values.astype(np.float32).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    train_dataset = GlucoseDataset(X_train_seq, X_train_static, y_train_scaled)
    test_dataset = GlucoseDataset(X_test_seq, X_test_static, np.zeros_like(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = LSTMRegressor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        static_size=len(static_cols),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for seq_batch, static_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(seq_batch, static_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses))
        scheduler.step(epoch_loss)

        if args.verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(f"lstm | {feature_set_name} | epoch {epoch + 1:02d}/{args.epochs} | loss {epoch_loss:.4f} | lr {lr:.2e}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for seq_batch, static_batch, _ in test_loader:
            seq_batch = seq_batch.to(device)
            static_batch = static_batch.to(device)
            preds = model(seq_batch, static_batch)
            all_preds.append(preds.cpu().numpy())

    y_pred_scaled = np.vstack(all_preds)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_test.flatten()

    metrics = evaluate(y_true, y_pred)

    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose"] = y_true
    pred_df["predicted_glucose"] = y_pred
    pred_df["model"] = "lstm"
    pred_df["feature_set"] = feature_set_name
    pred_df.to_csv(os.path.join(args.output_dir, f"predictions_lstm_{feature_set_name}.csv"), index=False)

    return {
        "model": "lstm",
        "feature_set": feature_set_name,
        "features": ", ".join(lag_cols + static_cols),
        "best_params": "fixed architecture",
        "n_train": len(train_sub),
        "n_test": len(test_sub),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="train_features_ablation.csv")
    parser.add_argument("--test-file", default="test_features_ablation.csv")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--output-dir", default="baseline_ablation_outputs")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train = pd.read_csv(args.train_file, parse_dates=["timestamp"])
    test = pd.read_csv(args.test_file, parse_dates=["timestamp"])

    feature_sets = get_feature_sets(train.columns)
    results = []

    for feature_set_name, cols in feature_sets.items():
        lag_cols = cols["lag_cols"]
        static_cols = cols["static_cols"]
        all_cols = cols["all_cols"]

        print(f"\n=== Feature set: {feature_set_name} ===")
        print("Features:", ", ".join(all_cols))

        print("Running Ridge...")
        ridge_result = run_ridge(feature_set_name, train, test, all_cols, args.target_col, args.output_dir)
        results.append(ridge_result)
        print(f"Ridge:   MAE={ridge_result['MAE']:.3f}, RMSE={ridge_result['RMSE']:.3f}, R2={ridge_result['R2']:.3f}")

        print("Running XGBoost...")
        xgb_result = run_xgboost(feature_set_name, train, test, all_cols, args.target_col, args.output_dir)
        if xgb_result is not None:
            results.append(xgb_result)
            print(f"XGBoost: MAE={xgb_result['MAE']:.3f}, RMSE={xgb_result['RMSE']:.3f}, R2={xgb_result['R2']:.3f}")

        print("Running LSTM...")
        lstm_result = run_lstm(feature_set_name, train, test, lag_cols, static_cols, args, device)
        results.append(lstm_result)
        print(f"LSTM:    MAE={lstm_result['MAE']:.3f}, RMSE={lstm_result['RMSE']:.3f}, R2={lstm_result['R2']:.3f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "baseline_ablation_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\nFinal baseline ablation results:")
    print(results_df[["model", "feature_set", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
