"""
Clean rerun: LSTM baseline with glucose + insulin features for OhioT1DM.

Expected files in the same folder:
    train_features.csv
    test_features.csv

Features:
    Sequential input:
        glucose_lag_12 ... glucose_lag_1  (60 minutes history, oldest -> newest)
    Static/context input:
        basal
        bolus
        bolus_30min
        optional glucose_slope, controlled by USE_GLUCOSE_SLOPE

Target:
    target = glucose 30 minutes ahead

Outputs:
    baseline_lstm_results.csv
    baseline_lstm_predictions.csv
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"
TARGET_COL = "target"

EPOCHS = 50
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
GRAD_CLIP = 1.0

# Set to True only if you want to reproduce the older Table 1 setup.
# Set to False for a cleaner glucose + insulin setup without slope.
USE_GLUCOSE_SLOPE = True


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------
# DATASET
# -----------------------------
class GlucoseDataset(Dataset):
    def __init__(self, seq_x, static_x, y):
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


# -----------------------------
# MODEL
# -----------------------------
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


def prepare_data(train, test, lag_cols, static_cols):
    needed_cols = ["patient", "timestamp"] + lag_cols + static_cols + [TARGET_COL]

    missing_train = [c for c in needed_cols if c not in train.columns]
    missing_test = [c for c in needed_cols if c not in test.columns]
    if missing_train or missing_test:
        raise ValueError(f"Missing columns. Train: {missing_train}, Test: {missing_test}")

    train = train[needed_cols].dropna().reset_index(drop=True)
    test = test[needed_cols].dropna().reset_index(drop=True)

    X_train_seq = train[lag_cols].values.astype(np.float32)
    X_test_seq = test[lag_cols].values.astype(np.float32)

    # Global scaling across all lag values keeps one glucose scale for the sequence.
    seq_scaler = StandardScaler()
    n_train, seq_len = X_train_seq.shape
    n_test = X_test_seq.shape[0]

    X_train_seq = seq_scaler.fit_transform(X_train_seq.reshape(-1, 1)).reshape(n_train, seq_len, 1)
    X_test_seq = seq_scaler.transform(X_test_seq.reshape(-1, 1)).reshape(n_test, seq_len, 1)

    static_scaler = StandardScaler()
    X_train_static = static_scaler.fit_transform(train[static_cols].values.astype(np.float32))
    X_test_static = static_scaler.transform(test[static_cols].values.astype(np.float32))

    y_train = train[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    y_test = test[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    return train, test, X_train_seq, X_train_static, y_train_scaled, X_test_seq, X_test_static, y_test, y_scaler


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train = pd.read_csv(TRAIN_FILE, parse_dates=["timestamp"])
    test = pd.read_csv(TEST_FILE, parse_dates=["timestamp"])

    lag_cols = [f"glucose_lag_{i}" for i in range(12, 0, -1)]
    static_cols = ["basal", "bolus", "bolus_30min"]
    if USE_GLUCOSE_SLOPE and "glucose_slope" in train.columns and "glucose_slope" in test.columns:
        static_cols.append("glucose_slope")

    print("Sequential features:", lag_cols)
    print("Static features:", static_cols)

    train_sub, test_sub, X_train_seq, X_train_static, y_train_scaled, X_test_seq, X_test_static, y_test, y_scaler = prepare_data(
        train, test, lag_cols, static_cols
    )

    train_dataset = GlucoseDataset(X_train_seq, X_train_static, y_train_scaled)
    test_dataset = GlucoseDataset(X_test_seq, X_test_static, np.zeros_like(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(
        seq_input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        static_size=len(static_cols),
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    for epoch in range(EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses))
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | train loss: {epoch_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

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

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\nClean LSTM baseline: glucose + insulin")
    print("--------------------------------------")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    results = pd.DataFrame([{
        "model": "LSTM",
        "feature_set": "glucose_insulin",
        "sequential_features": ", ".join(lag_cols),
        "static_features": ", ".join(static_cols),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "n_train": len(train_sub),
        "n_test": len(test_sub),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }])
    results.to_csv("baseline_lstm_results.csv", index=False)

    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose_30min"] = y_true
    pred_df["predicted_glucose_30min"] = y_pred
    pred_df.to_csv("baseline_lstm_predictions.csv", index=False)

    print("\nSaved baseline_lstm_results.csv")
    print("Saved baseline_lstm_predictions.csv")


if __name__ == "__main__":
    main()
