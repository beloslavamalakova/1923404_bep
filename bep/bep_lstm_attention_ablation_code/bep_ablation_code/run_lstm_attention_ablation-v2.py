"""
Run BEP ablation study for:
  1. LSTM
  2. LSTM + temporal attention

Feature sets:
  1. glucose_only
  2. glucose_insulin
  3. glucose_meal

Input files:
  train_features_ablation.csv and test_features_ablation.csv
  OR your existing train_features.csv and test_features.csv, if they contain needed columns.

Outputs are written to ablation_outputs/:
  - ablation_results.csv
  - predictions_*.csv
  - attention_weights_*.csv for attention models
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


class AttentionLSTMRegressor(nn.Module):
    def __init__(self, seq_input_size=1, hidden_size=64, num_layers=1, static_size=0, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_size, 1)
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
        lstm_out, _ = self.lstm(seq_x)              # (batch, seq_len, hidden)
        attn_scores = self.attention(lstm_out)      # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        combined = torch.cat([context, static_x], dim=1)
        return self.fc(combined), attn_weights


def get_feature_sets(train_columns):
    # lag_cols ordered oldest→newest so the LSTM sees the sequence in
    # chronological order: lag_12 (60 min ago) ... lag_1 (5 min ago).
    lag_cols = [f"glucose_lag_{i}" for i in range(12, 0, -1)]

    feature_sets = {
        "glucose_only": [],
        "glucose_insulin": ["basal", "bolus", "bolus_30min", "time_since_last_bolus_min"],
        "glucose_meal": ["carbs", "carbs_30min", "time_since_last_meal_min"],
    }

    # Be forgiving if you run on old train_features.csv without timing/meal columns.
    cleaned = {}
    for name, static_cols in feature_sets.items():
        available_static = [c for c in static_cols if c in train_columns]
        missing = [c for c in static_cols if c not in train_columns]
        if missing:
            print(f"Warning for {name}: missing columns ignored: {missing}")
        cleaned[name] = {"lag_cols": lag_cols, "static_cols": available_static}

    return cleaned


def prepare_data(train, test, lag_cols, static_cols, target_col="target"):
    needed = ["patient", "timestamp"] + lag_cols + static_cols + [target_col]
    train_sub = train[needed].dropna().reset_index(drop=True).copy()
    test_sub = test[needed].dropna().reset_index(drop=True).copy()

    X_train_seq = train_sub[lag_cols].values.astype(np.float32)
    X_test_seq = test_sub[lag_cols].values.astype(np.float32)

    # FIX: fit a single scaler on the flattened sequence values so that all
    # lag columns share one mean/std.  The original code fitted StandardScaler
    # column-by-column (one per lag), which is fine but can slightly differ
    # per-lag; using a global scale is cleaner and more common for time-series.
    seq_scaler = StandardScaler()
    n_train, seq_len = X_train_seq.shape
    X_train_seq = seq_scaler.fit_transform(
        X_train_seq.reshape(-1, 1)
    ).reshape(n_train, seq_len, 1)
    X_test_seq = seq_scaler.transform(
        X_test_seq.reshape(-1, 1)
    ).reshape(len(test_sub), seq_len, 1)

    if len(static_cols) > 0:
        X_train_static = train_sub[static_cols].values.astype(np.float32)
        X_test_static = test_sub[static_cols].values.astype(np.float32)
        static_scaler = StandardScaler()
        X_train_static = static_scaler.fit_transform(X_train_static)
        X_test_static = static_scaler.transform(X_test_static)
    else:
        X_train_static = np.zeros((len(train_sub), 0), dtype=np.float32)
        X_test_static = np.zeros((len(test_sub), 0), dtype=np.float32)

    y_train = train_sub[target_col].values.astype(np.float32).reshape(-1, 1)
    y_test = test_sub[target_col].values.astype(np.float32).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    # y_test is kept in original scale for evaluation; scaled version unused.

    return train_sub, test_sub, X_train_seq, X_train_static, y_train_scaled, X_test_seq, X_test_static, y_test, y_scaler


def train_one_model(model_name, feature_set_name, train, test, lag_cols, static_cols, args, device):
    set_seed(args.seed)

    train_sub, test_sub, X_train_seq, X_train_static, y_train_scaled, X_test_seq, X_test_static, y_test, y_scaler = prepare_data(
        train, test, lag_cols, static_cols, target_col=args.target_col
    )

    train_dataset = GlucoseDataset(X_train_seq, X_train_static, y_train_scaled)
    test_dataset = GlucoseDataset(X_test_seq, X_test_static, np.zeros_like(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    if model_name == "lstm":
        model = LSTMRegressor(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            static_size=len(static_cols),
            dropout=args.dropout,
        ).to(device)
    elif model_name == "attention_lstm":
        model = AttentionLSTMRegressor(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            static_size=len(static_cols),
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # FIX: add a ReduceLROnPlateau scheduler so the learning rate decays when
    # training loss plateaus, helping models with more features converge fully.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for seq_batch, static_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            if model_name == "attention_lstm":
                preds, _ = model(seq_batch, static_batch)
            else:
                preds = model(seq_batch, static_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            # FIX: gradient clipping prevents exploding gradients, which can
            # occur when the additional static features have high variance.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = np.mean(losses)
        scheduler.step(epoch_loss)

        if args.verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"{model_name} | {feature_set_name} | epoch {epoch+1:02d}/{args.epochs}"
                f" | loss {epoch_loss:.4f} | lr {current_lr:.2e}"
            )

    model.eval()
    all_preds = []
    all_attn = []
    with torch.no_grad():
        for seq_batch, static_batch, _ in test_loader:
            seq_batch = seq_batch.to(device)
            static_batch = static_batch.to(device)
            if model_name == "attention_lstm":
                preds, attn = model(seq_batch, static_batch)
                all_attn.append(attn.cpu().numpy())
            else:
                preds = model(seq_batch, static_batch)
            all_preds.append(preds.cpu().numpy())

    y_pred_scaled = np.vstack(all_preds)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_test.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose"] = y_true
    pred_df["predicted_glucose"] = y_pred
    pred_df["model"] = model_name
    pred_df["feature_set"] = feature_set_name
    pred_df.to_csv(os.path.join(args.output_dir, f"predictions_{model_name}_{feature_set_name}.csv"), index=False)

    if model_name == "attention_lstm" and all_attn:
        # attn_array shape: (N, seq_len)
        # Index 0 = lag_12 (oldest, 60 min ago), index 11 = lag_1 (5 min ago).
        attn_array = np.vstack(all_attn).squeeze(axis=2)
        attn_df = pd.DataFrame({
            "lag": lag_cols,                              # lag_12 … lag_1
            "minutes_before_prediction": list(range(60, 0, -5)),   # 60, 55 … 5
            "average_attention_weight": attn_array.mean(axis=0),
        })
        attn_df.to_csv(os.path.join(args.output_dir, f"attention_weights_{feature_set_name}.csv"), index=False)

    return {
        "model": model_name,
        "feature_set": feature_set_name,
        "static_features": ", ".join(static_cols) if static_cols else "none",
        "n_train": len(train_sub),
        "n_test": len(test_sub),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="train_features_ablation.csv")
    parser.add_argument("--test-file", default="test_features_ablation.csv")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--output-dir", default="ablation_outputs")
    parser.add_argument("--epochs", type=int, default=50)   # FIX: raised default from 30→50
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train = pd.read_csv(args.train_file, parse_dates=["timestamp"])
    test = pd.read_csv(args.test_file, parse_dates=["timestamp"])

    feature_sets = get_feature_sets(train.columns)
    results = []

    for feature_set_name, cols in feature_sets.items():
        lag_cols = cols["lag_cols"]
        static_cols = cols["static_cols"]
        for model_name in ["lstm", "attention_lstm"]:
            print(f"\nRunning {model_name} with {feature_set_name}...")
            result = train_one_model(model_name, feature_set_name, train, test, lag_cols, static_cols, args, device)
            results.append(result)
            print(f"MAE={result['MAE']:.3f}, RMSE={result['RMSE']:.3f}, R2={result['R2']:.3f}")

    results_df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ablation_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\nFinal ablation results:")
    print(results_df[["model", "feature_set", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
