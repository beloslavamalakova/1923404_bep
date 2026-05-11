"""
Transformer ablation study for OhioT1DM.

Runs Transformer Encoder on:
1. glucose_only
2. glucose_insulin
3. glucose_meal

Input:
  train_features_ablation.csv
  test_features_ablation.csv

Output:
  transformer_outputs/transformer_results.csv
  transformer_outputs/predictions_transformer_*.csv
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GlucoseDataset(Dataset):
    def __init__(self, seq_x, static_x, y):
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        seq_input_size=1,
        static_size=0,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(seq_input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model + static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq_x, static_x):
        x = self.input_projection(seq_x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        # Mean pooling over timesteps
        seq_repr = x.mean(dim=1)

        combined = torch.cat([seq_repr, static_x], dim=1)
        return self.fc(combined)


def get_feature_sets(train_columns):
    lag_cols = [f"glucose_lag_{i}" for i in range(12, 0, -1)]

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
        available_static = [c for c in static_cols if c in train_columns]
        missing = [c for c in static_cols if c not in train_columns]

        if missing:
            print(f"Warning for {name}: missing columns ignored: {missing}")

        cleaned[name] = {
            "lag_cols": lag_cols,
            "static_cols": available_static,
        }

    return cleaned


def prepare_data(train, test, lag_cols, static_cols, target_col="target"):
    needed_cols = ["patient", "timestamp"] + lag_cols + static_cols + [target_col]

    train_sub = train[needed_cols].dropna().reset_index(drop=True)
    test_sub = test[needed_cols].dropna().reset_index(drop=True)

    X_train_seq = train_sub[lag_cols].values.astype(np.float32)
    X_test_seq = test_sub[lag_cols].values.astype(np.float32)

    seq_scaler = StandardScaler()

    n_train, seq_len = X_train_seq.shape
    n_test = X_test_seq.shape[0]

    X_train_seq = seq_scaler.fit_transform(
        X_train_seq.reshape(-1, 1)
    ).reshape(n_train, seq_len, 1)

    X_test_seq = seq_scaler.transform(
        X_test_seq.reshape(-1, 1)
    ).reshape(n_test, seq_len, 1)

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

    return (
        train_sub,
        test_sub,
        X_train_seq,
        X_train_static,
        y_train_scaled,
        X_test_seq,
        X_test_static,
        y_test,
        y_scaler,
    )


def train_transformer(feature_set_name, train, test, lag_cols, static_cols, args, device):
    set_seed(args.seed)

    (
        train_sub,
        test_sub,
        X_train_seq,
        X_train_static,
        y_train_scaled,
        X_test_seq,
        X_test_static,
        y_test,
        y_scaler,
    ) = prepare_data(train, test, lag_cols, static_cols, args.target_col)

    train_dataset = GlucoseDataset(X_train_seq, X_train_static, y_train_scaled)
    test_dataset = GlucoseDataset(X_test_seq, X_test_static, np.zeros_like(y_test))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    model = TransformerRegressor(
        seq_input_size=1,
        static_size=len(static_cols),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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

        epoch_loss = np.mean(losses)
        scheduler.step(epoch_loss)

        if args.verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Transformer | {feature_set_name} | "
                f"epoch {epoch + 1:02d}/{args.epochs} | "
                f"loss {epoch_loss:.4f} | lr {lr:.2e}"
            )

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

    os.makedirs(args.output_dir, exist_ok=True)

    pred_df = test_sub[["patient", "timestamp"]].copy()
    pred_df["true_glucose"] = y_true
    pred_df["predicted_glucose"] = y_pred
    pred_df["model"] = "transformer"
    pred_df["feature_set"] = feature_set_name

    pred_df.to_csv(
        os.path.join(
            args.output_dir,
            f"predictions_transformer_{feature_set_name}.csv",
        ),
        index=False,
    )

    return {
        "model": "transformer",
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
    parser.add_argument("--output-dir", default="transformer_outputs")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
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
        print(f"\nRunning Transformer with {feature_set_name}...")

        result = train_transformer(
            feature_set_name=feature_set_name,
            train=train,
            test=test,
            lag_cols=cols["lag_cols"],
            static_cols=cols["static_cols"],
            args=args,
            device=device,
        )

        results.append(result)

        print(
            f"MAE={result['MAE']:.3f}, "
            f"RMSE={result['RMSE']:.3f}, "
            f"R2={result['R2']:.3f}"
        )

    results_df = pd.DataFrame(results)

    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, "transformer_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\nFinal Transformer results:")
    print(results_df[["model", "feature_set", "MAE", "RMSE", "R2"]].to_string(index=False))

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
