import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train = pd.read_csv("train_features.csv")
test = pd.read_csv("test_features.csv")

lag_cols = [f"glucose_lag_{i}" for i in range(12, 0, -1)]   # oldest -> most recent
static_cols = ["basal", "bolus", "bolus_30min", "glucose_slope"]
target_col = "target"

# keep only needed columns
needed_cols = ["patient", "timestamp"] + lag_cols + static_cols + [target_col]
train = train[needed_cols].copy()
test = test[needed_cols].copy()

# drop any remaining missing values just in case
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)


# --------------------------------------------------
# PREPARE INPUTS
# --------------------------------------------------

# sequence data: (n_samples, seq_len, 1)
X_train_seq = train[lag_cols].values.astype(np.float32)
X_test_seq = test[lag_cols].values.astype(np.float32)

# static data: (n_samples, n_static_features)
X_train_static = train[static_cols].values.astype(np.float32)
X_test_static = test[static_cols].values.astype(np.float32)

y_train = train[target_col].values.astype(np.float32).reshape(-1, 1)
y_test = test[target_col].values.astype(np.float32).reshape(-1, 1)

# scale sequence values
seq_scaler = StandardScaler()
X_train_seq_2d = seq_scaler.fit_transform(X_train_seq)
X_test_seq_2d = seq_scaler.transform(X_test_seq)

# reshape back to sequence format
X_train_seq = X_train_seq_2d.reshape(-1, len(lag_cols), 1)
X_test_seq = X_test_seq_2d.reshape(-1, len(lag_cols), 1)

# scale static features
static_scaler = StandardScaler()
X_train_static = static_scaler.fit_transform(X_train_static)
X_test_static = static_scaler.transform(X_test_static)

# scale target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class GlucoseDataset(Dataset):
    def __init__(self, seq_x, static_x, y):
        self.seq_x = torch.tensor(seq_x, dtype=torch.float32)
        self.static_x = torch.tensor(static_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq_x[idx], self.static_x[idx], self.y[idx]


train_dataset = GlucoseDataset(X_train_seq, X_train_static, y_train_scaled)
test_dataset = GlucoseDataset(X_test_seq, X_test_static, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, seq_input_size=1, hidden_size=64, num_layers=1, static_size=4, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, seq_x, static_x):
        lstm_out, (h_n, c_n) = self.lstm(seq_x)

        # take final hidden state from last LSTM layer
        seq_repr = h_n[-1]  # shape: (batch, hidden_size)

        combined = torch.cat([seq_repr, static_x], dim=1)
        out = self.fc(combined)
        return out


model = LSTMRegressor(
    seq_input_size=1,
    hidden_size=64,
    num_layers=1,
    static_size=len(static_cols),
    dropout=0.2
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for seq_batch, static_batch, y_batch in train_loader:
        seq_batch = seq_batch.to(device)
        static_batch = static_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(seq_batch, static_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

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

print("\nLSTM Test Performance")
print("------------------")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

results = test[["patient", "timestamp"]].copy()
results["true_glucose_30min"] = y_true
results["lstm_predicted_glucose_30min"] = y_pred

results.to_csv("lstm_predictions.csv", index=False)
print("\nSaved lstm_predictions.csv")
