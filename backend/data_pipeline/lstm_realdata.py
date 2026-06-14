"""
AeroCool AI — Member 3 Pipeline (Real Data Version)
=====================================================
Runs the full LSTM pipeline on merged_master_data.csv
(20 sensor locations across Delhi, 41 timesteps @ 15-min intervals)

Pipeline:
  1. Load + inspect merged_master_data.csv
  2. Normalize features (MinMaxScaler)
  3. Build per-sensor sliding-window sequences
  4. Train PyTorch LSTM
  5. Forecast next temperature per location
  6. Export per-location forecasts (feeds heat_map.py, ndvi_analysis.py, scenario_simulator.py)

HOW TO RUN:
    pip install torch pandas numpy scikit-learn
    python lstm_pipeline_real_data.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)

# ═════════════════════════════════════════════════════════════
# STEP 1 — LOAD THE MERGED DATA
# ═════════════════════════════════════════════════════════════

df = pd.read_csv("merged_master_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

n_sensors = df["sensor_id"].nunique()
n_timesteps = df["timestamp"].nunique()

print("=" * 60)
print("STEP 1 — Loaded merged_master_data.csv")
print("=" * 60)
print(f"Rows:            {len(df)}")
print(f"Sensors:         {n_sensors}")
print(f"Timesteps:       {n_timesteps}  ({df['timestamp'].min()} -> {df['timestamp'].max()})")
print(f"Columns:         {list(df.columns)}")
print()

# ═════════════════════════════════════════════════════════════
# STEP 2 — NORMALIZE FEATURES
# ═════════════════════════════════════════════════════════════

FEATURES = ["actual_temp", "diurnal_factor", "solar_interaction", "ndvi", "priority_score"]
TARGET = "actual_temp"
target_idx = FEATURES.index(TARGET)

scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df[FEATURES])
scaled_df = df[["timestamp", "sensor_id", "location", "latitude", "longitude"]].copy()
for i, f in enumerate(FEATURES):
    scaled_df[f] = scaled_array[:, i]

print("=" * 60)
print("STEP 2 — Normalized features to [0,1]")
print("=" * 60)
print(f"Features: {FEATURES}")
print(f"Target:   {TARGET}")
print()

# ═════════════════════════════════════════════════════════════
# STEP 3 — BUILD PER-SENSOR SEQUENCES
#
# Each of the 20 sensors has its own 41-step time series.
# Sliding windows are built PER SENSOR (so the LSTM learns
# general temporal patterns across all locations), then
# combined into one training set.
# ═════════════════════════════════════════════════════════════

SEQ_LEN = 6    # 6 readings of history
HORIZON = 4    # predict 4 readings ahead

X_list, y_list, sensor_ids_list = [], [], []
rows_per_sensor = None

for sid in scaled_df["sensor_id"].unique():
    s = scaled_df[scaled_df["sensor_id"] == sid].sort_values("timestamp", kind="stable")
    vals = s[FEATURES].to_numpy()
    rows_per_sensor = len(vals)
    for i in range(len(vals) - SEQ_LEN - HORIZON):
        X_list.append(vals[i:i + SEQ_LEN])
        y_list.append(vals[i + SEQ_LEN + HORIZON - 1, target_idx])
        sensor_ids_list.append(sid)

X = np.array(X_list)
y = np.array(y_list)
sensor_ids_arr = np.array(sensor_ids_list)

seqs_per_sensor = rows_per_sensor - SEQ_LEN - HORIZON
print("=" * 60)
print("STEP 3 — Built sliding-window sequences (per sensor)")
print("=" * 60)
print(f"Rows per sensor: {rows_per_sensor}")
print(f"SEQ_LEN={SEQ_LEN} readings history  HORIZON={HORIZON} readings ahead")
print(f"Sequences per sensor: {seqs_per_sensor}  x  {n_sensors} sensors = {len(X)} total")
print(f"X shape: {X.shape}   y shape: {y.shape}")
print()

# Time-ordered split per sensor: first 80% -> train, last 20% -> test
train_mask = np.zeros(len(X), dtype=bool)
split_point = int(seqs_per_sensor * 0.8)
for sid in scaled_df["sensor_id"].unique():
    idxs = np.where(sensor_ids_arr == sid)[0]
    train_mask[idxs[:split_point]] = True

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Train sequences: {X_train_t.shape}   Test sequences: {X_test_t.shape}")
print()

# ═════════════════════════════════════════════════════════════
# STEP 4 — DEFINE & TRAIN THE LSTM
# ═════════════════════════════════════════════════════════════

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = TemperatureLSTM(input_size=len(FEATURES), hidden_size=32, num_layers=2)

print("=" * 60)
print("STEP 4 — PyTorch LSTM Architecture")
print("=" * 60)
print(model)
print()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 100
best_test_loss = float("inf")
best_state = None
best_epoch = 0

print(f"Training for up to {EPOCHS} epochs (keeping best model on test loss)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test_t), y_test_t).item()

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train Loss: {loss.item():.5f} | Test Loss: {test_loss:.5f}")

model.load_state_dict(best_state)
print(f"Training complete! Best model from epoch {best_epoch} (test loss {best_test_loss:.5f})\n")

# ═════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE ON TEST SET (inverse-transformed to °C)
# ═════════════════════════════════════════════════════════════

def inverse_temp(scaled_vals, scaler, target_idx, n_features):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, target_idx] = scaled_vals
    return scaler.inverse_transform(dummy)[:, target_idx]

model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test_t).numpy().flatten()
    test_actual_scaled = y_test_t.numpy().flatten()

test_pred_temps = inverse_temp(test_pred_scaled, scaler, target_idx, len(FEATURES))
test_actual_temps = inverse_temp(test_actual_scaled, scaler, target_idx, len(FEATURES))
mae = float(np.mean(np.abs(test_pred_temps - test_actual_temps)))

print("=" * 60)
print("STEP 5 — Test Set Accuracy")
print("=" * 60)
print(f"Mean Absolute Error ({HORIZON}-step-ahead forecast): {mae:.3f} °C")
print()
print("Sample test predictions:")
for p, a in list(zip(test_pred_temps, test_actual_temps))[:6]:
    print(f"   Predicted: {p:6.2f}°C   |   Actual: {a:6.2f}°C   |   Diff: {abs(p-a):.2f}°C")
print()

# ═════════════════════════════════════════════════════════════
# STEP 6 — FORECAST NEXT TEMPERATURE PER LOCATION (export)
#
# For each of the 20 sensors, take its LAST SEQ_LEN window and
# predict the temperature HORIZON steps ahead -> one forecast
# per real location (matches the columns heat_map.py,
# ndvi_analysis.py, and scenario_simulator.py expect).
# ═════════════════════════════════════════════════════════════

results = []
for sid in scaled_df["sensor_id"].unique():
    s = scaled_df[scaled_df["sensor_id"] == sid].sort_values("timestamp", kind="stable")
    last_window = s[FEATURES].to_numpy()[-SEQ_LEN:]
    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(x).item()
    pred_temp = inverse_temp(np.array([pred_scaled]), scaler, target_idx, len(FEATURES))[0]

    row = df[df["sensor_id"] == sid].iloc[-1]
    results.append({
        "name": row["location"],
        "sensor_id": int(sid),
        "lat": row["latitude"],
        "lon": row["longitude"],
        "pred_temp": round(float(pred_temp), 1),
        "ndvi": round(float(row["ndvi"]), 3),
        "priority_score": round(float(row["priority_score"]), 2),
    })

forecast_df = pd.DataFrame(results).sort_values("pred_temp", ascending=False).reset_index(drop=True)

print("=" * 60)
print(f"STEP 6 — Per-Location Forecasts ({HORIZON} steps ahead)")
print("=" * 60)
print(forecast_df.to_string(index=False))
print()

forecast_df.to_csv("lstm_zone_forecasts.csv", index=False)
print("EXPORTED: lstm_zone_forecasts.csv")
print("  -> Ready for heat_map.py, ndvi_analysis.py, scenario_simulator.py")
print("=" * 60)
