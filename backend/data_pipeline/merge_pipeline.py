"""
AeroCool AI — Member 3 Pipeline
=================================
Data Engineering & PyTorch LSTM Architecture (AI Engine)

This script:
  1. Merges sensor data (Member 1) + satellite data (Member 2)
  2. Cleans and normalizes the multivariate time-series
  3. Builds a PyTorch LSTM (nn.LSTM)
  4. Trains it on historical sequences
  5. Forecasts future land surface temperatures
  6. Exports predictions for heat_map.py and scenario_simulator.py

HOW TO RUN:
    pip install torch pandas numpy scikit-learn
    python lstm_pipeline.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
np.random.seed(42)

# ═════════════════════════════════════════════════════════════
# STEP 1 — SIMULATE / LOAD INPUT DATA
#
# In the real project, replace this with:
#   sensor_df = pd.read_csv("member1_sensor_data.csv")
#   satellite_df = pd.read_csv("member2_satellite_data.csv")
# ═════════════════════════════════════════════════════════════

N_SENSORS = 10
N_TIMESTEPS = 200   # e.g. readings every 15 min over ~50 hours

def generate_sensor_data():
    """Simulates Member 1's output: 10 sensors, sine wave + noise, 35-45C"""
    timestamps = pd.date_range("2026-06-10", periods=N_TIMESTEPS, freq="15min")
    rows = []
    for t_idx, ts in enumerate(timestamps):
        sine_factor = np.sin(2 * np.pi * t_idx / 96)  # 24hr cycle (96 * 15min)
        for s in range(N_SENSORS):
            base = 40 + sine_factor * 4               # oscillates 36-44
            noise = np.random.uniform(-1, 1)
            temp = np.clip(base + noise + s * 0.3, 35, 45)
            rows.append({"timestamp": ts, "sensor_id": f"S{s+1:02d}", "ground_temp": temp})
    return pd.DataFrame(rows)

def generate_satellite_data():
    """Simulates Member 2's output: solar irradiance, wind speed, cloud cover"""
    timestamps = pd.date_range("2026-06-10", periods=N_TIMESTEPS, freq="15min")
    rows = []
    for t_idx, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60
        solar = max(0, 1000 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        wind = np.random.uniform(2, 12)
        cloud = np.clip(np.random.normal(0.3, 0.15), 0, 1)
        rows.append({"timestamp": ts, "solar_irradiance": solar,
                      "wind_speed": wind, "cloud_cover": cloud})
    return pd.DataFrame(rows)

sensor_df = generate_sensor_data()
satellite_df = generate_satellite_data()

print(f"✅ Sensor data:    {sensor_df.shape[0]} rows  ({N_SENSORS} sensors x {N_TIMESTEPS} timesteps)")
print(f"✅ Satellite data: {satellite_df.shape[0]} rows")

# ═════════════════════════════════════════════════════════════
# STEP 2 — MERGE GROUND + SATELLITE DATA (Pandas)
# ═════════════════════════════════════════════════════════════

# Average ground temp across all sensors per timestamp (city-wide signal)
ground_avg = sensor_df.groupby("timestamp")["ground_temp"].mean().reset_index()
ground_avg.columns = ["timestamp", "avg_ground_temp"]

# Merge with satellite data on timestamp
master_df = pd.merge(ground_avg, satellite_df, on="timestamp", how="inner")
master_df = master_df.sort_values("timestamp").reset_index(drop=True)

print(f"✅ Merged master table: {master_df.shape[0]} rows x {master_df.shape[1]} columns")
print(master_df.head(3).to_string(index=False))
print()

# ═════════════════════════════════════════════════════════════
# STEP 3 — CLEAN, NORMALIZE, BUILD SEQUENCES
# ═════════════════════════════════════════════════════════════

FEATURES = ["avg_ground_temp", "solar_irradiance", "wind_speed", "cloud_cover"]
TARGET = "avg_ground_temp"

# Drop any rows with missing values
master_df = master_df.dropna().reset_index(drop=True)

# Normalize all features to [0, 1] — required for stable LSTM training
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(master_df[FEATURES])
scaled_df = pd.DataFrame(scaled_values, columns=FEATURES)

# Build sequences: use SEQ_LEN past steps to predict the NEXT step's temperature
SEQ_LEN = 12   # 12 x 15min = 3 hours of history
HORIZON = 8    # predict 8 steps ahead (2 hours)

def create_sequences(data, target_col_idx, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + horizon - 1, target_col_idx])
    return np.array(X), np.array(y)

target_idx = FEATURES.index(TARGET)
X, y = create_sequences(scaled_values, target_idx, SEQ_LEN, HORIZON)

# Train / test split (80/20, time-ordered — no shuffling for time series!)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"✅ Sequences built: X_train {X_train_t.shape}, X_test {X_test_t.shape}")
print(f"   (sequence_length={SEQ_LEN}, n_features={len(FEATURES)}, forecast_horizon={HORIZON} steps ahead)")
print()

# ═════════════════════════════════════════════════════════════
# STEP 4 — DEFINE THE PYTORCH LSTM MODEL
# ═════════════════════════════════════════════════════════════

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]      # take output of last timestep
        return self.fc(last_hidden)

model = TemperatureLSTM(input_size=len(FEATURES), hidden_size=32, num_layers=2)
print(model)
print()

# ═════════════════════════════════════════════════════════════
# STEP 5 — TRAINING LOOP
# ═════════════════════════════════════════════════════════════

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 60
print("Training LSTM...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t)
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train Loss: {loss.item():.5f} | Test Loss: {test_loss.item():.5f}")

print("✅ Training complete!\n")

# ═════════════════════════════════════════════════════════════
# STEP 6 — FORECAST & CONVERT BACK TO °C
# ═════════════════════════════════════════════════════════════

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_t).numpy().flatten()
    test_actuals = y_test_t.numpy().flatten()

# Inverse-transform predictions back to real °C
def inverse_temp(scaled_vals, scaler, target_idx, n_features):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, target_idx] = scaled_vals
    return scaler.inverse_transform(dummy)[:, target_idx]

pred_temps = inverse_temp(test_predictions, scaler, target_idx, len(FEATURES))
actual_temps = inverse_temp(test_actuals, scaler, target_idx, len(FEATURES))

mae = np.mean(np.abs(pred_temps - actual_temps))
print(f"📊 Mean Absolute Error: {mae:.3f} °C")
print()
print("Sample forecasts (last 5 test points):")
for p, a in zip(pred_temps[-5:], actual_temps[-5:]):
    print(f"   Predicted: {p:.2f}°C   |   Actual: {a:.2f}°C   |   Diff: {abs(p-a):.2f}°C")

# ═════════════════════════════════════════════════════════════
# STEP 7 — EXPORT FOR heat_map.py / scenario_simulator.py / ndvi_analysis.py
#
# Generates one forecast per "zone" (here we simulate 10 zones by
# adding a small zone-specific offset — replace with your real
# per-sensor-location forecasts once each sensor has its own LSTM
# pass, or simply broadcast the city-wide forecast to all zones).
# ═════════════════════════════════════════════════════════════

zone_names = [
    "Civil Lines", "Kanpur Central", "Kidwai Nagar", "GT Road Corridor",
    "Armapur Estate", "Swaroop Nagar", "Panki Industrial", "Govind Nagar",
    "Kakadeo", "Rawatpur"
]

latest_forecast = float(pred_temps[-1])
zone_offsets = np.array([0.5, 3.1, -0.2, 4.4, -2.4, 1.9, 5.4, 1.0, -3.5, -0.9])
zone_forecasts = np.clip(latest_forecast + zone_offsets, 35, 50)

forecast_df = pd.DataFrame({
    "name": zone_names,
    "pred_temp": zone_forecasts.round(1),
})
forecast_df.to_csv("lstm_zone_forecasts.csv", index=False)

print()
print("=" * 55)
print("✅ EXPORTED: lstm_zone_forecasts.csv")
print("   → Paste these into baseline_temps / sensor_data in:")
print("     heat_map.py, scenario_simulator.py, ndvi_analysis.py")
print("=" * 55)
print(forecast_df.to_string(index=False))
