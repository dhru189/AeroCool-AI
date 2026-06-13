# AeroCool AI — System Architecture

## Data flow

```
Satellite Data          Ground Sensors        Weather APIs
(ECOSTRESS, Landsat,   (10 virtual sensors   (ERA5 reanalysis,
 Sentinel-2, Bhuvan)    35–45°C, Delhi)       CPCB stations)
        │                      │                     │
        └──────────────────────┴─────────────────────┘
                               │
                    data_pipeline/merge_pipeline.py
                    (Pandas merge on timestamp)
                               │
                    Physics Feature Engineering
                    ┌──────────────────────────────┐
                    │ UHI Intensity                │
                    │ Heat Stress Index (Steadman) │
                    │ Stefan-Boltzmann radiation   │
                    │ Vegetation Cooling Potential │
                    │ NDVI × Solar heat load       │
                    └──────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
               XGBoost              PINN (custom)
               MAE 0.053°C          Physics loss:
               R² 0.9998            NDVI↑ → LST↓
                    │                     │
                    └──────────┬──────────┘
                               │
                    SHAP Driver Attribution
                    (per-zone: solar 38.6%, diurnal 24.7% ...)
                               │
                    Heat Vulnerability Index
                    (Temp × 0.30 + UHI × 0.20 +
                     Pop × 0.15 + Elderly × 0.15 +
                     Schools × 0.10 + Hospitals × 0.05 +
                     LowNDVI × 0.05)
                               │
                    Scenario Simulation Engine
                    (Cool Roofs / Trees / Misting /
                     Green Roofs / Reflective Pavement)
                               │
                    Cooling ROI Engine
                    (ROI = ΔTemp / Cost)
                               │
                    Budget Optimization
                    (Knapsack: max cooling ≤ ₹50 Cr)
                               │
                    Before / After ΔT Export
                               │
                    ┌──────────┴──────────┐
                    │                     │
               FastAPI REST          Streamlit
               (api/main.py)         (demo only)
                    │
               React Frontend
               (5 dashboard pages)
```

## Backend modules

| File | Responsibility |
|---|---|
| `data_pipeline/merge_pipeline.py` | Merges sensor + satellite data, builds LSTM sequences |
| `data_pipeline/lstm_realdata.py` | LSTM adapted for real Delhi sensor data |
| `data_pipeline/physics_engine.py` | Physics features, hotspot scoring, optimization |
| `models/train_models.py` | XGBoost, PINN training, SHAP attribution |
| `hvi_engine/hvi_scorer.py` | Heat Vulnerability Index computation |
| `scenario_simulator/simulator.py` | Streamlit scenario simulator |
| `scenario_simulator/heat_map.py` | Folium interactive heat map |
| `scenario_simulator/ndvi_analysis.py` | NDVI vegetation gap analysis |
| `api/main.py` | FastAPI serving all outputs to React frontend |

## Frontend pages

| Component | Data source |
|---|---|
| `HeatMap.jsx` | `GET /api/zones` |
| `ModelStats.jsx` | `GET /api/models` + `GET /api/shap` |
| `HVIRanking.jsx` | `GET /api/hvi` |
| `ScenarioSimulator.jsx` | `POST /api/simulate` (live) |
| `OptimizationPlan.jsx` | `GET /api/optimal-plan` + `/api/scenarios` + `/api/before-after` |

## Physics formulas used

| Formula | Source | Effect |
|---|---|---|
| Cool Roofs: −0.10°C per 10% coverage | Akbari et al. 2001 | Urban albedo increase |
| Trees: −0.02°C per 100 trees | Bowler et al. 2010 | Evapotranspiration + shade |
| Misting: −0.40°C per station | Montazeri et al. 2017 | Evaporative cooling |
| Green Roofs: −0.15°C per 10% coverage | EPA 2008 | Combined ET + insulation |
| UHI = T_zone − T_city_avg | Standard definition | Relative heat intensity |
| Priority = T × (1−NDVI) × (Solar/800) | AeroCool formula | Combined hotspot score |
