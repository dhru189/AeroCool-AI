# AeroCool AI — Urban Heat Mitigation Optimizer

**Team BharatX** · Bharatiya Antariksh Hackathon 2026 · ISRO · Problem Statement 1

> Optimizing Urban Heat Mitigation and cooling strategies via AI/ML — geospatial, physics-informed, scenario-based.

---

## Architecture

```
aerocool-ai/
├── backend/                  ← Python AI engine 
│   ├── data_pipeline/        merge + LSTM forecasting pipeline
│   ├── models/               XGBoost + custom PINN training
│   ├── hvi_engine/           Heat Vulnerability Index scorer
│   ├── scenario_simulator/   cooling scenarios + heat map + NDVI
│   ├── api/                  FastAPI REST endpoints
│   │   └── main.py           serves all AI outputs to frontend
│   ├── outputs/              exported CSVs + charts
│   └── demo_dashboard.py     internal Streamlit demo
│
├── frontend/                 ← React dashboard
│   ├── src/
│   │   ├── App.jsx           main app + navigation
│   │   └── components/
│   │       ├── HeatMap.jsx           zone heat map
│   │       ├── ModelStats.jsx        AI model comparison + SHAP
│   │       ├── HVIRanking.jsx        Heat Vulnerability Index
│   │       ├── ScenarioSimulator.jsx cooling sliders (calls API live)
│   │       └── OptimizationPlan.jsx  budget optimizer results
│   ├── index.html
│   └── package.json
│
└── docs/                     architecture diagrams + notes
```

---

## What the system does

1. **Heat Stress Mapping** — identifies urban heat hotspots from sensor + satellite data
2. **Driver Attribution** — SHAP values show *why* each zone is hot (NDVI, solar, building density)
3. **Physics-Informed AI** — XGBoost (MAE 0.053°C, R² 0.9998) + custom PINN with NDVI/solar constraints
4. **Heat Vulnerability Index** — combines temperature with population, elderly %, schools, hospitals
5. **Cooling Scenario Simulation** — 4 intervention scenarios with literature-backed physics formulas
6. **Budget Optimization** — knapsack algorithm: maximize cooling within ₹50 Cr constraint
7. **Before/After Maps** — visual proof of expected temperature reduction per zone

---

## Running locally

### Backend (Python / FastAPI)

```bash
cd backend
pip install -r requirements.txt

# Run the AI pipeline first to generate outputs
python data_pipeline/merge_pipeline.py
python models/train_models.py
python hvi_engine/hvi_scorer.py

# Start the API server
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend (React / Vite)

```bash
cd frontend
npm install
npm run dev
```

Opens at: http://localhost:3000

### Internal demo only (Streamlit)

```bash
cd backend
streamlit run demo_dashboard.py
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/zones` | All 20 zone scores + predicted temperatures |
| GET | `/api/hvi` | Heat Vulnerability Index per zone |
| GET | `/api/scenarios` | 4 cooling scenario results |
| GET | `/api/optimal-plan` | Budget-optimized intervention plan |
| GET | `/api/models` | Model comparison (RF / XGBoost / PINN) |
| GET | `/api/before-after` | Before/After ΔT per zone |
| GET | `/api/shap` | SHAP driver attribution per zone |
| POST | `/api/simulate` | Custom scenario with slider values |

---

## Team

| Member | Role |
|---|---|
| Member 1 |Varsha - Satellite & API Integration (ISRO Bhuvan / NASA|
| Member 2 |Shikhar - 3D Visualization, VFX Simulation & Documentation |
| **Member 3 (me)** |Dhruv - **Data Engineering & AI Engine (this repo's backend)** ,Sensor Integration (Ground Data)|

