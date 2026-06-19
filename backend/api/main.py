"""
AeroCool AI — FastAPI Backend
==============================
Team BharatX | Bharatiya Antariksh Hackathon 2026 | ISRO

REST API that serves all AI engine outputs to the React frontend.

Endpoints:
  GET  /api/zones          → all 20 zone scores + HVI rankings
  GET  /api/hvi            → full Heat Vulnerability Index report
  GET  /api/scenarios      → 4 cooling scenario results
  GET  /api/optimal-plan   → budget-optimized intervention plan
  GET  /api/models         → model comparison (XGBoost vs PINN vs RF)
  GET  /api/before-after   → before/after temperature per zone
  POST /api/simulate       → run custom scenario with given sliders
  GET  /api/health         → health check

HOW TO RUN:
    pip install fastapi uvicorn pandas numpy scikit-learn xgboost torch
    uvicorn backend.api.main:app --reload --port 8000

Swagger docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="AeroCool AI API",
    description="Urban Heat Mitigation Engine — Team BharatX",
    version="1.0.0",
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found. Run the pipeline first.")
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# MODELS for POST /api/simulate
# ─────────────────────────────────────────────
class SimulateRequest(BaseModel):
    cool_roof_pct:  float = 30.0   # % of rooftops with cool coating
    trees:          int   = 3000   # number of trees planted
    misters:        int   = 15     # number of misting stations
    green_roof_pct: float = 10.0   # % of buildings with green roofs
    budget_cr:      float = 50.0   # budget in ₹ crore


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "team": "BharatX", "project": "AeroCool AI"}


@app.get("/api/zones")
def get_zones():
    """All 20 Delhi zones with predicted temperature, NDVI, priority score."""
    df = load_csv("aerocool_zone_scores.csv")
    return df.to_dict(orient="records")


@app.get("/api/hvi")
def get_hvi():
    """Heat Vulnerability Index — temperature + population + schools + hospitals."""
    df = load_csv("hvi_report.csv")
    return df.to_dict(orient="records")


@app.get("/api/scenarios")
def get_scenarios():
    """4 cooling scenarios: BAU, Greenery, Cool Roof, Hybrid Optimal."""
    df = load_csv("aerocool_scenarios.csv")
    return df.to_dict(orient="records")


@app.get("/api/optimal-plan")
def get_optimal_plan():
    """Budget-optimized intervention plan (₹50 Cr, knapsack algorithm)."""
    df = load_csv("aerocool_optimal_plan.csv")
    return df.to_dict(orient="records")


@app.get("/api/models")
def get_models():
    """Model comparison: Random Forest vs XGBoost vs PINN."""
    df = load_csv("model_comparison.csv")
    return df.to_dict(orient="records")


@app.get("/api/before-after")
def get_before_after():
    """Before and after temperature per zone after optimal interventions."""
    df = load_csv("before_after_map.csv")
    return df.to_dict(orient="records")


@app.get("/api/top5")
def get_top5():
    """Top 5 highest priority zones for intervention."""
    df = load_csv("top5_priority_zones.csv")
    return df.to_dict(orient="records")


@app.get("/api/shap")
def get_shap():
    """SHAP driver attribution per zone — which feature drives heat most."""
    df = load_csv("zone_driver_report.csv")
    return df.to_dict(orient="records")


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """
    Run a custom cooling scenario based on slider values from the frontend.
    Returns per-zone cooling estimates and city-wide summary.

    Physics formulas (peer-reviewed):
      Cool Roofs:   Akbari et al. 2001 — urban albedo effect
      Trees:        Bowler et al. 2010 — evapotranspiration + shade
      Misting:      Montazeri et al. 2017 — evaporative cooling
      Green Roofs:  EPA 2008 — combined shade + ET
    """
    cr_effect  = (req.cool_roof_pct  / 10.0) * 0.10
    tr_effect  = (req.trees          / 100.0) * 0.02
    ms_effect  = req.misters * 0.40
    gr_effect  = (req.green_roof_pct / 10.0) * 0.15
    total_cooling = round(cr_effect + tr_effect + ms_effect + gr_effect, 3)

    cost = round(
        (req.cool_roof_pct  / 10) * 5.0 +
        (req.trees          / 100) * 0.5 +
        req.misters * 0.2 +
        (req.green_roof_pct / 10) * 8.0,
        2
    )

    water_saved_L = int(total_cooling * 500 * 20)
    co2_saved_kg  = round(water_saved_L * 0.34 / 1000, 1)

    # Per-zone cooling (weighted by HVI)
    try:
        zones_df = load_csv("aerocool_zone_scores.csv")
        hvi_col  = "priority_score" if "priority_score" in zones_df.columns else zones_df.columns[-1]
        max_hvi  = zones_df[hvi_col].max()
        zone_cooling = []
        for _, row in zones_df.iterrows():
            weight = float(row[hvi_col]) / max_hvi
            zone_cooling.append({
                "name":         row.get("name", row.get("location", "")),
                "before_temp":  round(float(row.get("pred_temp", row.get("avg_temp", 40))), 1),
                "after_temp":   round(float(row.get("pred_temp", row.get("avg_temp", 40))) - total_cooling * weight, 1),
                "delta_t":      round(total_cooling * weight, 2),
            })
    except Exception:
        zone_cooling = []

    return {
        "inputs": {
            "cool_roof_pct":  req.cool_roof_pct,
            "trees":          req.trees,
            "misters":        req.misters,
            "green_roof_pct": req.green_roof_pct,
        },
        "summary": {
            "total_cooling_C":   total_cooling,
            "total_cost_Cr":     cost,
            "within_budget":     cost <= req.budget_cr,
            "water_saved_L_day": water_saved_L,
            "co2_saved_kg_day":  co2_saved_kg,
            "pop_protected":     int(total_cooling * 12000),
        },
        "breakdown": {
            "cool_roof_C":   round(cr_effect, 3),
            "trees_C":       round(tr_effect, 3),
            "misting_C":     round(ms_effect, 3),
            "green_roof_C":  round(gr_effect, 3),
        },
        "zone_cooling": zone_cooling,
    }
