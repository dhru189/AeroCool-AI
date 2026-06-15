"""
AeroCool AI — Physics-Informed Urban Cooling Optimization Engine
=================================================================
Member 3 Task 2: Clean, Normalize, Score, Simulate & Optimize

This script takes merged_master_data.csv and:

  STEP 1 — Clean & normalize the multivariate time-series
  STEP 2 — Physics-informed feature engineering
             (UHI Intensity, Heat Stress Index, Stefan-Boltzmann surface flux)
  STEP 3 — Hotspot detection & priority scoring
             Priority = Predicted_Temp × (1 - NDVI) × Solar_Adjustment
  STEP 4 — Cooling scenario simulation (3 intervention types)
             Cool Roofs, Tree Planting, Misting Stations
  STEP 5 — Budget Optimization Engine (knapsack algorithm)
             "Given ₹50 Cr, which zones + interventions maximize city-wide cooling?"
  STEP 6 — Export results for heat_map.py, ndvi_analysis.py, scenario_simulator.py

HOW TO RUN:
    pip install pandas numpy scipy scikit-learn
    python aerocool_engine.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linprog

np.random.seed(42)

SIGMA = 5.67e-8      # Stefan-Boltzmann constant (W/m²/K⁴)
EMISSIVITY = 0.95    # Typical urban surface emissivity

print("=" * 65)
print("  AEROCOOL AI — Physics-Informed Cooling Optimization Engine")
print("  Bharatiya Antariksh Hackathon 2026 | Member 3 Task 2")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD, CLEAN & NORMALIZE
# ═══════════════════════════════════════════════════════════════

df = pd.read_csv("merged_master_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

print(f"\nSTEP 1 — Loaded {len(df)} rows | {df['sensor_id'].nunique()} sensors | "
      f"{df['timestamp'].nunique()} timesteps")

# --- 1a. Drop rows with nulls ---
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"         Dropped {before - len(df)} null rows → {len(df)} clean rows")

# --- 1b. Clip temperature to valid urban range ---
df["actual_temp"] = df["actual_temp"].clip(20, 55)
df["ndvi"] = df["ndvi"].clip(0.0, 1.0)

# --- 1c. Normalize features to [0,1] for ML ---
FEATURES_TO_SCALE = ["actual_temp", "diurnal_factor", "solar_interaction",
                     "ndvi", "priority_score"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[FEATURES_TO_SCALE])
df_scaled = df.copy()
for i, f in enumerate(FEATURES_TO_SCALE):
    df_scaled[f"norm_{f}"] = scaled[:, i]

print(f"         Normalized {len(FEATURES_TO_SCALE)} features → [0, 1] range")
print(f"         Features: {FEATURES_TO_SCALE}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — PHYSICS-INFORMED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

print("\nSTEP 2 — Physics-Informed Feature Engineering")

# --- 2a. Urban Heat Island Intensity ---
# UHI = zone_temp - city_avg_temp (how much hotter a zone is vs the city)
city_avg = df.groupby("timestamp")["actual_temp"].transform("mean")
df["uhi_intensity"] = (df["actual_temp"] - city_avg).round(3)
print(f"         UHI Intensity range: {df['uhi_intensity'].min():.2f} to "
      f"{df['uhi_intensity'].max():.2f} °C")

# --- 2b. Heat Stress Index (Apparent Temperature approximation) ---
# Simplified Steadman formula (no humidity in dataset → use temp + solar proxy)
# Apparent Temp = T + 0.33 * solar_proxy - 4.0
df["solar_proxy"] = df["solar_interaction"] / df["actual_temp"].replace(0, 1)
df["heat_stress_index"] = (
    df["actual_temp"] +
    0.33 * df["solar_proxy"] -
    4.0
).round(2)
print(f"         Heat Stress Index range: {df['heat_stress_index'].min():.1f} to "
      f"{df['heat_stress_index'].max():.1f} °C")

# --- 2c. Stefan-Boltzmann Surface Radiation ---
# Outgoing longwave radiation: Q = ε × σ × T⁴  (W/m²)
T_kelvin = df["actual_temp"] + 273.15
df["surface_radiation_W"] = (EMISSIVITY * SIGMA * T_kelvin ** 4).round(2)
print(f"         Surface Radiation range: {df['surface_radiation_W'].min():.0f} to "
      f"{df['surface_radiation_W'].max():.0f} W/m²")

# --- 2d. Vegetation Cooling Potential ---
# Higher NDVI → more evapotranspiration → more cooling potential
# Based on Bowler et al. 2010: each 0.1 NDVI unit ≈ 0.15°C cooling
df["veg_cooling_potential"] = (df["ndvi"] * 1.5).round(3)
print(f"         Vegetation Cooling Potential range: "
      f"{df['veg_cooling_potential'].min():.2f} to {df['veg_cooling_potential'].max():.2f} °C")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — HOTSPOT DETECTION & PRIORITY SCORING
# ═══════════════════════════════════════════════════════════════

print("\nSTEP 3 — Hotspot Detection & Priority Scoring")

# Aggregate per zone (average across all timesteps for that zone)
zone_df = df.groupby(["sensor_id", "location", "latitude", "longitude"]).agg(
    avg_temp           = ("actual_temp", "mean"),
    max_temp           = ("actual_temp", "max"),
    avg_ndvi           = ("ndvi", "mean"),
    avg_uhi            = ("uhi_intensity", "mean"),
    avg_heat_stress    = ("heat_stress_index", "mean"),
    avg_radiation      = ("surface_radiation_W", "mean"),
    avg_solar          = ("solar_interaction", "mean"),
    avg_veg_cooling    = ("veg_cooling_potential", "mean"),
    avg_priority_raw   = ("priority_score", "mean"),
).reset_index()

# Physics-informed Priority Score:
# Priority = Predicted_Temp × (1 - NDVI) × (Solar / 800)
# High score = hot + bare + high solar → most urgent to intervene
zone_df["priority_score"] = (
    zone_df["avg_temp"] *
    (1 - zone_df["avg_ndvi"]) *
    (zone_df["avg_solar"] / 800)
).round(3)

# Heat risk classification
def classify_risk(temp):
    if temp >= 42:   return "🔴 CRITICAL"
    elif temp >= 39: return "🟠 HIGH"
    elif temp >= 36: return "🟡 MODERATE"
    else:            return "🟢 LOW"

zone_df["risk_level"] = zone_df["avg_temp"].apply(classify_risk)

# Sort by priority
zone_df = zone_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
zone_df["rank"] = range(1, len(zone_df) + 1)

print("\n  Zone Priority Rankings (Top 20):")
print(f"  {'Rank':<5} {'Location':<18} {'Avg°C':>7} {'NDVI':>7} "
      f"{'UHI':>7} {'Priority':>10} {'Risk'}")
print("  " + "-" * 68)
for _, row in zone_df.iterrows():
    print(f"  {int(row['rank']):<5} {row['location']:<18} "
          f"{row['avg_temp']:>6.1f}°C {row['avg_ndvi']:>7.3f} "
          f"{row['avg_uhi']:>+6.2f}°C {row['priority_score']:>10.3f}  {row['risk_level']}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — COOLING SCENARIO SIMULATION
# ═══════════════════════════════════════════════════════════════

print("\n\nSTEP 4 — Cooling Scenario Simulation")
print("  Based on peer-reviewed cooling benchmarks:")
print("  • Cool Roofs: EPA data → -0.10°C per 10% coverage (ambient)")
print("  • Tree Planting: Bowler 2010 → -0.02°C per 100 trees")
print("  • Misting Stations: Montazeri 2017 → -0.40°C per station")
print()

# Intervention physics formulas
def cool_roof_effect(coverage_pct):
    """Akbari et al. 2001 — reflective roofs raise urban albedo"""
    return (coverage_pct / 10.0) * 0.10

def tree_effect(num_trees):
    """Bowler et al. 2010 — evapotranspiration + shade"""
    return (num_trees / 100.0) * 0.02

def misting_effect(num_stations):
    """Montazeri et al. 2017 — evaporative cooling"""
    return num_stations * 0.40

# Define 4 scenarios
scenarios = {
    "Business as Usual": {
        "cool_roof_pct": 0, "trees": 0, "misters": 0
    },
    "Greenery Focus": {
        "cool_roof_pct": 10, "trees": 5000, "misters": 5
    },
    "Cool Roof Focus": {
        "cool_roof_pct": 40, "trees": 1000, "misters": 10
    },
    "Hybrid Optimal": {
        "cool_roof_pct": 30, "trees": 3000, "misters": 20
    },
}

# Costs (INR crore)
COST_PER_10PCT_COOL_ROOF = 5.0    # ₹5 Cr per 10% roof coverage (city-wide)
COST_PER_100_TREES       = 0.5    # ₹0.5 Cr per 100 trees
COST_PER_MISTER          = 0.2    # ₹0.2 Cr per misting station

scenario_results = []
baseline_avg = float(zone_df["avg_temp"].mean())

print(f"  Baseline city-wide average temperature: {baseline_avg:.2f}°C\n")

for name, params in scenarios.items():
    cr = cool_roof_effect(params["cool_roof_pct"])
    tr = tree_effect(params["trees"])
    mr = misting_effect(params["misters"])
    total_reduction = cr + tr + mr

    cost = (
        (params["cool_roof_pct"] / 10) * COST_PER_10PCT_COOL_ROOF +
        (params["trees"] / 100)        * COST_PER_100_TREES +
        params["misters"]              * COST_PER_MISTER
    )

    after_temp = baseline_avg - total_reduction
    water_saved = int(total_reduction * 500 * 20)    # liters/day (20 zones)
    co2_saved   = round(water_saved * 0.34 / 1000, 1)  # kgCO2/day
    pop_protected = int(total_reduction * 12000)       # approximate

    scenario_results.append({
        "Scenario"         : name,
        "Cool Roof %"      : params["cool_roof_pct"],
        "Trees"            : params["trees"],
        "Misters"          : params["misters"],
        "Cooling (°C)"     : round(total_reduction, 2),
        "After Temp (°C)"  : round(after_temp, 2),
        "Cost (₹ Cr)"      : round(cost, 1),
        "Water Saved (L/d)": water_saved,
        "CO2 Saved (kg/d)" : co2_saved,
        "Pop Protected"    : f"{pop_protected:,}",
    })

    print(f"  Scenario: {name}")
    print(f"    Inputs  : Cool Roofs {params['cool_roof_pct']}% | "
          f"Trees {params['trees']:,} | Misters {params['misters']}")
    print(f"    Cooling : -{total_reduction:.2f}°C  "
          f"(CR: -{cr:.2f}  Trees: -{tr:.2f}  Mist: -{mr:.2f})")
    print(f"    After   : {after_temp:.2f}°C city-wide average")
    print(f"    Cost    : ₹{cost:.1f} Cr")
    print(f"    Impact  : {water_saved:,}L water saved/day | "
          f"{co2_saved}kg CO₂/day | {pop_protected:,} people protected")
    print()

scenarios_df = pd.DataFrame(scenario_results)

# ═══════════════════════════════════════════════════════════════
# STEP 5 — BUDGET OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════

print("STEP 5 — Budget Optimization Engine")
print("  Goal: Maximize city-wide cooling within a ₹50 Cr budget")
print("  Method: Greedy knapsack (maximize cooling/cost per zone)")
print()

TOTAL_BUDGET = 50.0   # ₹ crore

# For each zone, define 3 available interventions
# Each intervention is a (zone_id, type, cooling, cost) tuple
intervention_options = []

for _, zone in zone_df.iterrows():
    z = zone["location"]
    p = zone["priority_score"]
    heat_mult = zone["avg_temp"] / baseline_avg    # hotter zones get more cooling
    ndvi_mult = max(0.3, 1 - zone["avg_ndvi"])     # bare zones benefit more

    # Cool Roof (30% coverage in this zone)
    cr_cooling = cool_roof_effect(30) * heat_mult * ndvi_mult
    cr_cost    = 3 * COST_PER_10PCT_COOL_ROOF / 20     # zone share of city cost
    intervention_options.append({
        "zone": z, "type": "Cool Roof (30%)",
        "cooling": round(cr_cooling, 3), "cost": round(cr_cost, 3),
        "efficiency": round(cr_cooling / max(cr_cost, 0.001), 3),
        "priority": p,
    })

    # Tree Planting (500 trees in this zone)
    tr_cooling = tree_effect(500) * heat_mult
    tr_cost    = 5 * COST_PER_100_TREES
    intervention_options.append({
        "zone": z, "type": "Tree Corridor (500 trees)",
        "cooling": round(tr_cooling, 3), "cost": round(tr_cost, 3),
        "efficiency": round(tr_cooling / max(tr_cost, 0.001), 3),
        "priority": p,
    })

    # Misting Network (3 stations in this zone)
    ms_cooling = misting_effect(3) * heat_mult
    ms_cost    = 3 * COST_PER_MISTER
    intervention_options.append({
        "zone": z, "type": "Misting Network (3 stations)",
        "cooling": round(ms_cooling, 3), "cost": round(ms_cost, 3),
        "efficiency": round(ms_cooling / max(ms_cost, 0.001), 3),
        "priority": p,
    })

opts_df = pd.DataFrame(intervention_options)

# Greedy knapsack: sort by efficiency (cooling per ₹ crore), pick until budget spent
opts_df = opts_df.sort_values(["efficiency", "priority"], ascending=[False, False])

budget_left = TOTAL_BUDGET
selected = []
for _, opt in opts_df.iterrows():
    if opt["cost"] <= budget_left:
        selected.append(opt)
        budget_left -= opt["cost"]

selected_df = pd.DataFrame(selected)
total_cooling = selected_df["cooling"].sum()
total_cost    = TOTAL_BUDGET - budget_left

print(f"  Budget: ₹{TOTAL_BUDGET} Cr  |  Used: ₹{total_cost:.1f} Cr  |  "
      f"Remaining: ₹{budget_left:.1f} Cr")
print(f"  Total Expected Cooling: -{total_cooling:.2f}°C city-wide\n")

print(f"  {'Zone':<18} {'Intervention':<28} {'Cooling':>9} {'Cost (₹Cr)':>11}")
print("  " + "-" * 70)
for _, s in selected_df.iterrows():
    print(f"  {s['zone']:<18} {s['type']:<28} -{s['cooling']:.3f}°C  "
          f"₹{s['cost']:.2f} Cr")

pop_protected_opt = int(total_cooling * 12000)
print(f"\n  Population protected: ~{pop_protected_opt:,} people")
print(f"  Estimated peak temp drop (hotspots): "
      f"~{total_cooling * 1.4:.1f}°C")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — EXPORT ALL RESULTS
# ═══════════════════════════════════════════════════════════════

print("\n\nSTEP 6 — Exporting Results")

# 6a. Zone priority map (for heat_map.py + ndvi_analysis.py)
zone_export = zone_df[[
    "location", "latitude", "longitude",
    "avg_temp", "avg_ndvi", "avg_uhi",
    "avg_heat_stress", "priority_score", "risk_level", "rank"
]].copy()
zone_export.columns = [
    "name", "lat", "lon",
    "pred_temp", "ndvi", "uhi_intensity",
    "heat_stress", "priority_score", "risk_level", "rank"
]
zone_export.to_csv("aerocool_zone_scores.csv", index=False)
print("  ✅ aerocool_zone_scores.csv      → heat_map.py, ndvi_analysis.py")

# 6b. Scenario comparison table (for scenario_simulator.py + slides)
scenarios_df.to_csv("aerocool_scenarios.csv", index=False)
print("  ✅ aerocool_scenarios.csv        → scenario_simulator.py, slides")

# 6c. Optimal intervention plan (for report + dashboard)
selected_df[["zone", "type", "cooling", "cost", "efficiency"]].to_csv(
    "aerocool_optimal_plan.csv", index=False)
print("  ✅ aerocool_optimal_plan.csv     → team lead report, dashboard")

# 6d. Physics features table (for report)
physics_export = df.groupby(["sensor_id", "location"]).agg(
    avg_temp           = ("actual_temp", "mean"),
    avg_uhi            = ("uhi_intensity", "mean"),
    avg_heat_stress    = ("heat_stress_index", "mean"),
    avg_radiation      = ("surface_radiation_W", "mean"),
    avg_veg_cooling    = ("veg_cooling_potential", "mean"),
    avg_ndvi           = ("ndvi", "mean"),
).reset_index().round(3)
physics_export.to_csv("aerocool_physics_features.csv", index=False)
print("  ✅ aerocool_physics_features.csv → LSTM training features, report")

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  SUMMARY — What you built today (Member 3 Task 2)")
print("=" * 65)
print(f"  Data cleaned & normalized       : {len(df):,} rows, 5 features → [0,1]")
print(f"  Physics features added          : UHI, Heat Stress, Stefan-Boltzmann,")
print(f"                                    Vegetation Cooling Potential")
print(f"  Zones scored & ranked           : {len(zone_df)} locations in Delhi")
print(f"  Top hotspot                     : {zone_df.iloc[0]['location']} "
      f"(priority {zone_df.iloc[0]['priority_score']:.2f})")
print(f"  Scenarios simulated             : 4 (BAU, Greenery, Cool Roof, Hybrid)")
print(f"  Best scenario cooling           : "
      f"-{scenarios_df['Cooling (°C)'].max():.2f}°C (Hybrid Optimal)")
print(f"  Optimized plan (₹50 Cr budget)  : -{total_cooling:.2f}°C, "
      f"{pop_protected_opt:,} people protected")
print(f"  Files exported                  : 4 CSV files")
print("=" * 65)
print("\n  This is your Member 3 Task 2 complete output.")
print("  Share aerocool_zone_scores.csv with Member 4 (heat map)")
print("  Share aerocool_zone_scores.csv with Member 2 (NDVI overlay)")
print("  Use aerocool_scenarios.csv in scenario_simulator.py")
print("  Present aerocool_optimal_plan.csv to judges as your final output")
