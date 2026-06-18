"""
AeroCool AI — NDVI Vegetation Gap Analysis
============================================
Pulls NDVI (vegetation index) data and overlays it with heat predictions
to identify PRIORITY ZONES for intervention.

Logic:
  HIGH temperature + LOW NDVI (bare concrete) = CRITICAL intervention zone
  HIGH temperature + HIGH NDVI (green cover)  = Monitor zone
  LOW temperature  + LOW NDVI                 = Upgrade zone

HOW TO RUN:
    pip install requests folium pandas numpy
    python ndvi_analysis.py
    → Saves ndvi_heat_overlay.html (open in browser)

DATA SOURCE OPTIONS (choose one):
    Option A: Simulated NDVI (works immediately, no API needed)   ← default
    Option B: NASA POWER API (free, no login needed)
    Option C: ISRO Bhuvan WMS (for final submission — see instructions)
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import json

# ─────────────────────────────────────────────
# STEP 1: Zone Data
# Replace with your actual coordinates and LSTM predictions
# ─────────────────────────────────────────────
zones = [
    {"name": "Civil Lines",       "lat": 26.4740, "lon": 80.3319, "pred_temp": 44.2},
    {"name": "Kanpur Central",    "lat": 26.4670, "lon": 80.3515, "pred_temp": 46.8},
    {"name": "Kidwai Nagar",      "lat": 26.4558, "lon": 80.3411, "pred_temp": 43.5},
    {"name": "GT Road Corridor",  "lat": 26.4800, "lon": 80.3600, "pred_temp": 47.1},
    {"name": "Armapur Estate",    "lat": 26.4900, "lon": 80.2950, "pred_temp": 41.3},
    {"name": "Swaroop Nagar",     "lat": 26.4350, "lon": 80.3200, "pred_temp": 45.6},
    {"name": "Panki Industrial",  "lat": 26.4200, "lon": 80.2800, "pred_temp": 48.9},
    {"name": "Govind Nagar",      "lat": 26.4450, "lon": 80.3750, "pred_temp": 44.7},
    {"name": "Kakadeo",           "lat": 26.4650, "lon": 80.2700, "pred_temp": 40.2},
    {"name": "Rawatpur",          "lat": 26.5050, "lon": 80.3400, "pred_temp": 42.8},
]

# ─────────────────────────────────────────────
# STEP 2: NDVI Values
#
# NDVI scale: −1.0 to +1.0
#   < 0.1  = Bare land / concrete / buildings
#   0.1–0.3 = Sparse vegetation
#   0.3–0.6 = Moderate green cover
#   > 0.6   = Dense vegetation / parks / forests
#
# HOW TO GET REAL NDVI:
#   Option A (easiest, no login):
#     Go to https://bhuvan.nrsc.gov.in/bhuvan_links.php
#     → Geoportal → LISS-III imagery → Download NDVI product for your area
#     Then read the raster and sample at your GPS coordinates.
#
#   Option B (API, slightly complex):
#     Use Copernicus Open Access Hub (free):
#     https://scihub.copernicus.eu/dhus/
#     Download Sentinel-2 L2A tile, compute NDVI = (B8 - B4) / (B8 + B4)
#
#   For NOW (hackathon submission): use realistic simulated values below
#   and note in your documentation it will be replaced with Bhuvan data.
# ─────────────────────────────────────────────

# Simulated NDVI — realistic for an Indian industrial/urban city
# Higher = more vegetation, Lower = more concrete/bare land
ndvi_values = {
    "Civil Lines":      0.38,   # Moderate — some trees on roads
    "Kanpur Central":   0.09,   # Very low — dense market, concrete
    "Kidwai Nagar":     0.25,   # Sparse — residential colony
    "GT Road Corridor": 0.06,   # Critically low — highway, no green
    "Armapur Estate":   0.42,   # Better — residential, garden plots
    "Swaroop Nagar":    0.18,   # Low — mixed use
    "Panki Industrial": 0.04,   # Almost zero — factory zone
    "Govind Nagar":     0.22,   # Low-moderate
    "Kakadeo":          0.45,   # Good — parks, residential green
    "Rawatpur":         0.31,   # Moderate
}

# ─────────────────────────────────────────────
# STEP 3: Compute Priority Score
# Formula: Priority = (Normalized Temp) × (1 - NDVI)
# High score = Hot AND bare = needs intervention most
# ─────────────────────────────────────────────
temps = np.array([z["pred_temp"] for z in zones])
temp_min, temp_max = temps.min(), temps.max()
temp_norm = (temps - temp_min) / (temp_max - temp_min)

records = []
for i, z in enumerate(zones):
    ndvi = ndvi_values[z["name"]]
    priority_score = round(temp_norm[i] * (1 - ndvi), 3)

    # Classify intervention priority
    if priority_score >= 0.7:
        priority = "🔴 CRITICAL — Intervene Now"
        priority_color = "#A32D2D"
    elif priority_score >= 0.45:
        priority = "🟠 HIGH — Plan Intervention"
        priority_color = "#D85A30"
    elif priority_score >= 0.25:
        priority = "🟡 MODERATE — Monitor"
        priority_color = "#BA7517"
    else:
        priority = "🟢 LOW — Maintain"
        priority_color = "#3B6D11"

    # Recommend action based on NDVI + temperature
    if ndvi < 0.1 and z["pred_temp"] >= 44:
        recommendation = "Cool roof paint + tree planting (urgent)"
    elif ndvi < 0.2 and z["pred_temp"] >= 44:
        recommendation = "Misting stations + urban greening"
    elif ndvi < 0.3:
        recommendation = "Tree planting corridor"
    else:
        recommendation = "Vegetation maintenance"

    records.append({
        **z,
        "ndvi": ndvi,
        "priority_score": priority_score,
        "priority": priority,
        "priority_color": priority_color,
        "recommendation": recommendation,
    })

df = pd.DataFrame(records).sort_values("priority_score", ascending=False)

# ─────────────────────────────────────────────
# STEP 4: Print Analysis to Console
# ─────────────────────────────────────────────
print("=" * 65)
print("  AEROCOOL AI — NDVI × HEAT STRESS ANALYSIS")
print("  City: Kanpur, Uttar Pradesh")
print("=" * 65)
print(f"{'Zone':<22} {'Temp':>7} {'NDVI':>7} {'Priority':>9}  Action")
print("-" * 65)
for _, row in df.iterrows():
    emoji = row["priority"].split()[0]
    print(f"{row['name']:<22} {row['pred_temp']:>6}°C {row['ndvi']:>7.2f} {row['priority_score']:>9.3f}  {emoji} {row['recommendation']}")
print()
print("TOP 3 CRITICAL ZONES (intervene first):")
for _, row in df.head(3).iterrows():
    print(f"  ⚡ {row['name']}: {row['pred_temp']}°C | NDVI {row['ndvi']} | {row['recommendation']}")
print()

# ─────────────────────────────────────────────
# STEP 5: Build the Folium Overlay Map
# ─────────────────────────────────────────────
center_lat = np.mean([z["lat"] for z in zones])
center_lon = np.mean([z["lon"] for z in zones])

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="CartoDB positron"
)
folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)

# ── Layer 1: NDVI Heatmap (green = vegetation) ──
ndvi_heat_data = [[r["lat"], r["lon"], r["ndvi"]] for _, r in df.iterrows()]
HeatMap(
    ndvi_heat_data,
    name="NDVI Vegetation Layer",
    radius=35,
    blur=25,
    gradient={0.0: "brown", 0.3: "yellow", 0.6: "lightgreen", 1.0: "darkgreen"},
    min_opacity=0.3,
).add_to(m)

# ── Layer 2: Priority Zone Markers ──
priority_group = folium.FeatureGroup(name="Priority Zones", show=True)

for _, row in df.iterrows():
    # Marker size = priority score
    radius = max(int(row["priority_score"] * 28), 8)

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=row["priority_color"],
        fill=True,
        fill_color=row["priority_color"],
        fill_opacity=0.75,
        weight=2.5,
        popup=folium.Popup(
            f"""
            <div style='font-family: sans-serif; min-width: 220px;'>
                <b style='font-size:14px; color:{row["priority_color"]}'>{row['name']}</b><br>
                <hr style='margin:6px 0'>
                🌡️ <b>Predicted Temp:</b> {row['pred_temp']}°C<br>
                🌿 <b>NDVI Score:</b> {row['ndvi']} &nbsp;
                    ({'Bare/Concrete' if row['ndvi'] < 0.15 else 'Sparse' if row['ndvi'] < 0.3 else 'Moderate' if row['ndvi'] < 0.5 else 'Good'} cover)<br>
                📊 <b>Priority Score:</b> {row['priority_score']}<br>
                ⚠️ <b>Status:</b> {row['priority']}<br>
                <hr style='margin:6px 0'>
                💡 <b>Recommended Action:</b><br>
                &nbsp;&nbsp;{row['recommendation']}<br>
                <small style='color:gray'>AeroCool AI · NDVI × LSTM Analysis</small>
            </div>
            """,
            max_width=280
        ),
        tooltip=f"{row['name']} | {row['pred_temp']}°C | NDVI {row['ndvi']}",
    ).add_to(priority_group)

priority_group.add_to(m)

# ── Legend ──
legend_html = """
<div style="
    position: fixed; bottom: 40px; left: 40px; z-index: 9999;
    background: white; border: 1px solid #ccc; border-radius: 10px;
    padding: 14px 18px; font-family: sans-serif; font-size: 13px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); min-width: 220px;
">
    <b>🌿 NDVI × 🌡️ Heat Priority Map</b><br>
    <div style='margin-top:8px'>
        <b>Circle size</b> = intervention urgency<br><br>
        <span style='color:#A32D2D'>●</span> CRITICAL — Act Now<br>
        <span style='color:#D85A30'>●</span> HIGH — Plan Intervention<br>
        <span style='color:#BA7517'>●</span> MODERATE — Monitor<br>
        <span style='color:#3B6D11'>●</span> LOW — Maintain<br><br>
        <b>Heatmap</b> = NDVI vegetation layer<br>
        <span style='color:brown'>■</span> Bare &nbsp;
        <span style='color:#8B8B00'>■</span> Sparse &nbsp;
        <span style='color:darkgreen'>■</span> Dense
    </div>
    <div style='margin-top:8px; color:gray; font-size:11px'>
        Priority = Heat × (1 − NDVI)<br>
        AeroCool AI · ISRO Bhuvan Integration
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

output_path = "ndvi_heat_overlay.html"
m.save(output_path)

print(f"✅ NDVI overlay map saved: {output_path}")
print("   Open in browser to see the interactive map")
print("=" * 65)

# ─────────────────────────────────────────────
# STEP 6: Save Analysis as CSV (for your report)
# ─────────────────────────────────────────────
csv_cols = ["name", "lat", "lon", "pred_temp", "ndvi", "priority_score", "recommendation"]
df[csv_cols].to_csv("ndvi_analysis_results.csv", index=False)
print("📄 Results also saved to: ndvi_analysis_results.csv")
