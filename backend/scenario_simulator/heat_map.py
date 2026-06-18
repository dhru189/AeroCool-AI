"""
AeroCool AI — Urban Heat Hotspot Map
=====================================
Generates an interactive HTML choropleth map of predicted heat
stress zones across city areas using Folium.

HOW TO RUN:
    pip install folium pandas numpy
    python heat_map.py
    → Opens heat_map_output.html in your browser

REPLACE:
    sensor_data below with your actual LSTM predictions
    lat/lon values with your real city sensor coordinates
"""

import folium
import json
import numpy as np
import webbrowser
import os

# ─────────────────────────────────────────────
# STEP 1: Your sensor locations + LSTM predicted temperatures
# Replace these with your actual sensor GPS coordinates & predictions
# ─────────────────────────────────────────────
sensor_data = [
    {"name": "Sensor_01 — Civil Lines",        "lat": 26.4740, "lon": 80.3319, "pred_temp": 44.2, "zone": "North"},
    {"name": "Sensor_02 — Kanpur Central",      "lat": 26.4670, "lon": 80.3515, "pred_temp": 46.8, "zone": "Central"},
    {"name": "Sensor_03 — Kidwai Nagar",        "lat": 26.4558, "lon": 80.3411, "pred_temp": 43.5, "zone": "South-West"},
    {"name": "Sensor_04 — GT Road Corridor",    "lat": 26.4800, "lon": 80.3600, "pred_temp": 47.1, "zone": "East"},
    {"name": "Sensor_05 — Armapur Estate",      "lat": 26.4900, "lon": 80.2950, "pred_temp": 41.3, "zone": "North-West"},
    {"name": "Sensor_06 — Swaroop Nagar",       "lat": 26.4350, "lon": 80.3200, "pred_temp": 45.6, "zone": "South"},
    {"name": "Sensor_07 — Panki Industrial",    "lat": 26.4200, "lon": 80.2800, "pred_temp": 48.9, "zone": "Industrial"},
    {"name": "Sensor_08 — Govind Nagar",        "lat": 26.4450, "lon": 80.3750, "pred_temp": 44.7, "zone": "East"},
    {"name": "Sensor_09 — Kakadeo",             "lat": 26.4650, "lon": 80.2700, "pred_temp": 40.2, "zone": "West"},
    {"name": "Sensor_10 — Rawatpur",            "lat": 26.5050, "lon": 80.3400, "pred_temp": 42.8, "zone": "North-East"},
]

# ─────────────────────────────────────────────
# STEP 2: Heat Risk Classification
# ─────────────────────────────────────────────
def classify_risk(temp):
    if temp >= 47:
        return "CRITICAL", "#A32D2D", "🔴"
    elif temp >= 44:
        return "HIGH", "#D85A30", "🟠"
    elif temp >= 41:
        return "MODERATE", "#BA7517", "🟡"
    else:
        return "LOW", "#3B6D11", "🟢"

# ─────────────────────────────────────────────
# STEP 3: Build the Folium Map
# ─────────────────────────────────────────────
center_lat = np.mean([s["lat"] for s in sensor_data])
center_lon = np.mean([s["lon"] for s in sensor_data])

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles="CartoDB positron",   # Clean light base map
)

# Add a dark satellite overlay option
folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)
folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

# ─────────────────────────────────────────────
# STEP 4: Add sensor markers with popups
# ─────────────────────────────────────────────
for s in sensor_data:
    risk_label, color, emoji = classify_risk(s["pred_temp"])

    # Circle marker — radius = heat intensity visual
    radius = (s["pred_temp"] - 38) * 2.5   # Scale: hotter = bigger circle

    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=max(radius, 8),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=2,
        popup=folium.Popup(
            f"""
            <div style='font-family: sans-serif; min-width: 200px;'>
                <b style='font-size:14px'>{s['name']}</b><br>
                <hr style='margin: 6px 0'>
                🌡️ <b>Predicted Temp:</b> {s['pred_temp']}°C<br>
                📍 <b>Zone:</b> {s['zone']}<br>
                ⚠️ <b>Risk Level:</b>
                <span style='color:{color}; font-weight:bold'>{emoji} {risk_label}</span><br>
                <hr style='margin: 6px 0'>
                <small style='color:gray'>LSTM 2-hour forecast</small>
            </div>
            """,
            max_width=250
        ),
        tooltip=f"{s['name'].split('—')[1].strip()} | {s['pred_temp']}°C | {risk_label}",
    ).add_to(m)

# ─────────────────────────────────────────────
# STEP 5: Legend (HTML injected into map)
# ─────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed; bottom: 40px; left: 40px; z-index: 9999;
    background: white; border: 1px solid #ccc; border-radius: 10px;
    padding: 14px 18px; font-family: sans-serif; font-size: 13px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
">
    <b style='font-size:14px'>🌡️ Heat Risk Level</b><br>
    <div style='margin-top:8px'>
        <span style='color:#A32D2D'>●</span>&nbsp; CRITICAL ≥ 47°C<br>
        <span style='color:#D85A30'>●</span>&nbsp; HIGH 44–47°C<br>
        <span style='color:#BA7517'>●</span>&nbsp; MODERATE 41–44°C<br>
        <span style='color:#3B6D11'>●</span>&nbsp; LOW &lt; 41°C
    </div>
    <div style='margin-top:8px; color:gray; font-size:11px'>
        Circle size = heat intensity<br>
        AeroCool AI · LSTM 2hr Forecast
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ─────────────────────────────────────────────
# STEP 6: Add Layer Control & Save
# ─────────────────────────────────────────────
folium.LayerControl().add_to(m)

output_path = "heat_map_output.html"
m.save(output_path)

print("=" * 50)
print("✅ Heat map saved!")
print(f"   File: {os.path.abspath(output_path)}")
print()
print("📊 Summary:")
for s in sensor_data:
    risk_label, _, emoji = classify_risk(s["pred_temp"])
    print(f"   {emoji} {s['name'].split('—')[1].strip():<22} {s['pred_temp']}°C  [{risk_label}]")
print()
print("🌐 Open heat_map_output.html in your browser")
print("=" * 50)
