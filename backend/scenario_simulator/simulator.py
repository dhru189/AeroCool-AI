"""
AeroCool AI — Cooling Scenario Simulator
==========================================
A Streamlit dashboard with 3 interactive sliders:
  - Cool Roof Coverage %
  - Trees Planted
  - Water Misting Stations

Each slider adjusts a physics-based formula to show how much
the predicted temperature would drop with those interventions.

HOW TO RUN:
    pip install streamlit pandas numpy plotly
    streamlit run scenario_simulator.py
    → Opens in browser at http://localhost:8501
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AeroCool AI — Scenario Simulator",
    page_icon="🌡️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🌡️ AeroCool AI — Cooling Scenario Simulator")
st.markdown("**Adjust interventions below to see how much the predicted peak temperature drops.**")
st.markdown("---")

# ─────────────────────────────────────────────
# Baseline: Your LSTM predicted temperatures for each zone
# REPLACE these with your actual LSTM outputs
# ─────────────────────────────────────────────
zone_names = [
    "Civil Lines", "Kanpur Central", "Kidwai Nagar",
    "GT Road Corridor", "Armapur Estate", "Swaroop Nagar",
    "Panki Industrial", "Govind Nagar", "Kakadeo", "Rawatpur"
]
baseline_temps = np.array([44.2, 46.8, 43.5, 47.1, 41.3, 45.6, 48.9, 44.7, 40.2, 42.8])

# ─────────────────────────────────────────────
# PHYSICS FORMULAS
# Based on peer-reviewed urban cooling studies:
#
#  Cool Roofs:  Each 10% coverage → ~0.3°C drop (urban albedo effect)
#               Source: Akbari et al., 2001 — "Cool surfaces and shade trees"
#
#  Trees:       Each 100 trees in a zone → ~0.15°C drop (evapotranspiration + shade)
#               Source: Bowler et al., 2010 — "Urban greening as a climate adaptation"
#
#  Misting:     Each station → ~0.4°C in target zone (evaporative cooling)
#               Source: Montazeri et al., 2017 — "Outdoor misting system evaluation"
# ─────────────────────────────────────────────
def calculate_temp_reduction(cool_roof_pct, num_trees, num_misters):
    cool_roof_effect  = (cool_roof_pct / 10) * 0.30    # per 10% coverage
    tree_effect       = (num_trees / 100) * 0.15        # per 100 trees
    misting_effect    = num_misters * 0.40              # per misting station
    total_reduction   = cool_roof_effect + tree_effect + misting_effect
    return round(total_reduction, 2)

def calculate_water_saved_liters(baseline, final_temps, area_km2=50):
    """
    Reactive cooling (sprinklers at peak) vs AeroCool proactive cooling
    Estimate: each 1°C reduction avoids ~500L/zone/day of reactive water use
    """
    avg_reduction = np.mean(baseline - final_temps)
    zones = len(baseline)
    liters_saved = avg_reduction * 500 * zones
    return int(liters_saved)

def calculate_carbon_saved_kg(liters_saved):
    """
    Pumping + treating water: ~0.34 kgCO2 per 1000L (India grid average)
    """
    return round(liters_saved * 0.34 / 1000, 1)

# ─────────────────────────────────────────────
# Sidebar — Intervention Sliders
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Intervention Controls")
    st.markdown("Adjust each slider to simulate a cooling scenario.")
    st.markdown("---")

    cool_roof_pct = st.slider(
        "🏠 Cool Roof Coverage (%)",
        min_value=0, max_value=100, value=20, step=5,
        help="Percentage of rooftops painted with reflective cool roof coating. Increases urban albedo."
    )
    st.caption(f"Effect: −{round((cool_roof_pct/10)*0.30, 2)}°C average city-wide")

    st.markdown("---")

    num_trees = st.slider(
        "🌳 Trees Planted (city-wide)",
        min_value=0, max_value=5000, value=500, step=100,
        help="Number of trees planted across all zones. Provides shade and evapotranspiration cooling."
    )
    st.caption(f"Effect: −{round((num_trees/100)*0.15, 2)}°C average city-wide")

    st.markdown("---")

    num_misters = st.slider(
        "💧 Water Misting Stations",
        min_value=0, max_value=50, value=5, step=1,
        help="Misting stations installed in high-traffic zones. Each station cools its local area."
    )
    st.caption(f"Effect: −{round(num_misters*0.40, 2)}°C average city-wide")

    st.markdown("---")
    st.markdown("**Sources:** Akbari et al. 2001, Bowler et al. 2010, Montazeri et al. 2017")

# ─────────────────────────────────────────────
# Calculations
# ─────────────────────────────────────────────
reduction = calculate_temp_reduction(cool_roof_pct, num_trees, num_misters)
final_temps = np.clip(baseline_temps - reduction, 35.0, 55.0)

zones_above_threshold_before = int(np.sum(baseline_temps >= 44.0))
zones_above_threshold_after  = int(np.sum(final_temps >= 44.0))

liters_saved = calculate_water_saved_liters(baseline_temps, final_temps)
carbon_saved = calculate_carbon_saved_kg(liters_saved)
max_reduction_zone = zone_names[np.argmax(baseline_temps - final_temps)]

# ─────────────────────────────────────────────
# Top Metrics Row
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🌡️ Avg Temperature Drop",
        value=f"−{reduction}°C",
        delta=f"from {round(float(np.mean(baseline_temps)),1)}°C baseline",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="🔴 High-Risk Zones",
        value=f"{zones_above_threshold_after} zones",
        delta=f"{zones_above_threshold_before - zones_above_threshold_after} zones resolved",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="💧 Water Saved Daily",
        value=f"{liters_saved:,} L",
        delta="vs reactive cooling",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="🌱 CO₂ Avoided",
        value=f"{carbon_saved} kg/day",
        delta="pumping + treatment",
        delta_color="normal"
    )

st.markdown("---")

# ─────────────────────────────────────────────
# Main Chart — Before vs After Bar Chart
# ─────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📊 Predicted Temperature — Before vs After Intervention")

    fig = go.Figure()

    # Danger threshold line
    fig.add_hline(
        y=44.0, line_dash="dash", line_color="red",
        annotation_text="⚠️ High Risk Threshold (44°C)",
        annotation_position="top right"
    )

    fig.add_trace(go.Bar(
        name="Baseline (no intervention)",
        x=zone_names,
        y=baseline_temps,
        marker_color=["#A32D2D" if t >= 47 else "#D85A30" if t >= 44 else "#BA7517" for t in baseline_temps],
        opacity=0.6,
        text=[f"{t}°C" for t in baseline_temps],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="After Intervention",
        x=zone_names,
        y=final_temps,
        marker_color=["#0C447C" if t >= 44 else "#0F6E56" for t in final_temps],
        opacity=0.9,
        text=[f"{round(float(t),1)}°C" for t in final_temps],
        textposition="outside",
    ))

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-35,
        yaxis=dict(title="Temperature (°C)", range=[35, 52]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=0, r=0, t=10, b=80),
        font=dict(family="sans-serif", size=12),
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 🎯 Intervention Breakdown")

    cool_roof_contribution = (cool_roof_pct / 10) * 0.30
    tree_contribution      = (num_trees / 100) * 0.15
    misting_contribution   = num_misters * 0.40

    if reduction > 0:
        pie_fig = go.Figure(data=[go.Pie(
            labels=["Cool Roofs", "Tree Planting", "Misting Stations"],
            values=[
                cool_roof_contribution,
                tree_contribution,
                misting_contribution
            ],
            hole=0.45,
            marker_colors=["#1D9E75", "#639922", "#378ADD"],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Cooling: %{value:.2f}°C<extra></extra>"
        )])

        pie_fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"−{reduction}°C", x=0.5, y=0.5,
                              font_size=20, showarrow=False)]
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("Move the sliders to see cooling breakdown.")

    st.markdown(f"""
    **Biggest impact zone:** {max_reduction_zone}

    **Total cooling:** −{reduction}°C
    - Cool Roofs: −{round(cool_roof_contribution, 2)}°C
    - Trees: −{round(tree_contribution, 2)}°C
    - Misting: −{round(misting_contribution, 2)}°C
    """)

# ─────────────────────────────────────────────
# Bottom — Zone Status Table
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗺️ Zone-by-Zone Status")

import pandas as pd

def risk_label(t):
    if t >= 47: return "🔴 CRITICAL"
    if t >= 44: return "🟠 HIGH"
    if t >= 41: return "🟡 MODERATE"
    return "🟢 LOW"

df = pd.DataFrame({
    "Zone":             zone_names,
    "Baseline (°C)":   baseline_temps.round(1),
    "After Scenario (°C)": final_temps.round(1),
    "Reduction (°C)":  (baseline_temps - final_temps).round(2),
    "Risk Before":     [risk_label(t) for t in baseline_temps],
    "Risk After":      [risk_label(t) for t in final_temps],
})

st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("AeroCool AI · Bharatiya Antariksh Hackathon 2026 · Physics-informed cooling scenario simulator")
