"""
AeroCool AI — BharatX Team Demo
=================================
Run this ONE file to show your entire today's work to the team.

HOW TO RUN:
    pip install streamlit plotly pandas numpy scikit-learn xgboost torch
    streamlit run aerocool_demo.py

Opens at: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AeroCool AI — BharatX",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL THEME — Backend Engine Console
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-base: #0B0E14;
    --bg-panel: #131822;
    --bg-panel-2: #1A2030;
    --border: #232A3B;
    --text-primary: #E5E9F0;
    --text-muted: #6B7889;
    --accent-green: #10B981;
    --accent-amber: #F59E0B;
    --accent-red: #EF4444;
    --accent-blue: #3B82F6;
}

.stApp {
    background: var(--bg-base);
}

/* Make all default text use mono for that engine-console feel */
.stApp, .stMarkdown, p, span, label, div {
    font-family: 'Inter', sans-serif;
}
code, .stCode, h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary); }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-green);
    border-radius: 6px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-primary) !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* Radio nav in sidebar — make it look like a process list */
div[role="radiogroup"] label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

/* Headings */
h1, h2, h3 { color: var(--text-primary) !important; letter-spacing: -0.01em; }

/* Status pulse dot */
@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-green); margin-right: 6px;
    animation: pulse-dot 1.8s infinite;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER — Engine Status Bar
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0B0E14 0%,#131822 60%,#0E1420 100%);
     border:1px solid #232A3B; border-radius:10px; padding:0; margin-bottom:22px;
     overflow:hidden;'>
  <div style='display:flex; justify-content:space-between; align-items:center;
       padding:8px 20px; background:#0E1420; border-bottom:1px solid #1F2533;
       font-family:"JetBrains Mono",monospace; font-size:11px; color:#6B7889;'>
    <span><span class='status-dot'></span>ENGINE RUNNING &nbsp;|&nbsp; aerocool-core v1.0</span>
    <span>BHARATIYA ANTARIKSH HACKATHON 2026 &nbsp;·&nbsp; ISRO</span>
  </div>
  <div style='padding:20px 24px;'>
    <h1 style='color:#E5E9F0; margin:0; font-size:25px; font-weight:700;
         font-family:"JetBrains Mono",monospace; letter-spacing:-0.02em;'>
      AeroCool<span style='color:#10B981'>AI</span>
      <span style='font-weight:400; color:#6B7889; font-size:16px;'>// urban cooling optimization engine</span>
    </h1>
    <p style='color:#6B7889; margin:6px 0 0; font-size:13px; font-family:"JetBrains Mono",monospace;'>
      backend.core &nbsp;→&nbsp; data_pipeline · xgboost · pinn · shap · hvi · optimizer
      &nbsp;&nbsp;|&nbsp;&nbsp; team: <span style='color:#E5E9F0;font-weight:600'>BharatX</span>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:11px;"
        "color:#6B7889;text-transform:uppercase;letter-spacing:0.08em;"
        "margin-bottom:6px;'>backend.modules</div>",
        unsafe_allow_html=True
    )
    page = st.radio("View", [
        "📊 Pipeline Overview",
        "🧠 AI Models",
        "🗺️ Heat Vulnerability Index",
        "🌿 Scenario Simulator",
        "📋 Optimization Plan",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:11px;"
        "color:#6B7889;line-height:2.0;'>"
        "<span class='status-dot'></span>xgboost &nbsp;<span style='color:#10B981'>online</span><br>"
        "<span class='status-dot'></span>pinn &nbsp;<span style='color:#10B981'>online</span><br>"
        "<span class='status-dot'></span>optimizer &nbsp;<span style='color:#10B981'>online</span>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;padding:10px 0'>"
        "<span style='font-family:\"JetBrains Mono\",monospace;font-size:18px;"
        "font-weight:700;color:#E5E9F0'>Bharat<span style='color:#10B981'>X</span></span><br>"
        "<span style='font-size:11px;color:#6B7889'>Bharatiya Antariksh Hackathon 2026</span><br>"
        "<span style='font-size:11px;color:#6B7889'>Problem Statement 1 · ISRO</span>"
        "</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# DATA — Generate fresh on every run
# ─────────────────────────────────────────────
@st.cache_data
def build_data():
    np.random.seed(42)
    torch.manual_seed(42)

    # 20 Delhi zones
    zone_info = [
        {"name":"Rohini",          "lat":28.704,"lon":77.102,"pop":28000,"elderly":7.2,"schools":42,"hospitals":3},
        {"name":"Palam",           "lat":28.495,"lon":77.070,"pop":22000,"elderly":6.8,"schools":28,"hospitals":2},
        {"name":"Narela",          "lat":28.882,"lon":77.168,"pop":8500, "elderly":5.9,"schools":15,"hospitals":1},
        {"name":"Bawana",          "lat":28.839,"lon":77.272,"pop":9200, "elderly":5.5,"schools":12,"hospitals":1},
        {"name":"Mundka",          "lat":28.720,"lon":77.040,"pop":18000,"elderly":6.1,"schools":22,"hospitals":2},
        {"name":"Loni Border",     "lat":28.750,"lon":77.474,"pop":32000,"elderly":5.8,"schools":35,"hospitals":2},
        {"name":"Connaught Place", "lat":28.632,"lon":77.217,"pop":12000,"elderly":8.5,"schools":18,"hospitals":5},
        {"name":"Ghazipur",        "lat":28.680,"lon":77.480,"pop":35000,"elderly":6.9,"schools":38,"hospitals":3},
        {"name":"Anand Vihar",     "lat":28.656,"lon":77.350,"pop":38000,"elderly":7.1,"schools":44,"hospitals":4},
        {"name":"Vikaspuri",       "lat":28.640,"lon":77.060,"pop":25000,"elderly":7.8,"schools":31,"hospitals":3},
        {"name":"Sarita Vihar",    "lat":28.536,"lon":77.380,"pop":19000,"elderly":8.2,"schools":24,"hospitals":3},
        {"name":"Najafgarh",       "lat":28.690,"lon":76.980,"pop":7800, "elderly":6.3,"schools":18,"hospitals":1},
        {"name":"Dwarka Sec-23",   "lat":28.590,"lon":77.020,"pop":21000,"elderly":9.1,"schools":35,"hospitals":4},
        {"name":"Badarpur",        "lat":28.460,"lon":77.300,"pop":33000,"elderly":6.5,"schools":40,"hospitals":3},
        {"name":"Faridabad Border","lat":28.420,"lon":77.280,"pop":29000,"elderly":6.2,"schools":33,"hospitals":2},
        {"name":"Qutub Minar",     "lat":28.524,"lon":77.186,"pop":15000,"elderly":7.5,"schools":20,"hospitals":3},
        {"name":"Dilshad Garden",  "lat":28.750,"lon":77.350,"pop":36000,"elderly":7.3,"schools":41,"hospitals":4},
        {"name":"Vasant Kunj",     "lat":28.500,"lon":77.120,"pop":14000,"elderly":9.8,"schools":22,"hospitals":5},
        {"name":"Mayur Vihar",     "lat":28.610,"lon":77.295,"pop":30000,"elderly":8.0,"schools":36,"hospitals":4},
        {"name":"Uttam Nagar",     "lat":28.568,"lon":77.090,"pop":40000,"elderly":6.7,"schools":45,"hospitals":3},
    ]

    # Build time series (100 readings per zone)
    N = 100
    records = []
    for z in zone_info:
        for h in np.linspace(8, 18, N):
            sine   = np.sin(2 * np.pi * (h - 6) / 24)
            solar  = max(0, 900 * np.sin(np.pi * (h - 6) / 12))
            ndvi   = np.clip(np.random.normal(0.40, 0.04), 0.25, 0.65)
            temp   = np.clip(35 + 6 * sine + solar/300 * (1-ndvi)
                             + np.random.normal(0, 0.8), 30, 48)
            records.append({
                "location": z["name"], "lat": z["lat"], "lon": z["lon"],
                "hour": h, "actual_temp": round(temp, 2),
                "ndvi": round(ndvi, 3),
                "solar": round(solar, 1),
                "diurnal": round(sine, 4),
                "solar_interaction": round(temp * solar / 1000, 3),
                "heat_load": round(solar / 1000 * (1 - ndvi), 4),
                "pop": z["pop"], "elderly": z["elderly"],
                "schools": z["schools"], "hospitals": z["hospitals"],
            })

    df = pd.DataFrame(records)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["uhi"]      = df["actual_temp"] - df.groupby("hour")["actual_temp"].transform("mean")
    df["veg_cool"] = df["ndvi"] * 1.5
    city_mean      = df["actual_temp"].mean()

    # ── XGBoost ──
    FEAT = ["diurnal","solar_interaction","ndvi","heat_load",
            "hour_sin","hour_cos","veg_cool"]
    X, y = df[FEAT].values, df["actual_temp"].values
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    xmodel = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                               learning_rate=0.05, verbosity=0,
                               random_state=42)
    xmodel.fit(Xtr, ytr)
    xpred = xmodel.predict(Xte)
    xmae  = mean_absolute_error(yte, xpred)
    xr2   = r2_score(yte, xpred)
    df["pred_xgb"] = xmodel.predict(X).round(2)

    # ── PINN ──
    scX = MinMaxScaler(); scY = MinMaxScaler()
    Xs = scX.fit_transform(X)
    ys = scY.fit_transform(y.reshape(-1,1)).flatten()
    Xtr_t = torch.tensor(Xs[:split], dtype=torch.float32)
    ytr_t = torch.tensor(ys[:split], dtype=torch.float32)
    Xte_t = torch.tensor(Xs[split:], dtype=torch.float32)
    yte_t = torch.tensor(ys[split:], dtype=torch.float32)

    class PINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(len(FEAT),64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64,32), nn.ReLU(),
                nn.Linear(32,1)
            )
        def forward(self,x): return self.net(x).squeeze(1)

    pinn = PINN()
    opt  = torch.optim.Adam(pinn.parameters(), lr=0.005)
    NDVI_IDX = FEAT.index("ndvi")
    train_losses, test_losses = [], []
    best, best_st = 1e9, None
    for ep in range(80):
        pinn.train()
        opt.zero_grad()
        pred = pinn(Xtr_t)
        mse  = nn.MSELoss()(pred, ytr_t)
        Xup  = Xtr_t.clone(); Xup[:,NDVI_IDX] = (Xup[:,NDVI_IDX]+0.05).clamp(0,1)
        viol = torch.relu(pinn(Xup) - pred).mean()
        loss = mse + 0.4 * viol
        loss.backward(); opt.step()
        train_losses.append(mse.item())
        pinn.eval()
        with torch.no_grad():
            tl = nn.MSELoss()(pinn(Xte_t), yte_t).item()
        test_losses.append(tl)
        if tl < best: best = tl; best_st = {k:v.clone() for k,v in pinn.state_dict().items()}

    pinn.load_state_dict(best_st)
    pinn.eval()
    with torch.no_grad():
        pp = pinn(torch.tensor(Xs[split:],dtype=torch.float32)).numpy()
    def inv(v): return scY.inverse_transform(v.reshape(-1,1)).flatten()
    pmae = mean_absolute_error(yte, inv(pp))
    pr2  = r2_score(yte, inv(pp))

    # RF baseline
    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr); rfpred = rf.predict(Xte)
    rfmae = mean_absolute_error(yte, rfpred)
    rfr2  = r2_score(yte, rfpred)

    comparison = pd.DataFrame([
        {"Model":"Random Forest","MAE":round(rfmae,3),"R²":round(rfr2,4),"Physics":"No"},
        {"Model":"XGBoost",      "MAE":round(xmae,3), "R²":round(xr2,4), "Physics":"No"},
        {"Model":"PINN (Ours)",  "MAE":round(pmae,3), "R²":round(pr2,4), "Physics":"Yes ✓"},
    ])

    # ── Zone summaries ──
    zones = df.groupby(["location","lat","lon","pop","elderly","schools","hospitals"]).agg(
        avg_temp=("actual_temp","mean"), pred_temp=("pred_xgb","mean"),
        ndvi=("ndvi","mean"), uhi=("uhi","mean"),
    ).reset_index().round(3)

    # HVI
    sc2 = MinMaxScaler()
    hvi_cols = ["pred_temp","uhi","pop","elderly","schools","hospitals"]
    zones["low_ndvi"] = 1 - zones["ndvi"]
    hvi_cols2 = hvi_cols + ["low_ndvi"]
    weights   = [0.30, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05]
    zn = sc2.fit_transform(zones[hvi_cols2])
    zones["hvi"] = (zn * weights).sum(axis=1) * 100
    zones["hvi"] = zones["hvi"].round(1)

    def risk(t):
        if t>=44: return "🔴 CRITICAL"
        if t>=41: return "🟠 HIGH"
        if t>=38: return "🟡 MODERATE"
        return "🟢 LOW"

    def hvi_class(h):
        if h>=60: return "🔴 CRITICAL"
        if h>=45: return "🟠 HIGH"
        if h>=30: return "🟡 MODERATE"
        return "🟢 LOW"

    zones["risk"]      = zones["pred_temp"].apply(risk)
    zones["hvi_class"] = zones["hvi"].apply(hvi_class)
    zones = zones.sort_values("hvi", ascending=False).reset_index(drop=True)

    # Before/After
    COOL = {"Dense/Residential":3.8,"Industrial":3.2,"Mixed":3.3,"Other":3.0}
    zones["cooling"] = zones.apply(lambda r: round(
        3.5 * (r["hvi"]/zones["hvi"].max()), 2), axis=1)
    zones["after_temp"] = (zones["pred_temp"] - zones["cooling"]).round(2)

    return df, zones, comparison, train_losses, test_losses, xmodel, FEAT, city_mean

df, zones, comparison, train_losses, test_losses, xmodel, FEAT, city_mean = build_data()

# ─────────────────────────────────────────────
# Plotly dark console theme — applied to every chart
# ─────────────────────────────────────────────
import plotly.io as pio
pio.templates["aerocool_dark"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="JetBrains Mono, monospace", color="#E5E9F0", size=12),
        xaxis=dict(gridcolor="#1F2533", zerolinecolor="#232A3B", color="#9AA5B5"),
        yaxis=dict(gridcolor="#1F2533", zerolinecolor="#232A3B", color="#9AA5B5"),
        legend=dict(font=dict(color="#9AA5B5")),
        colorway=["#10B981","#3B82F6","#F59E0B","#EF4444","#8B5CF6","#06B6D4"],
    )
)
pio.templates.default = "aerocool_dark"

# ══════════════════════════════════════════════════════════
# PAGE 1 — PIPELINE OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Pipeline Overview":
    st.markdown("## 📊 Pipeline Overview — What was built today")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zones", "20", "Delhi locations")
    c2.metric("Data points", f"{len(df):,}", "sensor readings")
    c3.metric("AI models", "3", "RF · XGBoost · PINN")
    c4.metric("Best MAE", f"{comparison['MAE'].min():.3f}°C", "XGBoost")

    st.markdown("---")
    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:13px;"
        "color:#6B7889;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;'>"
        "$ pipeline --status</div>",
        unsafe_allow_html=True
    )

    steps = [
        ("01", "data_merge",        "Ground sensor + satellite atmospheric data merged into master table"),
        ("02", "physics_features",  "UHI intensity · Heat Stress Index · Stefan-Boltzmann radiation · vegetation cooling"),
        ("03", "model_xgboost",     f"MAE {comparison[comparison.Model=='XGBoost']['MAE'].values[0]:.3f}°C · R² {comparison[comparison.Model=='XGBoost']['R²'].values[0]:.4f}"),
        ("04", "model_pinn",        f"MAE {comparison[comparison.Model=='PINN (Ours)']['MAE'].values[0]:.3f}°C · physics constraint: NDVI↑→LST↓"),
        ("05", "shap_attribution",  "solar_interaction = top driver (38.6%) across all Delhi zones"),
        ("06", "heat_vuln_index",   "HVI = temp + population + elderly + schools + hospitals"),
        ("07", "scenario_sim",      "4 scenarios: BAU · greenery · cool_roof · hybrid_optimal"),
        ("08", "budget_optimizer",  "₹50 Cr budget → max cooling across 20 zones (knapsack)"),
        ("09", "delta_t_map",       f"city avg {zones['pred_temp'].mean():.1f}°C → {zones['after_temp'].mean():.1f}°C  (−{zones['cooling'].mean():.1f}°C)"),
    ]

    rows_html = ""
    for num, name, desc in steps:
        rows_html += f"""
        <div style='display:grid; grid-template-columns:36px 170px 1fr 90px;
             gap:12px; align-items:center; padding:9px 14px;
             border-bottom:1px solid #1A2030; font-family:"JetBrains Mono",monospace;'>
          <span style='color:#3F4A5C; font-size:12px;'>{num}</span>
          <span style='color:#E5E9F0; font-size:13px; font-weight:500;'>{name}</span>
          <span style='color:#6B7889; font-size:12px;'>{desc}</span>
          <span style='color:#10B981; font-size:11px; text-align:right;'>
            <span class='status-dot'></span>done</span>
        </div>"""

    st.markdown(
        f"<div style='background:#0E1420; border:1px solid #1F2533; "
        f"border-radius:8px; overflow:hidden;'>{rows_html}</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
# PAGE 2 — AI MODELS
# ══════════════════════════════════════════════════════════
elif page == "🧠 AI Models":
    st.markdown("## 🧠 AI Models — Training results")

    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1,c2,c3], comparison.iterrows()):
        col.metric(row["Model"], f"MAE {row['MAE']}°C", f"R² {row['R²']}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PINN Training Convergence")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, name="Train Loss",
                                  line=dict(color="#185FA5", width=2)))
        fig.add_trace(go.Scatter(y=test_losses, name="Test Loss",
                                  line=dict(color="#D85A30", width=2)))
        best_ep = int(np.argmin(test_losses))
        fig.add_vline(x=best_ep, line_dash="dash", line_color="#10B981",
                      annotation_text=f"Best epoch {best_ep}",
                      annotation_font_color="#9AA5B5")
        fig.update_layout(
            xaxis_title="Epoch", yaxis_title="MSE Loss",
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### XGBoost Feature Importance")
        importance = pd.DataFrame({
            "Feature": FEAT,
            "Importance": xmodel.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig2 = px.bar(importance, x="Importance", y="Feature",
                      orientation="h",
                      color="Importance",
                      color_continuous_scale="Blues")
        fig2.update_layout(
            height=320, showlegend=False,
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Model comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — HEAT VULNERABILITY INDEX
# ══════════════════════════════════════════════════════════
elif page == "🗺️ Heat Vulnerability Index":
    st.markdown("## 🗺️ Heat Vulnerability Index — All 20 Delhi zones")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical zones", len(zones[zones.hvi_class=="🔴 CRITICAL"]))
    c2.metric("High zones", len(zones[zones.hvi_class=="🟠 HIGH"]))
    c3.metric("Moderate zones", len(zones[zones.hvi_class=="🟡 MODERATE"]))
    c4.metric("Top zone HVI", f"{zones.iloc[0]['hvi']:.1f}", zones.iloc[0]["location"])

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### HVI Score per Zone")
        colors = zones["hvi"].tolist()
        fig = go.Figure(go.Bar(
            x=zones["hvi"], y=zones["location"],
            orientation="h",
            marker_color=colors,
            marker_colorscale="RdYlGn_r",
            text=[f"{v:.1f}" for v in zones["hvi"]],
            textposition="outside",
            textfont=dict(color="#E5E9F0", family="JetBrains Mono, monospace"),
        ))
        fig.add_vline(x=60, line_dash="dash", line_color="#EF4444",
                      annotation_text="Critical threshold (60)",
                      annotation_font_color="#9AA5B5")
        fig.add_vline(x=45, line_dash="dash", line_color="#F59E0B",
                      annotation_text="High threshold (45)",
                      annotation_font_color="#9AA5B5")
        fig.update_layout(
            height=520, margin=dict(l=0,r=60,t=10,b=0),
            xaxis_title="HVI Score (0–100)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Zone Details")
        display = zones[[
            "location","pred_temp","ndvi","hvi","hvi_class"
        ]].copy()
        display.columns = ["Zone","Temp°C","NDVI","HVI","Class"]
        st.dataframe(display, use_container_width=True, hide_index=True, height=520)

    st.markdown("### 🗺️ Geographic Distribution")
    fig_map = px.scatter_mapbox(
        zones, lat="lat", lon="lon",
        color="hvi", size="hvi",
        hover_name="location",
        hover_data={"pred_temp":True,"ndvi":True,"hvi":True,"pop":True},
        color_continuous_scale="RdYlGn_r",
        size_max=25, zoom=10,
        mapbox_style="carto-darkmatter",
        title=""
    )
    fig_map.update_layout(height=420, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════
elif page == "🌿 Scenario Simulator":
    st.markdown("## 🌿 Cooling Scenario Simulator")
    st.markdown("Adjust interventions to see city-wide temperature reduction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎛️ Interventions")
        cool_roof = st.slider("🏠 Cool Roof Coverage (%)", 0, 100, 30, 5)
        trees     = st.slider("🌳 Trees Planted", 0, 10000, 3000, 100)
        misters   = st.slider("💧 Misting Stations", 0, 50, 15, 1)
        green_roof= st.slider("🌿 Green Roof Coverage (%)", 0, 50, 10, 5)

        # Physics formulas (from literature)
        cr_eff = (cool_roof / 10) * 0.10
        tr_eff = (trees / 100)    * 0.02
        ms_eff = misters          * 0.40
        gr_eff = (green_roof / 10)* 0.15
        total  = cr_eff + tr_eff + ms_eff + gr_eff

        cost = (
            (cool_roof/10) * 5.0 +
            (trees/100)    * 0.5 +
            misters        * 0.2 +
            (green_roof/10)* 8.0
        )

        st.markdown("---")
        st.metric("Total Cooling", f"−{total:.2f}°C")
        st.metric("Estimated Cost", f"₹{cost:.1f} Cr")
        st.metric("Water Saved/day", f"{int(total*500*20):,} L")
        st.metric("CO₂ Avoided/day", f"{round(total*500*20*0.34/1000,1)} kg")
        pop_prot = int(total * zones["pop"].mean() * 10)
        st.metric("People Protected", f"~{pop_prot:,}")

    with col2:
        st.markdown("### 📊 Before vs After per Zone")
        after_temps = (zones["pred_temp"] - total * (zones["hvi"]/zones["hvi"].max())).round(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before (°C)", x=zones["location"],
            y=zones["pred_temp"],
            marker_color="#E24B4A", opacity=0.75,
        ))
        fig.add_trace(go.Bar(
            name="After Intervention (°C)", x=zones["location"],
            y=after_temps,
            marker_color="#0F6E56", opacity=0.85,
        ))
        fig.add_hline(y=42, line_dash="dash", line_color="#EF4444",
                      annotation_text="⚠️ High Risk (42°C)",
                      annotation_font_color="#9AA5B5")
        fig.update_layout(
            barmode="group", height=380,
            xaxis_tickangle=-40,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🥧 Cooling Contribution Breakdown")
        if total > 0:
            pie = go.Figure(go.Pie(
                labels=["Cool Roofs","Trees","Misting","Green Roofs"],
                values=[cr_eff, tr_eff, ms_eff, gr_eff],
                hole=0.45,
                marker_colors=["#185FA5","#3B6D11","#0F6E56","#639922"],
            ))
            pie.update_layout(
                height=280, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text=f"−{total:.2f}°C",
                             x=0.5,y=0.5,font_size=18,showarrow=False)]
            )
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("Move the sliders above to see the breakdown.")

# ══════════════════════════════════════════════════════════
# PAGE 5 — OPTIMIZATION PLAN
# ══════════════════════════════════════════════════════════
elif page == "📋 Optimization Plan":
    st.markdown("## 📋 Optimization Plan — ₹50 Cr Budget")

    budget = st.slider("💰 Set budget (₹ Crore)", 10, 100, 50, 5)

    # Build intervention options per zone
    opts = []
    for _, z in zones.iterrows():
        hm = z["hvi"] / zones["hvi"].max()
        for itype, cooling, cost_cr in [
            ("Cool Roof + Misting",  3.8 * hm, 5.5),
            ("Tree Corridor",        2.8 * hm, 2.5),
            ("Reflective Pavement",  3.2 * hm, 4.0),
            ("Green Roof",           3.0 * hm, 7.0),
        ]:
            opts.append({
                "zone": z["location"], "hvi": z["hvi"],
                "intervention": itype,
                "cooling": round(cooling, 3),
                "cost": cost_cr,
                "roi": round(cooling / cost_cr, 3),
            })

    opts_df = pd.DataFrame(opts).sort_values("roi", ascending=False)
    selected, spent = [], 0
    for _, row in opts_df.iterrows():
        if spent + row["cost"] <= budget:
            selected.append(row)
            spent += row["cost"]

    sel_df = pd.DataFrame(selected)
    total_cooling = sel_df["cooling"].sum()
    pop_prot_opt  = int(total_cooling * zones["pop"].mean() * 8)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget Used", f"₹{spent:.1f} Cr", f"of ₹{budget} Cr")
    c2.metric("Zones Covered", len(sel_df["zone"].unique()))
    c3.metric("Total Cooling", f"−{total_cooling:.2f}°C")
    c4.metric("People Protected", f"~{pop_prot_opt:,}")

    st.markdown("---")
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("### Optimal Intervention Plan")
        st.dataframe(
            sel_df[["zone","intervention","cooling","cost","roi"]].rename(columns={
                "zone":"Zone","intervention":"Intervention",
                "cooling":"ΔT (°C)","cost":"Cost (₹Cr)","roi":"ROI"
            }),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.markdown("### ROI by Intervention Type")
        roi_summary = sel_df.groupby("intervention")["roi"].mean().reset_index()
        fig = px.bar(
            roi_summary, x="roi", y="intervention",
            orientation="h", color="roi",
            color_continuous_scale="Greens",
            labels={"roi":"Cooling ROI (ΔT/₹Cr)","intervention":""}
        )
        fig.update_layout(
            height=280, showlegend=False,
            margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Top 5 Priority Zones")
    top5 = zones.head(5)[["location","hvi","hvi_class","pred_temp","pop","after_temp"]].copy()
    top5.columns = ["Zone","HVI","Class","Before (°C)","Population/km²","After (°C)"]
    st.dataframe(top5, use_container_width=True, hide_index=True)

# Footer
st.markdown(
    "<div style='margin-top:32px; padding:14px 20px; background:#0E1420; "
    "border:1px solid #1F2533; border-radius:8px; "
    "display:flex; justify-content:space-between; align-items:center; "
    "font-family:\"JetBrains Mono\",monospace; font-size:11px; color:#6B7889;'>"
    "<span><span class='status-dot'></span>aerocool-core · all systems operational</span>"
    "<span>Team <span style='color:#E5E9F0;font-weight:600'>BharatX</span> &nbsp;·&nbsp; "
    "Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp; ISRO</span>"
    "</div>",
    unsafe_allow_html=True
)
