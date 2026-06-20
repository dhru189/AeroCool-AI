import { useState, useEffect } from "react";
import HeatMap from "./components/HeatMap";
import ModelStats from "./components/ModelStats";
import ScenarioSimulator from "./components/ScenarioSimulator";
import HVIRanking from "./components/HVIRanking";
import OptimizationPlan from "./components/OptimizationPlan";

const API = "http://localhost:8000/api";

const NAV = [
  { id: "heatmap",   label: "Heat Map",      icon: "🗺️" },
  { id: "models",    label: "AI Models",     icon: "🧠" },
  { id: "hvi",       label: "Vulnerability", icon: "⚠️" },
  { id: "scenario",  label: "Simulator",     icon: "🌿" },
  { id: "optimizer", label: "Optimizer",     icon: "📋" },
];

export default function App() {
  const [page, setPage]     = useState("heatmap");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  return (
    <div style={styles.root}>
      {/* ── Top Bar ── */}
      <header style={styles.topbar}>
        <div style={styles.brand}>
          <span style={styles.brandName}>Bharat<span style={styles.x}>X</span></span>
          <span style={styles.brandSub}>AeroCool AI · Urban Heat Mitigation Engine</span>
        </div>
        <div style={styles.statusPill}>
          <span style={{
            ...styles.dot,
            background: health?.status === "ok" ? "#10B981" : "#EF4444"
          }} />
          {health?.status === "ok" ? "API online" : "API offline"}
        </div>
      </header>

      {/* ── Nav ── */}
      <nav style={styles.nav}>
        {NAV.map(n => (
          <button key={n.id}
            onClick={() => setPage(n.id)}
            style={{ ...styles.navBtn, ...(page === n.id ? styles.navActive : {}) }}>
            {n.icon} {n.label}
          </button>
        ))}
      </nav>

      {/* ── Page Content ── */}
      <main style={styles.main}>
        {page === "heatmap"   && <HeatMap   api={API} />}
        {page === "models"    && <ModelStats api={API} />}
        {page === "hvi"       && <HVIRanking api={API} />}
        {page === "scenario"  && <ScenarioSimulator api={API} />}
        {page === "optimizer" && <OptimizationPlan api={API} />}
      </main>

      {/* ── Footer ── */}
      <footer style={styles.footer}>
        Team BharatX · Bharatiya Antariksh Hackathon 2026 · ISRO · Problem Statement 1
      </footer>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0B0E14",
    color: "#E5E9F0",
    fontFamily: "'Inter', sans-serif",
    display: "flex", flexDirection: "column",
  },
  topbar: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "14px 28px",
    background: "#0E1420",
    borderBottom: "1px solid #1F2533",
  },
  brand: { display: "flex", alignItems: "baseline", gap: 14 },
  brandName: {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 22, fontWeight: 700, color: "#E5E9F0",
  },
  x: { color: "#10B981" },
  brandSub: { fontSize: 13, color: "#6B7889" },
  statusPill: {
    display: "flex", alignItems: "center", gap: 7,
    padding: "4px 14px",
    background: "#131822",
    border: "1px solid #1F2533",
    borderRadius: 20,
    fontSize: 12, color: "#9AA5B5",
  },
  dot: {
    width: 8, height: 8, borderRadius: "50%",
    display: "inline-block",
    animation: "pulse 2s infinite",
  },
  nav: {
    display: "flex", gap: 4, padding: "10px 24px",
    background: "#0E1420",
    borderBottom: "1px solid #1F2533",
    overflowX: "auto",
  },
  navBtn: {
    padding: "7px 18px",
    background: "transparent",
    border: "1px solid #1F2533",
    borderRadius: 8,
    color: "#6B7889",
    cursor: "pointer",
    fontSize: 13,
    fontFamily: "inherit",
    whiteSpace: "nowrap",
    transition: "all 0.15s",
  },
  navActive: {
    background: "#131822",
    border: "1px solid #3B82F6",
    color: "#E5E9F0",
  },
  main: {
    flex: 1,
    padding: "24px 28px",
    maxWidth: 1280,
    width: "100%",
    margin: "0 auto",
  },
  footer: {
    padding: "12px 28px",
    background: "#0E1420",
    borderTop: "1px solid #1F2533",
    fontSize: 11,
    color: "#3F4A5C",
    fontFamily: "'JetBrains Mono', monospace",
    textAlign: "center",
  },
};
