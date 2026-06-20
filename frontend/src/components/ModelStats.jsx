import { useState, useEffect } from "react";

export default function ModelStats({ api }) {
  const [models, setModels] = useState([]);
  const [shap, setShap]     = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${api}/models`).then(r => r.json()).catch(() => []),
      fetch(`${api}/shap`).then(r => r.json()).catch(() => []),
    ]).then(([m, s]) => { setModels(m); setShap(s); setLoading(false); });
  }, [api]);

  if (loading) return <div style={s.loading}>Loading model data...</div>;

  const bestModel = [...models].sort((a,b) => a["MAE (°C)"] - b["MAE (°C)"])[0];

  return (
    <div>
      <h2 style={s.heading}>🧠 AI Model Performance</h2>
      <p style={s.sub}>Three models trained on real Delhi sensor data · Best model: {bestModel?.Model}</p>

      <div style={s.modelGrid}>
        {models.map((m, i) => (
          <div key={i} style={{ ...s.card, borderTop: m.Model === bestModel?.Model ? "2px solid #10B981" : "2px solid #1F2533" }}>
            <div style={s.modelName}>{m.Model}</div>
            <div style={s.modelType}>{m.Type ?? (m.Model === "PINN (Ours)" ? "Custom Neural Net" : "Ensemble ML")}</div>
            <div style={s.metrics}>
              <div style={s.metRow}>
                <span style={s.metKey}>MAE</span>
                <span style={{ ...s.metVal, color: "#10B981" }}>{m["MAE (°C)"] ?? m.MAE}°C</span>
              </div>
              <div style={s.metRow}>
                <span style={s.metKey}>R²</span>
                <span style={s.metVal}>{m["R²"] ?? m.R2}</span>
              </div>
              <div style={s.metRow}>
                <span style={s.metKey}>Physics constraints</span>
                <span style={{ ...s.metVal, color: m["Physics Constraints"]?.includes("Yes") ? "#10B981" : "#6B7889" }}>
                  {m["Physics Constraints"] ?? (m.Model === "PINN (Ours)" ? "Yes ✓" : "No")}
                </span>
              </div>
            </div>
            {m.Model === bestModel?.Model && (
              <div style={s.bestBadge}>Best MAE ✓</div>
            )}
          </div>
        ))}
      </div>

      {shap.length > 0 && (
        <div style={s.panel}>
          <div style={s.panelHead}>SHAP Driver Attribution — why each zone is hot</div>
          <div style={{ padding: 16, overflowX: "auto" }}>
            <table style={s.table}>
              <thead>
                <tr>
                  {["Zone", "Avg Temp", "Top Driver", "Driver %", "NDVI"].map(h => (
                    <th key={h} style={s.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shap.map((row, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? "#0E1420" : "transparent" }}>
                    <td style={s.td}>{row.location}</td>
                    <td style={{ ...s.td, color: "#EF4444", fontFamily: "'JetBrains Mono',monospace" }}>{(row.avg_temp ?? 0).toFixed(1)}°C</td>
                    <td style={{ ...s.td, color: "#F59E0B" }}>{row.top_driver}</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace" }}>{(row.top_driver_pct ?? 0).toFixed(1)}%</td>
                    <td style={{ ...s.td, color: "#10B981", fontFamily: "'JetBrains Mono',monospace" }}>{(row.avg_ndvi ?? 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

const s = {
  heading: { margin: "0 0 4px", fontSize: 20, fontWeight: 600 },
  sub: { margin: "0 0 20px", fontSize: 13, color: "#6B7889" },
  loading: { color: "#6B7889", padding: 40, textAlign: "center" },
  modelGrid: { display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 20 },
  card: { background: "#0E1420", border: "1px solid #1F2533", borderRadius: 10, padding: 20, position: "relative" },
  modelName: { fontSize: 16, fontWeight: 600, color: "#E5E9F0", marginBottom: 4 },
  modelType: { fontSize: 12, color: "#6B7889", marginBottom: 16 },
  metrics: { display: "flex", flexDirection: "column", gap: 8 },
  metRow: { display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid #131822" },
  metKey: { fontSize: 12, color: "#6B7889" },
  metVal: { fontSize: 13, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color: "#E5E9F0" },
  bestBadge: { marginTop: 14, padding: "4px 10px", background: "#10B98122", color: "#10B981", fontSize: 11, fontWeight: 600, borderRadius: 20, display: "inline-block" },
  panel: { background: "#0E1420", border: "1px solid #1F2533", borderRadius: 10, overflow: "hidden" },
  panelHead: { padding: "12px 16px", borderBottom: "1px solid #1F2533", fontSize: 12, fontWeight: 500, color: "#9AA5B5", fontFamily: "'JetBrains Mono',monospace", textTransform: "uppercase", letterSpacing: "0.05em" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: { textAlign: "left", padding: "8px 12px", color: "#6B7889", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500, borderBottom: "1px solid #1F2533" },
  td: { padding: "8px 12px", color: "#E5E9F0", borderBottom: "1px solid #131822" },
};
