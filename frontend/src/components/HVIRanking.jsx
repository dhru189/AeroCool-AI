import { useState, useEffect } from "react";

// ── Heat Vulnerability Index ──────────────────────────────────
export function HVIRanking({ api }) {
  const [data, setData]     = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${api}/hvi`).then(r => r.json()).then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [api]);

  if (loading) return <div style={s.loading}>Loading HVI data...</div>;

  const sorted = [...data].sort((a, b) => (b.hvi_score ?? 0) - (a.hvi_score ?? 0));
  const nameKey = data[0]?.name ? "name" : "location";
  const critical = sorted.filter(z => (z.hvi_class ?? "").includes("CRITICAL")).length;
  const high     = sorted.filter(z => (z.hvi_class ?? "").includes("HIGH")).length;

  return (
    <div>
      <h2 style={s.heading}>⚠️ Heat Vulnerability Index</h2>
      <p style={s.sub}>Combines temperature + population density + elderly % + schools + hospitals</p>

      <div style={s.metrics}>
        {[
          ["Critical zones",    critical,                 "#EF4444"],
          ["High zones",        high,                     "#F59E0B"],
          ["Most vulnerable",   sorted[0]?.[nameKey] ?? "—", "#E5E9F0"],
          ["Top HVI score",     `${(sorted[0]?.hvi_score ?? 0).toFixed(1)} / 100`, "#10B981"],
        ].map(([k, v, c], i) => (
          <div key={i} style={s.metCard}>
            <div style={{ ...s.metVal, color: c }}>{v}</div>
            <div style={s.metKey}>{k}</div>
          </div>
        ))}
      </div>

      <div style={s.panel}>
        <div style={s.panelHead}>Zone vulnerability rankings</div>
        <div style={{ overflowX: "auto" }}>
          <table style={s.table}>
            <thead>
              <tr>
                {["Rank", "Zone", "HVI", "Class", "Temp", "Pop/km²", "Elderly %", "Recommendation"].map(h => (
                  <th key={h} style={s.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => {
                const cls   = row.hvi_class ?? "";
                const color = cls.includes("CRITICAL") ? "#EF4444" : cls.includes("HIGH") ? "#F59E0B" : cls.includes("MODERATE") ? "#EAB308" : "#10B981";
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? "#0E1420" : "transparent" }}>
                    <td style={{ ...s.td, color: "#6B7889" }}>#{i + 1}</td>
                    <td style={{ ...s.td, fontWeight: 500 }}>{row[nameKey]}</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace", color: color, fontWeight: 700 }}>{(row.hvi_score ?? 0).toFixed(1)}</td>
                    <td style={{ ...s.td }}><span style={{ background: color + "22", color, padding: "2px 8px", borderRadius: 20, fontSize: 11, fontWeight: 600 }}>{cls.split(" ").slice(1).join(" ")}</span></td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace" }}>{(row.pred_temp ?? row.avg_temp ?? 0).toFixed(1)}°C</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace" }}>{(row.pop_density ?? 0).toLocaleString()}</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace" }}>{(row.elderly_pct ?? 0).toFixed(1)}%</td>
                    <td style={{ ...s.td, fontSize: 11, color: "#9AA5B5" }}>{row.recommended_action ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Optimization Plan ─────────────────────────────────────────
export function OptimizationPlan({ api }) {
  const [plan, setPlan]       = useState([]);
  const [scenarios, setScen]  = useState([]);
  const [baMap, setBaMap]     = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${api}/optimal-plan`).then(r => r.json()).catch(() => []),
      fetch(`${api}/scenarios`).then(r => r.json()).catch(() => []),
      fetch(`${api}/before-after`).then(r => r.json()).catch(() => []),
    ]).then(([p, sc, ba]) => { setPlan(p); setScen(sc); setBaMap(ba); setLoading(false); });
  }, [api]);

  if (loading) return <div style={s.loading}>Loading optimization data...</div>;

  const totalCooling = plan.reduce((sum, r) => sum + (r.cooling ?? r["Cooling (°C)"] ?? 0), 0).toFixed(2);
  const totalCost    = plan.reduce((sum, r) => sum + (r.cost ?? r["Cost (₹ Cr)"] ?? 0), 0).toFixed(1);
  const nameKey      = baMap[0]?.name ? "name" : "location";

  return (
    <div>
      <h2 style={s.heading}>📋 Budget Optimization Plan</h2>
      <p style={s.sub}>Knapsack algorithm · Maximize city-wide cooling within ₹50 Cr budget constraint</p>

      <div style={s.metrics}>
        {[
          ["Total cooling",     `−${totalCooling}°C`, "#10B981"],
          ["Budget used",       `₹${totalCost} Cr`,  "#3B82F6"],
          ["Zones covered",     new Set(plan.map(r => r.zone)).size, "#E5E9F0"],
          ["People protected",  `~${(parseFloat(totalCooling) * 12000).toLocaleString()}`, "#F59E0B"],
        ].map(([k, v, c], i) => (
          <div key={i} style={s.metCard}>
            <div style={{ ...s.metVal, color: c }}>{v}</div>
            <div style={s.metKey}>{k}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Scenario comparison */}
        <div style={s.panel}>
          <div style={s.panelHead}>Scenario comparison</div>
          <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
            {scenarios.map((sc, i) => {
              const cooling = sc["Cooling (°C)"] ?? sc.cooling ?? 0;
              const cost    = sc["Cost (₹ Cr)"]  ?? sc.cost    ?? 0;
              const maxC    = Math.max(...scenarios.map(x => x["Cooling (°C)"] ?? x.cooling ?? 0));
              const colors  = ["#6B7889","#3B82F6","#F59E0B","#10B981"];
              return (
                <div key={i} style={{ marginBottom: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontSize: 13, color: "#E5E9F0" }}>{sc.Scenario ?? sc.scenario}</span>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, color: colors[i], fontWeight: 700 }}>−{cooling}°C · ₹{cost}Cr</span>
                  </div>
                  <div style={s.barTrack}>
                    <div style={{ ...s.barFill, width: `${(cooling / maxC) * 100}%`, background: colors[i] }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Before/After */}
        <div style={s.panel}>
          <div style={s.panelHead}>Before vs After — top 10 zones</div>
          <div style={{ overflowX: "auto", maxHeight: 340, overflowY: "auto" }}>
            <table style={s.table}>
              <thead>
                <tr>
                  {["Zone", "Before", "After", "ΔT"].map(h => (
                    <th key={h} style={s.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...baMap].sort((a, b) => (b.delta_temp ?? 0) - (a.delta_temp ?? 0)).slice(0, 10).map((row, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? "#0E1420" : "transparent" }}>
                    <td style={s.td}>{row[nameKey]}</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace", color: "#EF4444" }}>{(row.before_temp ?? 0).toFixed(1)}°C</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace", color: "#10B981" }}>{(row.after_temp ?? 0).toFixed(1)}°C</td>
                    <td style={{ ...s.td, fontFamily: "'JetBrains Mono',monospace", color: "#10B981", fontWeight: 700 }}>−{(row.delta_temp ?? 0).toFixed(2)}°C</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HVIRanking;

const s = {
  heading: { margin: "0 0 4px", fontSize: 20, fontWeight: 600 },
  sub: { margin: "0 0 20px", fontSize: 13, color: "#6B7889" },
  loading: { color: "#6B7889", padding: 40, textAlign: "center" },
  metrics: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 },
  metCard: { background: "#0E1420", border: "1px solid #1F2533", borderLeft: "2px solid #10B981", borderRadius: 8, padding: "12px 16px" },
  metVal: { fontSize: 22, fontWeight: 600, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.2 },
  metKey: { fontSize: 11, color: "#6B7889", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.05em" },
  panel: { background: "#0E1420", border: "1px solid #1F2533", borderRadius: 10, overflow: "hidden", marginBottom: 16 },
  panelHead: { padding: "12px 16px", borderBottom: "1px solid #1F2533", fontSize: 12, fontWeight: 500, color: "#9AA5B5", fontFamily: "'JetBrains Mono',monospace", textTransform: "uppercase", letterSpacing: "0.05em" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: { textAlign: "left", padding: "8px 12px", color: "#6B7889", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500, borderBottom: "1px solid #1F2533" },
  td: { padding: "8px 12px", color: "#E5E9F0", borderBottom: "1px solid #131822" },
  barTrack: { height: 5, background: "#1F2533", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 3, transition: "width 0.5s" },
};
