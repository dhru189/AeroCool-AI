import { useState, useEffect } from "react";

const RISK_COLOR = {
  "🔴 CRITICAL": "#EF4444",
  "🟠 HIGH":     "#F59E0B",
  "🟡 MODERATE": "#EAB308",
  "🟢 LOW":      "#10B981",
};

export default function HeatMap({ api }) {
  const [zones, setZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch(`${api}/zones`)
      .then(r => r.json())
      .then(data => { setZones(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [api]);

  if (loading) return <div style={s.loading}>Loading zone data...</div>;

  const sorted = [...zones].sort((a, b) =>
    (b.pred_temp ?? b.avg_temp ?? 0) - (a.pred_temp ?? a.avg_temp ?? 0)
  );

  const nameKey = zones[0] ? (zones[0].name ? "name" : "location") : "name";
  const tempKey = zones[0] ? (zones[0].pred_temp != null ? "pred_temp" : "avg_temp") : "pred_temp";

  return (
    <div>
      <h2 style={s.heading}>🗺️ Urban Heat Stress Map — Delhi NCR</h2>
      <p style={s.sub}>20 sensor zones · Predicted land surface temperature · NDVI vegetation overlay</p>

      {/* Summary metrics */}
      <div style={s.metrics}>
        {[
          { label: "Zones monitored", value: zones.length },
          { label: "Critical zones", value: zones.filter(z => (z.risk_level ?? "").includes("CRITICAL")).length },
          { label: "Avg city temp", value: `${(zones.reduce((s,z)=>s+(z[tempKey]||0),0)/Math.max(zones.length,1)).toFixed(1)}°C` },
          { label: "Hottest zone", value: sorted[0]?.[nameKey] ?? "—" },
        ].map((m,i) => (
          <div key={i} style={s.metricCard}>
            <div style={s.metricVal}>{m.value}</div>
            <div style={s.metricLabel}>{m.label}</div>
          </div>
        ))}
      </div>

      <div style={s.grid}>
        {/* Zone list */}
        <div style={s.panel}>
          <div style={s.panelHead}>Zone rankings by temperature</div>
          <div style={s.zoneList}>
            {sorted.map((z, i) => {
              const temp = z[tempKey] ?? 0;
              const risk = z.risk_level ?? (temp >= 44 ? "🔴 CRITICAL" : temp >= 41 ? "🟠 HIGH" : temp >= 38 ? "🟡 MODERATE" : "🟢 LOW");
              const color = RISK_COLOR[risk] ?? "#6B7889";
              const isSelected = selected?.[nameKey] === z[nameKey];
              return (
                <div key={i} onClick={() => setSelected(isSelected ? null : z)}
                  style={{ ...s.zoneRow, borderLeft: `3px solid ${color}`, background: isSelected ? "#131822" : "transparent" }}>
                  <span style={{ ...s.rank, color }}>{i + 1}</span>
                  <div style={s.zoneInfo}>
                    <span style={s.zoneName}>{z[nameKey]}</span>
                    <span style={s.zoneTemp}>{temp.toFixed(1)}°C</span>
                  </div>
                  <span style={{ ...s.badge, background: color + "22", color }}>{risk.split(" ").slice(1).join(" ")}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Detail panel */}
        <div style={s.panel}>
          {selected ? (
            <>
              <div style={s.panelHead}>Zone details — {selected[nameKey]}</div>
              <div style={s.detailGrid}>
                {[
                  ["Predicted temp", `${(selected[tempKey]??0).toFixed(1)}°C`],
                  ["NDVI", (selected.ndvi ?? selected.avg_ndvi ?? 0).toFixed(3)],
                  ["Priority score", (selected.priority_score ?? 0).toFixed(2)],
                  ["UHI intensity", `${(selected.uhi ?? selected.avg_uhi ?? 0).toFixed(2)}°C`],
                  ["HVI score", selected.hvi_score ?? "—"],
                  ["Rank", `#${selected.rank ?? "—"}`],
                ].map(([k, v], i) => (
                  <div key={i} style={s.detailRow}>
                    <span style={s.detailKey}>{k}</span>
                    <span style={s.detailVal}>{v}</span>
                  </div>
                ))}
              </div>
              <div style={s.barContainer}>
                <div style={s.barLabel}>Temperature intensity</div>
                <div style={s.barTrack}>
                  <div style={{ ...s.barFill, width: `${Math.min(100, ((selected[tempKey]??35) - 25) / 25 * 100)}%` }} />
                </div>
                <div style={s.barLabel}>NDVI (vegetation)</div>
                <div style={s.barTrack}>
                  <div style={{ ...s.barFillGreen, width: `${(selected.ndvi ?? selected.avg_ndvi ?? 0) * 100}%` }} />
                </div>
              </div>
            </>
          ) : (
            <div style={s.hint}>
              <div style={{ fontSize: 36, marginBottom: 10 }}>👆</div>
              Click any zone in the list to see its detailed breakdown
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const s = {
  heading: { margin: "0 0 4px", fontSize: 20, fontWeight: 600 },
  sub: { margin: "0 0 20px", fontSize: 13, color: "#6B7889" },
  loading: { color: "#6B7889", padding: 40, textAlign: "center" },
  metrics: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 },
  metricCard: { background: "#0E1420", border: "1px solid #1F2533", borderLeft: "2px solid #10B981", borderRadius: 8, padding: "12px 16px" },
  metricVal: { fontSize: 22, fontWeight: 600, color: "#E5E9F0", fontFamily: "'JetBrains Mono',monospace" },
  metricLabel: { fontSize: 11, color: "#6B7889", marginTop: 2, textTransform: "uppercase", letterSpacing: "0.05em" },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
  panel: { background: "#0E1420", border: "1px solid #1F2533", borderRadius: 10, overflow: "hidden" },
  panelHead: { padding: "12px 16px", borderBottom: "1px solid #1F2533", fontSize: 12, fontWeight: 500, color: "#9AA5B5", fontFamily: "'JetBrains Mono',monospace", textTransform: "uppercase", letterSpacing: "0.05em" },
  zoneList: { maxHeight: 440, overflowY: "auto" },
  zoneRow: { display: "flex", alignItems: "center", gap: 12, padding: "10px 14px", cursor: "pointer", borderBottom: "1px solid #0E1420", transition: "background 0.1s" },
  rank: { fontFamily: "'JetBrains Mono',monospace", fontSize: 12, fontWeight: 700, width: 20, flexShrink: 0 },
  zoneInfo: { flex: 1, display: "flex", justifyContent: "space-between", alignItems: "center" },
  zoneName: { fontSize: 13, color: "#E5E9F0" },
  zoneTemp: { fontFamily: "'JetBrains Mono',monospace", fontSize: 13, color: "#E5E9F0", fontWeight: 600 },
  badge: { fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 20 },
  hint: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 300, color: "#3F4A5C", fontSize: 13 },
  detailGrid: { padding: 16, display: "flex", flexDirection: "column", gap: 8 },
  detailRow: { display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #131822" },
  detailKey: { fontSize: 12, color: "#6B7889" },
  detailVal: { fontSize: 13, color: "#E5E9F0", fontFamily: "'JetBrains Mono',monospace", fontWeight: 500 },
  barContainer: { padding: "0 16px 16px" },
  barLabel: { fontSize: 11, color: "#6B7889", marginBottom: 4, marginTop: 10 },
  barTrack: { height: 6, background: "#1F2533", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", background: "#EF4444", borderRadius: 3, transition: "width 0.5s" },
  barFillGreen: { height: "100%", background: "#10B981", borderRadius: 3, transition: "width 0.5s" },
};
