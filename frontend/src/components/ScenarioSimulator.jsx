import { useState } from "react";

const DEFAULT = { cool_roof_pct: 30, trees: 3000, misters: 15, green_roof_pct: 10, budget_cr: 50 };

export default function ScenarioSimulator({ api }) {
  const [inputs, setInputs]   = useState(DEFAULT);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  const update = (key, val) => setInputs(p => ({ ...p, [key]: val }));

  const simulate = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${api}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      });
      setResult(await res.json());
    } catch {
      setResult({ error: "API offline — run the backend first" });
    }
    setLoading(false);
  };

  const SLIDERS = [
    { key: "cool_roof_pct",  label: "🏠 Cool Roof Coverage",  unit: "%",   min: 0, max: 100, step: 5  },
    { key: "trees",          label: "🌳 Trees Planted",        unit: "",    min: 0, max: 10000, step: 100 },
    { key: "misters",        label: "💧 Misting Stations",     unit: "",    min: 0, max: 50,   step: 1  },
    { key: "green_roof_pct", label: "🌿 Green Roof Coverage",  unit: "%",   min: 0, max: 50,   step: 5  },
    { key: "budget_cr",      label: "💰 Budget",               unit: "₹ Cr", min: 10, max: 100, step: 5 },
  ];

  return (
    <div>
      <h2 style={s.heading}>🌿 Cooling Scenario Simulator</h2>
      <p style={s.sub}>Adjust interventions and click Simulate to see expected city-wide temperature reduction</p>

      <div style={s.grid}>
        {/* Sliders */}
        <div style={s.panel}>
          <div style={s.panelHead}>Intervention controls</div>
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
            {SLIDERS.map(sl => (
              <div key={sl.key}>
                <div style={s.sliderLabel}>
                  {sl.label}
                  <span style={s.sliderVal}>{inputs[sl.key].toLocaleString()}{sl.unit}</span>
                </div>
                <input type="range"
                  min={sl.min} max={sl.max} step={sl.step}
                  value={inputs[sl.key]}
                  onChange={e => update(sl.key, Number(e.target.value))}
                  style={s.range}
                />
                <div style={s.rangeLabels}>
                  <span>{sl.min}{sl.unit}</span><span>{sl.max}{sl.unit}</span>
                </div>
              </div>
            ))}
            <button onClick={simulate} disabled={loading} style={s.btn}>
              {loading ? "⏳ Simulating..." : "▶ Run Simulation"}
            </button>
          </div>
        </div>

        {/* Results */}
        <div style={s.panel}>
          <div style={s.panelHead}>Simulation output</div>
          {!result && !loading && (
            <div style={s.hint}>Set your interventions and click Run Simulation</div>
          )}
          {loading && <div style={s.hint}>⏳ Running physics model...</div>}
          {result?.error && <div style={s.error}>{result.error}</div>}
          {result?.summary && (
            <div style={{ padding: 20 }}>
              {/* Big cooling number */}
              <div style={s.bigCooling}>
                <span style={s.bigNum}>−{result.summary.total_cooling_C}°C</span>
                <span style={s.bigLabel}>city-wide average cooling</span>
              </div>

              {/* Summary metrics */}
              <div style={s.summaryGrid}>
                {[
                  ["Cost", `₹${result.summary.total_cost_Cr} Cr`, result.summary.within_budget ? "#10B981" : "#EF4444"],
                  ["Within budget", result.summary.within_budget ? "Yes ✓" : "Over budget ✗", result.summary.within_budget ? "#10B981" : "#EF4444"],
                  ["Water saved/day", `${result.summary.water_saved_L_day.toLocaleString()} L`, "#3B82F6"],
                  ["CO₂ avoided/day", `${result.summary.co2_saved_kg_day} kg`, "#10B981"],
                  ["People protected", `~${result.summary.pop_protected.toLocaleString()}`, "#E5E9F0"],
                ].map(([k, v, c], i) => (
                  <div key={i} style={s.summaryRow}>
                    <span style={s.summaryKey}>{k}</span>
                    <span style={{ ...s.summaryVal, color: c ?? "#E5E9F0" }}>{v}</span>
                  </div>
                ))}
              </div>

              {/* Breakdown bars */}
              <div style={s.breakSection}>
                <div style={s.breakHead}>Cooling breakdown</div>
                {Object.entries(result.breakdown).map(([k, v]) => {
                  const total = result.summary.total_cooling_C || 1;
                  const pct = Math.round(v / total * 100);
                  const label = k.replace("_C","").replace(/_/g," ").replace(/\b\w/g,l=>l.toUpperCase());
                  return (
                    <div key={k} style={{ marginBottom: 10 }}>
                      <div style={s.breakLabel}><span>{label}</span><span>−{v}°C · {pct}%</span></div>
                      <div style={s.barTrack}>
                        <div style={{ ...s.barFill, width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
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
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
  panel: { background: "#0E1420", border: "1px solid #1F2533", borderRadius: 10, overflow: "hidden" },
  panelHead: { padding: "12px 16px", borderBottom: "1px solid #1F2533", fontSize: 12, fontWeight: 500, color: "#9AA5B5", fontFamily: "'JetBrains Mono',monospace", textTransform: "uppercase", letterSpacing: "0.05em" },
  sliderLabel: { display: "flex", justifyContent: "space-between", fontSize: 13, color: "#E5E9F0", marginBottom: 8 },
  sliderVal: { fontFamily: "'JetBrains Mono',monospace", color: "#10B981", fontWeight: 600 },
  range: { width: "100%", accentColor: "#10B981", cursor: "pointer" },
  rangeLabels: { display: "flex", justifyContent: "space-between", fontSize: 11, color: "#3F4A5C", marginTop: 2 },
  btn: { width: "100%", padding: "12px", background: "#10B981", color: "#0B0E14", border: "none", borderRadius: 8, fontSize: 14, fontWeight: 700, cursor: "pointer", fontFamily: "inherit", marginTop: 8 },
  hint: { display: "flex", alignItems: "center", justifyContent: "center", minHeight: 300, color: "#3F4A5C", fontSize: 13, padding: 20, textAlign: "center" },
  error: { padding: 20, color: "#EF4444", fontSize: 13 },
  bigCooling: { textAlign: "center", padding: "20px 0", borderBottom: "1px solid #131822", marginBottom: 16 },
  bigNum: { display: "block", fontSize: 48, fontWeight: 700, color: "#10B981", fontFamily: "'JetBrains Mono',monospace", lineHeight: 1 },
  bigLabel: { fontSize: 12, color: "#6B7889", marginTop: 4, display: "block" },
  summaryGrid: { display: "flex", flexDirection: "column", gap: 6, marginBottom: 20 },
  summaryRow: { display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #0E1420" },
  summaryKey: { fontSize: 12, color: "#6B7889" },
  summaryVal: { fontSize: 13, fontFamily: "'JetBrains Mono',monospace", fontWeight: 600 },
  breakSection: { borderTop: "1px solid #131822", paddingTop: 16 },
  breakHead: { fontSize: 11, color: "#6B7889", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 12 },
  breakLabel: { display: "flex", justifyContent: "space-between", fontSize: 12, color: "#9AA5B5", marginBottom: 5 },
  barTrack: { height: 5, background: "#1F2533", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", background: "#10B981", borderRadius: 3, transition: "width 0.5s" },
};
