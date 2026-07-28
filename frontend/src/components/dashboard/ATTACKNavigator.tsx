import React, { useMemo } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store";

const TACTICS = [
  { id: "TA0001", name: "Initial\nAccess" },
  { id: "TA0002", name: "Execution" },
  { id: "TA0005", name: "Defense\nEvasion" },
  { id: "TA0006", name: "Credential\nAccess" },
  { id: "TA0007", name: "Discovery" },
  { id: "TA0009", name: "Collection" },
  { id: "TA0010", name: "Exfiltration" },
  { id: "TA0040", name: "Impact" },
  { id: "TA0011", name: "C2" },
  { id: "TA0003", name: "Persistence" },
  { id: "TA0004", name: "Priv.\nEscalation" },
  { id: "TA0008", name: "Lateral\nMovement" },
];

const THREAT_TACTIC_MAP: Record<string, string[]> = {
  PHISHING: ["TA0001"],
  BEC: ["TA0001", "TA0009"],
  MALWARE: ["TA0002", "TA0005"],
  NETWORK_ATTACK: ["TA0040", "TA0006", "TA0007"],
};

export default function ATTACKNavigator() {
  const events = useSelector((s: RootState) => s.threats.events);

  const tacticCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    events.slice(0, 200).forEach((event: any) => {
      (THREAT_TACTIC_MAP[event.threat_class] || []).forEach(
        (t) => { counts[t] = (counts[t] || 0) + 1; }
      );
    });
    return counts;
  }, [events]);

  const maxCount = Math.max(...Object.values(tacticCounts), 1);

  const getColor = (id: string) => {
    const count = tacticCounts[id] || 0;
    if (count === 0) return "rgba(255,255,255,0.05)";
    const intensity = count / maxCount;
    return `rgba(${Math.round(255 * intensity)}, ${Math.round(50 * (1 - intensity))}, 30, ${0.3 + intensity * 0.5})`;
  };

  return (
    <div className="panel">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-soc-muted uppercase tracking-wider">
          ATT&amp;CK Coverage
        </h2>
        <span className="text-xs text-soc-muted">{events.length} events</span>
      </div>
      <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(6, 1fr)" }}>
        {TACTICS.map((tactic) => {
          const count = tacticCounts[tactic.id] || 0;
          return (
            <div key={tactic.id}
              className="rounded p-1.5 text-center cursor-pointer transition-all hover:scale-105"
              style={{
                background: getColor(tactic.id),
                border: `1px solid ${count > 0 ? "rgba(239,68,68,0.4)" : "rgba(255,255,255,0.08)"}`,
                minHeight: 52,
              }}
              title={`${tactic.name.replace("\n", " ")}: ${count} detections`}
            >
              <div style={{ fontSize: 8, color: count > 0 ? "#fca5a5" : "#64748b" }}>
                {tactic.name.split("\n").map((line, i) => <div key={i}>{line}</div>)}
              </div>
              {count > 0 && (
                <div className="font-bold font-mono mt-0.5" style={{ fontSize: 11, color: "#ef4444" }}>
                  {count}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3 mt-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded"
            style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }} />
          <span className="text-xs text-soc-muted">No detections</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ background: "rgba(200,50,30,0.6)" }} />
          <span className="text-xs text-soc-muted">Active</span>
        </div>
      </div>
    </div>
  );
}