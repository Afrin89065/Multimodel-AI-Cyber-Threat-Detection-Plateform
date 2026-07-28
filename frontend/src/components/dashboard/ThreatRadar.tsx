import React from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip
} from "recharts";
import { useSelector } from "react-redux";
import { RootState } from "../../store";

const AXES = [
  { key: "nlp",     label: "NLP/Email"  },
  { key: "vision",  label: "Visual"     },
  { key: "network", label: "Network"    },
  { key: "malware", label: "Malware"    },
];

export default function ThreatRadar() {
  const events = useSelector((s: RootState) => s.threats.events);
  const latest = events[0];

  const data = AXES.map(({ key, label }) => ({
    subject: label,
    score: latest
      ? Math.round((latest.module_scores as any)[key] * 100)
      : 0,
    uncertainty: latest?.uncertainties
      ? Math.round((latest.uncertainties as any)[key] * 100)
      : 0,
  }));

  const severityColor =
    latest?.severity === "CRITICAL" ? "#ef4444" :
    latest?.severity === "HIGH"     ? "#f97316" :
    latest?.severity === "MEDIUM"   ? "#eab308" :
    "#22c55e";

  return (
    <div className="panel h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-soc-muted uppercase tracking-wider">
          Module Threat Radar
        </h2>
        {latest && (
          <span className={`severity-badge severity-${latest.severity}`}>
            {latest.severity}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <RadarChart data={data}>
          <PolarGrid stroke="#1e2d4a" />
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fill: "#64748b", fontSize: 11 }}
          />
          <Radar
            name="Score"
            dataKey="score"
            stroke={severityColor}
            fill={severityColor}
            fillOpacity={0.25}
          />
          <Radar
            name="Uncertainty"
            dataKey="uncertainty"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.1}
            strokeDasharray="4 2"
          />
          <Tooltip
            contentStyle={{
              background: "#0f1629",
              border: "1px solid #1e2d4a",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v: any) => [`${v}%`]}
          />
        </RadarChart>
      </ResponsiveContainer>
      <div className="flex gap-4 justify-center mt-1">
        <div className="flex items-center gap-1 text-xs text-soc-muted">
          <span
            className="w-3 h-0.5 inline-block rounded"
            style={{ background: severityColor }}
          />
          Threat Score
        </div>
        <div className="flex items-center gap-1 text-xs text-soc-muted">
          <span className="w-3 h-0.5 inline-block rounded bg-blue-500" />
          Uncertainty
        </div>
      </div>
    </div>
  );
}