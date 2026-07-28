import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, CartesianGrid
} from "recharts";
import { useSelector } from "react-redux";
import { RootState } from "../../store";

const MODULE_COLORS: Record<string, string> = {
  NLP:     "#3b82f6",
  Vision:  "#8b5cf6",
  Network: "#06b6d4",
  Malware: "#ef4444",
};

export default function ModuleScores() {
  const events = useSelector((s: RootState) => s.threats.events);

  const recentN = Math.min(events.length, 20);
  const recent = events.slice(0, recentN);

  const avgScores = {
    NLP:     0,
    Vision:  0,
    Network: 0,
    Malware: 0,
  };

  if (recentN > 0) {
    for (const e of recent) {
      avgScores.NLP     += e.module_scores.nlp;
      avgScores.Vision  += e.module_scores.vision;
      avgScores.Network += e.module_scores.network;
      avgScores.Malware += e.module_scores.malware;
    }
    Object.keys(avgScores).forEach((k) => {
      avgScores[k as keyof typeof avgScores] =
        Math.round((avgScores[k as keyof typeof avgScores] / recentN) * 100);
    });
  }

  const data = Object.entries(avgScores).map(([name, score]) => ({
    name,
    score,
    fill: MODULE_COLORS[name],
  }));

  return (
    <div className="panel">
      <h2 className="text-sm font-semibold text-soc-muted uppercase tracking-wider mb-3">
        Average Module Scores (Last {recentN} events)
      </h2>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} barSize={32}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2d4a" />
          <XAxis
            dataKey="name"
            tick={{ fill: "#64748b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#64748b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              background: "#0f1629",
              border: "1px solid #1e2d4a",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v: any) => [`${v}%`, "Avg Score"]}
          />
          <Bar dataKey="score" radius={[4, 4, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.fill} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}