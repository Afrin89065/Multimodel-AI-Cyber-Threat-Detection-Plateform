import React from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store";
import {
  AlertTriangle, ShieldAlert, Eye, Activity
} from "lucide-react";

const CARDS = [
  {
    key: "total",
    label: "Total Events",
    icon: Activity,
    color: "border-blue-800 text-blue-400",
  },
  {
    key: "critical",
    label: "Critical",
    icon: AlertTriangle,
    color: "border-red-800 text-red-400",
  },
  {
    key: "high",
    label: "High",
    icon: ShieldAlert,
    color: "border-orange-800 text-orange-400",
  },
  {
    key: "needs_review",
    label: "Needs Review",
    icon: Eye,
    color: "border-yellow-800 text-yellow-400",
  },
];

export default function StatCards() {
  const stats = useSelector((s: RootState) => s.threats.stats);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {CARDS.map(({ key, label, icon: Icon, color }) => (
        <div
          key={key}
          className={`panel border ${color} flex items-center gap-3`}
        >
          <Icon size={28} className="flex-shrink-0 opacity-80" />
          <div>
            <div className="text-xs text-soc-muted uppercase tracking-wider">
              {label}
            </div>
            <div className="text-3xl font-bold font-mono mt-0.5">
              {stats ? (stats as any)[key] ?? 0 : "—"}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}