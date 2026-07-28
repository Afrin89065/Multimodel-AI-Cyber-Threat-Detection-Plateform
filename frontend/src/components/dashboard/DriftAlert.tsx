import React from "react";
import { AlertTriangle } from "lucide-react";

interface DriftCheck {
  drift_detected: boolean;
  psi_score: number;
  drifted_features: { feature: string; psi: number }[];
  recommendation: string;
  module: string;
}

interface Props {
  drift?: DriftCheck;
}

export default function DriftAlert({ drift }: Props) {
  if (!drift || !drift.drift_detected) return null;

  return (
    <div className="panel border border-yellow-800 bg-yellow-900/20">
      <div className="flex items-start gap-3">
        <AlertTriangle size={20} className="text-yellow-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-yellow-400">
              Data Drift Detected — {drift.module.toUpperCase()} Module
            </span>
            <span className="text-xs bg-yellow-900/60 text-yellow-400 border border-yellow-700 px-2 py-0.5 rounded font-mono">
              PSI={drift.psi_score.toFixed(3)}
            </span>
          </div>
          <p className="text-xs text-soc-muted mt-1">
            Recommendation:{" "}
            <span className="text-yellow-400 font-semibold">
              {drift.recommendation}
            </span>
          </p>
          {drift.drifted_features.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {drift.drifted_features.slice(0, 5).map((f) => (
                <span
                  key={f.feature}
                  className="text-xs bg-soc-bg border border-soc-border px-2 py-0.5 rounded font-mono"
                >
                  {f.feature} ({f.psi.toFixed(2)})
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}