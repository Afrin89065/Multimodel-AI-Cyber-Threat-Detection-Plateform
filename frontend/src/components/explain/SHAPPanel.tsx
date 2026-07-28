import React from "react";
import { SHAPFeature } from "../../types";

interface Props {
  features: SHAPFeature[];
  title?: string;
}

export default function SHAPPanel({
  features,
  title = "SHAP Feature Attribution",
}: Props) {
  if (!features || features.length === 0) {
    return (
      <div className="panel text-soc-muted text-sm text-center py-6">
        No SHAP explanation available
      </div>
    );
  }

  const maxMag = Math.max(...features.map((f) => f.magnitude));

  return (
    <div className="panel">
      <h3 className="text-sm font-semibold text-soc-muted uppercase tracking-wider mb-3">
        {title}
      </h3>
      <div className="space-y-2">
        {features.slice(0, 10).map((f, i) => (
          <div key={i} className="grid grid-cols-[1fr_auto] gap-2 items-center">
            <div>
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-mono text-soc-text truncate">
                  {f.feature}
                </span>
                <span
                  className={`text-xs font-mono ml-2 ${
                    f.direction === "threat" || f.direction === "attack"
                      ? "text-red-400"
                      : "text-green-400"
                  }`}
                >
                  {f.shap_value > 0 ? "+" : ""}
                  {f.shap_value.toFixed(3)}
                </span>
              </div>
              <div className="relative h-1.5 bg-soc-border rounded-full overflow-hidden">
                <div
                  className={`absolute top-0 h-full rounded-full transition-all ${
                    f.direction === "threat" || f.direction === "attack"
                      ? "bg-red-500"
                      : "bg-green-500"
                  }`}
                  style={{
                    width: `${(f.magnitude / maxMag) * 100}%`,
                    left: 0,
                  }}
                />
              </div>
            </div>
            <span className="text-xs font-mono text-soc-muted w-16 text-right">
              val={f.feature_value.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}