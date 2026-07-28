import React from "react";
import { CounterfactualChange } from "../../types";
import { ArrowDown, CheckCircle } from "lucide-react";

interface Props {
  counterfactual: {
    original_severity: string;
    target_severity: string;
    counterfactual_found: boolean;
    changes_required: CounterfactualChange[];
    interpretation: string;
  } | null;
}

export default function CounterfactualPanel({ counterfactual: cf }: Props) {
  if (!cf) {
    return (
      <div className="panel text-soc-muted text-sm text-center py-6">
        No counterfactual explanation available
      </div>
    );
  }

  return (
    <div className="panel">
      <h3 className="text-sm font-semibold text-soc-muted uppercase tracking-wider mb-3">
        Counterfactual — What Would Lower This Alert?
      </h3>

      <div className="flex items-center gap-3 mb-4">
        <span className={`severity-badge severity-${cf.original_severity}`}>
          {cf.original_severity}
        </span>
        <ArrowDown size={14} className="text-soc-muted" />
        <span className={`severity-badge severity-${cf.target_severity}`}>
          {cf.target_severity}
        </span>
        {cf.counterfactual_found ? (
          <CheckCircle size={14} className="text-green-500 ml-auto" />
        ) : (
          <span className="text-xs text-red-400 ml-auto">Not found</span>
        )}
      </div>

      {cf.changes_required.length > 0 && (
        <div className="space-y-2 mb-4">
          {cf.changes_required.map((c, i) => (
            <div
              key={i}
              className="flex items-center gap-3 bg-soc-bg rounded-lg px-3 py-2"
            >
              <div className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />
              <div className="flex-1">
                <span className="text-xs text-soc-text">{c.description}</span>
                <span className="text-xs text-red-400 font-mono ml-2">
                  ↓ {Math.abs(c.required_change * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-soc-muted italic leading-relaxed">
        {cf.interpretation}
      </p>
    </div>
  );
}