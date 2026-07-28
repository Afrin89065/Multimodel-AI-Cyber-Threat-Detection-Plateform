import React, { useRef, useEffect } from "react";
import { useSelector } from "react-redux";
import { RootState } from "../../store";
import { ThreatEvent } from "../../types";
import { formatDistanceToNow } from "date-fns";

const SEV_DOT: Record<string, string> = {
  CRITICAL: "bg-red-500",
  HIGH: "bg-orange-500",
  MEDIUM: "bg-yellow-500",
  LOW: "bg-green-500",
};

function EventRow({ event }: { event: ThreatEvent }) {
  return (
    <div
      className={`flex items-start gap-3 py-2.5 px-3 border-b border-soc-border
                  hover:bg-white/5 transition-colors cursor-pointer
                  ${event.needs_human_review ? "border-l-2 border-l-yellow-500" : ""}`}
    >
      {/* Severity dot */}
      <div className="flex-shrink-0 mt-1.5">
        <span
          className={`inline-block w-2 h-2 rounded-full ${SEV_DOT[event.severity] || "bg-gray-500"}`}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`severity-badge severity-${event.severity}`}>
            {event.severity}
          </span>
          <span className="text-xs text-soc-muted font-mono">
            {event.threat_class}
          </span>
          {event.needs_human_review && (
            <span className="text-xs bg-yellow-900/40 text-yellow-400 border border-yellow-800 px-1.5 py-0.5 rounded">
              REVIEW
            </span>
          )}
        </div>
        <p className="text-xs text-soc-text mt-1 truncate" title={event.reason}>
          {event.reason || "No threat detected"}
        </p>
        <div className="flex gap-3 mt-1">
          {Object.entries(event.module_scores || {}).map(([k, v]) => (
            <div key={k} className="flex items-center gap-1">
              <span className="text-xs text-soc-muted uppercase">{k[0]}</span>
              <div className="w-12 bg-soc-border rounded-full h-1">
                <div
                  className="h-1 rounded-full bg-blue-500"
                  style={{ width: `${Math.round((v as number) * 100)}%` }}
                />
              </div>
              <span className="text-xs font-mono text-soc-muted">
                {Math.round((v as number) * 100)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Risk score */}
      <div className="flex-shrink-0 text-right">
        <div className="text-lg font-bold font-mono text-soc-text">
          {Math.round(event.risk_score * 100)}
          <span className="text-xs text-soc-muted">%</span>
        </div>
        <div className="text-xs text-soc-muted">
          {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
        </div>
      </div>
    </div>
  );
}

export default function LiveFeed() {
  const events = useSelector((s: RootState) => s.threats.events);
  const isConnected = useSelector((s: RootState) => s.threats.wsConnected);
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="panel flex flex-col h-full">
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <h2 className="text-sm font-semibold text-soc-muted uppercase tracking-wider">
          Live Threat Feed
        </h2>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-soc-muted">
            {isConnected ? "LIVE" : "RECONNECTING"}
          </span>
          <span className="text-xs text-soc-muted font-mono">
            {events.length} events
          </span>
        </div>
      </div>

      <div ref={containerRef} className="overflow-y-auto flex-1 -mx-4 px-0">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-soc-muted">
            <div className="text-4xl mb-3">🛡️</div>
            <div className="text-sm">Monitoring... No threats detected</div>
          </div>
        ) : (
          events.slice(0, 100).map((event) => (
            <EventRow key={event.id || event.request_id} event={event} />
          ))
        )}
      </div>
    </div>
  );
}