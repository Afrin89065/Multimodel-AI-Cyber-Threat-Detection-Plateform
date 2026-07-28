import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../../store";
import { fetchStats, fetchEvents } from "../../store/slices/threatSlice";
import { logout } from "../../store/slices/authSlice";
import { useThreatStream } from "../../hooks/useThreatStream";
import StatCards from "./StatCards";
import ThreatRadar from "./ThreatRadar";
import LiveFeed from "./LiveFeed";
import ModuleScores from "../charts/ModuleScores";
import { Shield, LogOut, RefreshCw } from "lucide-react";
import toast from "react-hot-toast";

export default function Dashboard() {
  const dispatch = useDispatch<AppDispatch>();
  const { username, role } = useSelector((s: RootState) => s.auth);
  const { isConnected } = useThreatStream();

  useEffect(() => {
    dispatch(fetchStats(24));
    dispatch(fetchEvents());

    // Refresh stats every 60 seconds
    const interval = setInterval(() => {
      dispatch(fetchStats(24));
    }, 60000);
    return () => clearInterval(interval);
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchStats(24));
    dispatch(fetchEvents());
    toast.success("Dashboard refreshed");
  };

  return (
    <div className="min-h-screen bg-soc-bg flex flex-col">
      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="bg-soc-panel border-b border-soc-border px-6 py-3
                         flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <Shield size={24} className="text-blue-500" />
          <div>
            <span className="font-bold text-soc-text tracking-wider">AIDTECT</span>
            <span className="text-soc-muted text-xs ml-2">
              Multi-Modal XAI Cyber Threat Detection
            </span>
          </div>
          <span
            className={`ml-4 flex items-center gap-1.5 text-xs px-2 py-1 rounded-full
                        border ${
                          isConnected
                            ? "border-green-800 text-green-400"
                            : "border-red-800 text-red-400"
                        }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
            />
            {isConnected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-soc-muted">
            {username}{" "}
            <span className="text-blue-400 uppercase">[{role}]</span>
          </span>
          <button
            onClick={handleRefresh}
            className="text-soc-muted hover:text-soc-text transition-colors"
            title="Refresh"
          >
            <RefreshCw size={16} />
          </button>
          <button
            onClick={() => dispatch(logout())}
            className="text-soc-muted hover:text-red-400 transition-colors"
            title="Logout"
          >
            <LogOut size={16} />
          </button>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────────── */}
      <main className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
        {/* Left column */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <StatCards />
          <ModuleScores />
          <div className="flex-1 min-h-0">
            <LiveFeed />
          </div>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          <ThreatRadar />
          {/* Threat distribution ring */}
          <ThreatDistribution />
        </div>
      </main>
    </div>
  );
}

function ThreatDistribution() {
  const stats = useSelector((s: RootState) => s.threats.stats);
  if (!stats) return null;

  const items = [
    { label: "CRITICAL", value: stats.critical, color: "#ef4444" },
    { label: "HIGH",     value: stats.high,     color: "#f97316" },
    { label: "MEDIUM",   value: stats.medium,   color: "#eab308" },
    { label: "LOW",      value: stats.low,       color: "#22c55e" },
  ];
  const total = items.reduce((s, i) => s + i.value, 0) || 1;

  return (
    <div className="panel">
      <h2 className="text-sm font-semibold text-soc-muted uppercase tracking-wider mb-3">
        Severity Distribution (24h)
      </h2>
      <div className="space-y-2">
        {items.map(({ label, value, color }) => (
          <div key={label} className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: color }}
            />
            <span className="text-xs text-soc-muted w-16">{label}</span>
            <div className="flex-1 bg-soc-border rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all duration-700"
                style={{
                  width: `${(value / total) * 100}%`,
                  background: color,
                  opacity: 0.8,
                }}
              />
            </div>
            <span className="text-xs font-mono text-soc-text w-8 text-right">
              {value}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 border-t border-soc-border">
        <div className="flex justify-between text-xs text-soc-muted">
          <span>Avg Risk Score</span>
          <span className="font-mono text-soc-text">
            {(stats.avg_risk_score * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}