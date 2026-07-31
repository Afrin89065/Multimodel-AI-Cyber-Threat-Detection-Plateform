export interface ModuleScores {
  nlp: number;
  vision: number;
  network: number;
  malware: number;
}

export interface Uncertainties {
  nlp: number;
  vision: number;
  network: number;
  malware: number;
  fusion: number;
  aggregate: number;
}

export interface ThreatEvent {
  id: string;
  request_id?: string;
  timestamp: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  risk_score: number;
  threat_class: string;
  reason: string;
  module_scores: ModuleScores;
  uncertainties?: Uncertainties;
  needs_human_review?: boolean;
  analyst_verdict?: string;
  // v3 FIX: these already exist on the ThreatEvent DB row
  // (backend/models/db_models.py) and are already returned by
  // GET /dashboard/events (SELECT *) — just weren't declared here, so
  // nothing on the frontend could read them.
  network_result?: {
    drift_check?: {
      drift_detected: boolean;
      psi_score: number;
      drifted_features: { feature: string; psi: number }[];
      recommendation: string;
      module: string;
    };
    [key: string]: any;
  };
  nlp_result?: Record<string, any>;
  vision_result?: Record<string, any>;
  malware_result?: Record<string, any>;
  fusion_result?: {
    mitre_tags?: string[];
    mitre_tactics?: string[];
    [key: string]: any;
  };
  shap_values?: Record<string, any> | null;
  counterfactual?: Record<string, any> | null;
  // Present only on live websocket messages (see fusion.py's broadcast),
  // not on rows fetched from /dashboard/events — optional accordingly.
  drift_check?: {
    drift_detected: boolean;
    psi_score: number;
    drifted_features: { feature: string; psi: number }[];
    recommendation: string;
    module: string;
  } | null;
}

export interface DashboardStats {
  period_hours: number;
  total: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  needs_review: number;
  avg_risk_score: number;
}

export interface SHAPFeature {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "threat" | "clean" | "attack" | "benign";
  magnitude: number;
}

export interface CounterfactualChange {
  module: string;
  description: string;
  required_change: number;
  direction: string;
}

export interface AuthState {
  token: string | null;
  username: string | null;
  role: string | null;
  isAuthenticated: boolean;
}