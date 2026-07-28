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