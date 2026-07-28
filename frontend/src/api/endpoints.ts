import axios from "axios";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const authHeader = () => ({
  Authorization: `Bearer ${localStorage.getItem("aidtect_token")}`,
});

export const api = {
  // Auth
  login: (username: string, password: string) =>
    axios.post(
      `${BASE}/auth/token`,
      new URLSearchParams({ username, password }),
      { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
    ),

  // Analysis
  analyseNLP: (text: string, url: string, explain = false) =>
    axios.post(
      `${BASE}/analyse/nlp`,
      { text, url, explain },
      { headers: authHeader() }
    ),

  analyseVision: (file: File, explain = false) => {
    const fd = new FormData();
    fd.append("file", file);
    return axios.post(`${BASE}/analyse/vision?explain=${explain}`, fd, {
      headers: { ...authHeader(), "Content-Type": "multipart/form-data" },
    });
  },

  analyseNetwork: (features: number[], computeShap = true) =>
    axios.post(
      `${BASE}/analyse/network`,
      { features, compute_shap: computeShap },
      { headers: authHeader() }
    ),

  analyseMalware: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return axios.post(`${BASE}/analyse/malware`, fd, {
      headers: { ...authHeader(), "Content-Type": "multipart/form-data" },
    });
  },

  analyseFullPipeline: (body: any) =>
    axios.post(`${BASE}/analyse/full`, body, { headers: authHeader() }),

  explainFusion: (body: any) =>
    axios.post(`${BASE}/explain/fusion`, body, { headers: authHeader() }),

  // Dashboard
  getStats: (hours = 24) =>
    axios.get(`${BASE}/dashboard/stats?hours=${hours}`, {
      headers: authHeader(),
    }),

  getEvents: (limit = 50, severity?: string) =>
    axios.get(
      `${BASE}/dashboard/events?limit=${limit}${severity ? `&severity=${severity}` : ""}`,
      { headers: authHeader() }
    ),

  updateVerdict: (eventId: string, verdict: string, notes?: string) =>
    axios.patch(
      `${BASE}/dashboard/events/${eventId}/verdict`,
      null,
      {
        params: { verdict, notes },
        headers: authHeader(),
      }
    ),

  health: () => axios.get(`${BASE}/health`),
};