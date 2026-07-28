import { createSlice, createAsyncThunk, PayloadAction } from "@reduxjs/toolkit";
import { ThreatEvent, DashboardStats } from "../../types";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

export const fetchStats = createAsyncThunk(
  "threats/fetchStats",
  async (hours: number = 24, { getState }: any) => {
    const token = getState().auth.token;
    const res = await axios.get(`${API}/dashboard/stats?hours=${hours}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return res.data as DashboardStats;
  }
);

export const fetchEvents = createAsyncThunk(
  "threats/fetchEvents",
  async (_, { getState }: any) => {
    const token = getState().auth.token;
    const res = await axios.get(`${API}/dashboard/events?limit=100`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return res.data as ThreatEvent[];
  }
);

interface ThreatState {
  events: ThreatEvent[];
  stats: DashboardStats | null;
  loading: boolean;
  wsConnected: boolean;
}

const initialState: ThreatState = {
  events: [],
  stats: null,
  loading: false,
  wsConnected: false,
};

const threatSlice = createSlice({
  name: "threats",
  initialState,
  reducers: {
    addLiveEvent: (state, action: PayloadAction<ThreatEvent>) => {
      state.events.unshift(action.payload);
      if (state.events.length > 500) state.events.pop();
      // Update stats counters
      if (state.stats) {
        state.stats.total += 1;
        const sev = action.payload.severity.toLowerCase() as keyof DashboardStats;
        if (sev in state.stats) {
          (state.stats as any)[sev] += 1;
        }
        if (action.payload.needs_human_review) {
          state.stats.needs_review += 1;
        }
      }
    },
    setWsConnected: (state, action: PayloadAction<boolean>) => {
      state.wsConnected = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchStats.fulfilled, (state, action) => {
        state.stats = action.payload;
      })
      .addCase(fetchEvents.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchEvents.fulfilled, (state, action) => {
        state.events = action.payload;
        state.loading = false;
      })
      .addCase(fetchEvents.rejected, (state) => {
        state.loading = false;
      });
  },
});

export const { addLiveEvent, setWsConnected } = threatSlice.actions;
export default threatSlice.reducer;