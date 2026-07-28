import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { AuthState } from "../../types";

const initialState: AuthState = {
  token: localStorage.getItem("aidtect_token"),
  username: localStorage.getItem("aidtect_user"),
  role: localStorage.getItem("aidtect_role"),
  isAuthenticated: !!localStorage.getItem("aidtect_token"),
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginSuccess: (
      state,
      action: PayloadAction<{ token: string; username: string; role: string }>
    ) => {
      state.token = action.payload.token;
      state.username = action.payload.username;
      state.role = action.payload.role;
      state.isAuthenticated = true;
      localStorage.setItem("aidtect_token", action.payload.token);
      localStorage.setItem("aidtect_user", action.payload.username);
      localStorage.setItem("aidtect_role", action.payload.role);
    },
    logout: (state) => {
      state.token = null;
      state.username = null;
      state.role = null;
      state.isAuthenticated = false;
      localStorage.removeItem("aidtect_token");
      localStorage.removeItem("aidtect_user");
      localStorage.removeItem("aidtect_role");
    },
  },
});

export const { loginSuccess, logout } = authSlice.actions;
export default authSlice.reducer;