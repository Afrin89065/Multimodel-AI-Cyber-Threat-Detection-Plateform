import { configureStore } from "@reduxjs/toolkit";
import threatReducer from "./slices/threatSlice";
import authReducer from "./slices/authSlice";

export const store = configureStore({
  reducer: {
    threats: threatReducer,
    auth: authReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;