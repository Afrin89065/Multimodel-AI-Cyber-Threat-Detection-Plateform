import React from "react";
import { Provider } from "react-redux";
import { useSelector } from "react-redux";
import { store, RootState } from "./store";
import Dashboard from "./components/dashboard/Dashboard";
import LoginPage from "./components/auth/LoginPage";
import { Toaster } from "react-hot-toast";

function AppContent() {
  const isAuthenticated = useSelector(
    (s: RootState) => s.auth.isAuthenticated
  );
  return isAuthenticated ? <Dashboard /> : <LoginPage />;
}

export default function App() {
  return (
    <Provider store={store}>
      <AppContent />
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#0f1629",
            color: "#e2e8f0",
            border: "1px solid #1e2d4a",
            fontSize: "13px",
          },
        }}
      />
    </Provider>
  );
}