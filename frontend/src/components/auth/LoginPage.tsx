import React, { useState } from "react";
import { useDispatch } from "react-redux";
import { loginSuccess } from "../../store/slices/authSlice";
import { api } from "../../api/endpoints";
import toast from "react-hot-toast";
import { Shield } from "lucide-react";

export default function LoginPage() {
  const dispatch = useDispatch();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await api.login(username, password);
      dispatch(
        loginSuccess({
          token: res.data.access_token,
          username: res.data.username,
          role: res.data.role,
        })
      );
      toast.success("Logged in successfully");
    } catch {
      toast.error("Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-soc-bg flex items-center justify-center">
      <div className="panel w-full max-w-sm">
        <div className="flex flex-col items-center mb-8">
          <Shield className="text-blue-500 mb-2" size={48} />
          <h1 className="text-2xl font-bold tracking-wider text-soc-text">
            AIDTECT
          </h1>
          <p className="text-soc-muted text-sm mt-1">
            SOC Analyst Login
          </p>
        </div>
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-xs text-soc-muted mb-1 uppercase tracking-wider">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-soc-bg border border-soc-border rounded-lg px-3 py-2
                         text-soc-text focus:outline-none focus:border-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-soc-muted mb-1 uppercase tracking-wider">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-soc-bg border border-soc-border rounded-lg px-3 py-2
                         text-soc-text focus:outline-none focus:border-blue-500"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50
                       text-white font-semibold py-2 rounded-lg transition-colors"
          >
            {loading ? "Authenticating..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}