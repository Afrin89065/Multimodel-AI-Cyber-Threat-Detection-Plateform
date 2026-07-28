import { useEffect, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import { addLiveEvent, setWsConnected } from "../store/slices/threatSlice";
import { RootState } from "../store";
import toast from "react-hot-toast";

const WS_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8000/api/v1/ws/soc";

export function useThreatStream() {
  const dispatch = useDispatch();
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(1000);
  const mountedRef = useRef(true);
  const isConnected = useSelector((s: RootState) => s.threats.wsConnected);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      retryRef.current = 1000;
      dispatch(setWsConnected(true));
      console.log("SOC WebSocket connected");
    };

    ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        dispatch(addLiveEvent(event));
        // Toast for CRITICAL threats
        if (event.severity === "CRITICAL") {
          toast.error(
            `🚨 CRITICAL: ${event.reason?.slice(0, 60)}`,
            { duration: 8000, id: event.request_id }
          );
        } else if (event.severity === "HIGH") {
          toast(
            `⚠️ HIGH: ${event.reason?.slice(0, 60)}`,
            { duration: 5000, icon: "⚠️", id: event.request_id }
          );
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      dispatch(setWsConnected(false));
      if (mountedRef.current) {
        setTimeout(connect, retryRef.current);
        retryRef.current = Math.min(retryRef.current * 2, 30000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [dispatch]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      wsRef.current?.close();
    };
  }, [connect]);

  return { isConnected };
}