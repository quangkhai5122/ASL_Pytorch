import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import type { ASLMode, FontSize, Theme, BufferToken, SignPrediction } from "@/lib/mockData";
import { useAuth } from "@/hooks/useAuth";
import { usePredictions } from "@/hooks/usePredictions";
import { useWebSocket, type WebSocketMessage } from "@/hooks/useWebSocket";
import type { PredictionResponse } from "@/services/api";
import { apiClient } from "@/services/api";

export interface BackendStats {
  modelLoaded: boolean;
  device: string;
  predictions: number;
  latency: number;
}

type ASLContextType = {
  // Original state
  mode: ASLMode;
  setMode: (m: ASLMode) => void;
  fontSize: FontSize;
  setFontSize: (f: FontSize) => void;
  theme: Theme;
  setTheme: (t: Theme) => void;
  buffer: BufferToken[];
  setBuffer: React.Dispatch<React.SetStateAction<BufferToken[]>>;
  addToBuffer: (token: BufferToken) => void;
  removeFromBuffer: (id: string) => void;
  clearBuffer: () => void;
  undoBuffer: () => void;
  cameraActive: boolean;
  setCameraActive: (v: boolean) => void;
  generatedSentence: string;
  setGeneratedSentence: (s: string) => void;
  status: string;
  setStatus: (s: string) => void;
  settingsOpen: boolean;
  setSettingsOpen: (v: boolean) => void;
  onboardingOpen: boolean;
  setOnboardingOpen: (v: boolean) => void;

  // Backend integration
  isAuthenticated: boolean;
  authLoading: boolean;
  authError: string | null;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  clearAuthError: () => void;

  // Predictions
  lastPrediction: PredictionResponse | null;
  predictionLoading: boolean;
  predictionError: string | null;
  predictFrame: (frameBase64: string) => Promise<void>;
  clearPredictionError: () => void;

  // WebSocket
  wsConnected: boolean;
  wsError: string | null;
  wsLastPrediction: WebSocketMessage | null;
  sendFrameViaWebSocket: (canvas: HTMLCanvasElement) => Promise<boolean>;
  wsStats: { frames: number; predictions: number; latency: number } | null;

  // Backend Stats
  backendStats: BackendStats | null;
  backendLoading: boolean;
};

const ASLContext = createContext<ASLContextType | null>(null);

function getStatusFromWebSocketMessage(message: WebSocketMessage | null): string | null {
  if (!message) return null;

  if (message.type === "connected") {
    return "Connected. Start camera to stream.";
  }

  if (message.type === "frame_received") {
    return "Listening";
  }

  if (message.type === "status") {
    if (message.status === "no_hands" || message.status === "landmark_failed") return "Ready";
    if (message.status === "idle") return "Listening";
    return "Processing";
  }

  if (message.type === "error") {
    return message.error || "Streaming error";
  }

  if (message.type === "prediction" && message.sign) {
    const confidence = message.confidence ?? 0;
    const sign = String(message.sign).toUpperCase();
    if (confidence >= 0.7) return `${sign} (${(confidence * 100).toFixed(0)}%)`;
    if (confidence >= 0.3) return `Maybe: ${sign} (${(confidence * 100).toFixed(0)}%)`;
    if (message.status === "unknown") return "Unknown sign";
    return "Listening";
  }

  return null;
}

export function ASLProvider({ children }: { children: React.ReactNode }) {
  // Original state — start with empty buffer (no mock data)
  const [mode, setMode] = useState<ASLMode>("automatic");
  const [fontSize, setFontSizeState] = useState<FontSize>("a");
  const [theme, setThemeState] = useState<Theme>("light");
  const [buffer, setBuffer] = useState<BufferToken[]>([]);
  const [bufferHistory, setBufferHistory] = useState<BufferToken[][]>([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [generatedSentence, setGeneratedSentence] = useState("");
  const [status, setStatus] = useState("Ready");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(true);
  const [backendStats, setBackendStats] = useState<BackendStats | null>(null);
  const [backendLoading, setBackendLoading] = useState(true);

  // Backend hooks
  const auth = useAuth();
  const predictions = usePredictions();
  const webSocket = useWebSocket(
    auth.isAuthenticated ? apiClient.getToken() : null,
    import.meta.env.VITE_WEBSOCKET_ENABLED !== 'false'
  );

  // Track previous wsLastPrediction to detect commit events
  const prevPredictionRef = useRef<WebSocketMessage | null>(null);

  // Listen for WebSocket prediction messages and handle commit events
  useEffect(() => {
    const pred = webSocket.lastPrediction;
    if (!pred || pred === prevPredictionRef.current) return;
    prevPredictionRef.current = pred;

    const nextStatus = getStatusFromWebSocketMessage(pred);
    if (nextStatus) {
      setStatus(nextStatus);
    }

    // Handle commit events from the backend
    if (pred.commit) {
      const commitData = pred.commit;
      const newToken: BufferToken = {
        id: `commit-${Date.now()}`,
        gloss: commitData.sign.toUpperCase(),
        confidence: commitData.confidence || pred.confidence || 0,
        timestamp: Date.now(),
      };
      setBuffer((prev) => {
        const lastToken = prev[prev.length - 1];
        if (lastToken?.gloss.toUpperCase() === newToken.gloss.toUpperCase()) {
          return prev;
        }
        setBufferHistory((h) => [...h, prev]);
        return [...prev, newToken];
      });
    }
  }, [webSocket.lastPrediction]);

  // Load backend stats on mount
  useEffect(() => {
    const loadBackendStats = async () => {
      try {
        const health = await apiClient.getHealth();
        const metrics = await apiClient.getMetrics();
        setBackendStats({
          modelLoaded: health.model_loaded,
          device: health.device,
          predictions: metrics.predictions_count,
          latency: metrics.avg_latency_ms,
        });
      } catch (error) {
        console.error("Failed to load backend stats:", error);
        setBackendStats({
          modelLoaded: false,
          device: "unknown",
          predictions: 0,
          latency: 0,
        });
      } finally {
        setBackendLoading(false);
      }
    };

    loadBackendStats();

    // Refresh stats every 10 seconds
    const interval = setInterval(loadBackendStats, 10000);
    return () => clearInterval(interval);
  }, []);

  // Original callbacks
  const setFontSize = useCallback((f: FontSize) => {
    setFontSizeState(f);
    document.documentElement.className = document.documentElement.className
      .replace(/font-a(-plus(-plus)?)?/g, "")
      .trim();
    document.documentElement.classList.add(`font-${f}`);
  }, []);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    document.documentElement.classList.remove("high-contrast");
    if (t === "high-contrast") {
      document.documentElement.classList.add("high-contrast");
    }
  }, []);

  const addToBuffer = useCallback((token: BufferToken) => {
    setBuffer(prev => {
      const lastToken = prev[prev.length - 1];
      if (lastToken?.gloss.toUpperCase() === token.gloss.toUpperCase()) {
        return prev;
      }
      setBufferHistory(h => [...h, prev]);
      return [...prev, token];
    });
  }, []);

  const removeFromBuffer = useCallback((id: string) => {
    setBuffer(prev => {
      setBufferHistory(h => [...h, prev]);
      return prev.filter(t => t.id !== id);
    });
  }, []);

  const clearBuffer = useCallback(() => {
    setBuffer(prev => {
      setBufferHistory(h => [...h, prev]);
      return [];
    });
  }, []);

  const undoBuffer = useCallback(() => {
    setBufferHistory(h => {
      if (h.length === 0) return h;
      const prev = h[h.length - 1];
      setBuffer(prev);
      return h.slice(0, -1);
    });
  }, []);

  return (
    <ASLContext.Provider value={{
      // Original
      mode, setMode, fontSize, setFontSize, theme, setTheme,
      buffer, setBuffer, addToBuffer, removeFromBuffer, clearBuffer, undoBuffer,
      cameraActive, setCameraActive,
      generatedSentence, setGeneratedSentence, status, setStatus,
      settingsOpen, setSettingsOpen, onboardingOpen, setOnboardingOpen,

      // Backend integration
      isAuthenticated: auth.isAuthenticated,
      authLoading: auth.isLoading,
      authError: auth.error,
      username: auth.username,
      login: auth.login,
      logout: auth.logout,
      clearAuthError: auth.clearError,

      // Predictions
      lastPrediction: predictions.predictions,
      predictionLoading: predictions.isLoading,
      predictionError: predictions.error,
      predictFrame: predictions.predictFrame,
      clearPredictionError: predictions.clearError,

      // WebSocket
      wsConnected: webSocket.isConnected,
      wsError: webSocket.error,
      wsLastPrediction: webSocket.lastPrediction,
      sendFrameViaWebSocket: webSocket.sendFrameFromCanvas,
      wsStats: webSocket.stats,

      // Backend Stats
      backendStats,
      backendLoading,
    }}>
      {children}
    </ASLContext.Provider>
  );
}

export function useASL() {
  const ctx = useContext(ASLContext);
  if (!ctx) throw new Error("useASL must be used within ASLProvider");
  return ctx;
}
