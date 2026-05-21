import { useEffect, useRef, useState, useCallback } from 'react';
import { apiClient } from '@/services/api';

export interface WebSocketMessage {
  type: 'frame' | 'frame_received' | 'prediction' | 'heartbeat' | 'error' | 'connected' | 'status' | 'pong';
  sign?: string;
  confidence?: number;
  top5?: Array<{ sign: string; confidence: number }>;
  frame_id?: number;
  processing_time_ms?: number;
  error?: string;
  commit?: { sign: string; confidence: number; word_buffer: string[] };
  motion?: number;
  message?: string;
  frames_in_buffer?: number;
  status?: string;
  landmarks?: (number[] | null)[];  // [543] entries, each [x, y, z] or null
  [key: string]: unknown;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastPrediction: WebSocketMessage | null;
  sendFrame: (frameData: Blob) => Promise<boolean>;
  sendFrameFromCanvas: (canvas: HTMLCanvasElement) => Promise<boolean>;
  connect: () => void;
  disconnect: () => void;
  stats: { frames: number; predictions: number; latency: number } | null;
}

/**
 * Hook for WebSocket real-time streaming predictions.
 * Fixed: uses refs for connection state to avoid reconnect loops.
 */
export function useWebSocket(token: string | null, enabled: boolean = true): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastPrediction, setLastPrediction] = useState<WebSocketMessage | null>(null);
  const [stats, setStats] = useState<{ frames: number; predictions: number; latency: number } | null>(null);
  const frameIdRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Use refs to track connection state without triggering re-renders in callbacks
  const isConnectedRef = useRef(false);
  const isConnectingRef = useRef(false);
  const enabledRef = useRef(enabled);
  const tokenRef = useRef(token);

  // Keep refs in sync
  enabledRef.current = enabled;
  tokenRef.current = token;

  const doConnect = useCallback(() => {
    const currentToken = tokenRef.current;
    const currentEnabled = enabledRef.current;

    if (!currentToken || !currentEnabled) {
      return;
    }

    if (isConnectedRef.current || isConnectingRef.current) {
      return;
    }

    // Cleanup any existing connection
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* Ignore stale socket close errors. */ }
      wsRef.current = null;
    }

    isConnectingRef.current = true;
    setIsConnecting(true);
    setError(null);

    try {
      const wsUrl = apiClient.getWebSocketUrl(currentToken);
      console.log('[WS] Connecting to:', wsUrl.substring(0, 60) + '...');
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connected successfully');
        isConnectedRef.current = true;
        isConnectingRef.current = false;
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          if (import.meta.env.VITE_DEBUG === 'true') {
            console.log('[WS] Message:', message.type, message.sign || '');
          }

          switch (message.type) {
            case 'connected':
              console.log('[WS] Server says:', message.message);
              setLastPrediction(message);
              break;
            case 'prediction':
              setLastPrediction(message);
              setStats((prev) => ({
                frames: (prev?.frames || 0) + 1,
                predictions: (prev?.predictions || 0) + 1,
                latency: message.processing_time_ms ?? prev?.latency ?? 0,
              }));
              break;
            case 'frame_received':
              setLastPrediction(message);
              setStats((prev) => ({
                frames: (prev?.frames || 0) + 1,
                predictions: prev?.predictions || 0,
                latency: message.processing_time_ms ?? prev?.latency ?? 0,
              }));
              break;
            case 'heartbeat':
              console.log('[WS] Heartbeat');
              break;
            case 'error':
              console.warn('[WS] Error:', message.error);
              setError(message.error || 'Unknown error');
              setLastPrediction(message);
              break;
            case 'status':
              console.log('[WS] Status:', message);
              setLastPrediction(message);
              if (message.processing_time_ms !== undefined) {
                setStats((prev) => ({
                  frames: (prev?.frames || 0) + 1,
                  predictions: prev?.predictions || 0,
                  latency: message.processing_time_ms ?? prev?.latency ?? 0,
                }));
              }
              break;
          }
        } catch (err) {
          console.error('[WS] Parse error:', err);
        }
      };

      ws.onerror = (_error) => {
        console.error('[WS] Connection error');
        setError('WebSocket connection error');
      };

      ws.onclose = (event) => {
        console.log(`[WS] Closed (code: ${event.code}, reason: ${event.reason || 'none'})`);
        isConnectedRef.current = false;
        isConnectingRef.current = false;
        wsRef.current = null;
        setIsConnected(false);
        setIsConnecting(false);

        // Auto-reconnect after 3 seconds (only if still enabled)
        if (enabledRef.current && tokenRef.current && !reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            doConnect();
          }, 3000);
        }
      };
    } catch (err) {
      console.error('[WS] Setup error:', err);
      isConnectingRef.current = false;
      setError(err instanceof Error ? err.message : 'Connection failed');
      setIsConnecting(false);
    }
  }, []); // No dependencies — uses refs instead

  const doDisconnect = useCallback(() => {
    // Clear reconnect timer
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      try {
        wsRef.current.close(1000, 'User disconnect');
      } catch { /* Ignore stale socket close errors. */ }
      wsRef.current = null;
    }

    isConnectedRef.current = false;
    isConnectingRef.current = false;
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const sendFrame = useCallback(async (frameData: Blob): Promise<boolean> => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      const buffer = await frameData.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const frameBase64 = btoa(binary);

      frameIdRef.current++;

      wsRef.current.send(
        JSON.stringify({
          type: 'frame',
          data: {
            frame_base64: frameBase64,
            frame_id: frameIdRef.current,
          },
        })
      );

      return true;
    } catch (err) {
      console.error('[WS] Send error:', err);
      return false;
    }
  }, []);

  const sendFrameFromCanvas = useCallback(async (canvas: HTMLCanvasElement): Promise<boolean> => {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          sendFrame(blob).then(resolve);
        } else {
          resolve(false);
        }
      }, 'image/jpeg', 0.8);
    });
  }, [sendFrame]);

  // Connect when token/enabled change
  useEffect(() => {
    if (enabled && token) {
      doConnect();
    } else {
      doDisconnect();
    }

    return () => {
      // Only disconnect on unmount, not on every re-render
    };
  }, [enabled, token, doConnect, doDisconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      doDisconnect();
    };
  }, [doDisconnect]);

  return {
    isConnected,
    isConnecting,
    error,
    lastPrediction,
    sendFrame,
    sendFrameFromCanvas,
    connect: doConnect,
    disconnect: doDisconnect,
    stats,
  };
}
