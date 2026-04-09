# WebSocket Integration Guide for React Frontend

## Overview

This guide explains how to integrate real-time WebSocket streaming with your Lovable React frontend for live sign language recognition.

---

## Architecture

```
┌─────────────────────────────────────┐
│   React Frontend (Lovable)          │
│  - Video Input Stream               │
│  - WebSocket Connection Manager     │
│  - Real-time Prediction Display     │
└────────────────┬────────────────────┘
                 │ HTTP + WebSocket
                 ↓ (JWT token)
┌─────────────────────────────────────┐
│   FastAPI Backend                   │
│  - WebSocket /api/v1/ws/stream      │
│  - Model Inference Service          │
│  - Landmark Extraction              │
└─────────────────────────────────────┘
```

---

## 1. Project Setup

### Install Dependencies

```bash
# For TypeScript/React with WebSocket
npm install ws  # or use browser WebSocket API (recommended)
```

### Environment Configuration

```typescript
// .env.local (development)
VITE_API_BASE_URL=http://localhost:8000

// .env.production
VITE_API_BASE_URL=https://api.your-domain.com
```

---

## 2. Create WebSocket Service

### File: `src/services/websocketService.ts`

```typescript
import { useCallback, useEffect, useRef, useState } from 'react';

interface WebSocketMessage {
  type: 'frame' | 'prediction' | 'heartbeat' | 'error' | 'connected';
  sign?: string;
  confidence?: number;
  top5?: Array<{ sign: string; confidence: number }>;
  frame_id?: number;
  processing_time_ms?: number;
  error?: string;
  [key: string]: any;
}

export class SignLanguageWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private token: string;
  private frameId = 0;
  private onPrediction: ((msg: WebSocketMessage) => void) | null = null;
  private onError: ((msg: WebSocketMessage) => void) | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(apiUrl: string, token: string) {
    // Convert HTTP/HTTPS to WS/WSS
    this.url = apiUrl
      .replace('http://', 'ws://')
      .replace('https://', 'wss://');
    this.token = token;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        const wsUrl = `${this.url}/api/v1/ws/stream?token=${this.token}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('✓ WebSocket connected');
          this.reconnectAttempts = 0;
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          resolve(false);
        };

        this.ws.onclose = () => {
          console.log('WebSocket closed');
          this.attemptReconnect();
        };

        // Timeout after 5 seconds
        setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            resolve(false);
          }
        }, 5000);
      } catch (error) {
        console.error('Connection failed:', error);
        resolve(false);
      }
    });
  }

  /**
   * Send frame for prediction
   */
  async sendFrame(frameData: Blob): Promise<boolean> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return false;
    }

    try {
      // Convert Blob to base64
      const buffer = await frameData.arrayBuffer();
      const frameBase64 = btoa(
        String.fromCharCode.apply(null, new Uint8Array(buffer) as any)
      );

      this.frameId++;

      const message = {
        type: 'frame',
        data: {
          frame_base64: frameBase64,
          frame_id: this.frameId,
        },
      };

      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending frame:', error);
      return false;
    }
  }

  /**
   * Send frame from canvas (video element)
   */
  async sendFrameFromCanvas(canvas: HTMLCanvasElement): Promise<boolean> {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          this.sendFrame(blob).then(resolve);
        } else {
          resolve(false);
        }
      }, 'image/jpeg', 0.9);
    });
  }

  /**
   * Request connection status
   */
  async getStatus(): Promise<any> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return null;
    }

    return new Promise((resolve) => {
      this.ws!.send(JSON.stringify({ type: 'status' }));

      // Wait for status response (simplified - in production use event listeners)
      setTimeout(() => resolve(null), 1000);
    });
  }

  /**
   * Send ping
   */
  async ping(): Promise<boolean> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Disconnect WebSocket
   */
  async disconnect(): Promise<void> {
    if (this.ws) {
      try {
        this.ws.send(JSON.stringify({ type: 'close' }));
      } catch {}
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Set prediction callback
   */
  onPredictionReceived(callback: (msg: WebSocketMessage) => void): void {
    this.onPrediction = callback;
  }

  /**
   * Set error callback
   */
  onErrorReceived(callback: (msg: WebSocketMessage) => void): void {
    this.onError = callback;
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'prediction':
        if (this.onPrediction) {
          this.onPrediction(message);
        }
        break;

      case 'error':
        if (this.onError) {
          this.onError(message);
        }
        break;

      case 'heartbeat':
        console.log(`♥ Heartbeat: ${message.frames_received} frames`);
        break;

      case 'connected':
        console.log('✓ Server welcome:', message.message);
        break;

      default:
        console.log('Message:', message);
    }
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
      console.log(`Reconnecting in ${delay}ms...`);

      setTimeout(() => {
        this.connect();
      }, delay);
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

/**
 * React Hook for WebSocket
 */
export function useWebSocket(apiUrl: string, token: string | null) {
  const clientRef = useRef<SignLanguageWebSocketClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;

    const initWebSocket = async () => {
      clientRef.current = new SignLanguageWebSocketClient(apiUrl, token);

      clientRef.current.onPredictionReceived((msg) => {
        setLastPrediction(msg);
      });

      clientRef.current.onErrorReceived((msg) => {
        setError(msg.error || 'Unknown error');
      });

      const connected = await clientRef.current.connect();
      setIsConnected(connected);
    };

    initWebSocket();

    return () => {
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
    };
  }, [token, apiUrl]);

  const sendFrame = useCallback(
    async (canvas: HTMLCanvasElement) => {
      if (!clientRef.current?.isConnected()) {
        setError('WebSocket not connected');
        return false;
      }
      return clientRef.current.sendFrameFromCanvas(canvas);
    },
    []
  );

  return {
    isConnected,
    lastPrediction,
    error,
    sendFrame,
    client: clientRef.current,
  };
}
```

---

## 3. Create React Component for Real-Time Streaming

### File: `src/components/LivePredictionPanel.tsx`

```typescript
import React, { useRef, useEffect, useState } from 'react';
import { useWebSocket } from '@/services/websocketService';

interface LivePredictionPanelProps {
  apiUrl: string;
  token: string | null;
  isEnabled: boolean;
}

export function LivePredictionPanel({
  apiUrl,
  token,
  isEnabled,
}: LivePredictionPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { isConnected, lastPrediction, error, sendFrame } = useWebSocket(
    apiUrl,
    token
  );
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  // Start video stream
  useEffect(() => {
    if (!isEnabled || !isConnected) return;

    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Camera access denied:', error);
      }
    };

    startVideo();
  }, [isEnabled, isConnected]);

  // Capture and send frames
  useEffect(() => {
    if (!isEnabled || !isConnected || !videoRef.current || !canvasRef.current) {
      return;
    }

    const interval = setInterval(async () => {
      const context = canvasRef.current!.getContext('2d');
      if (context && videoRef.current) {
        // Draw video frame to canvas
        context.drawImage(
          videoRef.current,
          0,
          0,
          canvasRef.current!.width,
          canvasRef.current!.height
        );

        // Send frame
        await sendFrame(canvasRef.current!);

        // Update FPS
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastTimeRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastTimeRef.current = now;
        }
      }
    }, 100); // Send frame every 100ms (~10 FPS)

    return () => clearInterval(interval);
  }, [isEnabled, isConnected, sendFrame]);

  return (
    <div className="live-prediction-panel">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ display: 'none' }}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="preview-canvas"
        />
      </div>

      <div className="prediction-info">
        {!isConnected && (
          <div className="status-message warning">
            Connecting to WebSocket...
          </div>
        )}

        {error && (
          <div className="status-message error">{error}</div>
        )}

        {isConnected && lastPrediction?.type === 'prediction' && (
          <div className="prediction-result">
            <div className="sign-name">{lastPrediction.sign}</div>
            <div className="confidence">
              {(lastPrediction.confidence * 100).toFixed(1)}%
            </div>
            <div className="top5">
              {lastPrediction.top5?.map((pred, idx) => (
                <div key={idx} className="alternative">
                  {pred.sign}: {(pred.confidence * 100).toFixed(1)}%
                </div>
              ))}
            </div>
            <div className="latency">
              {lastPrediction.processing_time_ms?.toFixed(2)}ms
            </div>
          </div>
        )}

        {isConnected && (
          <div className="status-message success">
            ✓ Connected • FPS: {fps}
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## 4. Integration with Existing App

### Update: `src/App.tsx` or Main Component

```typescript
import { useState, useEffect } from 'react';
import { LivePredictionPanel } from '@/components/LivePredictionPanel';

function App() {
  const [token, setToken] = useState<string | null>(null);
  const [streamMode, setStreamMode] = useState<'upload' | 'realtime'>('upload');

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  // Login
  const handleLogin = async (username: string, password: string) => {
    const response = await fetch(`${apiUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (response.ok) {
      const data = await response.json();
      setToken(data.access_token);
      localStorage.setItem('access_token', data.access_token);
    }
  };

  return (
    <div className="app">
      {!token ? (
        <LoginForm onLogin={handleLogin} />
      ) : (
        <>
          <div className="mode-selector">
            <button
              onClick={() => setStreamMode('upload')}
              className={streamMode === 'upload' ? 'active' : ''}
            >
              Upload Video
            </button>
            <button
              onClick={() => setStreamMode('realtime')}
              className={streamMode === 'realtime' ? 'active' : ''}
            >
              Live Stream
            </button>
          </div>

          {streamMode === 'upload' && (
            <VideoUploadPanel apiUrl={apiUrl} token={token} />
          )}

          {streamMode === 'realtime' && (
            <LivePredictionPanel
              apiUrl={apiUrl}
              token={token}
              isEnabled={true}
            />
          )}
        </>
      )}
    </div>
  );
}

export default App;
```

---

## 5. WebSocket Message Protocol

### Client → Server

**Send Frame:**
```json
{
  "type": "frame",
  "data": {
    "frame_base64": "/9j/4AAQSkZJRg...",
    "frame_id": 1,
    "timestamp": "2024-01-15T10:30:45.123Z"
  }
}
```

**Request Status:**
```json
{
  "type": "status"
}
```

**Ping:**
```json
{
  "type": "ping",
  "timestamp": 1705322445123
}
```

**Close Connection:**
```json
{
  "type": "close"
}
```

### Server → Client

**Prediction Result:**
```json
{
  "type": "prediction",
  "sign": "hello",
  "confidence": 0.95,
  "top5": [
    {"sign": "hello", "confidence": 0.95},
    {"sign": "hi", "confidence": 0.03}
  ],
  "frame_id": 1,
  "processing_time_ms": 45.2
}
```

**Heartbeat (Every 30 seconds):**
```json
{
  "type": "heartbeat",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "uptime_seconds": 123.45,
  "frames_received": 45,
  "predictions_made": 10,
  "avg_latency_ms": 42.5
}
```

**Error:**
```json
{
  "type": "error",
  "error": "Invalid frame format",
  "frame_id": 1
}
```

---

## 6. Testing

### Test WebSocket Connection

```typescript
// In browser console
const token = localStorage.getItem('access_token');
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/ws/stream?token=${token}`
);

ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onerror = (e) => console.error(e);

// Send test frame
ws.send(JSON.stringify({
  type: 'frame',
  data: { frame_base64: 'test', frame_id: 1 }
}));
```

### Python Test Client

```bash
pip install websockets
python -m app.utils.websocket_client
```

---

## 7. Performance Optimization

### Frame Rate Control

```typescript
// Send frames at controlled rate (10 FPS)
const FRAME_INTERVAL = 100; // milliseconds

setInterval(() => {
  captureAndSendFrame();
}, FRAME_INTERVAL);
```

### Frame Compression

```typescript
// Use JPEG compression for faster transmission
canvas.toBlob((blob) => {
  // blob is JPEG compressed
}, 'image/jpeg', 0.85); // 85% quality
```

### Network Optimization

```typescript
// Scale down video for faster processing
<video
  ref={videoRef}
  style={{ width: '320px', height: '240px' }}
/>
```

---

## 8. Error Handling

```typescript
// Check connection status before sending
if (!isConnected) {
  console.error('WebSocket not connected');
  // Attempt reconnect
}

// Handle network errors
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  showErrorUI('Connection lost');
};

// Handle disconnection
ws.onclose = () => {
  console.log('Connection closed');
  attemptReconnect();
};
```

---

## 9. Deployment URLs

### Development
```
WS: ws://localhost:8000/api/v1/ws/stream
REST: http://localhost:8000/api/v1
```

### Production (AWS/GCP)
```
WS: wss://api.yourdomain.com/api/v1/ws/stream
REST: https://api.yourdomain.com/api/v1
```

---

## 10. WebSocket Stats Endpoint

Get real-time connection statistics:

```bash
curl http://localhost:8000/api/v1/ws/stats
```

Response:
```json
{
  "active_connections": 3,
  "connections": {
    "user1": {
      "total_frames": 150,
      "total_predictions": 30,
      "avg_latency_ms": 42.5,
      "uptime_seconds": 300
    }
  }
}
```

---

## Summary

1. **Login** to get JWT token
2. **Connect** to WebSocket with token
3. **Stream frames** every ~100ms
4. **Receive predictions** in real-time
5. **Display results** in UI

**Latency**: ~100-150ms per frame (Internet + inference + roundtrip)

For production deployment, ensure:
- ✅ HTTPS/WSS encryption
- ✅ Connection pooling
- ✅ Error recovery
- ✅ Rate limiting
- ✅ Monitoring & logging
