# Frontend-Backend Integration Guide

## Overview

Your React frontend is now fully integrated with the FastAPI backend. This guide explains how to use the integration in your components.

---

## 1. Setup & Configuration

### Environment Setup

Copy the environment template and configure for your backend:

```bash
cd frontend
cp .env.example .env.local
```

Edit `.env.local`:

```env
# Development (local backend)
VITE_API_URL=http://localhost:8000
VITE_WEBSOCKET_ENABLED=true
VITE_DEBUG=false
VITE_LOG_PREDICTIONS=true
```

### Start the Backend

```bash
# From project root or backend directory
docker-compose up -d

# Wait for model to load (~30 seconds)
curl http://localhost:8000/api/v1/health
```

### Start the Frontend

```bash
cd frontend
npm install  # If needed
npm run dev
```

Access at: http://localhost:5173

---

## 2. Authentication

The `useAuth` hook manages login and token handling:

```typescript
import { useAuth } from "@/hooks/useAuth";

export function LoginComponent() {
  const { isAuthenticated, login, logout, error, username } = useAuth();

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    try {
      await login(
        formData.get("username") as string,
        formData.get("password") as string
      );
    } catch (err) {
      console.error("Login failed:", err);
    }
  };

  if (isAuthenticated) {
    return (
      <div>
        <p>Welcome, {username}!</p>
        <button onClick={logout}>Logout</button>
      </div>
    );
  }

  return (
    <form onSubmit={handleLogin}>
      <input name="username" placeholder="Username" required />
      <input name="password" type="password" placeholder="Password" required />
      <button type="submit">Login</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </form>
  );
}
```

**Test Credentials (Development):**
```
Username: testuser
Password: testpass123

Username: demo
Password: demo123
```

---

## 3. Frame Predictions

Use the `usePredictions` hook to send frames for prediction:

```typescript
import { usePredictions } from "@/hooks/usePredictions";

export function FramePredictionComponent() {
  const { predictions, isLoading, error, predictFrame } = usePredictions();

  const handleCameraFrame = async (canvas: HTMLCanvasElement) => {
    const frameBase64 = canvas.toDataURL("image/jpeg").split(",")[1];
    await predictFrame(frameBase64);
  };

  return (
    <div>
      <canvas id="camera-canvas" ref={canvasRef} />
      <button onClick={() => handleCameraFrame(canvasRef.current!)}>
        Predict
      </button>

      {isLoading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {predictions && (
        <div>
          <p>Sign: <strong>{predictions.sign}</strong></p>
          <p>Confidence: {(predictions.confidence * 100).toFixed(1)}%</p>
          <p>Latency: {predictions.processing_time_ms}ms</p>
          <ul>
            {predictions.top5.map((pred) => (
              <li key={pred.sign}>
                {pred.sign}: {(pred.confidence * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

## 4. WebSocket Real-Time Streaming

For continuous streaming predictions, use the `useWebSocket` hook:

```typescript
import { useWebSocket } from "@/hooks/useWebSocket";
import { apiClient } from "@/services/api";

export function LiveStreamComponent() {
  const token = apiClient.getToken();
  const { 
    isConnected, 
    lastPrediction, 
    sendFrameFromCanvas, 
    error,
    stats 
  } = useWebSocket(token, true);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      if (canvasRef.current) {
        sendFrameFromCanvas(canvasRef.current);
      }
    }, 100); // Send frame every 100ms

    return () => clearInterval(interval);
  }, [isConnected, sendFrameFromCanvas]);

  return (
    <div>
      <div style={{ 
        color: isConnected ? "green" : "red" 
      }}>
        {isConnected ? "✓ Connected" : "✗ Disconnected"}
      </div>

      <canvas ref={canvasRef} width={640} height={480} />

      {error && <p style={{ color: "red" }}>{error}</p>}

      {lastPrediction?.type === 'prediction' && (
        <div>
          <p>Sign: <strong>{lastPrediction.sign}</strong></p>
          <p>Confidence: {(lastPrediction.confidence * 100).toFixed(1)}%</p>
          <p>Latency: {lastPrediction.processing_time_ms}ms</p>
        </div>
      )}

      {stats && (
        <p>Frames: {stats.frames} | Predictions: {stats.predictions}</p>
      )}
    </div>
  );
}
```

---

## 5. Using the Enhanced ASL Context

The `ASLContext` now includes backend integration. Use it like before:

```typescript
import { useASL } from "@/context/ASLContext";

export function MyComponent() {
  const {
    // Original properties still work
    mode, buffer, generatedSentence,

    // New backend properties
    isAuthenticated, username,
    lastPrediction, predictFrame,
    wsConnected, wsLastPrediction, sendFrameViaWebSocket,
    backendStats,
  } = useASL();

  if (!isAuthenticated) {
    return <LoginPrompt />;
  }

  if (!backendStats?.modelLoaded) {
    return <LoadingModel />;
  }

  return (
    <div>
      <p>Model Device: {backendStats.device}</p>
      <p>Total Predictions: {backendStats.predictions}</p>
      <p>Avg Latency: {backendStats.latency}ms</p>
    </div>
  );
}
```

---

## 6. Integration Examples by Mode

### Automatic Mode (Real-Time Streaming)

```typescript
export function AutomaticModePanel() {
  const { wsConnected, wsLastPrediction, sendFrameViaWebSocket } = useASL();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      });
  }, []);

  useEffect(() => {
    if (!wsConnected) return;

    const interval = setInterval(() => {
      if (canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        ctx?.drawImage(
          videoRef.current,
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );
        sendFrameViaWebSocket(canvasRef.current);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [wsConnected, sendFrameViaWebSocket]);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline style={{ display: "none" }} />
      <canvas ref={canvasRef} width={640} height={480} />

      {wsLastPrediction?.type === 'prediction' && (
        <div>
          <h3>{wsLastPrediction.sign}</h3>
          <p>{(wsLastPrediction.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
```

### Manual Mode (Frame by Frame)

```typescript
export function ManualModePanel() {
  const { lastPrediction, predictFrame } = useASL();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleCapture = async () => {
    if (canvasRef.current) {
      const frameBase64 = canvasRef.current.toDataURL("image/jpeg").split(",")[1];
      await predictFrame(frameBase64);
    }
  };

  return (
    <div>
      <canvas ref={canvasRef} width={640} height={480} />
      <button onClick={handleCapture}>Capture & Predict</button>

      {lastPrediction && (
        <div>
          <h3>{lastPrediction.sign}</h3>
          <p>{(lastPrediction.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
```

---

## 7. Error Handling

Each hook provides error states:

```typescript
export function PredictionWithErrorHandling() {
  const { 
    lastPrediction, 
    predictionError, 
    clearPredictionError 
  } = useASL();

  useEffect(() => {
    if (predictionError) {
      // Show error notification
      console.error("Prediction error:", predictionError);

      // Auto-clear after 5 seconds
      const timeout = setTimeout(clearPredictionError, 5000);
      return () => clearTimeout(timeout);
    }
  }, [predictionError, clearPredictionError]);

  return (
    <div>
      {predictionError && (
        <div style={{ 
          padding: "10px", 
          backgroundColor: "#fee", 
          color: "red", 
          borderRadius: "4px" 
        }}>
          Error: {predictionError}
        </div>
      )}
    </div>
  );
}
```

---

## 8. API Client Usage (Advanced)

For direct API access without hooks:

```typescript
import { apiClient } from "@/services/api";

// Login
const token = await apiClient.login("testuser", "testpass123");

// Get health status
const health = await apiClient.getHealth();

// Get metrics
const metrics = await apiClient.getMetrics();

// Make prediction
const prediction = await apiClient.predictFrame(frameBase64);

// Get WebSocket URL
const wsUrl = apiClient.getWebSocketUrl(token.access_token);

// Batch prediction
const batch = await apiClient.predictBatch(landmarks, true);
```

---

## 9. Import Paths & File Locations

Make sure you have the correct import paths in your components:

```typescript
// Hooks
import { useAuth } from "@/hooks/useAuth";
import { usePredictions } from "@/hooks/usePredictions";
import { useWebSocket } from "@/hooks/useWebSocket";

// Services
import { apiClient } from "@/services/api";
import { authService } from "@/services/auth";

// Context
import { useASL, ASLProvider } from "@/context/ASLContext";
```

---

## 10. Testing the Integration

### 1. Backend Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

### 3. Frame Prediction

```bash
# Get test frame (or use real frame)
curl -X POST http://localhost:8000/api/v1/predict/frame \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"frame":"..."}'
```

### 4. WebSocket Connection

```bash
wscat -c 'ws://localhost:8000/api/v1/ws/stream?token=YOUR_TOKEN'
```

---

## 11. Troubleshooting

### "API connection refused"
- ✅ Backend not running: `docker-compose up -d`
- ✅ Wrong URL in .env.local: `VITE_API_URL=http://localhost:8000`
- ✅ Port conflict: Check if port 8000 is available

### "401 Unauthorized"
- ✅ Token expired: Login again
- ✅ Token not sent: Check Authorization header
- ✅ Invalid credentials: Use testuser/testpass123

### "WebSocket connection failed"
- ✅ JWT token invalid/expired
- ✅ Token not passed in query string
- ✅ Backend WebSocket endpoint not running

### "Model not loaded"
- ✅ Backend still loading: Wait 30-60 seconds
- ✅ GPU issues: Check VITE_LOG_PREDICTIONS and backend logs
- ✅ Check `/health` endpoint for status

---

## 12. Performance Optimization

### Reduce Frame Rate
```typescript
// Send frames every 200ms instead of 100ms
const interval = setInterval(() => {
  sendFrameViaWebSocket(canvas);
}, 200);
```

### Compress Frames
```typescript
// Use lower JPEG quality
canvas.toBlob((blob) => {
  sendFrame(blob);
}, 'image/jpeg', 0.75); // 75% quality
```

### Monitor Performance
```typescript
// Enable debug logging
export VITE_DEBUG=true
export VITE_LOG_PREDICTIONS=true
```

---

## 13. Production Deployment

When deploying to production:

1. **Update environment variables:**
```env
VITE_API_URL=https://api.your-domain.com
VITE_WEBSOCKET_ENABLED=true
VITE_DEBUG=false
```

2. **Build frontend:**
```bash
npm run build
```

3. **Deploy to CDN/Server:**
```bash
# Output is in dist/
deploy dist/ to your web server
```

4. **Add HTTPS/WSS support:**
- Use nginx/Apache with SSL certificate
- Enable WebSocket proxy support

---

## 14. Next Steps

- Integrate predictions into your components
- Add Gemini text generation for sentences
- Implement video upload functionality
- Add dictionary search with backend data
- Deploy to production

For API details, see [WEBSOCKET_INTEGRATION.md](WEBSOCKET_INTEGRATION.md) and [README.md](../README.md)
