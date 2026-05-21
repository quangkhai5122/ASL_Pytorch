# Frontend Development Guide

## Quick Start

### 1. Install & Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Start development server
npm run dev
```

Access at: http://localhost:5173

### 2. Backend Requirements

Start the backend first:

```bash
# From project root
docker-compose up -d

# Wait for model to load (~30-60 seconds)
curl http://localhost:8000/api/v1/health
```

### 3. Environment Configuration

Edit `.env.local`:

```env
VITE_API_URL=http://localhost:8000
VITE_WEBSOCKET_ENABLED=true
VITE_DEBUG=false
VITE_LOG_PREDICTIONS=true
```

---

## Project Structure

```
frontend/src/
├── services/
│   ├── api.ts              # API client (new)
│   └── auth.ts             # Auth service (new)
├── hooks/
│   ├── useAuth.ts          # Auth hook (new)
│   ├── usePredictions.ts   # Predictions hook (new)
│   └── useWebSocket.ts     # WebSocket hook (new)
├── context/
│   └── ASLContext.tsx      # Enhanced with backend (modified)
├── components/
│   ├── asl/
│   │   ├── BackendIntegrationExamples.tsx  # Example components (new)
│   │   ├── CameraCard.tsx
│   │   ├── PredictionsList.tsx
│   │   └── ...
│   └── ui/                 # shadcn/ui components
├── pages/
│   └── Index.tsx           # Main page
└── App.tsx                 # App component
```

---

## Key Features

### 1. Authentication

```typescript
import { useASL } from "@/context/ASLContext";

export function LoginExample() {
  const { isAuthenticated, username, login, logout } = useASL();

  if (!isAuthenticated) {
    return <LoginForm onSubmit={login} />;
  }

  return <p>Welcome, {username}!</p>;
}
```

**Test Credentials:**
- Username: `testuser`
- Password: `testpass123`

### 2. Frame Predictions

```typescript
const { predictFrame, lastPrediction } = useASL();

// Get frame from canvas and predict
const frameBase64 = canvas.toDataURL("image/jpeg").split(",")[1];
await predictFrame(frameBase64);

// Result
console.log(lastPrediction.sign);        // "HELLO"
console.log(lastPrediction.confidence);  // 0.95
```

### 3. Real-Time WebSocket Streaming

```typescript
const { wsConnected, wsLastPrediction, sendFrameViaWebSocket } = useASL();

// Send frame every 100ms
setInterval(() => {
  sendFrameViaWebSocket(canvas);
}, 100);

// Receive predictions
console.log(wsLastPrediction.sign);
```

### 4. Backend Status

```typescript
const { backendStats, isAuthenticated } = useASL();

console.log(backendStats?.modelLoaded);  // true
console.log(backendStats?.device);       // "cuda"
console.log(backendStats?.latency);      // 42.5
```

---

## Building Components

### Example: Prediction Display Component

```typescript
import { useASL } from "@/context/ASLContext";
import { Card } from "@/components/ui/card";

export function PredictionDisplay() {
  const { wsConnected, wsLastPrediction } = useASL();

  return (
    <Card>
      {!wsConnected && (
        <p className="text-yellow-600">Connecting...</p>
      )}

      {wsLastPrediction?.type === 'prediction' && (
        <div>
          <h2 className="text-2xl font-bold">
            {wsLastPrediction.sign}
          </h2>
          <p>
            Confidence: {(wsLastPrediction.confidence * 100).toFixed(1)}%
          </p>
          <p>
            Latency: {wsLastPrediction.processing_time_ms}ms
          </p>
        </div>
      )}
    </Card>
  );
}
```

---

## Integrating with Existing Components

### Update CameraCard Component

```typescript
// src/components/asl/CameraCard.tsx
import { useASL } from "@/context/ASLContext";

export function CameraCard() {
  const { 
    wsConnected, 
    wsLastPrediction, 
    sendFrameViaWebSocket,
    cameraActive,
    setCameraActive 
  } = useASL();

  // ... existing code ...

  // Add WebSocket frame sending in useEffect
  useEffect(() => {
    if (!wsConnected) return;
    
    const interval = setInterval(() => {
      if (canvasRef.current) {
        sendFrameViaWebSocket(canvasRef.current);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [wsConnected, sendFrameViaWebSocket]);

  // ... rest of component ...
}
```

### Update PredictionsList Component

```typescript
// src/components/asl/PredictionsList.tsx
import { useASL } from "@/context/ASLContext";

export function PredictionsList() {
  const { lastPrediction, predictionLoading } = useASL();

  if (predictionLoading) return <p>Loading...</p>;

  if (!lastPrediction) return <p>No predictions yet</p>;

  return (
    <div>
      <h3>{lastPrediction.sign}</h3>
      <p>Confidence: {(lastPrediction.confidence * 100).toFixed(1)}%</p>
      {lastPrediction.top5?.map((pred) => (
        <p key={pred.sign}>
          {pred.sign}: {(pred.confidence * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
}
```

---

## Testing

### Run Tests

```bash
npm run test
npm run test:watch
```

### Manual Testing Checklist

- [ ] Login with `testuser/testpass123`
- [ ] Backend status loads and shows model device
- [ ] Camera can be enabled
- [ ] Frame prediction works
- [ ] WebSocket connects
- [ ] Predictions appear in real-time
- [ ] Logout works

---

## Debugging

### Enable Debug Mode

```env
VITE_DEBUG=true
VITE_LOG_PREDICTIONS=true
```

Open browser console to see:
- API requests
- WebSocket messages
- Frame predictions
- Context state changes

### Check Backend Health

```bash
curl http://localhost:8000/api/v1/health

# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cuda",
#   "version": "1.0.0"
# }
```

### Check Authentication

```bash
# Get token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}' \
  | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/metrics
```

---

## Common Issues

### "Cannot find module" errors

Make sure you created the services and hooks:
- `src/services/api.ts`
- `src/services/auth.ts`
- `src/hooks/useAuth.ts`
- `src/hooks/usePredictions.ts`
- `src/hooks/useWebSocket.ts`

### Backend connection refused

1. Make sure backend is running: `docker-compose ps`
2. Check VITE_API_URL in .env.local
3. Check backend logs: `docker-compose logs backend`

### WebSocket keeps disconnecting

1. Check token is valid: Verify token in localStorage
2. Backend may have restarted: Refresh page to re-login
3. Network issue: Check browser network tab

### Predictions not appearing

1. Ensure authenticated: Check `isAuthenticated` in context
2. Check backend stats: `backendStats.modelLoaded` should be true
3. Check browser console for errors
4. Try manual frame prediction first, then WebSocket

---

## Next Steps

1. **Integrate more components** - Update CameraCard, PredictionsList, etc.
2. **Add Gemini integration** - Call `predictBatch` with `enable_gemini: true`
3. **Video upload** - Use `predictVideo` with file upload
4. **Dictionary search** - Fetch from backend if added
5. **Generate sentences** - Use Gemini sentence generation

For detailed API documentation, see:
- [FRONTEND_INTEGRATION.md](../FRONTEND_INTEGRATION.md) - Full API reference
- [WEBSOCKET_INTEGRATION.md](../WEBSOCKET_INTEGRATION.md) - WebSocket details
- [README.md](../README.md) - Project overview
