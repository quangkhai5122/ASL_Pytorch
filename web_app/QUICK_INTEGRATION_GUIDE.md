# Quick Integration Guide - Backend + Frontend

## Phase 1: Start Backend (5 minutes)

### Option A: Docker (Recommended)

```bash
cd d:\Python_Project\CV_GISLR\web_app
docker-compose up -d
```

Wait for this message in Docker logs:
```
✓ Model loaded successfully
✓ Listening on 0.0.0.0:8000
```

### Option B: Local Python Environment

```bash
# 1. Create .env file from template
cd d:\Python_Project\CV_GISLR\web_app\backend
copy .env.example .env

# 2. Edit .env and set:
#    DEVICE=cpu  (if no GPU, or cuda if you have GPU)

# 3. Create virtual environment
python -m venv backend_env
backend_env\Scripts\activate

# 4. Install requirements
pip install -r requirements.txt

# 5. Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Verify Backend is Running

```bash
# Should return {"status": "ok", "timestamp": "..."}
curl http://localhost:8000/api/v1/health

# Or open in browser:
# http://localhost:8000/api/v1/docs (interactive API docs)
```

---

## Phase 2: Frontend Integration (10 minutes)

### Step 1: Frontend .env Configuration

```bash
cd d:\Python_Project\CV_GISLR\web_app\frontend
copy .env.example .env.local
```

**Edit `.env.local`:**
```env
VITE_API_URL=http://localhost:8000
VITE_WEBSOCKET_ENABLED=true
VITE_DEBUG=true
VITE_LOG_PREDICTIONS=true
```

### Step 2: Verify Frontend Hooks are Initialized

The frontend already has all the integration hooks in place:
- `src/services/api.ts` - API client
- `src/hooks/useAuth.ts` - Authentication
- `src/hooks/usePredictions.ts` - Frame predictions
- `src/hooks/useWebSocket.ts` - Real-time streaming
- `src/context/ASLContext.tsx` - Unified state

### Step 3: Start Frontend

```bash
cd d:\Python_Project\CV_GISLR\web_app\frontend
npm run dev
# Open http://localhost:5173
```

---

## Phase 3: Wire Up Components (15 minutes)

### Component 1: Login (Already Done ✅)

The `AppLayout.tsx` should already use the ASL context:

```tsx
import { useASL } from '@/context/ASLContext';

export function AppLayout() {
  const { isAuthenticated, login, logout } = useASL();
  
  if (!isAuthenticated) {
    return <LoginPanel />;
  }
  
  return <MainUI />;
}
```

### Component 2: Camera Streaming (CameraCard.tsx)

Update to use WebSocket:

```tsx
import { useASL } from '@/context/ASLContext';

export function CameraCard() {
  const { wsConnected, sendFrameViaWebSocket, wsLastPrediction } = useASL();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!wsConnected) return;
    
    // Capture frame every 200ms
    const interval = setInterval(async () => {
      if (videoRef.current && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        ctx?.drawImage(videoRef.current, 0, 0);
        
        canvasRef.current.toBlob(async (blob) => {
          if (blob) {
            await sendFrameViaWebSocket(blob);
          }
        }, 'image/jpeg', 0.75);
      }
    }, 200);
    
    return () => clearInterval(interval);
  }, [wsConnected, sendFrameViaWebSocket]);
  
  return (
    <div>
      <video ref={videoRef} />
      <canvas ref={canvasRef} style={{display: 'none'}} />
      {wsLastPrediction && <div>{wsLastPrediction.sign}</div>}
    </div>
  );
}
```

### Component 3: Predictions List (PredictionsList.tsx)

```tsx
import { useASL } from '@/context/ASLContext';

export function PredictionsList() {
  const { wsLastPrediction, wsStats } = useASL();
  
  return (
    <div>
      <h3>Real-time Prediction</h3>
      {wsLastPrediction ? (
        <div>
          <p>Sign: <strong>{wsLastPrediction.sign}</strong></p>
          <p>Confidence: {(wsLastPrediction.confidence * 100).toFixed(1)}%</p>
        </div>
      ) : (
        <p>Waiting for prediction...</p>
      )}
      <p>Frames sent: {wsStats?.frames_sent}</p>
      <p>Avg latency: {wsStats?.avg_latency_ms.toFixed(0)}ms</p>
    </div>
  );
}
```

### Component 4: Status Panel (StatusPanel.tsx)

```tsx
import { useASL } from '@/context/ASLContext';

export function StatusPanel() {
  const { backendStats, wsConnected, isAuthenticated } = useASL();
  
  return (
    <div>
      <div>Login Status: {isAuthenticated ? '✅ Connected' : '❌ Not connected'}</div>
      <div>WebSocket: {wsConnected ? '✅ Connected' : '❌ Disconnected'}</div>
      {backendStats && (
        <>
          <div>Model Status: {backendStats.model_loaded ? '✅ Loaded' : '⏳ Loading...'}</div>
          <div>Device: {backendStats.device_info}</div>
          <div>Predictions: {backendStats.predictions_count}</div>
          <div>Avg Latency: {backendStats.avg_latency_ms.toFixed(0)}ms</div>
        </>
      )}
    </div>
  );
}
```

---

## Phase 4: Test Integration

### Test 1: Can you see the login page?
```bash
# Open http://localhost:5173
# Should display login form or already logged in screen
```

### Test 2: Can you login?
```bash
# Click Login button
# Username: testuser
# Password: testpass123
# Should see "Login successful" message
```

### Test 3: Check browser console
```
Open DevTools (F12) → Console tab
Should see:
✅ API client initialized
✅ Auth service initialized
✅ Backend connected
```

### Test 4: Check Network tab
```
DevTools → Network tab
Should see requests to:
- POST /api/v1/auth/login (200 OK)
- GET /api/v1/health (200 OK)
- WS /api/v1/ws/stream (101 Switching Protocols)
```

### Test 5: Test frame capture
```
1. Allow camera permissions
2. Camera should display video
3. WebSocket should show "connected"
4. Should see predictions updating
```

---

## Common Issues & Fixes

### ❌ "Connection refused" on localhost:8000

**Fix:** Backend isn't running
```bash
# Option 1: Docker
docker-compose up -d

# Option 2: Python
cd backend
backend_env\Scripts\activate
python -m uvicorn app.main:app --reload
```

### ❌ "Cannot GET /api/v1/health"

**Fix:** Wrong port or API URL
- Check backend is on 8000: `netstat -ano | findstr :8000`
- Check frontend .env: `VITE_API_URL=http://localhost:8000`

### ❌ Login error or 401

**Fix:** CORS or token issue
- Check CORS_ORIGINS in backend .env
- Verify JWT secret is consistent
- Clear localStorage: DevTools → Application → Clear All

### ❌ WebSocket connects then disconnects

**Fix:** Token expiration or server error
- Check backend logs for errors
- Verify token is being sent (Network tab, WS headers)
- Check model is loaded: `curl http://localhost:8000/api/v1/info`

### ❌ No predictions showing

**Fix:** Camera or model issue
- Check camera permissions in browser
- Verify model is loaded in backend
- Check DevTools Console for errors
- Try the `/api/v1/predict/frame` endpoint directly:

```bash
# Open DevTools Console and run:
fetch('http://localhost:8000/api/v1/predict/frame', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    frame_base64: '...',  // dummy base64
    metadata: {}
  })
}).then(r => r.json()).then(console.log)
```

---

## File Locations Reference

**Backend:**
- Main app: `backend/app/main.py`
- Routes: `backend/app/api/routes/`
- Config: `backend/app/config.py`
- Model: `backend/models/model_best_full_training.pth`

**Frontend:**
- Services: `frontend/src/services/`
  - `api.ts` - HTTP client
  - `auth.ts` - Auth wrapper
- Hooks: `frontend/src/hooks/`
  - `useAuth.ts`
  - `usePredictions.ts`
  - `useWebSocket.ts`
- Context: `frontend/src/context/ASLContext.tsx`
- Components: `frontend/src/components/asl/`

---

## Next Steps After Integration

✅ Basic login + WebSocket working?

1. **Gemini Integration** (Sentence generation)
   - Enable in backend: `ENABLE_GEMINI=true`
   - Add API key: `GEMINI_API_KEY=your-key`
   - Use in component: See FRONTEND_INTEGRATION.md

2. **Video Upload** (Batch processing)
   - Upload video file
   - Backend processes with model
   - Display results

3. **Text-to-Sign** (Reverse translation)
   - Generate sign video from text
   - Requires backend expansion

4. **Persistent Database**
   - Save user predictions
   - History + analytics

---

## Support Resources

- **API Documentation:** http://localhost:8000/api/v1/docs
- **Integration Reference:** [FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md)
- **Example Components:** [BackendIntegrationExamples.tsx](frontend/src/components/asl/BackendIntegrationExamples.tsx)
- **Architecture:** [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)

---

**Ready to start? Run the backend first!**
```bash
cd web_app
docker-compose up -d
# Then in another terminal:
cd web_app/frontend && npm run dev
```
