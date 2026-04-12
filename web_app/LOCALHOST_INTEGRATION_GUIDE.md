# 🎉 Frontend-Backend Integration Guide

## System Status

✅ **Backend Server**: Running on `http://localhost:8000`
✅ **Frontend Server**: Running on `http://localhost:8080`
✅ **Model**: Loaded and ready (20.28 MB, CUDA GPU)
✅ **WebSocket**: Configured and ready for real-time streaming

---

## 🚀 Quick Start

### 1. Open the Application
- Navigate to: **http://localhost:8080/**
- You should see the ASL-Bridge login page

### 2. Login with Test Credentials
**Option 1: Pre-filled Defaults**
- The login page comes pre-filled with:
  - Username: `testuser`
  - Password: `testpass123`
- Click **Login**

**Option 2: Alternative Test Account**
- Username: `demo`
- Password: `demo123`

### 3. Main Interface
After successful login, you'll see:
- **Top Header**: ASL-Bridge branding, username display, help/settings buttons
- **Left Column**: Camera feed with video stream
- **Middle Column**: Real-time predictions and sign recognition
- **Right Column**: Generated sentence and buffer management

---

## 📷 Camera & Recognition Workflow

### Using Automatic Mode (Default)
1. **Start Camera**:
   - Click the camera icon in the Camera Card (Left column)
   - Grant browser permission to access webcam
   - Green light indicates camera is active

2. **Frame Streaming**:
   - Frontend captures frames from your webcam at ~10 FPS
   - Frames are sent automatically via WebSocket to backend
   - Each frame goes through:
     - Landmark extraction (MediaPipe Holistic)
     - Model inference (Pre-trained transformer)
     - Confidence scoring

3. **Real-time Predictions**:
   - Predictions appear in the middle column under "Predictions"
   - Shows:
     - Sign name (recognized gesture)
     - Confidence percentage (0-100%)
     - Top 5 alternative predictions
     - Processing time in milliseconds

4. **Adding Signs to Buffer**:
   - Predictions with high confidence (≥70%) are highlighted
   - Click on a prediction to add it to the word buffer
   - Buffer shows accumulated signs/words

### Backend Recognition Pipeline
```
Camera Frame (640x480)
    ↓
Landmark Extraction (MediaPipe) → 543 hand/body landmarks
    ↓
Preprocessing Layer
    ↓
Transformer Model
    ↓
Sign Prediction + Confidence Score
```

---

## 🔐 Authentication Flow

1. **Login Request** → Backend validates credentials
2. **JWT Token** → Backend returns access token (30 min expiry)
3. **Token Storage** → Frontend stores in localStorage
4. **Auto-Authorization** → Token automatically added to all requests
5. **WebSocket Auth** → Token passed in WebSocket query parameter

**Test Credentials Stored In**: `web_app/backend/app/api/routes/auth.py`
- Can be customized for production use with database

---

## 🌐 API Endpoints (All Available)

### Health & Status
- `GET /api/v1/health` → Model status and device info
- `GET /api/v1/metrics` → Performance metrics
- `GET /api/v1/info` → API version information

### Authentication
- `POST /api/v1/auth/login` → User login
- `POST /api/v1/auth/verify` → Token verification
- `GET /api/v1/auth/test-credentials` → Dev credentials

### Predictions
- `POST /api/v1/predict/frame` → Single frame prediction
- `POST /api/v1/predict/batch` → Batch landmark prediction
- `POST /api/v1/predict/video` → Video file prediction
- `WS /api/v1/ws/stream?token=JWT` → Real-time WebSocket streaming

### Example API Calls

**Login:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

---

## 🔧 Configuration Files

### Frontend (.env)
**Location**: `web_app/frontend/.env`
```env
VITE_API_URL=http://localhost:8000
VITE_API_VERSION=/api/v1
VITE_WEBSOCKET_ENABLED=true
VITE_DEBUG=false
```

### Backend Config
**Location**: `web_app/backend/app/config.py`
- CORS Origins: localhost:3000, localhost:5173, localhost:8080
- Model Path: `./models/model_best_full_training.pth`
- Device: CUDA (GPU) - falls back to CPU if unavailable

---

## 🐛 Troubleshooting

### Backend Server Issues

**Port 8000 Already in Use:**
```bash
# Kill existing process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Model Not Loading:**
- Check `web_app/backend/models/model_best_full_training.pth` exists
- Verify CUDA is available (or falls back to CPU)
- Check console output for specific errors

**CORS Errors:**
- Frontend accessing wrong backend URL
- Verify CORS origins in `web_app/backend/app/config.py`
- Frontend should use `http://localhost:8000`

### Frontend Connection Issues

**Blank Login Page:**
- Clear browser cache: `Ctrl+Shift+Delete`
- Hard refresh: `Ctrl+Shift+R`
- Check console (F12) for JavaScript errors

**WebSocket Connection Failed:**
- Ensure backend is running
- Check that token is valid (not expired)
- Verify browser WebSocket support

**Camera Permission Denied:**
- Check browser camera permissions
- In browsers, sites need HTTPS for camera access in production
- For localhost development, HTTP works fine

### Prediction Issues

**"No Hands Detected":**
- Poor lighting conditions
- Hand too small or far from camera
- Gestures not in training set

**Low Confidence Predictions:**
- Check hand visibility and angle
- Ensure gestures match supported sign vocabulary
- Adjust distance from camera

---

## 📊 Performance Metrics

Monitor real-time performance in the Status Panel:

- **FPS**: Current frame rate (should be ~10 FPS streaming)
- **Latency**: Model inference time (typically 20-100ms GPU)
- **Predictions**: Total predictions processed
- **Device**: GPU (CUDA) or CPU
- **Model**: Loaded status and health

---

## 🎯 Key Features

### ✨ Live Camera Recognition
- Real-time sign language detection
- WebSocket-based low-latency streaming
- Temporal window processing (uses frame history)

### 🔤 Sign Buffer Management
- Accumulate recognized signs
- Edit buffer (remove, reorder)
- Manual addition of signs

### 📝 Text-to-Sign
- Type English text
- View corresponding sign animations
- Dictionary search functionality

### ⚙️ Settings & Customization
- Font size adjustment (a, a+, a++)
- Theme selection (light, dark, high-contrast)
- Toggle skeleton visualization
- Onboarding guide available

---

## 📋 File Structure

```
web_app/
├── frontend/
│   ├── .env                 ← Environment configuration
│   ├── src/
│   │   ├── services/api.ts  ← API client
│   │   ├── hooks/           ← useWebSocket, useAuth, usePredictions
│   │   ├── context/         ← ASLContext
│   │   └── components/asl/  ← UI components
│   └── vite.config.ts       ← Build config
│
├── backend/
│   ├── app/
│   │   ├── main.py          ← FastAPI app
│   │   ├── config.py        ← Configuration
│   │   ├── api/routes/      ← Endpoints
│   │   ├── services/        ← Business logic
│   │   └── core/auth.py     ← Authentication
│   └── models/              ← Pre-trained models
```

---

## 🚀 Next Steps

### For Testing
1. ✅ Verify login works
2. ✅ Test camera capture
3. ✅ Test WebSocket connection
4. ✅ Perform sign gestures and verify predictions
5. ✅ Test buffer management

### For Production Enhancement
1. Replace test credentials with database
2. Implement real user authentication
3. Add HTTPS/WSS for secure connections
4. Deploy to cloud server
5. Implement user profiles and history

---

## 💡 Tips & Best Practices

1. **For Best Recognition**:
   - Ensure good lighting
   - Position hands clearly in frame
   - Use slow, deliberate gestures
   - Maintain distance of 1-2 feet from camera

2. **For Development**:
   - Check browser console (F12) for errors
   - Use network tab to monitor WebSocket messages
   - Enable `VITE_DEBUG=true` for detailed logs

3. **Process Monitoring**:
   - Backend logs print to terminal
   - Frontend logs appear in browser console
   - Both are synchronized for full visibility

---

## 📞 Support

For issues:
1. Check this guide's troubleshooting section
2. Review console logs (backend terminal + browser console)
3. Verify all services are running (ports 8000 and 8080)
4. Ensure test credentials are correct

---

**🎉 You're all set! Open http://localhost:8080 to start!**
