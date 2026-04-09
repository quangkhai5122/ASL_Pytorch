# Quick Reference Guide - Getting Started

## 🚀 Start Backend (5 minutes)

### Option 1: With Docker (Recommended)

```bash
cd d:\Python_Project\CV_GISLR\web_app

# Start all services
docker-compose up -d

# See logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

**Access Points:**
- API Docs: http://localhost:8000/api/v1/docs ← **Click here to test**
- Health: http://localhost:8000/api/v1/health
- Info: http://localhost:8000/api/v1/info

### Option 2: Local Python (Requires setup)

```bash
cd d:\Python_Project\CV_GISLR\web_app\backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env
copy .env.example .env

# Run server
uvicorn app.main:app --reload --port 8000
```

---

## 🔑 Get API Token

**In Swagger UI (http://localhost:8000/api/v1/docs):**

1. Click "Authorize" button (top right)
2. Use test credentials:
   ```
   Username: testuser
   Password: testpass123
   ```
3. Click "Authorize" and close

**Or with curl:**

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## 🧪 Test Endpoints

### 1. Health Check (No Auth Required)

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Get API Info (No Auth Required)

```bash
curl http://localhost:8000/api/v1/info
```

### 3. Get Metrics (No Auth Required)

```bash
curl http://localhost:8000/api/v1/metrics
```

### 4. Predict from Frame

Upload a JPEG image to get prediction:

```bash
# Convert image to base64 (Windows PowerShell)
$imageData = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes('C:\path\to\image.jpg'))

# Make prediction (need TOKEN from login)
curl -X POST http://localhost:8000/api/v1/predict/frame \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"frame_base64\":\"$imageData\"}"
```

### 5. Test with Video File

```bash
curl -X POST http://localhost:8000/api/v1/predict/video \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/path/to/video.mp4"
```

---

## 📝 Key Files to Know

| File | Purpose |
|------|---------|
| `backend/.env` | Configuration (copy from .env.example) |
| `app/main.py` | FastAPI application entry |
| `app/services/model_inference.py` | Model loading & prediction |
| `app/api/routes/predict.py` | Prediction endpoints |
| `app/config.py` | Settings & constants |
| `docker-compose.yml` | Docker setup |

---

## 🐛 Troubleshooting

### Issue: Model not found
```
Error: Model not found at ./models/model_best_full_training.pth
```

**Solution:**
```bash
# Copy model to web_app
cp models/model_best_full_training.pth web_app/backend/models/
```

### Issue: Port 8000 already in use
```bash
# Change port in docker-compose.yml
# Or kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: GPU not available
```env
# If CUDA not available, Docker will fallback to CPU
# Or force CPU:
DEVICE=cpu
```

### Issue: Module not found

```
ModuleNotFoundError: No module named 'scripts'
```

**Solution:**
```bash
# Make sure scripts folder is in backend/
ls web_app/backend/scripts/
# Should show: config.py, model.py, preprocess.py, utils.py
```

---

## 📊 API Usage Examples

### Frontend (React) Integration

```typescript
// 1. Login
const loginResponse = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'testuser', password: 'testpass123' })
});
const { access_token } = await loginResponse.json();

// 2. Predict from frame
const frameResponse = await fetch('http://localhost:8000/api/v1/predict/frame', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ frame_base64: imageData })
});
const prediction = await frameResponse.json();
console.log(prediction.sign); // "hello"
```

---

## 🔍 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Just backend
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Check Health

```bash
# Quick health check
docker-compose exec backend curl http://localhost:8000/api/v1/health

# Full status
docker-compose exec backend curl http://localhost:8000/api/v1/info
```

### View Container Stats

```bash
docker stats
```

---

## 🧹 Cleanup

```bash
# Stop containers
docker-compose down

# Remove containers AND volumes
docker-compose down -v

# Remove images
docker rmi signlang-api:latest

# Clean build cache
docker builder prune
```

---

## 📚 Next Steps

After starting the backend:

1. **Test with Swagger UI**: http://localhost:8000/api/v1/docs
   - Scroll through endpoints
   - Click "Try it out" on each endpoint
   - See live responses

2. **Test with curl**: Use examples above

3. **Connect React Frontend**:
   - Set `VITE_API_BASE_URL=http://localhost:8000`
   - Implement login & API calls in React
   - Test real-time predictions

4. **Try Batch Predictions**:
   - Create JSON with landmark arrays
   - POST to `/predict/batch`
   - See sentence generation from Gemini

---

## ℹ️ Configuration Notes

### To use Gemini (sentence generation):
```env
ENABLE_GEMINI=true
GEMINI_API_KEY=<your-api-key>
```

### To use specific GPU:
```env
DEVICE=cuda  # or cpu
```

### To increase workers:
```env
WORKERS=8  # For high concurrency
```

### For production:
```env
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=<generate-random-32-chars>
```

See `backend/.env.example` for all 50+ options.

---

**Ready to start?** Run:
```bash
cd d:\Python_Project\CV_GISLR\web_app
docker-compose up -d
echo "Backend starting at http://localhost:8000/api/v1/docs"
```
