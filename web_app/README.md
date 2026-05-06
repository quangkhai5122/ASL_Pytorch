# Sign Language Recognition - Web Deployment

Complete web deployment package for ASL/GISLR (Vietnamese Sign Language) recognition system.

## Project Structure

```
web_app/
├── backend/                          # FastAPI backend service
│   ├── app/
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── config.py                # Configuration & environment variables
│   │   ├── dependencies.py          # Dependency injection (JWT, services)
│   │   ├── core/
│   │   │   ├── auth.py             # JWT token generation & validation
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py         # Authentication endpoints
│   │   │   │   ├── health.py       # Health & metrics endpoints
│   │   │   │   └── predict.py      # Prediction endpoints
│   │   │   └── websocket/           # WebSocket handlers (next phase)
│   │   ├── services/
│   │   │   ├── model_inference.py  # Model loading & inference
│   │   │   ├── landmark_extraction.py  # MediaPipe integration
│   │   │   ├── video_processing.py # Video file handling
│   │   │   └── gemini_service.py   # Sentence generation (optional)
│   │   ├── schemas/
│   │   │   ├── request.py          # Pydantic request models
│   │   │   └── response.py         # Pydantic response models
│   │   └── utils/                   # Utility functions
│   ├── models/                      # Pre-trained model weights
│   │   └── model_best_full_training.pth
│   ├── data/                        # Data files
│   │   └── sign_to_prediction_index_map.json
│   ├── scripts/                     # ML scripts (copied from parent)
│   ├── tests/                       # Unit tests
│   ├── Dockerfile                   # Docker image definition
│   ├── requirements.txt             # Python dependencies
│   └── .env.example                 # Environment template
├── frontend/                        # App Sign Language Trans
├── docker-compose.yml               # Local development setup
├── kubernetes/                      # K8s manifests (future)
├── deployment/                      # Deployment scripts
│   ├── aws-deploy.sh               # AWS ECS deployment
│   └── gcp-deploy.sh               # Google Cloud deployment
└── README.md                        # This file
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development without Docker)
- CUDA 12.1 runtime (for GPU support)

### 1. Setup Environment

```bash
cd web_app

# Tao moi truong ao backend
python -m venv backend_env

.\backend_env\Scripts\activate

conda\deactivate

pip install -r backend/requirements.txt
```
### 2. Run server

```bash
cd web_app\backend

..\web_app\backend_env\Scripts\pyxe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

npm run dev
```
### 3. Access API

- **API Docs**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **Health Check**: http://localhost:8000/api/v1/health


## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - Get JWT token
- `POST /api/v1/auth/verify` - Verify token
- `GET /api/v1/auth/test-credentials` - Get test credentials (dev only)

### Prediction
- `POST /api/v1/predict/frame` - Predict from single frame (base64)
- `POST /api/v1/predict/batch` - Batch prediction from landmarks
- `POST /api/v1/predict/video` - Upload video file for processing

### Health & Metrics
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Performance metrics
- `GET /api/v1/info` - API information

## Test Credentials

For development/testing:

```
Username: testuser
Password: testpass123
```

**Remove test credentials endpoint in production!**

## Frontend Integration

The backend is designed to work with the Lovable React frontend:
https://github.com/quangkhai5122/signlanguagetrans

### Connect Frontend to Backend

1. Update frontend environment:
```javascript
// .env.local (development)
VITE_API_BASE_URL=http://localhost:8000

// .env.production
VITE_API_BASE_URL=https://api.your-domain.com
```

2. Create API client in frontend:
```typescript
// src/services/api.ts
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL
});

// Add JWT token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

3. Implement authentication flow:
```typescript
// Login
const response = await apiClient.post('/api/v1/auth/login', {
  username: 'testuser',
  password: 'testpass123'
});
localStorage.setItem('access_token', response.data.access_token);

// Use API with token automatically included
const prediction = await apiClient.post('/api/v1/predict/frame', {
  frame_base64: frameData
});
```

## Docker Deployment

### Build Image

```bash
# Build locally
docker build -f backend/Dockerfile -t signlang-api:latest backend/

# Push to registry
docker tag signlang-api:latest your-registry/signlang-api:latest
docker push your-registry/signlang-api:latest
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DEVICE=cuda \
  -e GEMINI_API_KEY=your-key \
  --gpus all \
  signlang-api:latest
```

## Cloud Deployment

### AWS ECS + GPU

1. Create ECR repository:
```bash
aws ecr create-repository --repository-name signlang-api --region us-east-1
```

2. Build & push image:
```bash
./deployment/aws-deploy.sh
```

3. Create ECS service with GPU support

### Google Cloud Run

```bash
./deployment/gcp-deploy.sh
```

## Testing

### Unit Tests

```bash
pytest tests/ -v
```

### Integration Tests

```bash
# Start backend
docker-compose up -d

# Run integration tests
pytest tests/test_integration.py -v

# Check with sample requests
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

## Performance Metrics

- **Single frame inference**: ~50-100ms (GPU), ~200-300ms (CPU)
- **Batch processing**: ~30-40ms per frame (optimized)
- **Video upload**: ~2-5 seconds for 300-frame video
- **Memory usage**: ~2GB model + ~4GB runtime (GPU)

## Security

### Authentication
- JWT-based stateless authentication
- Tokens expire after 30 minutes (configurable)
- Passwords hashed with bcrypt

### Production Checklist
- [ ] Change `SECRET_KEY` to random 32+ char string
- [ ] Set `DEBUG=false`
- [ ] Enable HTTPS with SSL certificate
- [ ] Use PostgreSQL instead of SQLite
- [ ] Add database username/password
- [ ] Remove test credentials endpoint
- [ ] Configure CORS origins properly
- [ ] Enable rate limiting
- [ ] Add monitoring & logging

## Troubleshooting

### Model not loading
```
Error: Model not found at ./models/model_best_full_training.pth
→ Copy model file: cp models/model_best_full_training.pth web_app/backend/models/
```

### GPU not detected
```
⚠ GPU requested but not available, falling back to CPU
→ Ensure CUDA 12.1 installed: nvidia-smi
→ Or set DEVICE=cpu in .env
```

### Gemini API errors
```
⚠ Failed to initialize Gemini
→ Check GEMINI_API_KEY is set correctly
→ Or disable: ENABLE_GEMINI=false
```

### Import errors
```
ModuleNotFoundError: No module named 'scripts'
→ Ensure scripts/ folder copied to web_app/backend/scripts/
```

## Next Phase: WebSocket

Real-time streaming implementation:
- `WebSocket /api/v1/ws/stream` - Live frame streaming
- JSON message protocol for frame data
- Real-time predictions with sub-100ms latency

## Support

For issues or questions:
1. Check `.env` configuration
2. Review Docker logs: `docker-compose logs backend`
3. Check API docs: http://localhost:8000/api/v1/docs
4. Consult main project: [CV_GISLR](https://github.com/quangkhai5122/GISLR)

## License

Same as parent project
