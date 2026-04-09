# Testing Guide for Sign Language Recognition Backend

## Overview

This guide covers unit tests, integration tests, and load testing for the Sign Language Recognition API backend.

**Test Coverage:**
- ✅ Authentication (JWT, login, token verification)
- ✅ Health checks & metrics
- ✅ Prediction endpoints (frame, batch, video)
- ✅ WebSocket connections
- ✅ Error handling & validation
- ✅ Load testing & performance

---

## Setup

### Install Test Dependencies

```bash
# From project root
pip install pytest pytest-asyncio pytest-cov locust
```

### Environment Configuration

Tests automatically use CPU device and test configuration:

```python
# Automatically set by conftest.py
os.environ['ENVIRONMENT'] = 'testing'
os.environ['DEBUG'] = 'true'
os.environ['MODEL_DEVICE'] = 'cpu'
```

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_auth.py

# Run specific test class
pytest tests/test_auth.py::TestAuth

# Run specific test function
pytest tests/test_auth.py::TestAuth::test_login_valid_credentials
```

### Test Markers

```bash
# Run tests by category
pytest -m auth          # Authentication tests only
pytest -m prediction    # Prediction tests only
pytest -m integration   # Integration tests only
pytest -m unit          # Unit tests only

# Run excluding slow tests
pytest -m "not slow"
```

### Coverage Report

```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# Opens htmlcov/index.html in browser
# Shows which lines of code are tested
```

---

## Test Structure

### Fixture Hierarchy

```
conftest.py (Session-level fixtures)
├── test_app           # FastAPI test instance
├── client             # Test client
├── test_jwt_token     # Valid JWT token
├── authenticated_headers  # Headers with JWT
├── sample_frame       # Dummy JPEG frame
└── sample_video_path  # Test video file

test_auth.py          (Uses fixtures)
test_health.py        (Uses fixtures)
test_predict.py       (Uses fixtures)
...
```

### Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `test_auth.py` | 7 tests | Authentication endpoints |
| `test_health.py` | 7 tests | Health & metrics endpoints |
| `test_predict.py` | 11 tests | Prediction endpoints |
| `test_integration.py` | 10 tests | Complete workflows |
| `test_websocket.py` | 6 tests | WebSocket endpoints |
| `test_services.py` | 5 tests | Service layer units |

---

## Running Tests Locally

### Start Backend

```bash
# In one terminal
docker-compose up -d

# Wait for model to load (~30 seconds)
curl http://localhost:8000/api/v1/health
```

### Run Test Suite

```bash
# Terminal 2
cd web_app
pytest -v

# Example output:
# tests/test_auth.py::TestAuth::test_login_valid_credentials PASSED
# tests/test_auth.py::TestAuth::test_login_invalid_password PASSED
# ...
# ======================== 50 passed in 3.42s ========================
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=term-missing

# Shows which lines NOT covered
# Aim for >80% coverage on business logic
```

---

## Load Testing

### Setup Locust

```bash
pip install locust
```

### Run Load Tests

```bash
# Terminal 1: Start backend
docker-compose up -d

# Terminal 2: Start load test
cd web_app
locust -f tests/test_load.py --host=http://localhost:8000

# Opens browser at http://localhost:8089
```

### Load Test Configuration

In browser UI:
- **Number of users**: Start with 10
- **Spawn rate**: 2 users/second
- **Duration**: 5-10 minutes

### Expected Metrics

```
Endpoint          | Requests | P50 (ms) | P95 (ms) | Errors
/api/v1/predict/frame  | 1000    | 120      | 200      | 0%
/api/v1/health         | 500     | 10       | 50       | 0%
/api/v1/metrics        | 300     | 15       | 40       | 0%
```

### Load Test Scenarios

#### Simple Load (Development)
```
Users: 5
Duration: 2 minutes
Distribution: 70% frame prediction, 20% health, 10% metrics
```

#### Medium Load (Staging)
```
Users: 20
Duration: 5 minutes
Distribution: Mix of all endpoints
```

#### Heavy Load (Production-like)
```
Users: 100+
Duration: 10+ minutes
Distribution: Realistic user patterns
Goal: No errors, <200ms P95 latency
```

### Stopping Load Tests

```bash
# In Locust web UI
Click "Stop" button

# Or from terminal
Press Ctrl+C
```

---

## Authentication Tests

### Test Credentials

```
Development:
  Username: testuser
  Password: testpass123

  Username: demo
  Password: demo123
```

### What's Tested

1. ✅ Valid login returns JWT token
2. ✅ Invalid password rejected
3. ✅ Non-existent user rejected
4. ✅ Missing fields validation
5. ✅ Token verification works
6. ✅ Invalid token rejected
7. ✅ Test credentials endpoint

### Run Auth Tests Only

```bash
pytest tests/test_auth.py -v
```

---

## Prediction Endpoint Tests

### What's Tested

**Frame Prediction:**
- ✅ Valid frame processed
- ✅ Requires authentication
- ✅ Invalid base64 rejected
- ✅ Missing frame field rejected
- ✅ Response format validation

**Batch Prediction:**
- ✅ Valid landmarks processed
- ✅ Gemini integration (if enabled)
- ✅ Requires authentication
- ✅ Invalid shape rejected

**Video Prediction:**
- ✅ File upload handling
- ✅ Requires authentication
- ✅ Missing file rejected

### Run Prediction Tests

```bash
pytest tests/test_predict.py -v

# Or test specific aspect
pytest tests/test_predict.py::TestPredict::test_frame_predict_valid -v
```

---

## WebSocket Tests

### Limitations

WebSocket testing with `TestClient` has limitations:
- Can't easily send/receive multiple messages
- Requires async context
- Best tested with Python WebSocket client

### Run WebSocket Tests

```bash
pytest tests/test_websocket.py -v
```

### Manual WebSocket Testing

**Option 1: Python Client**

```bash
python -m app.utils.websocket_client
```

**Option 2: Browser Console**

```javascript
// Get JWT token first
const token = localStorage.getItem('access_token');

// Connect
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/stream?token=${token}`);

// Listen
ws.onmessage = (e) => console.log(JSON.parse(e.data));

// Send frame (in separate script)
ws.send(JSON.stringify({
  type: 'frame',
  data: { frame_base64: 'encoded_jpeg', frame_id: 1 }
}));
```

**Option 3: WebSocket Testing Tool**

```bash
# Install WebSocket client
pip install websockets

# Create test script
python -c "
import asyncio
import websockets
import json
import base64

async def test():
    async with websockets.connect('ws://localhost:8000/api/v1/ws/stream?token=YOUR_TOKEN') as ws:
        # Send frame
        await ws.send(json.dumps({
            'type': 'frame',
            'data': {'frame_base64': 'test', 'frame_id': 1}
        }))
        # Receive response
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(test())
"
```

---

## Integration Tests

### Test Workflows

1. **Authentication Flow**
   - Login → Get token → Use token → Verify access

2. **Prediction Workflow**
   - Login → Make prediction → Check metrics updated

3. **Error Handling**
   - Malformed JSON → Error response
   - Missing fields → 422 validation error
   - Invalid types → Type error response

### Run Integration Tests

```bash
pytest tests/test_integration.py -v
```

---

## Unit Tests

### Service Layer Tests

Tests for individual services:
- ModelInferenceService
- LandmarkExtractionService
- VideoProcessingService
- GeminiService

### Run Service Tests

```bash
pytest tests/test_services.py -v
```

---

## Debugging Failed Tests

### Enable Debug Logging

```bash
# Run with full traceback
pytest --tb=long -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

### Common Issues

**Issue: Model not loading in tests**
```
Solution: Tests use CPU device, so model may not load
         if GPU-only weights. Check app/config.py
```

**Issue: WebSocket tests fail**
```
Solution: WebSocket testing requires async context
         Use Python client instead of TestClient
```

**Issue: Fixtures not found**
```
Solution: Ensure conftest.py is in tests/ directory
         Run pytest from project root: pytest
```

**Issue: Import errors**
```
Solution: Add project root to PYTHONPATH
         export PYTHONPATH="${PWD}:${PYTHONPATH}"
         pytest
```

### Running Specific Failed Tests

```bash
# Get last failed tests
pytest --lf

# Get failed + passed
pytest --ff

# Exit after first failure for quick debugging
pytest -x
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          files: ./coverage.xml
```

---

## Performance Benchmarks

### Expected Performance (on GPU)

```
Operation              | Time (ms) | Max (ms)
-------------------------------------------
Frame prediction       | 45        | 150
Batch prediction (64)  | 200       | 500
Landmark extraction    | 30        | 100
Login                  | 5         | 20
Health check           | 1         | 5
```

### Performance on CPU

Expect 2-5x slower performance on CPU.

### Monitor During Tests

```bash
# In separate terminal (Linux/Mac)
watch -n 1 'nvidia-smi'

# Windows
Get-Process -Name python | foreach{Get-Counter "\Process($($_.Name)*)\% Processor Time"}
```

---

## Test Best Practices

### ✅ Do's

- ✅ Use fixtures for common setup
- ✅ Test both happy and sad paths
- ✅ Test error messages
- ✅ Keep tests isolated
- ✅ Use meaningful test names
- ✅ Group related tests in classes
- ✅ Mock external services (Gemini API)
- ✅ Test with realistic data

### ❌ Don'ts

- ❌ Don't test external APIs (mock them)
- ❌ Don't use hardcoded paths
- ❌ Don't leave test data in production
- ❌ Don't make tests dependent on each other
- ❌ Don't skip important test cases
- ❌ Don't test implementation details

---

## Mocking and Patching

### Mock Model Service

```python
from unittest.mock import patch, MagicMock

@patch('app.services.model_inference.ModelInferenceService.predict')
def test_with_mocked_model(mock_predict):
    mock_predict.return_value = (
        ['hello', 'world'],
        [0.95, 0.92]
    )
    # Test code here
```

### Mock Gemini Service

```python
@patch('app.services.gemini_service.GeminiService.generate_sentence')
def test_without_gemini(mock_gemini):
    mock_gemini.return_value = 'Hello world'
    # Test code here
```

---

## Troubleshooting

### Tests Pass Locally But Fail in CI

**Possible Causes:**
- Different Python version
- Missing GPU in CI environment
- Path differences

**Solution:**
```bash
# Test with exact CI environment
python --version  # Should match CI
pip list          # Should match requirements.txt
pytest --co -q    # List all tests
```

### Memory Leaks in Long Tests

```bash
# Memory profile
pip install memory-profiler
python -m memory_profiler tests/test_predict.py
```

### Slow Tests

```bash
# Find slowest tests
pytest --durations=10

# Skip slow tests
pytest -m "not slow"
```

---

## Summary

**Quick Test Commands:**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run load test
locust -f tests/test_load.py --host=http://localhost:8000

# Run specific test file
pytest tests/test_auth.py

# Run with detailed output
pytest -vv --tb=long
```

**Coverage Goals:**
- Aim for >80% code coverage
- >90% for critical paths (auth, prediction)
- 100% for error handling

For questions or issues, see the main [README.md](README.md)
