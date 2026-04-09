"""
Pytest configuration and fixtures for backend testing
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment
os.environ['ENVIRONMENT'] = 'testing'
os.environ['DEBUG'] = 'true'
os.environ['MODEL_DEVICE'] = 'cpu'  # Always use CPU for tests


@pytest.fixture(scope='session')
def test_app():
    """Create test FastAPI app instance"""
    from app.main import app

    return app


@pytest.fixture(scope='session')
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def test_jwt_token(client):
    """Get valid JWT token for testing"""
    response = client.post(
        '/api/v1/auth/login',
        json={'username': 'testuser', 'password': 'testpass123'},
    )
    assert response.status_code == 200
    return response.json()['access_token']


@pytest.fixture
def authenticated_headers(test_jwt_token):
    """Headers with valid JWT token"""
    return {
        'Authorization': f'Bearer {test_jwt_token}',
        'Content-Type': 'application/json',
    }


@pytest.fixture
def sample_frame():
    """Sample JPEG frame for testing (100x100 black image)"""
    import io

    from PIL import Image

    img = Image.new('RGB', (100, 100), color='black')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    import base64

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture
def sample_video_path():
    """Path to sample video for testing"""
    test_data_dir = Path(__file__).parent / 'data'
    test_data_dir.mkdir(exist_ok=True)

    video_path = test_data_dir / 'test_video.mp4'

    # Only create if it doesn't exist
    if not video_path.exists():
        import subprocess

        # Create a dummy video with ffmpeg (if available)
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-f',
                    'lavfi',
                    '-i',
                    'testsrc=s=640x480:d=1',
                    '-f',
                    'lavfi',
                    '-i',
                    'sine=f=1000:d=1',
                    str(video_path),
                    '-y',
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            # Skip video tests if ffmpeg not available
            pass

    return video_path
