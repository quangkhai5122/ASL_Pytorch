"""
Load testing script using Locust
Installation: pip install locust
Usage: locust -f tests/test_load.py --host=http://localhost:8000

This script simulates multiple concurrent users making predictions
"""
import base64
import io
import random
from locust import HttpUser, between, task
from PIL import Image


class SignLanguageUser(HttpUser):
    """Simulates a user interacting with the Sign Language API"""

    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts"""
        # Login and get token
        response = self.client.post(
            '/api/v1/auth/login',
            json={'username': 'testuser', 'password': 'testpass123'},
        )
        if response.status_code == 200:
            self.token = response.json()['access_token']
            self.headers = {'Authorization': f'Bearer {self.token}'}
        else:
            self.token = None
            self.headers = {}

    def create_dummy_frame(self):
        """Create a dummy JPEG frame"""
        img = Image.new('RGB', (640, 480), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @task(3)
    def predict_frame(self):
        """Make frame predictions"""
        if not self.token:
            return

        frame = self.create_dummy_frame()
        self.client.post(
            '/api/v1/predict/frame',
            json={'frame': frame},
            headers=self.headers,
            name='/api/v1/predict/frame',
        )

    @task(1)
    def batch_predict(self):
        """Make batch predictions"""
        if not self.token:
            return

        # Create dummy landmarks [64, 66, 3]
        landmarks = [[[0.5, 0.5, 0.5] * 22 for _ in range(66)] for _ in range(64)]

        self.client.post(
            '/api/v1/predict/batch',
            json={
                'landmarks': landmarks,
                'enable_gemini': False,
            },
            headers=self.headers,
            name='/api/v1/predict/batch',
        )

    @task(2)
    def check_health(self):
        """Check API health"""
        self.client.get('/api/v1/health', name='/api/v1/health')

    @task(1)
    def check_metrics(self):
        """Check API metrics"""
        if not self.token:
            return

        self.client.get(
            '/api/v1/metrics',
            headers=self.headers,
            name='/api/v1/metrics',
        )


class LightUser(HttpUser):
    """Light user that only checks health"""

    wait_time = between(5, 10)

    @task
    def check_health(self):
        """Check API health"""
        self.client.get('/api/v1/health')


# Optional: WebSocket load testing would require a different approach
# See: https://docs.locust.io/en/stable/extending-locust.html#custom-protocols
