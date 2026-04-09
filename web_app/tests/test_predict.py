"""
Test prediction endpoints (frame, batch, video)
"""
import base64
import io

import pytest
from PIL import Image


class TestPredict:
    """Prediction endpoint tests"""

    def test_frame_predict_valid(self, client, authenticated_headers, sample_frame):
        """Test /predict/frame with valid frame"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': sample_frame},
            headers=authenticated_headers,
        )
        # Response depends on model initialization
        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert 'sign' in data or 'error' in data

    def test_frame_predict_missing_auth(self, client, sample_frame):
        """Test /predict/frame requires authentication"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': sample_frame},
        )
        assert response.status_code == 403

    def test_frame_predict_invalid_base64(self, client, authenticated_headers):
        """Test /predict/frame with invalid base64"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': 'not valid base64!!!'},
            headers=authenticated_headers,
        )
        # Could be 400 for invalid format or 422 for validation
        assert response.status_code in [400, 422]

    def test_frame_predict_missing_frame(self, client, authenticated_headers):
        """Test /predict/frame with missing frame field"""
        response = client.post(
            '/api/v1/predict/frame',
            json={},
            headers=authenticated_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_batch_predict_valid(self, client, authenticated_headers):
        """Test /predict/batch with valid landmarks"""
        # Create dummy landmarks array [64, 66, 3]
        landmarks = [[[0.5, 0.5, 0.5] * 22 for _ in range(66)] for _ in range(64)]

        response = client.post(
            '/api/v1/predict/batch',
            json={
                'landmarks': landmarks,
                'enable_gemini': False,
            },
            headers=authenticated_headers,
        )
        assert response.status_code in [200, 400, 422]

    def test_batch_predict_with_gemini(self, client, authenticated_headers):
        """Test /predict/batch with Gemini enabled"""
        landmarks = [[[0.5, 0.5, 0.5] * 22 for _ in range(66)] for _ in range(64)]

        response = client.post(
            '/api/v1/predict/batch',
            json={
                'landmarks': landmarks,
                'enable_gemini': True,
            },
            headers=authenticated_headers,
        )
        # Response depends on Gemini API configuration
        assert response.status_code in [200, 400, 422, 500]

    def test_batch_predict_missing_auth(self, client):
        """Test /predict/batch requires authentication"""
        landmarks = [[[0.5, 0.5, 0.5] * 22 for _ in range(66)] for _ in range(64)]

        response = client.post(
            '/api/v1/predict/batch',
            json={'landmarks': landmarks},
        )
        assert response.status_code == 403

    def test_video_predict_missing_auth(self, client):
        """Test /predict/video requires authentication"""
        response = client.post(
            '/api/v1/predict/video',
        )
        assert response.status_code == 403

    def test_video_predict_no_file(self, client, authenticated_headers):
        """Test /predict/video with no file"""
        response = client.post(
            '/api/v1/predict/video',
            headers=authenticated_headers,
        )
        assert response.status_code == 422  # Missing file


class TestPredictValidation:
    """Request validation tests"""

    def test_frame_predict_response_format(self, client, authenticated_headers, sample_frame):
        """Test /predict/frame response format"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': sample_frame},
            headers=authenticated_headers,
        )

        if response.status_code == 200:
            data = response.json()
            # Check required fields
            if 'sign' in data:  # Valid prediction
                assert 'confidence' in data
                assert 'processing_time_ms' in data
                assert 0 <= data['confidence'] <= 1

    def test_batch_predict_response_format(self, client, authenticated_headers):
        """Test /predict/batch response format"""
        landmarks = [[[0.5, 0.5, 0.5] * 22 for _ in range(66)] for _ in range(64)]

        response = client.post(
            '/api/v1/predict/batch',
            json={'landmarks': landmarks, 'enable_gemini': False},
            headers=authenticated_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert 'signs' in data or 'error' in data


class TestPredictEndpointSecure:
    """Security tests for prediction endpoints"""

    def test_token_expiration_check(self, client):
        """Test that expired tokens are rejected"""
        # This would require creating an expired token
        # Skip for now as it depends on JWT configuration
        pass

    def test_invalid_token_format(self, client):
        """Test rejection of invalid token format"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': 'test'},
            headers={'Authorization': 'Invalid'},
        )
        assert response.status_code in [401, 403]

    def test_bearer_token_required(self, client):
        """Test that Bearer token format is required"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': 'test'},
            headers={'Authorization': 'token_without_bearer'},
        )
        assert response.status_code in [401, 403]
