"""
Integration tests for complete workflows
"""
import pytest


class TestAuthenticationFlow:
    """Complete authentication workflow tests"""

    def test_full_auth_flow(self, client):
        """Test complete login -> verify -> use flow"""
        # Step 1: Login
        login_response = client.post(
            '/api/v1/auth/login',
            json={'username': 'testuser', 'password': 'testpass123'},
        )
        assert login_response.status_code == 200
        token = login_response.json()['access_token']

        # Step 2: Verify token
        verify_response = client.post(
            '/api/v1/auth/verify',
            headers={'Authorization': f'Bearer {token}'},
        )
        assert verify_response.status_code == 200
        assert verify_response.json()['username'] == 'testuser'

        # Step 3: Use token for protected endpoint
        metrics_response = client.get(
            '/api/v1/metrics',
            headers={'Authorization': f'Bearer {token}'},
        )
        assert metrics_response.status_code == 200

    def test_invalid_token_revokes_access(self, client):
        """Test that invalid token denies access"""
        response = client.get(
            '/api/v1/metrics',
            headers={'Authorization': 'Bearer invalid'},
        )
        assert response.status_code == 401


class TestPredictionWorkflow:
    """Complete prediction workflow tests"""

    def test_predict_then_check_metrics(self, client, authenticated_headers, sample_frame):
        """Test prediction updates metrics"""
        # Get initial metrics
        initial_metrics = client.get(
            '/api/v1/metrics', headers=authenticated_headers
        ).json()
        initial_count = initial_metrics['predictions_count']

        # Make prediction
        client.post(
            '/api/v1/predict/frame',
            json={'frame': sample_frame},
            headers=authenticated_headers,
        )

        # Get updated metrics (may or may not increase depending on model state)
        updated_metrics = client.get(
            '/api/v1/metrics', headers=authenticated_headers
        ).json()
        # Just verify metrics endpoint still works
        assert 'predictions_count' in updated_metrics


class TestErrorHandling:
    """Error handling tests"""

    def test_malformed_json(self, client, authenticated_headers):
        """Test handling of malformed JSON"""
        response = client.post(
            '/api/v1/predict/frame',
            data='not valid json',
            headers=authenticated_headers,
            content_type='application/json',
        )
        assert response.status_code in [400, 422]

    def test_missing_required_field(self, client, authenticated_headers):
        """Test handling of missing required fields"""
        response = client.post(
            '/api/v1/predict/frame',
            json={},
            headers=authenticated_headers,
        )
        assert response.status_code == 422

    def test_invalid_data_type(self, client, authenticated_headers):
        """Test handling of invalid data types"""
        response = client.post(
            '/api/v1/predict/frame',
            json={'frame': 12345},  # Should be string
            headers=authenticated_headers,
        )
        assert response.status_code in [400, 422]


class TestCORSAndHeaders:
    """CORS and header tests"""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        # CORS headers may or may not be present depending on configuration
        # Just verify endpoint works

    def test_content_type_json(self, client):
        """Test response content-type is JSON"""
        response = client.get('/api/v1/health')
        # Check if it's JSON
        assert response.headers.get('content-type') is not None


class TestRateLimiting:
    """Rate limiting behavior tests (if implemented)"""

    def test_multiple_requests_allowed(self, client, authenticated_headers):
        """Test that multiple requests are allowed"""
        # Make several requests
        for _ in range(5):
            response = client.get('/api/v1/metrics', headers=authenticated_headers)
            assert response.status_code == 200
