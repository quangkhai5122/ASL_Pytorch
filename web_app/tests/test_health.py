"""
Test health check and metrics endpoints
"""
import pytest


class TestHealth:
    """Health check tests"""

    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'device' in data
        assert 'version' in data

    def test_health_model_status(self, client):
        """Test health endpoint returns model status"""
        response = client.get('/api/v1/health')
        data = response.json()
        # Should indicate no user provided (or model ready)
        assert data['model_loaded'] in [True, False]

    def test_health_device_info(self, client):
        """Test health endpoint returns device info"""
        response = client.get('/api/v1/health')
        data = response.json()
        assert data['device'] in ['cuda', 'cpu', 'mps']

    def test_metrics_endpoint(self, client, authenticated_headers):
        """Test /metrics endpoint"""
        response = client.get('/api/v1/metrics', headers=authenticated_headers)
        assert response.status_code == 200
        data = response.json()
        assert 'predictions_count' in data
        assert 'avg_latency_ms' in data

    def test_metrics_initial_values(self, client, authenticated_headers):
        """Test metrics endpoint initial values"""
        response = client.get('/api/v1/metrics', headers=authenticated_headers)
        data = response.json()
        assert data['predictions_count'] >= 0
        assert data['avg_latency_ms'] >= 0

    def test_info_endpoint(self, client, authenticated_headers):
        """Test /info endpoint"""
        response = client.get('/api/v1/info', headers=authenticated_headers)
        assert response.status_code == 200
        data = response.json()
        assert 'api_version' in data
        assert 'model_version' in data
        assert 'environment' in data

    def test_unauthorized_metrics(self, client):
        """Test metrics endpoint requires auth"""
        response = client.get('/api/v1/metrics')
        assert response.status_code == 403

    def test_unauthorized_info(self, client):
        """Test info endpoint requires auth"""
        response = client.get('/api/v1/info')
        assert response.status_code == 403
