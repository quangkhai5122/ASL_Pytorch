"""
Test authentication endpoints and JWT functionality
"""
import pytest


class TestAuth:
    """Authentication tests"""

    def test_login_valid_credentials(self, client):
        """Test login with valid credentials"""
        response = client.post(
            '/api/v1/auth/login',
            json={'username': 'testuser', 'password': 'testpass123'},
        )
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        assert data['token_type'] == 'bearer'
        assert data['expires_in'] == 1800  # 30 minutes

    def test_login_invalid_password(self, client):
        """Test login with invalid password"""
        response = client.post(
            '/api/v1/auth/login',
            json={'username': 'testuser', 'password': 'wrongpassword'},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user"""
        response = client.post(
            '/api/v1/auth/login',
            json={'username': 'nonexistent', 'password': 'password'},
        )
        assert response.status_code == 401

    def test_login_missing_fields(self, client):
        """Test login with missing fields"""
        response = client.post(
            '/api/v1/auth/login',
            json={'username': 'testuser'},
        )
        assert response.status_code == 422  # Validation error

    def test_verify_valid_token(self, client, test_jwt_token):
        """Test token verification with valid token"""
        response = client.post(
            '/api/v1/auth/verify',
            headers={'Authorization': f'Bearer {test_jwt_token}'},
        )
        assert response.status_code == 200
        data = response.json()
        assert data['valid'] is True
        assert data['username'] == 'testuser'

    def test_verify_invalid_token(self, client):
        """Test token verification with invalid token"""
        response = client.post(
            '/api/v1/auth/verify',
            headers={'Authorization': 'Bearer invalid_token_here'},
        )
        assert response.status_code == 401

    def test_verify_missing_token(self, client):
        """Test token verification without token"""
        response = client.post('/api/v1/auth/verify')
        assert response.status_code == 403

    def test_demo_user_login(self, client):
        """Test login with demo user"""
        response = client.post(
            '/api/v1/auth/login',
            json={'username': 'demo', 'password': 'demo123'},
        )
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data

    def test_test_credentials_endpoint(self, client):
        """Test /auth/test-credentials endpoint"""
        response = client.get('/api/v1/auth/test-credentials')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data['users'], list)
        assert len(data['users']) >= 2
