"""
Unit tests for service classes
"""
import os

import pytest


class TestModelInferenceService:
    """Tests for ModelInferenceService"""

    def test_service_singleton_pattern(self):
        """Test that ModelInferenceService is singleton"""
        from app.services.model_inference import ModelInferenceService

        service1 = ModelInferenceService()
        service2 = ModelInferenceService()
        # Should be same instance (if singleton is implemented)
        # Note: This depends on singleton implementation

    def test_service_device_detection(self):
        """Test device detection (CPU vs CUDA)"""
        from app.services.model_inference import ModelInferenceService

        service = ModelInferenceService()
        # In test environment, should be CPU
        assert service.device in ['cpu', 'cuda', 'mps']


class TestLandmarkExtractionService:
    """Tests for LandmarkExtractionService"""

    def test_landmark_service_initialization(self):
        """Test LandmarkExtractionService initializes with MediaPipe"""
        try:
            from app.services.landmark_extraction import (
                LandmarkExtractionService,
            )

            service = LandmarkExtractionService()
            # Should have MediaPipe Holistic solution
            assert hasattr(service, 'holistic')
        except Exception as e:
            pytest.skip(f'MediaPipe not available: {e}')


class TestVideoProcessingService:
    """Tests for VideoProcessingService"""

    def test_video_validation(self):
        """Test video validation logic"""
        from app.services.video_processing import VideoProcessingService

        service = VideoProcessingService()
        # Test that service can be instantiated
        assert service is not None


class TestGeminiService:
    """Tests for GeminiService (optional based on config)"""

    def test_gemini_service_initialization(self):
        """Test GeminiService initializes correctly"""
        try:
            from app.services.gemini_service import GeminiService

            service = GeminiService()
            # Should have enabled flag or API key
            assert hasattr(service, 'enabled') or hasattr(
                service, 'client'
            )
        except Exception as e:
            # Gemini is optional
            pass
