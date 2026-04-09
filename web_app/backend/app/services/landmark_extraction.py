"""
Landmark Extraction Service using MediaPipe Holistic.
Extracts 543 landmarks (face, hands, body) from video frames.

GISLR ordering: Face(468) → Left Hand(21) → Pose(33) → Right Hand(21) = 543
"""

import numpy as np
from typing import Dict, Optional, Tuple
import mediapipe as mp

from app.config import settings


class LandmarkExtractionService:
    """
    Service for extracting landmarks from video frames using MediaPipe.
    Implements singleton pattern for efficient holistic model usage.

    Uses fixed-offset indexing matching GISLR convention:
        Face:       indices  0..467     (468 points)
        Left hand:  indices  468..488   (21 points)
        Pose:       indices  489..521   (33 points)
        Right hand: indices  522..542   (21 points)
    """

    _instance: Optional["LandmarkExtractionService"] = None
    _holistic = None
    _extraction_count = 0

    def __new__(cls) -> "LandmarkExtractionService":
        """Ensure singleton instantiation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize MediaPipe Holistic model."""
        if self._holistic is None:
            self._initialize()

    def _initialize(self):
        """Initialize MediaPipe Holistic."""
        try:
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[OK] MediaPipe Holistic initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe: {str(e)}")

    def extract_landmarks(self, frame: np.ndarray) -> Dict:
        """
        Extract landmarks from a single frame in GISLR ordering.

        The output array uses NaN for missing landmarks (not zeros), which is
        critical for the PreprocessLayer to correctly detect absent body parts.

        Args:
            frame: numpy array BGR image [H, W, 3]

        Returns:
            Dict with landmarks and metadata:
            {
                "landmarks": np.ndarray[543, 3],  # x, y, z (NaN for missing)
                "has_hands": bool,
                "n_hand_points": int,
                "hand_motion_data": np.ndarray[42, 2],  # both hands x,y for motion calc
                "success": bool,
            }
        """
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1].copy()

            # Run holistic detection
            results = self._holistic.process(rgb_frame)

            # Initialize with NaN (critical — model's PreprocessLayer uses NaN
            # to detect missing landmarks)
            landmarks_array = np.full((543, 3), np.nan, dtype=np.float32)

            # Face landmarks: indices 0..467 (468 points)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    if i >= 468:
                        break
                    landmarks_array[i] = [lm.x, lm.y, lm.z]

            # Left hand landmarks: indices 468..488 (21 points)
            if results.left_hand_landmarks:
                base = 468
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    if i >= 21:
                        break
                    landmarks_array[base + i] = [lm.x, lm.y, lm.z]

            # Pose landmarks: indices 489..521 (33 points)
            if results.pose_landmarks:
                base = 468 + 21  # = 489
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    if i >= 33:
                        break
                    landmarks_array[base + i] = [lm.x, lm.y, lm.z]

            # Right hand landmarks: indices 522..542 (21 points)
            if results.right_hand_landmarks:
                base = 468 + 21 + 33  # = 522
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    if i >= 21:
                        break
                    landmarks_array[base + i] = [lm.x, lm.y, lm.z]

            # Calculate hand quality metrics
            has_hands = (
                results.left_hand_landmarks is not None
                or results.right_hand_landmarks is not None
            )

            # Count valid hand points (for minimum hand presence gating)
            lh_slice = landmarks_array[468:468 + 21, :2]
            rh_slice = landmarks_array[522:522 + 21, :2]
            n_hand_points = int(
                np.isfinite(lh_slice).all(axis=1).sum()
                + np.isfinite(rh_slice).all(axis=1).sum()
            )

            self._extraction_count += 1

            return {
                "landmarks": landmarks_array,
                "has_hands": has_hands,
                "n_hand_points": n_hand_points,
                "success": True,
            }

        except Exception as e:
            return {
                "landmarks": np.full((543, 3), np.nan, dtype=np.float32),
                "has_hands": False,
                "n_hand_points": 0,
                "success": False,
                "error": str(e),
            }

    def batch_extract(self, frames: list) -> list:
        """
        Extract landmarks from multiple frames.

        Args:
            frames: List of BGR frames

        Returns:
            List of extraction results
        """
        results = []
        for frame in frames:
            results.append(self.extract_landmarks(frame))
        return results

    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return {
            "total_extractions": self._extraction_count,
        }


# Singleton instance
landmark_service = LandmarkExtractionService()
