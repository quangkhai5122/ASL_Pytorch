"""
Video Processing Service for handling video file uploads and frame extraction.
Supports various video formats (mp4, avi, mov, etc.).
"""

import os
import cv2
import numpy as np
from typing import Iterator, Dict, List, Tuple, Optional
from pathlib import Path

from app.config import settings


class VideoProcessingService:
    """
    Service for processing video files and extracting frames.
    Includes validation, frame extraction, and batch processing.
    """

    def __init__(self):
        """Initialize video processing service."""
        self.supported_formats = tuple(
            f.lower() for f in settings.SUPPORTED_VIDEO_FORMATS
        )

    def validate_video_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if file is a supported video format.

        Args:
            file_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        file_ext = Path(file_path).suffix.lower().lstrip(".")
        if file_ext not in self.supported_formats:
            return (
                False,
                f"Unsupported format: {file_ext}. Supported: {self.supported_formats}",
            )

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_VIDEO_FILE_SIZE_MB:
            return (
                False,
                f"File too large: {file_size_mb:.2f}MB > {settings.MAX_VIDEO_FILE_SIZE_MB}MB",
            )

        return True, None

    def get_video_info(self, file_path: str) -> Dict:
        """
        Get video information (duration, FPS, resolution, etc.).

        Args:
            file_path: Path to video file

        Returns:
            Dict with video metadata
        """
        try:
            cap = cv2.VideoCapture(file_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_sec": duration_sec,
            }

        except Exception as e:
            return {
                "error": str(e),
                "frame_count": 0,
                "fps": 0,
                "width": 0,
                "height": 0,
                "duration_sec": 0,
            }

    def extract_frames(
        self, file_path: str, max_frames: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Extract frames from video file as iterator (memory-efficient).

        Args:
            file_path: Path to video file
            max_frames: Maximum number of frames to extract (None = all)

        Yields:
            BGR frames as numpy arrays
        """
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {file_path}")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                yield frame
                frame_count += 1

        finally:
            cap.release()

    def extract_frames_batch(
        self, file_path: str, max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract all frames from video into memory as batch.

        Args:
            file_path: Path to video file
            max_frames: Maximum number of frames to extract

        Returns:
            Tuple of (frames_list, metadata)
        """
        frames = []
        frame_times = []

        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                frames.append(frame)
                frame_times.append(frame_count / fps if fps > 0 else 0)
                frame_count += 1

        finally:
            cap.release()

        metadata = {
            "frame_count": len(frames),
            "fps": fps,
            "duration_sec": frame_times[-1] if frame_times else 0,
        }

        return frames, metadata

    def resize_frame(
        self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.

        Args:
            frame: BGR frame
            target_size: Target (width, height)

        Returns:
            Resized frame
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1] range.

        Args:
            frame: BGR frame [0, 255]

        Returns:
            Normalized frame [0, 1]
        """
        return frame.astype(np.float32) / 255.0

    def validate_duration(self, duration_sec: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if video duration is within limits.

        Args:
            duration_sec: Video duration in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        if duration_sec > settings.MAX_VIDEO_DURATION_SECONDS:
            return (
                False,
                f"Video too long: {duration_sec:.2f}s > {settings.MAX_VIDEO_DURATION_SECONDS}s",
            )

        return True, None


# Singleton instance
video_service = VideoProcessingService()
