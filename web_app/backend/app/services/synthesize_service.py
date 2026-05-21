"""
Synthesize Service — Ghép nối các ký hiệu ASL thành chuỗi landmarks.

Pipeline (giống text_to_sign_tab.py trong desktop app):
  1. SignDictionary.load_gloss() — đọc từng .parquet file
  2. MotionSynthesizer.synthesize_phrase() — Hermite transition + IK + smoothing
  3. Trả về frames dưới dạng JSON để frontend render trên Canvas
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Optional
import numpy as np

# Project root = ASL_Pytorch/ (4 levels up from this file)
# web_app/backend/app/services/synthesize_service.py
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Thêm project root vào sys.path để import scripts.*
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class SynthesizeService:
    """
    Wrap SignDictionary + MotionSynthesizer cho web API.
    Khởi tạo một lần, dùng nhiều lần (singleton pattern).
    """

    _instance: Optional["SynthesizeService"] = None

    def __init__(self, data_dir: str = "data/WLASL_Skeleton",
                 transition_frames: int = 12, context_frames: int = 5):
        self._loaded = False
        self.data_dir = data_dir
        self.transition_frames = transition_frames
        self.context_frames = context_frames
        self.dictionary = None
        self.synthesizer = None
        self._available: set = set()
        self._load()

    def _load(self):
        """Load dictionary + synthesizer (heavy, done once at startup)."""
        try:
            from scripts.sign_dictionary import SignDictionary
            from scripts.motion_synthesizer import MotionSynthesizer

            # Resolve path relative to project root
            data_path = _PROJECT_ROOT / self.data_dir
            if not data_path.exists():
                # fallback: try relative to cwd
                data_path = Path(self.data_dir)

            self.dictionary = SignDictionary(data_dir=str(data_path))
            self.synthesizer = MotionSynthesizer(
                self.dictionary,
                transition_frames=self.transition_frames,
                context_frames=self.context_frames,
            )
            self._available = set(self.dictionary.get_available_glosses())
            self._loaded = True
            print(f"[SynthesizeService] Loaded {len(self._available)} glosses from {data_path}")
        except Exception as e:
            print(f"[SynthesizeService] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and self.dictionary is not None

    def get_available_glosses(self) -> List[str]:
        return sorted(self._available)

    def synthesize(self, glosses: List[str], fps: int = 25) -> dict:
        """
        Tổng hợp chuỗi landmarks từ danh sách glosses.

        Args:
            glosses: Danh sách ký hiệu (đã lowercase)
            fps: FPS để frontend biết tốc độ play

        Returns:
            {
              "frames": List[List[List[float]]],  # (T, 153, 3) flattened
              "n_frames": int,
              "fps": int,
              "n_landmarks": int,       # 153
              "glosses_used": List[str],
              "missing_glosses": List[str],
              "success": bool,
              "error": str | None
            }
        """
        if not self.is_available:
            return {"success": False, "error": "SynthesizeService not loaded", "frames": []}

        # Normalize glosses
        glosses_lower = [g.lower().strip() for g in glosses if g.strip()]

        # Split valid / missing
        valid_glosses = [g for g in glosses_lower if g in self._available]
        missing_glosses = [g for g in glosses_lower if g not in self._available]

        if not valid_glosses:
            return {
                "success": False,
                "error": f"None of the glosses found in dictionary: {glosses_lower}",
                "glosses_used": [],
                "missing_glosses": missing_glosses,
                "frames": [],
                "n_frames": 0,
                "fps": fps,
                "n_landmarks": 153,
            }

        try:
            # Synthesize (same as text_to_sign_tab._translate_thread)
            sequence: Optional[np.ndarray] = self.synthesizer.synthesize_phrase(valid_glosses)

            if sequence is None:
                return {
                    "success": False,
                    "error": "MotionSynthesizer returned None",
                    "glosses_used": valid_glosses,
                    "missing_glosses": missing_glosses,
                    "frames": [],
                    "n_frames": 0,
                    "fps": fps,
                    "n_landmarks": 153,
                }

            # sequence shape: (T, 153, 3), dtype float32
            # Replace NaN with 0 for JSON serialization
            sequence = np.nan_to_num(sequence, nan=0.0)

            # Convert to Python list (T, 153, 3)
            # Use float16 precision → reduce payload size ~50%
            frames_list = sequence.astype(np.float16).tolist()

            return {
                "success": True,
                "error": None,
                "glosses_used": valid_glosses,
                "missing_glosses": missing_glosses,
                "frames": frames_list,
                "n_frames": int(sequence.shape[0]),
                "fps": fps,
                "n_landmarks": int(sequence.shape[1]),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "glosses_used": valid_glosses,
                "missing_glosses": missing_glosses,
                "frames": [],
                "n_frames": 0,
                "fps": fps,
                "n_landmarks": 153,
            }


# ---------------------------------------------------------------------------
# Singleton accessor — imported by FastAPI dependencies
# ---------------------------------------------------------------------------

_service_instance: Optional[SynthesizeService] = None


def get_synthesize_service() -> SynthesizeService:
    global _service_instance
    if _service_instance is None:
        _service_instance = SynthesizeService()
    return _service_instance
