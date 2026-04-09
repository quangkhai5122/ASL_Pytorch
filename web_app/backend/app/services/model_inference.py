"""
Model Inference Service for ASL Recognition.
Handles model loading, preprocessing, and inference operations.

The complete inference pipeline:
    1. Raw landmarks [T, 543, 3] (with NaN for missing)
    2. PreprocessLayer → [64, 66, 3] + [64] non_empty_frame_idxs
    3. ASLTransformerModel(frames, non_empty_frame_idxs) → logits [B, 250]
"""

import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.config import settings


class ModelInferenceService:
    """
    Service for loading and running inference with ASLTransformerModel.
    Implements singleton pattern with lazy loading.

    Owns both the ASLTransformerModel and the PreprocessLayer, ensuring
    they are always used together correctly.
    """

    _instance: Optional["ModelInferenceService"] = None
    _model = None
    _preprocess_layer = None
    _device = None
    _label_map: Dict[int, str] = {}
    _reverse_label_map: Dict[str, int] = {}
    _model_loaded = False
    _inference_count = 0
    _total_inference_time = 0.0

    def __new__(cls) -> "ModelInferenceService":
        """Ensure singleton instantiation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the service (lazy-loads model on first use)."""
        if not self._model_loaded:
            self._initialize()

    def _initialize(self):
        """Initialize model, preprocess layer, and labels on first use."""
        try:
            self._device = self._get_device()
            self._load_label_map()
            self._load_model()
            self._load_preprocess_layer()
            self._model_loaded = True
            self._warmup()
            print(
                f"[OK] Model initialized successfully on device: {self._device}"
            )
        except Exception as e:
            print(f"[ERROR] Error initializing model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _get_device(self) -> str:
        """Determine the device to use (GPU or CPU)."""
        if settings.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif settings.DEVICE == "cuda":
            if not torch.cuda.is_available():
                print("[WARN] GPU requested but not available, falling back to CPU")
                return "cpu"
            return "cuda"
        return "cpu"

    def _load_label_map(self):
        """Load the sign-to-index label mapping from JSON file."""
        # Try multiple paths
        candidates = [
            Path(settings.MODEL_PATH).parent.parent / "data" / "sign_to_prediction_index_map.json",
            Path("./data/sign_to_prediction_index_map.json"),
            Path("../data/sign_to_prediction_index_map.json"),
        ]

        label_path = None
        for p in candidates:
            if p.exists():
                label_path = p
                break

        if label_path and label_path.exists():
            with open(label_path, "r") as f:
                sign2idx = json.load(f)
            # sign2idx: {"hello": 0, "world": 1, ...}
            self._label_map = {int(v): k for k, v in sign2idx.items()}
            self._reverse_label_map = {k: int(v) for k, v in sign2idx.items()}
            print(f"[OK] Loaded {len(self._label_map)} sign labels from {label_path}")
        else:
            print(f"[WARN] Label mapping not found. Tried: {[str(p) for p in candidates]}")

    def _load_model(self):
        """Load the pre-trained ASLTransformerModel weights."""
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {settings.MODEL_PATH}")

        try:
            from scripts.model import ASLTransformerModel

            self._model = ASLTransformerModel()

            checkpoint = torch.load(
                settings.MODEL_PATH, map_location=self._device
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)

            self._model.to(self._device)
            self._model.eval()

            model_size_mb = sum(
                p.numel() for p in self._model.parameters()
            ) * 4 / (1024 * 1024)
            print(f"[OK] Model loaded from {settings.MODEL_PATH} ({model_size_mb:.2f} MB)")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_preprocess_layer(self):
        """Load and initialize the PreprocessLayer (torch.nn.Module)."""
        try:
            from scripts.preprocess import PreprocessLayer

            self._preprocess_layer = PreprocessLayer()
            self._preprocess_layer.to(self._device)
            self._preprocess_layer.eval()
            print("[OK] PreprocessLayer initialized")

        except Exception as e:
            raise RuntimeError(f"Failed to load PreprocessLayer: {str(e)}")

    def _warmup(self):
        """Run a dummy inference to warm up GPU/CUDA and verify pipeline."""
        print("[Warmup] Running initial inference to verify pipeline...")
        try:
            dummy_landmarks = np.zeros((settings.WINDOW_SIZE, 543, 3), dtype=np.float32)
            _ = self.predict_from_landmarks(dummy_landmarks)
            print("[Warmup] Pipeline verified successfully!")
        except Exception as e:
            print(f"[Warmup] Warning: warmup inference failed: {e}")
            import traceback
            traceback.print_exc()

    def predict_from_landmarks(
        self,
        landmarks_sequence: np.ndarray,
        return_top_k: int = 5,
    ) -> Dict:
        """
        Run the full inference pipeline: PreprocessLayer → Model.

        This is the primary prediction method. Takes raw landmark sequences
        and handles preprocessing + model inference internally.

        Args:
            landmarks_sequence: numpy array of shape [T, 543, 3] where T is
                                the number of frames in the temporal window.
                                Missing landmarks should be NaN.
            return_top_k: Number of top predictions to return

        Returns:
            Dict with prediction results:
            {
                "sign": str,
                "confidence": float,
                "top5": [{"sign": ..., "confidence": ...}, ...],
                "probabilities": np.ndarray,  # raw softmax probabilities
                "processing_time_ms": float,
            }
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Convert numpy landmarks to torch tensor
            input_tensor = torch.from_numpy(
                landmarks_sequence.astype(np.float32)
            ).to(self._device)

            with torch.inference_mode():
                # Step 1: Preprocess — [T, 543, 3] → [64, 66, 3] + [64]
                processed, non_empty_frame_idxs = self._preprocess_layer(input_tensor)

                # Step 2: Add batch dimension — [1, 64, 66, 3] + [1, 64]
                processed = processed.unsqueeze(0)
                non_empty_frame_idxs = non_empty_frame_idxs.unsqueeze(0)

                # Step 3: Model inference — logits [1, NUM_CLASSES]
                logits = self._model(processed, non_empty_frame_idxs)

                # Step 4: Softmax
                probs = F.softmax(logits, dim=-1)
                probs_np = probs.detach().cpu().float().numpy()[0]

            processing_time_ms = (time.time() - start_time) * 1000
            self._total_inference_time += processing_time_ms
            self._inference_count += 1

            # Top-K
            top_k_indices = np.argsort(-probs_np)[:return_top_k]
            top_predictions = []
            for idx in top_k_indices:
                sign_name = self._label_map.get(int(idx), f"UNKNOWN_{idx}")
                conf = float(probs_np[int(idx)])
                top_predictions.append({"sign": sign_name, "confidence": round(conf, 4)})

            return {
                "sign": top_predictions[0]["sign"],
                "confidence": top_predictions[0]["confidence"],
                "top5": top_predictions,
                "probabilities": probs_np,
                "processing_time_ms": round(processing_time_ms, 2),
            }

        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def get_device(self) -> str:
        """Get the current device."""
        return self._device or "cpu"

    def get_label_map(self) -> Dict[int, str]:
        """Get the label map (idx → sign name)."""
        return self._label_map

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        avg_latency = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0
        )
        return {
            "total_predictions": self._inference_count,
            "avg_latency_ms": round(avg_latency, 2),
            "total_time_ms": round(self._total_inference_time, 2),
        }

    def reset_stats(self):
        """Reset inference statistics."""
        self._inference_count = 0
        self._total_inference_time = 0.0


# Singleton instance
model_service = ModelInferenceService()
