"""
WebSocket stream handler for real-time sign language recognition.
Implements the full inference pipeline matching the desktop app (app_optimized.py):
  - Frame buffer with sliding window
  - Motion gating (suppresses inference when hands are idle)
  - Hand presence gating (minimum hand points required)
  - EMA probability smoothing with hysteresis
  - Stability gating (requires N consecutive same predictions to commit)
"""

import json
import base64
import asyncio
import time
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, Dict, Tuple
from collections import deque

from fastapi import WebSocket, WebSocketDisconnect, status

from app.config import settings
from app.services.model_inference import model_service
from app.services.landmark_extraction import landmark_service


# ============================================================================
# Probability Smoother (ported from app_optimized.py)
# ============================================================================

class ProbabilitySmoother:
    """EMA smoothing + hysteresis + stability gating.

    Produces stable predictions by:
    1. Exponential moving average of raw softmax probabilities
    2. Hysteresis (arm/disarm) to avoid flickering near threshold
    3. Requiring N consecutive identical predictions before "committing" a word
    """

    def __init__(
        self,
        n_classes: int,
        alpha: float = None,
        stability_n: int = None,
        confidence_on: float = None,
        confidence_off: float = None,
        confidence_unknown: float = None,
    ):
        self.alpha = alpha if alpha is not None else settings.PROB_EMA_ALPHA
        self.stability_n = stability_n if stability_n is not None else settings.STABILITY_N
        self.confidence_on = confidence_on if confidence_on is not None else settings.CONFIDENCE_ON
        self.confidence_off = confidence_off if confidence_off is not None else settings.CONFIDENCE_OFF
        self.confidence_unknown = confidence_unknown if confidence_unknown is not None else settings.CONFIDENCE_UNKNOWN
        self.n_classes = n_classes

        self.ema: Optional[np.ndarray] = None
        self.last_label: Optional[int] = None
        self.stable_count: int = 0
        self.armed: bool = False

    def update(self, probs: np.ndarray) -> Tuple[int, float, Optional[int]]:
        """
        Update EMA with current probabilities.

        Returns:
            (top1_label, top1_prob, commit_label_or_None)
        """
        if self.ema is None:
            self.ema = probs.astype(np.float32).copy()
        else:
            self.ema = self.alpha * self.ema + (1.0 - self.alpha) * probs.astype(np.float32)

        top1 = int(self.ema.argmax())
        top1p = float(self.ema[top1])

        # Hysteresis arm/disarm
        if not self.armed:
            if top1p >= self.confidence_on:
                self.armed = True
        else:
            if top1p < self.confidence_off:
                self.armed = False
                self.last_label = None
                self.stable_count = 0
                return top1, top1p, None

        if not self.armed:
            return top1, top1p, None

        # Stability count
        if self.last_label == top1:
            self.stable_count += 1
        else:
            self.last_label = top1
            self.stable_count = 1

        if self.stable_count >= self.stability_n:
            self.stable_count = 0
            return top1, top1p, top1

        return top1, top1p, None

    def get_top5(self, label_map: Dict[int, str]) -> list:
        """Get top-5 predictions from the current EMA."""
        if self.ema is None:
            return []
        top5_idx = np.argsort(-self.ema)[:5]
        return [
            {"sign": label_map.get(int(idx), f"UNKNOWN_{idx}"),
             "confidence": round(float(self.ema[int(idx)]), 4)}
            for idx in top5_idx
        ]

    def reset(self):
        """Reset smoother state."""
        self.ema = None
        self.last_label = None
        self.stable_count = 0
        self.armed = False


# ============================================================================
# Motion Detection (ported from app_optimized.py)
# ============================================================================

def compute_hand_motion(
    prev_xyz: Optional[np.ndarray],
    cur_xyz: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute mean per-point motion magnitude for hands between two frames.
    Uses both hands in GISLR order slices.

    Returns:
        (mean_motion, n_points_used)
    """
    if prev_xyz is None:
        return 0.0, 0

    # GISLR slices
    lh = slice(468, 468 + 21)
    rh = slice(468 + 21 + 33, 468 + 21 + 33 + 21)

    prev = np.concatenate([prev_xyz[lh, :2], prev_xyz[rh, :2]], axis=0)
    cur = np.concatenate([cur_xyz[lh, :2], cur_xyz[rh, :2]], axis=0)

    valid = np.isfinite(prev).all(axis=1) & np.isfinite(cur).all(axis=1)
    if not np.any(valid):
        return 0.0, 0

    d = cur[valid] - prev[valid]
    mag = np.sqrt((d ** 2).sum(axis=1))
    return float(mag.mean()), int(valid.sum())


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: Dict[str, "StreamConnection"] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> "StreamConnection":
        """Register new WebSocket connection."""
        connection = StreamConnection(websocket, client_id)
        self.active_connections[client_id] = connection
        await websocket.accept()
        return connection

    async def disconnect(self, client_id: str):
        """Unregister closed connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast(self, message: dict, exclude_client: Optional[str] = None):
        """Broadcast message to all connected clients."""
        disconnected = []
        for cid, conn in self.active_connections.items():
            if exclude_client and cid == exclude_client:
                continue
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(cid)
        for cid in disconnected:
            await self.disconnect(cid)

    def get_connection(self, client_id: str) -> Optional["StreamConnection"]:
        return self.active_connections.get(client_id)

    def get_active_count(self) -> int:
        return len(self.active_connections)


# ============================================================================
# Stream Connection (per-client state)
# ============================================================================

class StreamConnection:
    """
    Represents a single WebSocket client connection.
    Manages frame buffering, preprocessing, inference, and smoothing state.
    """

    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.connected_at = datetime.now()

        # Counters
        self.frame_count = 0
        self.prediction_count = 0
        self.total_processing_time = 0.0

        # Frame buffer — stores raw landmarks [543, 3] per frame
        self.frame_buffer: deque = deque(maxlen=settings.WINDOW_SIZE)

        # Motion tracking
        self.prev_xyz: Optional[np.ndarray] = None
        self.motion_hist: deque = deque(maxlen=settings.WINDOW_SIZE)

        # Inference scheduling
        self.frames_since_infer = 0
        self.last_infer_time = 0.0

        # Probability smoother
        n_classes = settings.NUM_CLASSES
        self.smoother = ProbabilitySmoother(n_classes=n_classes)

        # Current prediction state (for non-inference frames)
        self.prediction_state = {
            "current_sign": None,
            "confidence": 0.0,
            "top5": [],
            "status": "warming_up",
        }

        # Committed words buffer
        self.committed_words: list = []

    async def send_json(self, message: dict):
        """Send JSON message to client."""
        try:
            await self.websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message to {self.client_id}: {str(e)}")
            raise

    async def send_response(self, response):
        """Send a Pydantic response model to client."""
        await self.send_json(response.model_dump())

    async def receive_frame_message(self) -> Optional[Dict]:
        """Receive and parse frame message from client."""
        try:
            data = await self.websocket.receive_text()
            message = json.loads(data)
            return message
        except WebSocketDisconnect:
            raise
        except json.JSONDecodeError:
            await self.send_json({"type": "error", "error": "Invalid JSON format"})
            return None
        except Exception as e:
            print(f"Error receiving frame from {self.client_id}: {str(e)}")
            return None

    def _serialize_landmarks(self, landmarks: np.ndarray) -> list:
        """Serialize landmarks [543, 3] to a JSON-safe list of [x, y, z].

        NaN values are replaced with None so the frontend can detect missing
        landmarks. Returns full 543 points to support face/pose/hand rendering.
        Only includes the essential 75 points (pose + hands) to reduce payload
        size, padded to full 543 indices for correct GISLR offset mapping.
        """
        result = []
        # Indices to include: Left hand (468..488), Pose (489..521), Right hand (522..542)
        # Also include a sparse set of face points for face outline (optional)
        for i in range(543):
            pt = landmarks[i]
            if np.isfinite(pt[0]) and np.isfinite(pt[1]):
                result.append([round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)])
            else:
                result.append(None)
        return result

    async def process_frame(self, frame_base64: str, frame_id: int) -> Optional[Dict]:
        """
        Process incoming frame through the full inference pipeline.

        Pipeline:
        1. Decode base64 → BGR image
        2. Extract landmarks (GISLR order, NaN fill)
        3. Motion gating + hand presence check
        4. Append to frame buffer
        5. Run inference when conditions are met (min frames, stride)
        6. Smooth predictions (EMA + hysteresis + stability)
        7. Return prediction or status update

        Args:
            frame_base64: Base64 encoded JPEG frame
            frame_id: Frame ID for tracking

        Returns:
            Prediction result dict or status update
        """
        start_time = time.time()

        try:
            # 1. Decode frame
            frame_data = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {
                    "type": "error",
                    "error": "Invalid frame format",
                    "frame_id": frame_id,
                }

            # 2. Extract landmarks
            landmark_result = landmark_service.extract_landmarks(frame)

            if not landmark_result["success"]:
                return {
                    "type": "status",
                    "status": "landmark_failed",
                    "message": "Landmark extraction failed",
                    "frame_id": frame_id,
                }

            landmarks = landmark_result["landmarks"]  # [543, 3] with NaN
            n_hand_points = landmark_result["n_hand_points"]

            # Serialize landmarks for frontend skeleton drawing
            # Convert to list of [x, y, z] with None for NaN
            landmarks_for_client = self._serialize_landmarks(landmarks)

            # 3. Motion gating
            motion, n_motion_pts = compute_hand_motion(self.prev_xyz, landmarks)
            self.prev_xyz = landmarks.copy()
            self.motion_hist.append(motion if n_motion_pts > 0 else 0.0)
            motion_avg = float(np.mean(self.motion_hist)) if self.motion_hist else 0.0

            # 4. Append to frame buffer
            self.frame_buffer.append(landmarks)
            self.frame_count += 1
            self.frames_since_infer += 1

            # 5. Check if we should run inference
            now = time.time()
            should_infer = (
                len(self.frame_buffer) >= settings.MIN_FRAMES_FOR_INFER
                and self.frames_since_infer >= settings.INFER_STRIDE_FRAMES
            )

            processing_time_ms = (time.time() - start_time) * 1000

            if not should_infer:
                # Not enough data or too soon — return buffer status
                status_msg = "collecting"
                if len(self.frame_buffer) < settings.MIN_FRAMES_FOR_INFER:
                    status_msg = "collecting"
                return {
                    "type": "frame_received",
                    "frame_id": frame_id,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "frames_in_buffer": len(self.frame_buffer),
                    "status": status_msg,
                    "landmarks": landmarks_for_client,
                }

            # Hand presence check
            if n_hand_points < settings.MIN_HAND_POINTS:
                self.prediction_state["status"] = "no_hands"
                return {
                    "type": "status",
                    "status": "no_hands",
                    "message": "No hands detected",
                    "frame_id": frame_id,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "motion": round(motion_avg, 5),
                    "landmarks": landmarks_for_client,
                }

            # Motion threshold check
            if motion_avg < settings.MOVEMENT_THRESHOLD:
                self.prediction_state["status"] = "idle"
                return {
                    "type": "status",
                    "status": "idle",
                    "message": f"Idling (motion: {motion_avg:.4f})",
                    "frame_id": frame_id,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "motion": round(motion_avg, 5),
                    "landmarks": landmarks_for_client,
                }

            # === INFERENCE ===
            # Only reset the stride counter if we ACTUALLY run inference
            self.frames_since_infer = 0
            self.last_infer_time = now

            # 6. Stack frames and run model
            window_np = np.stack(list(self.frame_buffer), axis=0)  # [T, 543, 3]

            try:
                result = model_service.predict_from_landmarks(
                    window_np, return_top_k=5
                )
            except Exception as e:
                print(f"Inference error for {self.client_id}: {str(e)}")
                return {
                    "type": "error",
                    "error": f"Inference failed: {str(e)}",
                    "frame_id": frame_id,
                }

            # 7. Smooth predictions
            raw_probs = result["probabilities"]
            top1_label, top1_prob, commit_label = self.smoother.update(raw_probs)

            # Get top-5 from smoothed EMA
            label_map = model_service.get_label_map()
            top5 = self.smoother.get_top5(label_map)

            # Determine display sign
            pred_sign = label_map.get(top1_label, f"UNKNOWN_{top1_label}")
            if top1_prob < self.smoother.confidence_unknown:
                display_sign = f"(Unknown: {top1_prob:.2f})"
                status = "unknown"
            else:
                display_sign = pred_sign
                status = "predicting"

            self.prediction_count += 1
            total_time_ms = (time.time() - start_time) * 1000
            self.total_processing_time += total_time_ms

            # Update internal state
            self.prediction_state = {
                "current_sign": display_sign,
                "confidence": round(top1_prob, 4),
                "top5": top5,
                "status": status,
            }

            # Build response
            response = {
                "type": "prediction",
                "sign": display_sign,
                "confidence": round(top1_prob, 4),
                "top5": top5,
                "frame_id": frame_id,
                "processing_time_ms": round(total_time_ms, 2),
                "motion": round(motion_avg, 5),
                "frames_in_buffer": len(self.frame_buffer),
                "landmarks": landmarks_for_client,
            }

            # Commit event — word is stable enough to add to sentence
            if commit_label is not None:
                committed_sign = label_map.get(commit_label, f"UNKNOWN_{commit_label}")
                self.committed_words.append(committed_sign)
                response["commit"] = {
                    "sign": committed_sign,
                    "confidence": round(top1_prob, 4),
                    "word_buffer": list(self.committed_words),
                }

            return response

        except Exception as e:
            print(f"Frame processing error for {self.client_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "type": "error",
                "error": str(e),
                "frame_id": frame_id,
            }

    async def send_heartbeat(self) -> Dict:
        """Generate heartbeat message with connection stats."""
        uptime_sec = (datetime.now() - self.connected_at).total_seconds()
        avg_latency = (
            self.total_processing_time / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime_sec, 2),
            "frames_received": self.frame_count,
            "predictions_made": self.prediction_count,
            "avg_latency_ms": round(avg_latency, 2),
            "frames_in_buffer": len(self.frame_buffer),
            "committed_words": list(self.committed_words),
        }

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        uptime_sec = (datetime.now() - self.connected_at).total_seconds()
        avg_latency = (
            self.total_processing_time / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            "client_id": self.client_id,
            "connected_at": self.connected_at.isoformat(),
            "uptime_seconds": round(uptime_sec, 2),
            "total_frames": self.frame_count,
            "total_predictions": self.prediction_count,
            "total_processing_time_ms": round(self.total_processing_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "buffer_size": len(self.frame_buffer),
            "last_prediction": self.prediction_state,
            "committed_words": list(self.committed_words),
        }


# Global connection manager
connection_manager = ConnectionManager()
