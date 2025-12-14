"""app_optimized.py

Optimized real-time ASL recognition GUI.

Key optimizations vs. app.py
- Remove Pandas from the real-time loop (use NumPy arrays for landmarks).
- Move webcam + MediaPipe + inference to a background thread (UI stays responsive).
- Sliding-window inference with stride + probability smoothing + stability gating.
- Fast motion metric computed incrementally (no groupby/std).
- Non-blocking Gemini calls (background thread).
- Resize with OpenCV (cheaper than PIL resize each frame).

This file keeps the same external dependencies as app.py (scripts.* modules, model weights, etc.).
"""

from __future__ import annotations

import os
import sys
import time
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3

import google.generativeai as genai
from dotenv import load_dotenv

from scripts.config import N_ROWS, N_DIMS, DEVICE
from scripts.model import ASLTransformerModel
from scripts.preprocess import PreprocessLayer
from scripts.utils import load_data_maps


# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.path.abspath('.')
sys.path.append(os.path.join(_BASE_DIR, 'src'))

load_dotenv()

MODEL_PATH = os.path.join("models", "model_best_full_training.pth")
CSV_PATH = "data/train.csv"


# Inference / gating parameters (tune for your dataset)
WINDOW_SIZE = 64                 # number of frames in the rolling window
MIN_FRAMES_FOR_INFER = 16        # do not infer too early
INFER_STRIDE_FRAMES = 4          # run inference every N frames
MIN_INFER_GAP_SEC = 0.08         # time-based throttle

CONFIDENCE_ON = 0.40             # hysteresis ON threshold
CONFIDENCE_OFF = 0.20            # hysteresis OFF threshold
CONFIDENCE_UNKNOWN = 0.15        # show unknown below this

STABILITY_N = 2                  # stable predictions needed to commit a word
PROB_EMA_ALPHA = 0.80            # higher = smoother (slower reaction)

# Motion gating (cheap incremental metric on hand keypoints)
MOVEMENT_THRESHOLD = 0.010       # if avg motion below -> idling (tune)
MIN_HAND_POINTS = 6              # minimum detected hand points per frame (across both hands)


# GUI configuration
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 720
CAMERA_WEIGHT = 65
INFO_WEIGHT = 35

FONT_LARGE = ("Helvetica", 18, "bold")
FONT_MEDIUM = ("Helvetica", 14)
FONT_SMALL = ("Helvetica", 12)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def _safe_resize_bgr_to_fit(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize BGR frame to fit inside (target_w, target_h) preserving aspect ratio."""
    if frame_bgr is None or target_w <= 1 or target_h <= 1:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return frame_bgr
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_xyz_gislr_order(results) -> np.ndarray:
    """Extract MediaPipe Holistic landmarks into a (543, 3) array (x,y,z) with NaNs.

    Order MUST match GISLR convention used in your training pipeline:
        Face (468) -> Left hand (21) -> Pose (33) -> Right hand (21)

    Missing landmarks remain NaN.
    """
    xyz = np.full((N_ROWS, N_DIMS), np.nan, dtype=np.float32)

    # Face: 0..467
    if results.face_landmarks:
        for i, p in enumerate(results.face_landmarks.landmark):
            if i >= 468:
                break
            xyz[i, 0] = p.x
            xyz[i, 1] = p.y
            xyz[i, 2] = p.z

    # Left hand: 468..488
    if results.left_hand_landmarks:
        base = 468
        for i, p in enumerate(results.left_hand_landmarks.landmark):
            if i >= 21:
                break
            xyz[base + i, 0] = p.x
            xyz[base + i, 1] = p.y
            xyz[base + i, 2] = p.z

    # Pose: 489..521 (33)
    if results.pose_landmarks:
        base = 468 + 21
        for i, p in enumerate(results.pose_landmarks.landmark):
            if i >= 33:
                break
            xyz[base + i, 0] = p.x
            xyz[base + i, 1] = p.y
            xyz[base + i, 2] = p.z

    # Right hand: 522..542
    if results.right_hand_landmarks:
        base = 468 + 21 + 33
        for i, p in enumerate(results.right_hand_landmarks.landmark):
            if i >= 21:
                break
            xyz[base + i, 0] = p.x
            xyz[base + i, 1] = p.y
            xyz[base + i, 2] = p.z

    return xyz


def hand_motion(prev_xyz: Optional[np.ndarray], cur_xyz: np.ndarray) -> Tuple[float, int]:
    """Compute mean per-point motion magnitude for hands between two frames.

    Returns (mean_motion, n_points_used).
    Uses both hands in GISLR order slices.
    """
    if prev_xyz is None:
        return 0.0, 0

    # Slices in GISLR order
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


@dataclass
class PredictionState:
    current_sign: str = "Initializing..."
    current_sign_color: str = "blue"
    top5: Tuple[Tuple[str, float], ...] = ()
    commit_word: Optional[str] = None
    commit_conf: float = 0.0
    motion: float = 0.0
    fps: float = 0.0


class ProbabilitySmoother:
    """EMA smoothing + stability gating."""

    def __init__(self, n_classes: int, alpha: float = PROB_EMA_ALPHA, stability_n: int = STABILITY_N):
        self.alpha = float(alpha)
        self.stability_n = int(stability_n)
        self.ema: Optional[np.ndarray] = None
        self.last_label: Optional[int] = None
        self.stable_count: int = 0
        self.armed: bool = False  # hysteresis state
        self.n_classes = int(n_classes)

    def update(self, probs: np.ndarray) -> Tuple[int, float, Optional[int]]:
        """Update EMA with current probs.

        Returns:
            top1_label, top1_prob, commit_label_or_None
        """
        if self.ema is None:
            self.ema = probs.astype(np.float32)
        else:
            self.ema = self.alpha * self.ema + (1.0 - self.alpha) * probs.astype(np.float32)

        top1 = int(self.ema.argmax())
        top1p = float(self.ema[top1])

        # Hysteresis arm/disarm
        if not self.armed:
            if top1p >= CONFIDENCE_ON:
                self.armed = True
        else:
            if top1p < CONFIDENCE_OFF:
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
            # Emit commit event once, then require a change or disarm to commit again.
            self.stable_count = 0
            return top1, top1p, top1

        return top1, top1p, None


# ----------------------------------------------------------------------------
# Worker thread
# ----------------------------------------------------------------------------


class CameraInferenceWorker(threading.Thread):
    """Runs webcam capture, MediaPipe Holistic, and model inference off the UI thread."""

    def __init__(
        self,
        camera_index: int,
        model: ASLTransformerModel,
        preprocess_layer: PreprocessLayer,
        ord2sign: dict,
        state_queue: "queue.Queue[Tuple[np.ndarray, PredictionState]]",
        stop_event: threading.Event,
        target_capture_size: Tuple[int, int] = (640, 480),
    ):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.model = model
        self.preprocess_layer = preprocess_layer
        self.ord2sign = ord2sign
        self.state_queue = state_queue
        self.stop_event = stop_event
        self.target_capture_size = target_capture_size

        self.seq: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.motion_hist: Deque[float] = deque(maxlen=WINDOW_SIZE)
        self.prev_xyz: Optional[np.ndarray] = None

        self.frames_since_infer = 0
        self.last_infer_time = 0.0

        self.smoother = ProbabilitySmoother(n_classes=len(self.ord2sign))

        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Video capture
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Stabilize latency
        w, h = self.target_capture_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # FPS calc
        self._fps_t0 = time.time()
        self._fps_frames = 0
        self._fps = 0.0

        # Keep last inference state to avoid UI flicker between inference steps
        self._last_state = PredictionState(current_sign="(Warming up)", current_sign_color="gray")

    def close(self):
        try:
            self.holistic.close()
        except Exception:
            pass
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

    def _update_fps(self):
        self._fps_frames += 1
        now = time.time()
        dt = now - self._fps_t0
        if dt >= 1.0:
            self._fps = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_t0 = now

    def _draw_landmarks(self, frame_bgr: np.ndarray, results) -> None:
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
        )
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

    def _try_infer(self, motion_avg: float) -> PredictionState:
        state = PredictionState()
        state.motion = float(motion_avg)
        state.fps = float(self._fps)

        # Basic hand presence gating
        if not self.seq:
            state.current_sign = "(No Data)"
            state.current_sign_color = "red"
            return state

        # Require minimum hand points on the most recent frame
        last = self.seq[-1]
        lh = last[468:468 + 21, :2]
        rh = last[468 + 21 + 33:468 + 21 + 33 + 21, :2]
        n_hand = int(np.isfinite(lh).all(axis=1).sum() + np.isfinite(rh).all(axis=1).sum())
        if n_hand < MIN_HAND_POINTS:
            state.current_sign = "(No Hands Detected)"
            state.current_sign_color = "red"
            return state

        if motion_avg < MOVEMENT_THRESHOLD:
            state.current_sign = f"(Idling: {motion_avg:.4f})"
            state.current_sign_color = "red"
            return state

        # Stack frames: [T, 543, 3]
        window_np = np.stack(list(self.seq), axis=0).astype(np.float32)

        # Model inference
        try:
            input_tensor = torch.from_numpy(window_np).to(DEVICE, non_blocking=True)
        except Exception:
            state.current_sign = "(Tensor Error)"
            state.current_sign_color = "red"
            return state

        with torch.inference_mode():
            processed, non_empty = self.preprocess_layer(input_tensor)
            processed = processed.unsqueeze(0)
            non_empty = non_empty.unsqueeze(0)

            # Guard: if everything is empty, skip
            if non_empty.max().item() == -1.0:
                state.current_sign = "(Empty Sequence)"
                state.current_sign_color = "red"
                return state

            logits = self.model(processed, non_empty)
            probs_t = F.softmax(logits, dim=1)
            probs = probs_t.detach().float().cpu().numpy()[0]

            # Smoothing + stability
            top1, top1p, commit = self.smoother.update(probs)

            # Top-5
            top5_idx = np.argsort(-self.smoother.ema)[:5] if self.smoother.ema is not None else np.argsort(-probs)[:5]
            top5 = []
            for idx in top5_idx:
                sign = self.ord2sign.get(int(idx), "Unknown")
                conf = float(self.smoother.ema[int(idx)]) if self.smoother.ema is not None else float(probs[int(idx)])
                top5.append((sign, conf))
            state.top5 = tuple(top5)

            # Status string
            pred_sign = self.ord2sign.get(int(top1), "Unknown")
            if top1p < CONFIDENCE_UNKNOWN:
                state.current_sign = f"(Unknown: {top1p:.2f})"
                state.current_sign_color = "orange"
            else:
                state.current_sign = f"{pred_sign} ({top1p:.2f})"
                state.current_sign_color = "blue"

            if commit is not None:
                state.commit_word = self.ord2sign.get(int(commit), "Unknown")
                state.commit_conf = float(top1p)

        return state

    def run(self):
        try:
            while not self.stop_event.is_set():
                ok, frame_bgr = self.cap.read()
                if not ok or frame_bgr is None:
                    time.sleep(0.01)
                    continue

                # MediaPipe expects RGB
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = self.holistic.process(image_rgb)

                # Draw landmarks on original BGR
                self._draw_landmarks(frame_bgr, results)

                # Extract landmarks (NumPy)
                xyz = extract_xyz_gislr_order(results)
                self.seq.append(xyz)

                # Motion metric (incremental)
                m, n_pts = hand_motion(self.prev_xyz, xyz)
                self.prev_xyz = xyz
                if n_pts > 0:
                    self.motion_hist.append(m)
                else:
                    self.motion_hist.append(0.0)

                motion_avg = float(np.mean(self.motion_hist)) if self.motion_hist else 0.0

                # Inference scheduling
                self.frames_since_infer += 1
                now = time.time()
                do_infer = (
                    len(self.seq) >= MIN_FRAMES_FOR_INFER
                    and self.frames_since_infer >= INFER_STRIDE_FRAMES
                    and (now - self.last_infer_time) >= MIN_INFER_GAP_SEC
                )

                if do_infer:
                    self.frames_since_infer = 0
                    self.last_infer_time = now
                    pred_state = self._try_infer(motion_avg)
                    self._last_state = pred_state
                else:
                    # Reuse last inference output; only refresh motion/FPS and clear commit.
                    pred_state = replace(self._last_state)
                    pred_state.motion = motion_avg
                    pred_state.fps = float(self._fps)
                    pred_state.commit_word = None
                    pred_state.commit_conf = 0.0
                    if len(self.seq) < MIN_FRAMES_FOR_INFER:
                        pred_state.current_sign = "(Collecting...)"
                        pred_state.current_sign_color = "gray"

                # Flip for user-friendly mirror view
                frame_bgr = cv2.flip(frame_bgr, 1)

                # Update FPS
                self._update_fps()
                pred_state.fps = float(self._fps)

                # Publish newest (drop older if queue full)
                try:
                    while True:
                        self.state_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.state_queue.put_nowait((frame_bgr, pred_state))
                except queue.Full:
                    pass

        finally:
            self.close()


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------


class ASLAppOptimized:
    def __init__(self, window: tk.Tk, window_title: str, camera_index: int = 0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.window.minsize(800, 600)

        self.camera_index = camera_index
        self.running = True
        self.stop_event = threading.Event()

        # State
        self.recognised_words_buffer: List[str] = []
        self.display_sentence: str = ""
        self.current_sign: str = "Initializing..."
        self.current_sign_color: str = "blue"
        self.top_5_predictions: List[Tuple[str, float]] = []
        self.last_motion: float = 0.0
        self.last_fps: float = 0.0

        # Load models/mappings/APIs
        self.initialize_components()

        # GUI
        self.setup_gui()

        # Background worker
        self.state_queue: "queue.Queue[Tuple[np.ndarray, PredictionState]]" = queue.Queue(maxsize=1)
        self.worker = CameraInferenceWorker(
            camera_index=self.camera_index,
            model=self.model,
            preprocess_layer=self.preprocess_layer,
            ord2sign=self.ORD2SIGN,
            state_queue=self.state_queue,
            stop_event=self.stop_event,
        )
        self.worker.start()

        # Bind keys
        self.window.bind('<KeyPress-g>', self.generate_sentence)
        self.window.bind('<KeyPress-G>', self.generate_sentence)
        self.window.bind('<KeyPress-c>', self.clear_session)
        self.window.bind('<KeyPress-C>', self.clear_session)
        self.window.bind('<KeyPress-q>', self.quit_app)
        self.window.bind('<KeyPress-Q>', self.quit_app)

        # Loop
        self.delay = 15
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.update_loop()

    def initialize_components(self):
        # 1) Label mapping
        _, self.ORD2SIGN = load_data_maps(CSV_PATH)
        if not self.ORD2SIGN:
            raise RuntimeError("Failed to load label mappings")

        # 2) Model
        self.model = ASLTransformerModel()
        model_path_abs = os.path.abspath(MODEL_PATH)
        if not os.path.exists(model_path_abs):
            raise FileNotFoundError(f"Model weights not found at {model_path_abs}")
        state_dict = torch.load(model_path_abs, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        # 3) Preprocess layer
        self.preprocess_layer = PreprocessLayer().to(DEVICE)
        self.preprocess_layer.eval()

        # 4) Gemini
        self.gemini_model = None
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                print("Gemini API (2.5 Flash Lite) Initialized.")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}")
                import traceback
                traceback.print_exc()

        # 5) TTS
        try:
            self.tts_engine = pyttsx3.init()
        except Exception:
            self.tts_engine = None

    # ---------------- GUI layout ----------------

    def on_info_frame_resize(self, event):
        padding = 30
        new_wraplength = event.width - padding
        if new_wraplength > padding:
            if hasattr(self, 'buffer_label_auto'):
                self.buffer_label_auto.config(wraplength=new_wraplength)
            if hasattr(self, 'sentence_label_auto'):
                self.sentence_label_auto.config(wraplength=new_wraplength)
            if hasattr(self, 'buffer_label_manual'):
                self.buffer_label_manual.config(wraplength=new_wraplength)
            if hasattr(self, 'sentence_label_manual'):
                self.sentence_label_manual.config(wraplength=new_wraplength)

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== Top bar: Tab buttons (fixed at top-left) =====
        self.tab_bar = ttk.Frame(self.main_frame)
        self.tab_bar.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)
        
        self.current_tab = tk.IntVar(value=0)
        
        self.tab_btn_auto = ttk.Radiobutton(
            self.tab_bar, text="Automatic Mode", 
            variable=self.current_tab, value=0,
            command=lambda: self._switch_tab(0)
        )
        self.tab_btn_auto.pack(side=tk.LEFT, padx=5)
        
        self.tab_btn_manual = ttk.Radiobutton(
            self.tab_bar, text="Manual Mode",
            variable=self.current_tab, value=1,
            command=lambda: self._switch_tab(1)
        )
        self.tab_btn_manual.pack(side=tk.LEFT, padx=5)
        
        self.tab_btn_translate = ttk.Radiobutton(
            self.tab_bar, text="Text2Sign Mode",
            variable=self.current_tab, value=2,
            command=lambda: self._switch_tab(2)
        )
        self.tab_btn_translate.pack(side=tk.LEFT, padx=5)
        
        self.tab_btn_dictionary = ttk.Radiobutton(
            self.tab_bar, text="Dictionary Mode",
            variable=self.current_tab, value=3,
            command=lambda: self._switch_tab(3)
        )
        self.tab_btn_dictionary.pack(side=tk.LEFT, padx=5)

        # ===== Content area =====
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # ----- Recognition view (Tab 1 & 2): Camera + Info panel -----
        self.recognition_view = ttk.Frame(self.content_frame)
        self.recognition_view.grid_rowconfigure(0, weight=1)
        self.recognition_view.grid_columnconfigure(0, weight=CAMERA_WEIGHT)
        self.recognition_view.grid_columnconfigure(1, weight=INFO_WEIGHT)

        # Left: Camera
        self.camera_frame = ttk.Frame(self.recognition_view)
        self.camera_frame.grid(row=0, column=0, sticky="nsew")
        self.camera_frame.pack_propagate(False)
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, anchor='center')

        # Right: Info panel (content switches based on tab selection)
        self.info_frame = ttk.Frame(self.recognition_view)
        self.info_frame.grid(row=0, column=1, sticky="nsew")
        self.info_frame.pack_propagate(False)

        # Auto mode content
        self.tab_auto = ttk.Frame(self.info_frame)
        self.setup_tab_auto(self.tab_auto)

        # Manual mode content
        self.tab_manual = ttk.Frame(self.info_frame)
        self.setup_tab_manual(self.tab_manual)

        # ----- Translate view (Tab 3): Full width -----
        self.setup_tab_translate()
        
        # ----- Dictionary view (Tab 4): Full width -----
        self.setup_tab_dictionary()

        self.info_frame.bind("<Configure>", self.on_info_frame_resize)
        
        # Show Tab 1 by default
        self._switch_tab(0)
        self.update_gui_info()

    def setup_tab_auto(self, parent):
        tk.Label(parent, text="Current Sign / Status:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(20, 5))
        self.sign_var_auto = tk.StringVar()
        self.sign_label_auto = tk.Label(parent, textvariable=self.sign_var_auto, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT)
        self.sign_label_auto.pack(fill=tk.X, padx=15)

        tk.Label(parent, text="Word Buffer:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.buffer_var_auto = tk.StringVar()
        self.buffer_label_auto = tk.Label(parent, textvariable=self.buffer_var_auto, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, height=2)
        self.buffer_label_auto.pack(fill=tk.X, padx=15)

        tk.Label(parent, text="Generated Sentence:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.sentence_var_auto = tk.StringVar()
        self.sentence_label_auto = tk.Label(parent, textvariable=self.sentence_var_auto, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, fg="green")
        self.sentence_label_auto.pack(fill=tk.X, padx=15)

        tk.Button(parent, text="Play Audio", command=self.play_audio, font=FONT_SMALL).pack(anchor='w', padx=15, pady=5)

        instructions = "Controls:\n[G] Generate Sentence\n[C] Clear Session\n[Q] Quit"
        tk.Label(parent, text=instructions, font=FONT_SMALL, anchor='w', justify=tk.LEFT, fg="gray").pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=20)

    def setup_tab_manual(self, parent):
        tk.Label(parent, text="Top 5 Predictions (Click to Add):", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(20, 5))
        self.top5_frame = ttk.Frame(parent)
        self.top5_frame.pack(fill=tk.X, padx=15)
        self.top5_buttons = []
        for i in range(5):
            btn = tk.Button(self.top5_frame, text=f"{i+1}. ...", font=FONT_MEDIUM, command=lambda idx=i: self.manual_add_word(idx))
            btn.pack(fill=tk.X, pady=2)
            self.top5_buttons.append(btn)

        tk.Label(parent, text="Word Buffer:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.buffer_var_manual = tk.StringVar()
        self.buffer_label_manual = tk.Label(parent, textvariable=self.buffer_var_manual, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, height=2)
        self.buffer_label_manual.pack(fill=tk.X, padx=15)

        tk.Label(parent, text="Generated Sentence:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.sentence_var_manual = tk.StringVar()
        self.sentence_label_manual = tk.Label(parent, textvariable=self.sentence_var_manual, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, fg="green")
        self.sentence_label_manual.pack(fill=tk.X, padx=15)

        tk.Button(parent, text="Play Audio", command=self.play_audio, font=FONT_SMALL).pack(anchor='w', padx=15, pady=5)

        instructions = "Controls:\n[G] Generate Sentence\n[C] Clear Session\n[Q] Quit"
        tk.Label(parent, text=instructions, font=FONT_SMALL, anchor='w', justify=tk.LEFT, fg="gray").pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=20)

    def setup_tab_translate(self):
        """Setup Tab 3: Text to Sign Translation"""
        from scripts.text_to_sign_tab import TextToSignTab
        
        self.translate_view = TextToSignTab(self.content_frame, gemini_model=self.gemini_model)

    def setup_tab_dictionary(self):
        """Setup Tab 4: Dictionary Mode"""
        from scripts.dictionary_tab import DictionaryTab
        
        self.dictionary_view = DictionaryTab(self.content_frame)

    def _switch_tab(self, tab_index: int):
        """Switch between tabs - show/hide appropriate views"""
        # Hide all views first
        self.recognition_view.pack_forget()
        self.translate_view.pack_forget()
        self.dictionary_view.pack_forget()
        self.tab_auto.pack_forget()
        self.tab_manual.pack_forget()
        
        if tab_index == 0:  # Automatic mode
            # Show recognition view (camera + auto info)
            self.recognition_view.pack(fill=tk.BOTH, expand=True)
            self.tab_auto.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        elif tab_index == 1:  # Manual mode
            # Show recognition view (camera + manual info)
            self.recognition_view.pack(fill=tk.BOTH, expand=True)
            self.tab_manual.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        elif tab_index == 2:  # Text → Sign
            # Show translate view (full width)
            self.translate_view.pack(fill=tk.BOTH, expand=True)
        else:  # Tab 4: Dictionary Mode
            # Show dictionary view (full width)
            self.dictionary_view.pack(fill=tk.BOTH, expand=True)
        
        self.current_tab.set(tab_index)

    # ---------------- UI updates ----------------

    def update_gui_info(self):
        # Auto tab
        self.sign_var_auto.set(self.current_sign)
        self.sign_label_auto.config(fg=self.current_sign_color)

        buffer_text = " ".join(self.recognised_words_buffer) if self.recognised_words_buffer else "(Empty)"
        self.buffer_var_auto.set(buffer_text)
        self.sentence_var_auto.set(self.display_sentence)

        # Manual tab
        self.buffer_var_manual.set(buffer_text)
        self.sentence_var_manual.set(self.display_sentence)

        # Top-5 buttons
        if self.top_5_predictions:
            for i, btn in enumerate(self.top5_buttons):
                if i < len(self.top_5_predictions):
                    sign, conf = self.top_5_predictions[i]
                    btn.config(text=f"{i+1}. {sign} ({conf:.2f})", state=tk.NORMAL)
                else:
                    btn.config(text="...", state=tk.DISABLED)
        else:
            for btn in self.top5_buttons:
                btn.config(text="...", state=tk.DISABLED)

    def manual_add_word(self, idx: int):
        if self.top_5_predictions and idx < len(self.top_5_predictions):
            sign, _ = self.top_5_predictions[idx]
            self.recognised_words_buffer.append(sign)
            self.update_gui_info()

    def play_audio(self):
        if self.display_sentence and self.tts_engine:
            threading.Thread(target=self._speak_thread, args=(self.display_sentence,), daemon=True).start()

    def _speak_thread(self, text: str):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass

    # ---------------- Gemini / session ----------------

    def generate_sentence(self, event=None):
        if not self.recognised_words_buffer:
            return
        words = list(self.recognised_words_buffer)
        self.current_sign = "(Generating...)"
        self.current_sign_color = "green"
        self.update_gui_info()
        threading.Thread(target=self._gemini_thread, args=(words,), daemon=True).start()

    def _gemini_thread(self, recognised_words: List[str]):
        sentence = self.get_display_message_from_api(recognised_words)
        # Marshal back to UI thread
        def _apply():
            self.display_sentence = sentence
            self.recognised_words_buffer = []
            self.current_sign = "(Generated)"
            self.current_sign_color = "green"
            self.update_gui_info()
        self.window.after(0, _apply)

    def get_display_message_from_api(self, recognised_words: List[str]) -> str:
        if not self.gemini_model:
            return " ".join(recognised_words) + " (Gemini Error/Missing Key)"

        prompt = f"""
            Objective:
            Construct a coherent and meaningful English sentence from a list of recognized American Sign Language (ASL) words. The sentence should be simple and accurately convey the meaning.

            Instructions:
            - Input: A Python list of recognized ASL words.
            - Processing: Rearrange the words (if necessary) to form a grammatically correct sentence. Ignore the word \"TV\" if present.
            - Output: A concise English sentence.

            Input: {recognised_words}
            Output:
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            if getattr(response, 'text', None):
                return response.text.strip()
            return "[Empty Response] " + " ".join(recognised_words)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            import traceback
            traceback.print_exc()
            return "[API Error] " + " ".join(recognised_words)

    def clear_session(self, event=None):
        self.recognised_words_buffer = []
        self.display_sentence = ""
        self.current_sign = "(Cleared)"
        self.current_sign_color = "gray"
        self.top_5_predictions = []
        self.update_gui_info()

    def quit_app(self, event=None):
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        try:
            if hasattr(self, 'worker') and self.worker.is_alive():
                self.worker.join(timeout=1.0)
        except Exception:
            pass
        # Cleanup translate view
        try:
            if hasattr(self, 'translate_view'):
                self.translate_view.destroy()
        except Exception:
            pass
        # Cleanup dictionary view
        try:
            if hasattr(self, 'dictionary_view'):
                self.dictionary_view.destroy()
        except Exception:
            pass
        try:
            self.window.destroy()
        except Exception:
            self.window.quit()

    # ---------------- Main UI loop ----------------

    def update_loop(self):
        if not self.running:
            return

        # Pull latest worker update (non-blocking)
        frame_bgr = None
        pred_state = None
        try:
            while True:
                frame_bgr, pred_state = self.state_queue.get_nowait()
        except queue.Empty:
            pass

        if frame_bgr is not None:
            # Update prediction state
            if pred_state is not None:
                self.current_sign = pred_state.current_sign
                self.current_sign_color = pred_state.current_sign_color
                self.last_motion = pred_state.motion
                self.last_fps = pred_state.fps

                self.top_5_predictions = list(pred_state.top5)

                # Auto-commit word only in Auto tab (tab index 0)
                if pred_state.commit_word:
                    if self.current_tab.get() == 0:
                        if not self.recognised_words_buffer or self.recognised_words_buffer[-1] != pred_state.commit_word:
                            self.recognised_words_buffer.append(pred_state.commit_word)

                self.update_gui_info()

            # Overlay small debug line (FPS / motion)
            try:
                cv2.putText(
                    frame_bgr,
                    f"FPS: {self.last_fps:.1f} | Motion: {self.last_motion:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # Resize to fit panel using OpenCV (fast)
            panel_h = self.camera_frame.winfo_height()
            panel_w = self.camera_frame.winfo_width()
            frame_bgr_rs = _safe_resize_bgr_to_fit(frame_bgr, panel_w, panel_h)

            # BGR -> RGB -> Tk image
            frame_rgb = cv2.cvtColor(frame_bgr_rs, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        self.window.after(self.delay, self.update_loop)


def run_inference_gui(camera_index: int = 0):
    root = tk.Tk()
    app = ASLAppOptimized(root, "ASL Recognition System", camera_index=camera_index)
    root.mainloop()


if __name__ == "__main__":
    idx = int(os.getenv("CAMERA_INDEX", "0"))
    run_inference_gui(camera_index=idx)
