"""
Sign Video Player - Tkinter widget để play animation từ landmark sequence
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
from typing import Optional, Callable
import time

# Import visualization components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from scripts.slp_config import (
    IDX_LIPS, IDX_FACE_OVAL, IDX_EYEBROWS, IDX_EYES,
    IDX_LEFT_HAND, IDX_RIGHT_HAND, IDX_POSE,
    IDX_LEFT_EYE_LOCAL, IDX_RIGHT_EYE_LOCAL,
    IDX_LEFT_EYEBROW_LOCAL, IDX_RIGHT_EYEBROW_LOCAL,
    POSE_CONNECTIONS, POSE_HAND_CONNECTIONS,
    HAND_FINGER_CHAINS, FINGER_COLORS
)


class SignVideoPlayer(ttk.Frame):
    """
    Tkinter widget để play landmark animation.
    Supports: play, pause, seek, loop, speed control.
    """
    
    # Speed options like YouTube
    SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    def __init__(self, parent, width: int = 400, height: int = 400, fps: int = 15):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        self.base_fps = fps
        self.fps = fps
        self.speed = 1.0
        self.frame_delay = int(1000 / fps)  # ms
        
        # Animation data
        self.landmarks: Optional[np.ndarray] = None  # (T, 153, 3)
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.is_playing: bool = False
        self.loop: bool = True
        
        # Threading
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Setup UI
        self._setup_ui()
        
        # Draw empty frame
        self._draw_empty()
    
    def _setup_ui(self):
        """Setup UI components"""
        # Canvas for video display
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Matplotlib figure for drawing
        self.fig, self.ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Scale(
            self.progress_frame, 
            from_=0, to=100, 
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            command=self._on_progress_change
        )
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.frame_label = ttk.Label(self.progress_frame, text="0/0", width=10)
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Control buttons
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = ttk.Button(self.control_frame, text="⏮ Start", command=self.seek_start, width=10)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_play = ttk.Button(self.control_frame, text="▶ Play", command=self.toggle_play, width=10)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_end = ttk.Button(self.control_frame, text="End ⏭", command=self.seek_end, width=10)
        self.btn_end.pack(side=tk.LEFT, padx=2)
        
        # Loop checkbox
        self.loop_var = tk.BooleanVar(value=True)
        self.loop_check = ttk.Checkbutton(
            self.control_frame, text="Loop", 
            variable=self.loop_var,
            command=self._on_loop_change
        )
        self.loop_check.pack(side=tk.RIGHT, padx=5)
        
        # Speed control (like YouTube)
        self.speed_frame = ttk.Frame(self.control_frame)
        self.speed_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(self.speed_frame, text="Speed:").pack(side=tk.LEFT)
        
        self.speed_var = tk.StringVar(value="1.0x")
        self.speed_combo = ttk.Combobox(
            self.speed_frame,
            textvariable=self.speed_var,
            values=[f"{s}x" for s in self.SPEED_OPTIONS],
            width=8,
            state="readonly"
        )
        self.speed_combo.pack(side=tk.LEFT, padx=2)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)
    
    def set_landmarks(self, landmarks: np.ndarray, autoplay: bool = True):
        """
        Load landmark sequence để play.
        
        Args:
            landmarks: Shape (T, 153, 3) - Avatar landmarks
            autoplay: Tự động play sau khi load
        """
        # Stop any existing playback first
        self.stop()
        
        # Wait for thread to fully stop
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)
        self._play_thread = None
        
        self.landmarks = landmarks
        self.total_frames = landmarks.shape[0]
        self.current_frame = 0
        
        # Update progress bar range
        self.progress_bar.configure(to=max(1, self.total_frames - 1))
        self._update_frame_label()
        
        # Draw first frame
        self._draw_frame(0)
        
        if autoplay:
            self.play()
    
    def play(self):
        """Start playing animation"""
        if self.landmarks is None or self.is_playing:
            return
        
        self.is_playing = True
        self.btn_play.configure(text="⏸ Pause")
        self._stop_event.clear()
        
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()
    
    def pause(self):
        """Pause animation"""
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        self._stop_event.set()
        
        # Wait for thread to stop
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.3)
        self._play_thread = None
    
    def stop(self):
        """Stop animation and reset"""
        self.is_playing = False
        self._stop_event.set()
        
        # Wait for thread to fully stop
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.3)
        self._play_thread = None
        
        self.btn_play.configure(text="▶ Play")
        self.current_frame = 0
        self._update_progress()
    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def seek_start(self):
        """Seek to start"""
        was_playing = self.is_playing
        self.pause()
        self.current_frame = 0
        self._draw_frame(0)
        self._update_progress()
        if was_playing:
            self.play()
    
    def seek_end(self):
        """Seek to end"""
        self.pause()
        if self.total_frames > 0:
            self.current_frame = self.total_frames - 1
            self._draw_frame(self.current_frame)
            self._update_progress()
    
    def seek(self, frame: int):
        """Seek to specific frame"""
        if self.landmarks is None:
            return
        
        frame = max(0, min(frame, self.total_frames - 1))
        self.current_frame = frame
        self._draw_frame(frame)
        self._update_progress()
    
    def _play_loop(self):
        """Background thread for animation playback"""
        while not self._stop_event.is_set() and self.is_playing:
            if self.landmarks is None:
                break
            
            # Check stop event before drawing
            if self._stop_event.is_set():
                break
            
            # Draw current frame (schedule on main thread)
            try:
                self.after(0, self._draw_frame, self.current_frame)
                self.after(0, self._update_progress)
            except tk.TclError:
                # Widget destroyed
                break
            
            # Advance frame
            self.current_frame += 1
            
            # Handle end of video
            if self.current_frame >= self.total_frames:
                if self.loop:
                    self.current_frame = 0
                else:
                    try:
                        self.after(0, self.pause)
                    except tk.TclError:
                        pass
                    break
            
            # Wait for next frame with interruptible sleep
            sleep_time = self.frame_delay / 1000.0
            # Split sleep into smaller chunks for faster response to stop
            sleep_chunk = 0.02  # 20ms chunks
            elapsed = 0.0
            while elapsed < sleep_time and not self._stop_event.is_set():
                time.sleep(min(sleep_chunk, sleep_time - elapsed))
                elapsed += sleep_chunk
    
    def _draw_frame(self, frame_idx: int):
        """Draw a single frame"""
        if self.landmarks is None or frame_idx >= self.total_frames:
            return
        
        frame_data = self.landmarks[frame_idx]
        
        self.ax.clear()
        self._setup_axes()
        
        # Draw components
        self._draw_face(frame_data)
        self._draw_pose(frame_data)
        self._draw_hand(frame_data, IDX_LEFT_HAND)
        self._draw_hand(frame_data, IDX_RIGHT_HAND)
        
        # Refresh canvas
        self.canvas.draw_idle()
    
    def _draw_empty(self):
        """Draw empty placeholder"""
        self.ax.clear()
        self._setup_axes()
        self.ax.text(0.5, 0.5, "No video loaded", 
                    ha='center', va='center', fontsize=12, color='gray')
        self.canvas.draw_idle()
    
    def _setup_axes(self):
        """Setup axes for visualization"""
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(1.1, -0.1)  # Flip Y axis
        self.ax.set_aspect('equal')
        self.ax.axis('off')
    
    def _draw_face(self, frame: np.ndarray):
        """Draw face landmarks"""
        face_color = 'darkred'
        face_lw = 1.5
        
        # Face Oval
        oval_pts = frame[IDX_FACE_OVAL]
        if not self._is_invalid(oval_pts):
            xs, ys = oval_pts[:, 0], oval_pts[:, 1]
            self.ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                        color=face_color, linewidth=face_lw)
        
        # Lips
        lips_pts = frame[IDX_LIPS]
        if not self._is_invalid(lips_pts):
            xs, ys = lips_pts[:, 0], lips_pts[:, 1]
            self.ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                        color=face_color, linewidth=face_lw)
        
        # Eyebrows
        eyebrow_pts = frame[IDX_EYEBROWS]
        if not self._is_invalid(eyebrow_pts):
            left_eb = eyebrow_pts[IDX_LEFT_EYEBROW_LOCAL]
            self.ax.plot(left_eb[:, 0], left_eb[:, 1], color=face_color, linewidth=face_lw)
            right_eb = eyebrow_pts[IDX_RIGHT_EYEBROW_LOCAL]
            self.ax.plot(right_eb[:, 0], right_eb[:, 1], color=face_color, linewidth=face_lw)
        
        # Eyes
        eye_pts = frame[IDX_EYES]
        if not self._is_invalid(eye_pts):
            left_eye = eye_pts[IDX_LEFT_EYE_LOCAL]
            xs, ys = left_eye[:, 0], left_eye[:, 1]
            self.ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                        color=face_color, linewidth=face_lw)
            right_eye = eye_pts[IDX_RIGHT_EYE_LOCAL]
            xs, ys = right_eye[:, 0], right_eye[:, 1]
            self.ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                        color=face_color, linewidth=face_lw)
    
    def _draw_pose(self, frame: np.ndarray):
        """Draw pose skeleton"""
        pose_color = 'red'
        pose_lw = 2
        
        # Check hand validity
        left_hand_valid = self._is_hand_valid(frame, IDX_LEFT_HAND)
        right_hand_valid = self._is_hand_valid(frame, IDX_RIGHT_HAND)
        
        # Draw pose connections
        for idx1, idx2 in POSE_CONNECTIONS:
            p1, p2 = frame[idx1], frame[idx2]
            if self._is_invalid_point(p1) or self._is_invalid_point(p2):
                continue
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=pose_color, linewidth=pose_lw)
        
        # Draw pose-to-hand connections (only if hand valid)
        for idx1, idx2 in POSE_HAND_CONNECTIONS:
            if idx2 == IDX_LEFT_HAND[0] and not left_hand_valid:
                continue
            if idx2 == IDX_RIGHT_HAND[0] and not right_hand_valid:
                continue
            
            p1, p2 = frame[idx1], frame[idx2]
            if self._is_invalid_point(p1) or self._is_invalid_point(p2):
                continue
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=pose_color, linewidth=pose_lw)
    
    def _draw_hand(self, frame: np.ndarray, hand_indices: np.ndarray):
        """Draw hand with colored fingers"""
        if not self._is_hand_valid(frame, hand_indices):
            return
        
        hand_pts = frame[hand_indices]
        
        for chain, color in zip(HAND_FINGER_CHAINS, FINGER_COLORS):
            chain_pts = hand_pts[chain]
            if self._is_invalid(chain_pts):
                continue
            self.ax.plot(chain_pts[:, 0], chain_pts[:, 1], 
                        color=color, linewidth=1.5)
    
    def _is_invalid(self, pts: np.ndarray) -> bool:
        """Check if points array is invalid"""
        if np.isnan(pts).all():
            return True
        if np.allclose(pts[:, :2], 0):
            return True
        return False
    
    def _is_invalid_point(self, pt: np.ndarray) -> bool:
        """Check if single point is invalid"""
        if np.isnan(pt).any():
            return True
        if np.allclose(pt[:2], 0):
            return True
        return False
    
    def _is_hand_valid(self, frame: np.ndarray, hand_indices: np.ndarray) -> bool:
        """Check if hand data is valid"""
        hand_pts = frame[hand_indices]
        
        wrist = hand_pts[0]
        if self._is_invalid_point(wrist):
            return False
        
        if self._is_invalid(hand_pts):
            return False
        
        # Check if all points are same (rest pose)
        first_point = hand_pts[0, :2]
        for i in range(1, len(hand_pts)):
            if not np.allclose(hand_pts[i, :2], first_point, atol=1e-6):
                return True  # Found different point -> valid
        
        return False  # All same -> invalid
    
    def _update_progress(self):
        """Update progress bar and label"""
        if self.total_frames > 0:
            self.progress_var.set(self.current_frame)
        self._update_frame_label()
    
    def _update_frame_label(self):
        """Update frame counter label"""
        self.frame_label.configure(text=f"{self.current_frame + 1}/{self.total_frames}")
    
    def _on_progress_change(self, value):
        """Handle progress bar drag"""
        if self.landmarks is not None:
            frame = int(float(value))
            if frame != self.current_frame:
                self.current_frame = frame
                self._draw_frame(frame)
                self._update_frame_label()
    
    def _on_loop_change(self):
        """Handle loop checkbox change"""
        self.loop = self.loop_var.get()
    
    def _on_speed_change(self, event=None):
        """Handle speed combobox change"""
        speed_str = self.speed_var.get().replace("x", "")
        try:
            self.speed = float(speed_str)
            # Recalculate frame delay based on speed
            self.frame_delay = int(1000 / (self.base_fps * self.speed))
        except ValueError:
            self.speed = 1.0
            self.frame_delay = int(1000 / self.base_fps)
    
    def set_speed(self, speed: float):
        """Programmatically set playback speed"""
        if speed in self.SPEED_OPTIONS:
            self.speed = speed
            self.speed_var.set(f"{speed}x")
            self.frame_delay = int(1000 / (self.base_fps * self.speed))
    
    def destroy(self):
        """Clean up resources"""
        self.stop()
        self.landmarks = None
        
        # Close matplotlib figure to free memory
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                plt.close(self.fig)
                self.fig = None
        except Exception:
            pass
        
        super().destroy()
