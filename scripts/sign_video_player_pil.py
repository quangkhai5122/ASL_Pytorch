"""
Sign Video Player (PIL Version) - Vẽ skeleton bằng PIL thay vì Matplotlib
Nhanh hơn nhiều và không có memory leak từ matplotlib figures.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
from typing import Optional, List, Tuple
import time
from PIL import Image, ImageDraw, ImageTk

from scripts.slp_config import (
    IDX_LIPS, IDX_FACE_OVAL, IDX_EYEBROWS, IDX_EYES,
    IDX_LEFT_HAND, IDX_RIGHT_HAND, IDX_POSE,
    IDX_LEFT_EYE_LOCAL, IDX_RIGHT_EYE_LOCAL,
    IDX_LEFT_EYEBROW_LOCAL, IDX_RIGHT_EYEBROW_LOCAL,
    POSE_CONNECTIONS, POSE_HAND_CONNECTIONS,
    HAND_FINGER_CHAINS, FINGER_COLORS
)


# Convert color names to RGB tuples
COLOR_MAP = {
    'darkred': (139, 0, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 128, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'pink': (255, 192, 203),
    'lime': (0, 255, 0),
}

# Finger colors as RGB
FINGER_COLORS_RGB = [
    (255, 0, 0),      # Thumb - red
    (255, 165, 0),    # Index - orange
    (0, 255, 0),      # Middle - green
    (0, 0, 255),      # Ring - blue
    (128, 0, 128),    # Pinky - purple
]


class SignVideoPlayerPIL(ttk.Frame):
    """
    Tkinter widget để play landmark animation.
    Vẽ bằng PIL ImageDraw - nhanh hơn Matplotlib rất nhiều.
    """
    
    SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    def __init__(self, parent, width: int = 400, height: int = 400, fps: int = 25):
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
        
        # Playback control
        self._play_job: Optional[str] = None  # after() job id
        
        # Current image reference (to prevent garbage collection)
        self._current_imgtk: Optional[ImageTk.PhotoImage] = None
        
        # Setup UI
        self._setup_ui()
        
        # Show placeholder
        self._show_placeholder("No video loaded")
    
    def _setup_ui(self):
        """Setup UI components"""
        # Video display area
        self.video_frame = ttk.Frame(self)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame, bg="white")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Speed control
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
        """Load landmark sequence để play."""
        # Stop current playback
        self.stop()
        
        self.landmarks = landmarks
        self.total_frames = landmarks.shape[0]
        self.current_frame = 0
        
        # Update progress bar
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
        self._schedule_next_frame()
    
    def pause(self):
        """Pause animation"""
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        
        # Cancel scheduled frame
        if self._play_job is not None:
            try:
                self.after_cancel(self._play_job)
            except (tk.TclError, ValueError):
                pass
            self._play_job = None
    
    def stop(self):
        """Stop animation and reset"""
        self.pause()
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
    
    def _schedule_next_frame(self):
        """Schedule next frame display using after()"""
        if not self.is_playing or self.landmarks is None:
            return
        
        # Draw current frame
        self._draw_frame(self.current_frame)
        self._update_progress()
        
        # Advance frame
        self.current_frame += 1
        
        # Handle end of video
        if self.current_frame >= self.total_frames:
            if self.loop:
                self.current_frame = 0
            else:
                self.pause()
                return
        
        # Schedule next frame
        delay = int(self.frame_delay / self.speed)
        self._play_job = self.after(delay, self._schedule_next_frame)
    
    def _draw_frame(self, frame_idx: int):
        """Draw a single frame using PIL"""
        if self.landmarks is None or frame_idx >= self.total_frames:
            return
        
        frame_data = self.landmarks[frame_idx]
        
        # Create image
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw components
        self._draw_face_pil(draw, frame_data)
        self._draw_pose_pil(draw, frame_data)
        self._draw_hand_pil(draw, frame_data, IDX_LEFT_HAND)
        self._draw_hand_pil(draw, frame_data, IDX_RIGHT_HAND)
        
        # Convert to PhotoImage and display
        self._current_imgtk = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._current_imgtk, text='')
    
    def _to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized coords (0-1) to pixel coords"""
        # Add margin and scale
        margin = 0.1
        scale_x = self.width / (1 + 2 * margin)
        scale_y = self.height / (1 + 2 * margin)
        
        px = int((x + margin) * scale_x)
        py = int((y + margin) * scale_y)
        return px, py
    
    def _draw_line(self, draw: ImageDraw.Draw, p1: np.ndarray, p2: np.ndarray, 
                   color: Tuple[int, int, int], width: int = 2):
        """Draw a line between two points"""
        if self._is_invalid_point(p1) or self._is_invalid_point(p2):
            return
        
        x1, y1 = self._to_pixel(p1[0], p1[1])
        x2, y2 = self._to_pixel(p2[0], p2[1])
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    
    def _draw_polygon(self, draw: ImageDraw.Draw, pts: np.ndarray, 
                      color: Tuple[int, int, int], width: int = 2):
        """Draw a closed polygon"""
        if self._is_invalid(pts):
            return
        
        points = []
        for i in range(len(pts)):
            if not self._is_invalid_point(pts[i]):
                px, py = self._to_pixel(pts[i, 0], pts[i, 1])
                points.append((px, py))
        
        if len(points) >= 2:
            # Close the polygon
            points.append(points[0])
            draw.line(points, fill=color, width=width)
    
    def _draw_polyline(self, draw: ImageDraw.Draw, pts: np.ndarray,
                       color: Tuple[int, int, int], width: int = 2):
        """Draw a polyline (not closed)"""
        if self._is_invalid(pts):
            return
        
        points = []
        for i in range(len(pts)):
            if not self._is_invalid_point(pts[i]):
                px, py = self._to_pixel(pts[i, 0], pts[i, 1])
                points.append((px, py))
        
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
    
    def _draw_face_pil(self, draw: ImageDraw.Draw, frame: np.ndarray):
        """Draw face landmarks using PIL"""
        face_color = (139, 0, 0)  # darkred
        
        # Face Oval
        oval_pts = frame[IDX_FACE_OVAL]
        self._draw_polygon(draw, oval_pts, face_color, width=2)
        
        # Lips
        lips_pts = frame[IDX_LIPS]
        self._draw_polygon(draw, lips_pts, face_color, width=2)
        
        # Eyebrows
        eyebrow_pts = frame[IDX_EYEBROWS]
        if not self._is_invalid(eyebrow_pts):
            left_eb = eyebrow_pts[IDX_LEFT_EYEBROW_LOCAL]
            self._draw_polyline(draw, left_eb, face_color, width=2)
            right_eb = eyebrow_pts[IDX_RIGHT_EYEBROW_LOCAL]
            self._draw_polyline(draw, right_eb, face_color, width=2)
        
        # Eyes
        eye_pts = frame[IDX_EYES]
        if not self._is_invalid(eye_pts):
            left_eye = eye_pts[IDX_LEFT_EYE_LOCAL]
            self._draw_polygon(draw, left_eye, face_color, width=2)
            right_eye = eye_pts[IDX_RIGHT_EYE_LOCAL]
            self._draw_polygon(draw, right_eye, face_color, width=2)
    
    def _draw_pose_pil(self, draw: ImageDraw.Draw, frame: np.ndarray):
        """Draw pose skeleton using PIL"""
        pose_color = (255, 0, 0)  # red
        
        # Check hand validity
        left_hand_valid = self._is_hand_valid(frame, IDX_LEFT_HAND)
        right_hand_valid = self._is_hand_valid(frame, IDX_RIGHT_HAND)
        
        # Draw pose connections
        for idx1, idx2 in POSE_CONNECTIONS:
            self._draw_line(draw, frame[idx1], frame[idx2], pose_color, width=3)
        
        # Draw pose-to-hand connections (only if hand valid)
        for idx1, idx2 in POSE_HAND_CONNECTIONS:
            if idx2 == IDX_LEFT_HAND[0] and not left_hand_valid:
                continue
            if idx2 == IDX_RIGHT_HAND[0] and not right_hand_valid:
                continue
            self._draw_line(draw, frame[idx1], frame[idx2], pose_color, width=3)
    
    def _draw_hand_pil(self, draw: ImageDraw.Draw, frame: np.ndarray, hand_indices: np.ndarray):
        """Draw hand with colored fingers using PIL"""
        if not self._is_hand_valid(frame, hand_indices):
            return
        
        hand_pts = frame[hand_indices]
        
        for chain, color in zip(HAND_FINGER_CHAINS, FINGER_COLORS_RGB):
            chain_pts = hand_pts[chain]
            self._draw_polyline(draw, chain_pts, color, width=2)
    
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
                return True
        
        return False
    
    def _show_placeholder(self, text: str):
        """Show placeholder text"""
        self._current_imgtk = None
        self.video_label.configure(image='', text=text, fg="gray",
                                   font=("Helvetica", 11), compound="center")
    
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
        self._current_imgtk = None
        self.landmarks = None
        super().destroy()


# Export as SignVideoPlayer for backward compatibility
SignVideoPlayer = SignVideoPlayerPIL
