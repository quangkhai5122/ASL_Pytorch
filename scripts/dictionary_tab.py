"""
Dictionary Tab - Tab tra cứu từ điển ASL
Hiển thị video MP4 và skeleton visualization song song
"""
import tkinter as tk
from tkinter import ttk
import threading
import os
import glob
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image, ImageTk
import time

# Import for skeleton visualization
from scripts.sign_video_player import SignVideoPlayer
from scripts.slp_config import (
    FACE_RAW_IDXS, LEFT_HAND_RAW_IDXS, RIGHT_HAND_RAW_IDXS,
    POSE_RAW_IDXS, N_AVATAR_LANDMARKS
)

# Try to load parquet
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class VideoPlayer(ttk.Frame):
    """
    Simple MP4 video player using OpenCV and Tkinter.
    """
    
    SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    def __init__(self, parent, width: int = 400, height: int = 350):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.fps: float = 30.0
        self.speed: float = 1.0
        self.is_playing: bool = False
        self.loop: bool = True
        
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._setup_ui()
        self._show_placeholder("No video loaded")
    
    def _setup_ui(self):
        """Setup UI components"""
        # Video display area
        self.video_frame = ttk.Frame(self)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
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
    
    def load_video(self, video_path: str, autoplay: bool = True):
        """Load a video file"""
        self.stop()
        
        if not os.path.exists(video_path):
            self._show_placeholder("Video file not found")
            return False
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self._show_placeholder("Could not open video")
            return False
        
        self.video_path = video_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0
        
        # Update progress bar
        self.progress_bar.configure(to=max(1, self.total_frames - 1))
        self._update_frame_label()
        
        # Show first frame
        self._show_frame(0)
        
        if autoplay:
            self.play()
        
        return True
    
    def play(self):
        """Start playing video"""
        if self.cap is None or self.is_playing:
            return
        
        self.is_playing = True
        self.btn_play.configure(text="⏸ Pause")
        self._stop_event.clear()
        
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()
    
    def pause(self):
        """Pause video"""
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        self._stop_event.set()
    
    def stop(self):
        """Stop and reset video"""
        self.pause()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
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
        self._show_frame(0)
        self._update_progress()
        if was_playing:
            self.play()
    
    def seek_end(self):
        """Seek to end"""
        self.pause()
        if self.total_frames > 0:
            self.current_frame = self.total_frames - 1
            self._show_frame(self.current_frame)
            self._update_progress()
    
    def _play_loop(self):
        """Background thread for video playback"""
        while not self._stop_event.is_set() and self.is_playing:
            if self.cap is None:
                break
            
            # Show current frame
            self.after(0, self._show_frame, self.current_frame)
            self.after(0, self._update_progress)
            
            # Advance frame
            self.current_frame += 1
            
            # Handle end of video
            if self.current_frame >= self.total_frames:
                if self.loop:
                    self.current_frame = 0
                else:
                    self.after(0, self.pause)
                    break
            
            # Wait based on FPS and speed
            delay = 1.0 / (self.fps * self.speed)
            time.sleep(delay)
    
    def _show_frame(self, frame_idx: int):
        """Display a specific frame"""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit
        h, w = frame_rgb.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text='')  # Clear placeholder text
    
    def _show_placeholder(self, text: str):
        """Show placeholder text"""
        self.video_label.configure(image='', text=text, fg="gray", 
                                   font=("Helvetica", 12), compound="center")
    
    def _update_progress(self):
        """Update progress bar"""
        if self.total_frames > 0:
            self.progress_var.set(self.current_frame)
        self._update_frame_label()
    
    def _update_frame_label(self):
        """Update frame counter"""
        self.frame_label.configure(text=f"{self.current_frame + 1}/{self.total_frames}")
    
    def _on_progress_change(self, value):
        """Handle progress bar drag"""
        if self.cap is not None:
            frame = int(float(value))
            if frame != self.current_frame:
                self.current_frame = frame
                self._show_frame(frame)
                self._update_frame_label()
    
    def _on_loop_change(self):
        """Handle loop checkbox"""
        self.loop = self.loop_var.get()
    
    def _on_speed_change(self, event=None):
        """Handle speed change"""
        speed_str = self.speed_var.get().replace("x", "")
        try:
            self.speed = float(speed_str)
        except ValueError:
            self.speed = 1.0
    
    def destroy(self):
        """Clean up resources"""
        self.stop()
        super().destroy()


class DictionaryTab(ttk.Frame):
    """
    Dictionary Mode Tab - Tra cứu từ điển ASL
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Paths
        self.video_folder = "data/WLASL_Only1Video"
        self.skeleton_folder = "data/WLASL_Skeleton"
        
        # Build index of available words
        self._build_word_index()
        
        # Setup UI
        self._setup_ui()
    
    def _build_word_index(self):
        """Build index of available words from both folders"""
        self.video_files = {}  # word -> path
        self.skeleton_files = {}  # word -> path
        
        # Index video files
        if os.path.exists(self.video_folder):
            for filepath in glob.glob(os.path.join(self.video_folder, "**", "*.mp4"), recursive=True):
                word = os.path.splitext(os.path.basename(filepath))[0].lower()
                self.video_files[word] = filepath
        
        # Index skeleton files
        if os.path.exists(self.skeleton_folder):
            for filepath in glob.glob(os.path.join(self.skeleton_folder, "**", "*.parquet"), recursive=True):
                word = os.path.splitext(os.path.basename(filepath))[0].lower()
                self.skeleton_files[word] = filepath
        
        # All available words
        self.all_words = sorted(set(self.video_files.keys()) | set(self.skeleton_files.keys()))
        
        print(f"DictionaryTab: Found {len(self.video_files)} videos, {len(self.skeleton_files)} skeletons")
    
    def _setup_ui(self):
        """Setup the UI layout"""
        # Configure grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # ===== Search bar (Google-like) =====
        self.search_frame = ttk.Frame(self)
        self.search_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=15)
        self.search_frame.columnconfigure(0, weight=1)
        
        # Search container with border effect
        search_container = ttk.Frame(self.search_frame)
        search_container.grid(row=0, column=0, sticky="ew")
        search_container.columnconfigure(0, weight=1)
        
        ttk.Label(search_container, text="🔍", font=("Helvetica", 14)).pack(side=tk.LEFT, padx=(10, 5))
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(
            search_container,
            textvariable=self.search_var,
            font=("Helvetica", 14),
            width=50
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=8)
        self.search_entry.bind("<Return>", self._on_search)
        self.search_entry.bind("<KeyRelease>", self._on_key_release)
        
        self.search_btn = ttk.Button(
            search_container,
            text="Search",
            command=self._on_search,
            width=10
        )
        self.search_btn.pack(side=tk.RIGHT, padx=10)
        
        # Autocomplete listbox (hidden by default)
        self.autocomplete_frame = ttk.Frame(self.search_frame)
        self.autocomplete_listbox = tk.Listbox(
            self.autocomplete_frame,
            height=5,
            font=("Helvetica", 12)
        )
        self.autocomplete_listbox.pack(fill=tk.BOTH, expand=True)
        self.autocomplete_listbox.bind("<Double-Button-1>", self._on_autocomplete_select)
        self.autocomplete_listbox.bind("<Return>", self._on_autocomplete_select)
        
        # Tab key to move focus to autocomplete list
        self.search_entry.bind("<Tab>", self._on_tab_to_autocomplete)
        # Arrow keys to navigate autocomplete
        self.search_entry.bind("<Down>", self._on_arrow_down)
        self.autocomplete_listbox.bind("<Up>", self._on_arrow_up_in_list)
        self.autocomplete_listbox.bind("<Escape>", self._on_escape_autocomplete)
        
        # Word count info
        self.info_label = ttk.Label(
            self.search_frame,
            text=f"Dictionary contains {len(self.all_words)} words",
            foreground="gray"
        )
        self.info_label.grid(row=2, column=0, pady=(5, 0))
        
        # ===== Content area: Two panels =====
        self.content_frame = ttk.Frame(self)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        
        # ----- Left panel: MP4 Video -----
        self.left_panel = ttk.LabelFrame(self.content_frame, text="📹 Video (MP4)")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(0, weight=1)
        
        self.video_player = VideoPlayer(self.left_panel, width=450, height=350)
        self.video_player.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # ----- Right panel: Skeleton -----
        self.right_panel = ttk.LabelFrame(self.content_frame, text="🦴 Skeleton (Landmarks)")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        
        self.skeleton_player = SignVideoPlayer(self.right_panel, width=450, height=350, fps=25)
        self.skeleton_player.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Status labels for each panel
        self.video_status = tk.StringVar(value="Enter a word to search")
        self.skeleton_status = tk.StringVar(value="Enter a word to search")
    
    def _on_key_release(self, event=None):
        """Handle key release for autocomplete"""
        query = self.search_var.get().lower().strip()
        
        if len(query) < 1:
            self.autocomplete_frame.grid_forget()
            return
        
        # Find matching words
        matches = [w for w in self.all_words if w.startswith(query)][:10]
        
        if matches:
            self.autocomplete_listbox.delete(0, tk.END)
            for word in matches:
                self.autocomplete_listbox.insert(tk.END, word)
            self.autocomplete_frame.grid(row=1, column=0, sticky="ew", padx=40)
        else:
            self.autocomplete_frame.grid_forget()
    
    def _on_autocomplete_select(self, event=None):
        """Handle autocomplete selection"""
        selection = self.autocomplete_listbox.curselection()
        if selection:
            word = self.autocomplete_listbox.get(selection[0])
            self.search_var.set(word)
            self.autocomplete_frame.grid_forget()
            self._search_word(word)
    
    def _on_search(self, event=None):
        """Handle search button click or Enter key"""
        self.autocomplete_frame.grid_forget()
        word = self.search_var.get().lower().strip()
        if word:
            self._search_word(word)
    
    def _on_tab_to_autocomplete(self, event=None):
        """Handle Tab key - move focus to autocomplete list"""
        if self.autocomplete_listbox.size() > 0:
            self.autocomplete_listbox.focus_set()
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(0)
            self.autocomplete_listbox.activate(0)
            return "break"  # Prevent default Tab behavior
    
    def _on_arrow_down(self, event=None):
        """Handle Down arrow - move to autocomplete list"""
        if self.autocomplete_listbox.size() > 0:
            self.autocomplete_listbox.focus_set()
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(0)
            self.autocomplete_listbox.activate(0)
            return "break"
    
    def _on_arrow_up_in_list(self, event=None):
        """Handle Up arrow in list - go back to search entry if at top"""
        selection = self.autocomplete_listbox.curselection()
        if selection and selection[0] == 0:
            self.search_entry.focus_set()
            return "break"
    
    def _on_escape_autocomplete(self, event=None):
        """Handle Escape - close autocomplete and return to search"""
        self.autocomplete_frame.grid_forget()
        self.search_entry.focus_set()
    
    def _search_word(self, word: str):
        """Search and display a word"""
        word = word.lower().strip()
        
        # Update info label
        self.info_label.configure(text=f"Searching: {word}")
        
        # Load video
        if word in self.video_files:
            video_path = self.video_files[word]
            self.video_player.load_video(video_path, autoplay=True)
            self.left_panel.configure(text=f"📹 Video: {word}")
        else:
            self.video_player.pause()
            self.video_player._show_placeholder("Not yet updated in the dictionary.")
            self.left_panel.configure(text="📹 Video (MP4)")
        
        # Load skeleton
        if word in self.skeleton_files:
            skeleton_path = self.skeleton_files[word]
            landmarks = self._load_skeleton(skeleton_path)
            if landmarks is not None:
                self.skeleton_player.set_landmarks(landmarks, autoplay=True)
                self.right_panel.configure(text=f"🦴 Skeleton: {word}")
            else:
                self._show_skeleton_placeholder("Error loading skeleton data.")
                self.right_panel.configure(text="🦴 Skeleton (Landmarks)")
        else:
            self._show_skeleton_placeholder("Not yet updated in the dictionary.")
            self.right_panel.configure(text="🦴 Skeleton (Landmarks)")
        
        # Update info
        video_status = "✓" if word in self.video_files else "✗"
        skeleton_status = "✓" if word in self.skeleton_files else "✗"
        self.info_label.configure(
            text=f"Word: '{word}' | Video: {video_status} | Skeleton: {skeleton_status}"
        )
    
    def _show_skeleton_placeholder(self, message: str):
        """Show placeholder message in skeleton player without destroying controls"""
        self.skeleton_player.pause()
        self.skeleton_player.landmarks = None
        self.skeleton_player.total_frames = 0
        self.skeleton_player.current_frame = 0
        self.skeleton_player._update_frame_label()
        
        # Draw placeholder
        self.skeleton_player.ax.clear()
        self.skeleton_player._setup_axes()
        self.skeleton_player.ax.text(0.5, 0.5, message,
                                     ha='center', va='center', fontsize=11, color='gray')
        self.skeleton_player.canvas.draw_idle()
    
    def _load_skeleton(self, parquet_path: str) -> Optional[np.ndarray]:
        """Load skeleton data from parquet file and convert to avatar format"""
        if not PANDAS_AVAILABLE:
            print("Pandas not available for loading parquet")
            return None
        
        try:
            df = pd.read_parquet(parquet_path)
            
            # Format: frame, row_id, type, landmark_index, x, y, z
            # Need to pivot to (T, 543, 3) then convert to avatar (T, 153, 3)
            
            if 'frame' in df.columns and 'x' in df.columns:
                # Get unique frames
                frames = sorted(df['frame'].unique())
                n_frames = len(frames)
                
                # Determine number of landmarks per frame (should be 543)
                frame0_data = df[df['frame'] == frames[0]]
                n_landmarks = len(frame0_data)
                
                # Initialize full landmarks array (543)
                full_landmarks = np.full((n_frames, 543, 3), np.nan, dtype=np.float32)
                
                # Fill data
                for i, frame_id in enumerate(frames):
                    frame_data = df[df['frame'] == frame_id]
                    
                    # Sort by type and landmark_index to get correct order
                    # Order: face(0-467), left_hand(468-488), pose(489-521), right_hand(522-542)
                    for _, row in frame_data.iterrows():
                        ltype = row.get('type', '')
                        lidx = int(row.get('landmark_index', 0))
                        
                        if ltype == 'face' and lidx < 468:
                            global_idx = lidx
                        elif ltype == 'left_hand' and lidx < 21:
                            global_idx = 468 + lidx
                        elif ltype == 'pose' and lidx < 33:
                            global_idx = 489 + lidx
                        elif ltype == 'right_hand' and lidx < 21:
                            global_idx = 522 + lidx
                        else:
                            continue
                        
                        full_landmarks[i, global_idx, 0] = row['x']
                        full_landmarks[i, global_idx, 1] = row['y']
                        full_landmarks[i, global_idx, 2] = row['z']
                
                # Convert to avatar format (153 landmarks)
                avatar_landmarks = self._convert_to_avatar(full_landmarks)
                return avatar_landmarks
            
            return None
                
        except Exception as e:
            print(f"Error loading skeleton: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_to_avatar(self, full_landmarks: np.ndarray) -> np.ndarray:
        """Convert 543 landmarks to 153 avatar landmarks"""
        n_frames = full_landmarks.shape[0]
        avatar = np.full((n_frames, N_AVATAR_LANDMARKS, 3), np.nan, dtype=np.float32)
        
        # Avatar indices mapping (from slp_config):
        # Face: 0-101 (102 points from FACE_RAW_IDXS)
        # Left Hand: 102-122 (21 points)
        # Pose: 123-131 (9 points from POSE_RAW_IDXS)
        # Right Hand: 132-152 (21 points)
        
        for t in range(n_frames):
            # Face (102 points)
            avatar[t, :len(FACE_RAW_IDXS), :] = full_landmarks[t, FACE_RAW_IDXS, :]
            
            # Left Hand (21 points)
            avatar[t, 102:123, :] = full_landmarks[t, LEFT_HAND_RAW_IDXS, :]
            
            # Pose (9 points)
            avatar[t, 123:132, :] = full_landmarks[t, POSE_RAW_IDXS, :]
            
            # Right Hand (21 points)
            avatar[t, 132:153, :] = full_landmarks[t, RIGHT_HAND_RAW_IDXS, :]
        
        return avatar
    
    def destroy(self):
        """Clean up resources"""
        if hasattr(self, 'video_player'):
            self.video_player.destroy()
        if hasattr(self, 'skeleton_player'):
            self.skeleton_player.destroy()
        super().destroy()
