import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import torch
import torch.nn.functional as F
import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from scripts.config import N_ROWS, N_DIMS, DEVICE
from scripts.model import ASLTransformerModel
from scripts.preprocess import PreprocessLayer
from scripts.utils import load_data_maps

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, 'src'))
except NameError:
    sys.path.append(os.path.abspath('src'))

load_dotenv()

MODEL_PATH = os.path.join("models", "model_best_fold_1.pth")
CSV_PATH = "data/train.csv"

PREDICTION_INTERVAL = 1.5
CONFIDENCE_THRESHOLD = 0.3
MOVEMENT_THRESHOLD = 0.015

# GUI Configuration
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 720
CAMERA_WEIGHT = 65
INFO_WEIGHT = 35

FONT_LARGE = ("Helvetica", 18, "bold")
FONT_MEDIUM = ("Helvetica", 14)
FONT_SMALL = ("Helvetica", 12)

def get_xyz_skel():
    types = ["face"] * 468 + ["pose"] * 33 + ["left_hand"] * 21 + ["right_hand"] * 21
    landmark_indices = list(range(468)) + list(range(33)) + list(range(21)) + list(range(21))
    return pd.DataFrame({"type": types, "landmark_index": landmark_indices})

XYZ_SKEL = get_xyz_skel()

def create_frame_landmark_df(results, frame):
    face, pose, left_hand, right_hand = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ["x", "y", "z"]] = [point.x, point.y, point.z]
    
    face = face.reset_index().rename(columns={"index": "landmark_index"}).assign(type="face")
    pose = pose.reset_index().rename(columns={"index": "landmark_index"}).assign(type="pose")
    left_hand = left_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="left_hand")
    right_hand = right_hand.reset_index().rename(columns={"index": "landmark_index"}).assign(type="right_hand")
    
    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = XYZ_SKEL.merge(landmarks, on=["type", "landmark_index"], how="left").assign(frame=frame)
    
    return landmarks

class ASLApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.window.minsize(800, 600) 
        
        # Initialize Components 
        self.initialize_components()
        
        # Initialize State Variables
        self.sequence_data = []
        self.frame_num = 0
        self.last_pred_time = time.time()
        self.recognised_words_buffer = []
        self.current_sign = "Initializing..."
        self.display_sentence = ""
        self.current_sign_color = "blue"

        # Setup GUI Layout
        self.setup_gui()
        
        # Bind Keyboard Events
        self.window.bind('<KeyPress-g>', self.generate_sentence)
        self.window.bind('<KeyPress-G>', self.generate_sentence)
        self.window.bind('<KeyPress-c>', self.clear_session)
        self.window.bind('<KeyPress-C>', self.clear_session)
        self.window.bind('<KeyPress-q>', self.quit_app)
        self.window.bind('<KeyPress-Q>', self.quit_app)

        # Start the video update loop
        self.delay = 15 # milliseconds
        self.update_frame()
        
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()

    def initialize_components(self):
        """Loads models, mappings, and APIs."""
        # 1. Load Label Mappings
        _, self.ORD2SIGN = load_data_maps(CSV_PATH)
        if not self.ORD2SIGN: sys.exit(1)

        # 2. Initialize PyTorch Model
        self.model = ASLTransformerModel()
        model_path_abs = os.path.abspath(MODEL_PATH)
        if os.path.exists(model_path_abs):
            try:
                state_dict = torch.load(model_path_abs, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                self.model.to(DEVICE)
                self.model.eval()
                print(f"Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}."); sys.exit(1)
        else:
            print(f"Error: Model weights not found at {model_path_abs}."); sys.exit(1)

        # 3. Initialize Preprocessing Layer
        self.preprocess_layer = PreprocessLayer().to(DEVICE)
        self.preprocess_layer.eval()

        # 4. Initialize MediaPipe and OpenCV Capture
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam."); sys.exit(1)

        # 5. Initialize Gemini
        self.gemini_model = None
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
                print("Gemini API (2.5 Pro) Initialized.")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini API: {e}.")
        else:
            print("Warning: GOOGLE_API_KEY not found.")

    def on_info_frame_resize(self, event):
        padding = 30 # Tổng padding ngang (15px trái + 15px phải)
        new_wraplength = event.width - padding
        if new_wraplength > padding:
            # Cập nhật thuộc tính wraplength cho các nhãn cần xuống dòng
            self.buffer_label.config(wraplength=new_wraplength)
            self.sentence_label.config(wraplength=new_wraplength)

    def setup_gui(self):
        """Creates the Tkinter layout using Grid for better responsiveness."""
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Cấu hình Grid layout cho main_frame
        self.main_frame.grid_rowconfigure(0, weight=1)
        # Cấu hình cột với trọng số tương ứng
        self.main_frame.grid_columnconfigure(0, weight=CAMERA_WEIGHT)
        self.main_frame.grid_columnconfigure(1, weight=INFO_WEIGHT)

        # Left Panel (Camera)
        self.camera_frame = ttk.Frame(self.main_frame)
        self.camera_frame.grid(row=0, column=0, sticky="nsew")
        self.camera_frame.pack_propagate(False) 
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True, anchor='center')

        # Right Panel (Information)
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.grid(row=0, column=1, sticky="nsew")
        self.info_frame.pack_propagate(False) 

        # 1. Current Sign Status
        tk.Label(self.info_frame, text="Current Sign / Status:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(20, 5))
        self.sign_var = tk.StringVar()
        self.sign_label = tk.Label(self.info_frame, textvariable=self.sign_var, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT)
        self.sign_label.pack(fill=tk.X, padx=15)

        # 2. Word Buffer
        tk.Label(self.info_frame, text="Word Buffer:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.buffer_var = tk.StringVar()
        self.buffer_label = tk.Label(self.info_frame, textvariable=self.buffer_var, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, height=2)
        self.buffer_label.pack(fill=tk.X, padx=15)

        # 3. Generated Sentence
        tk.Label(self.info_frame, text="Generated Sentence:", font=FONT_LARGE, anchor='w').pack(fill=tk.X, padx=15, pady=(30, 5))
        self.sentence_var = tk.StringVar()

        self.sentence_label = tk.Label(self.info_frame, textvariable=self.sentence_var, font=FONT_MEDIUM, anchor='w', justify=tk.LEFT, wraplength=1, fg="green")
        self.sentence_label.pack(fill=tk.X, padx=15)
        
        # 4. Instructions
        instructions = "Controls:\n[G] Generate Sentence\n[C] Clear Session\n[Q] Quit"
        tk.Label(self.info_frame, text=instructions, font=FONT_SMALL, anchor='w', justify=tk.LEFT, fg="gray").pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=20)
        self.info_frame.bind("<Configure>", self.on_info_frame_resize)

        self.update_gui_info()

    def update_gui_info(self):
        """Updates the text variables for the GUI labels."""
        self.sign_var.set(self.current_sign)
        self.sign_label.config(fg=self.current_sign_color)
        self.buffer_var.set(" ".join(self.recognised_words_buffer) if self.recognised_words_buffer else "(Empty)")
        self.sentence_var.set(self.display_sentence)

    def generate_sentence(self, event=None):
        if self.recognised_words_buffer:
            print(f"\nGenerating sentence for: {self.recognised_words_buffer}")
            self.display_sentence = self.get_display_message_from_api(self.recognised_words_buffer)
            print(f"Generated: {self.display_sentence}\n")
            self.recognised_words_buffer = []
            self.current_sign = "(Generated)"
            self.current_sign_color = "green"
            self.update_gui_info()

    def clear_session(self, event=None):
        print("Clearing session.")
        self.recognised_words_buffer = []
        self.display_sentence = ""
        self.current_sign = "(Cleared)"
        self.current_sign_color = "gray"
        self.sequence_data = []
        self.frame_num = 0
        self.update_gui_info()

    def quit_app(self, event=None):
        print("Quitting application.")
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.window.quit()

    def get_display_message_from_api(self, recognised_words):
        if not self.gemini_model:
            return " ".join(recognised_words) + " (Gemini Error/Missing Key)"
        
        prompt = f"""
                Objective:
                Construct a coherent and meaningful English sentence from a list of recognized American Sign Language (ASL) words. The sentence should be simple and accurately convey the meaning.

                Instructions:
                - Input: A Python list of recognized ASL words.
                - Processing: Rearrange the words (if necessary) to form a grammatically correct sentence. Ignore the word "TV" if present.
                - Output: A concise English sentence.
                
                Input: {recognised_words}
                Output:
                """
        try:
            response = self.gemini_model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            else:
                return "[Empty Response] " + " ".join(recognised_words)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "[API Error] " + " ".join(recognised_words)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.quit_app()
            return

        self.frame_num += 1
        
        # Process frame with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)

        # Extract landmarks
        landmarks_df = create_frame_landmark_df(results, self.frame_num)
        self.sequence_data.append(landmarks_df)

        # Prediction Logic 
        current_time = time.time()
        if current_time - self.last_pred_time > PREDICTION_INTERVAL and len(self.sequence_data) > 5:
            self.last_pred_time = current_time
            self.process_sequence()
            self.update_gui_info()

        # Display Camera Feed 
        image_display = cv2.flip(image_rgb, 1)
        
        # Điều chỉnh kích thước ảnh để vừa với panel camera
        panel_h = self.camera_frame.winfo_height()
        panel_w = self.camera_frame.winfo_width()
        img_h, img_w, _ = image_display.shape
        if img_w > 0 and img_h > 0 and panel_w > 1 and panel_h > 1:
            ratio = min(panel_w/img_w, panel_h/img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)

            if new_w > 0 and new_h > 0:
                img_pil = Image.fromarray(image_display)
                try:
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except AttributeError:
                    img_pil = img_pil.resize((new_w, new_h), Image.ANTIALIAS)

                # Chuyển sang ImageTk và cập nhật Label
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

        # Lên lịch cho lần cập nhật tiếp theo
        self.window.after(self.delay, self.update_frame)

    def process_sequence(self):
        """Handles movement detection and model inference."""
        full_sequence_df = pd.concat(self.sequence_data).reset_index(drop=True)

        # Movement Detection Logic
        hand_data = full_sequence_df[full_sequence_df['type'].isin(['left_hand', 'right_hand'])]
        detected_hand_data = hand_data.dropna(subset=['x', 'y'])

        if detected_hand_data.empty:
            self.current_sign = "(No Hands Detected)"
            self.current_sign_color = "red"
            self.sequence_data = []; self.frame_num = 0
            return
        
        # Phân tích chuyển động (Standard Deviation)
        std_devs = detected_hand_data.groupby(['type', 'landmark_index'])[['x', 'y']].std()
        mean_movement = std_devs.mean().mean() 

        if mean_movement < MOVEMENT_THRESHOLD:
            self.current_sign = f"(Idling: {mean_movement:.4f})"
            self.current_sign_color = "red"
            self.sequence_data = []; self.frame_num = 0
            return
        
        # Model Inference
        n_frames = full_sequence_df['frame'].nunique()
        
        try:
            xyz_data = full_sequence_df[['x', 'y', 'z']].values
            input_data_np = xyz_data.reshape(n_frames, N_ROWS, N_DIMS).astype(np.float32)
        except ValueError:
            self.current_sign = "(Data Error)"; self.current_sign_color = "red"
            self.sequence_data = []; self.frame_num = 0
            return

        input_tensor = torch.tensor(input_data_np, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            # Preprocessing
            processed_data, non_empty_idxs = self.preprocess_layer(input_tensor)
            processed_data = processed_data.unsqueeze(0)
            non_empty_idxs = non_empty_idxs.unsqueeze(0)

            # Inference
            if non_empty_idxs.max() != -1.0: 
                output = self.model(processed_data, non_empty_idxs)
                
                probabilities = F.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, prediction].item()

                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_sign = self.ORD2SIGN.get(prediction, "Unknown")
                    self.current_sign = f"{predicted_sign} ({confidence:.2f})"
                    self.current_sign_color = "blue"
                    
                    # Deduplication logic
                    if not self.recognised_words_buffer or self.recognised_words_buffer[-1] != predicted_sign:
                        self.recognised_words_buffer.append(predicted_sign)
                        print(f"Added: {predicted_sign} (Movement: {mean_movement:.4f})")
                else:
                    self.current_sign = "(Low Confidence)"
                    self.current_sign_color = "orange"
            else:
                self.current_sign = "(Dominant Hand Error)"
                self.current_sign_color = "red"

        # Reset sequence data after prediction
        self.sequence_data = []
        self.frame_num = 0

def run_inference_gui():
    root = tk.Tk()
    app = ASLApp(root, "ASL Recognition System (Mediapipe + Transformer Model)")

if __name__ == "__main__":
    run_inference_gui()