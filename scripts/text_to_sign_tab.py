"""
Text to Sign Tab - Tab dịch text thành video ngôn ngữ ký hiệu
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
from typing import Optional, List
import numpy as np

# Import sign language components
from scripts.sign_dictionary import SignDictionary
from scripts.motion_synthesizer import MotionSynthesizer
from scripts.sign_video_player import SignVideoPlayer

# Try import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TextToSignTab(ttk.Frame):
    """
    Tab để dịch text thành video ngôn ngữ ký hiệu.
    Giao diện giống Google Translate: khung trái nhập text, khung phải hiển thị video.
    """
    
    def __init__(self, parent, gemini_model=None):
        super().__init__(parent)
        
        self.gemini_model = gemini_model
        
        # Initialize sign language components
        self._init_sign_components()
        
        # Setup UI
        self._setup_ui()
        
        # State
        self.current_glosses: List[str] = []
        self.is_translating: bool = False
    
    def _init_sign_components(self):
        """Initialize sign dictionary and synthesizer"""
        try:
            self.dictionary = SignDictionary()
            self.synthesizer = MotionSynthesizer(
                self.dictionary,
                transition_frames=10,
                context_frames=3
            )
            self.available_glosses = set(self.dictionary.get_available_glosses())
            print(f"TextToSignTab: Loaded {len(self.available_glosses)} glosses")
        except Exception as e:
            print(f"Error loading sign components: {e}")
            self.dictionary = None
            self.synthesizer = None
            self.available_glosses = set()
    
    def _setup_ui(self):
        """Setup the UI layout"""
        # Main container with 2 columns (60% text, 40% video)
        self.columnconfigure(0, weight=60)
        self.columnconfigure(1, weight=40)
        self.rowconfigure(1, weight=1)
        
        # ===== Top bar with Translate button =====
        self.top_frame = ttk.Frame(self)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        ttk.Label(self.top_frame, text="Text", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        self.translate_btn = ttk.Button(
            self.top_frame, 
            text="Translate",
            command=self.translate,
            width=15
        )
        self.translate_btn.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(self.top_frame, text="Sign Language Video", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # ===== Left panel: Text input =====
        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_frame.rowconfigure(0, weight=1)
        self.left_frame.columnconfigure(0, weight=1)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            self.left_frame,
            wrap=tk.WORD,
            font=("Helvetica", 14),
            height=10
        )
        self.text_input.grid(row=0, column=0, sticky="nsew")
        self.text_input.insert(tk.END, "Enter your sentence here...")
        self.text_input.bind("<FocusIn>", self._on_text_focus_in)
        
        # Glosses preview (shows what glosses were extracted)
        self.glosses_frame = ttk.LabelFrame(self.left_frame, text="Extracted Glosses")
        self.glosses_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        self.glosses_var = tk.StringVar(value="(No glosses yet)")
        self.glosses_label = ttk.Label(
            self.glosses_frame, 
            textvariable=self.glosses_var,
            wraplength=350,
            font=("Helvetica", 11)
        )
        self.glosses_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Info about available glosses
        info_text = f"Available glosses: {len(self.available_glosses)}"
        ttk.Label(self.glosses_frame, text=info_text, foreground="gray", font=("Helvetica", 9)).pack(pady=(0, 5))
        
        # ===== Right panel: Video player =====
        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        
        # Video player
        self.video_player = SignVideoPlayer(self.right_frame, width=400, height=400, fps=25)
        self.video_player.grid(row=0, column=0, sticky="nsew")
        
        # ===== Bottom: Example sentences =====
        self.bottom_frame = ttk.LabelFrame(self, text="Example Sentences (click to use)")
        self.bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        
        examples = [
            "Hello, how are you?",
            "I want to go home",
            "Thank you very much",
            "What is your name?",
            "I love learning sign language"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(
                self.bottom_frame,
                text=example,
                command=lambda e=example: self._use_example(e)
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    def _on_text_focus_in(self, event):
        """Clear placeholder text on focus"""
        if self.text_input.get("1.0", tk.END).strip() == "Enter your sentence here...":
            self.text_input.delete("1.0", tk.END)
    
    def _use_example(self, example: str):
        """Use an example sentence"""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert(tk.END, example)
    
    def translate(self):
        """Translate text to sign language video"""
        if self.is_translating:
            return
        
        if self.dictionary is None:
            messagebox.showerror("Error", "Sign dictionary not loaded!")
            return
        
        # Get input text
        text = self.text_input.get("1.0", tk.END).strip()
        if not text or text == "Enter your sentence here...":
            messagebox.showwarning("Warning", "Please enter a sentence to translate!")
            return
        
        # Start translation in background
        self.is_translating = True
        self.status_var.set("Translating...")
        self.translate_btn.configure(state=tk.DISABLED)
        
        threading.Thread(target=self._translate_thread, args=(text,), daemon=True).start()
    
    def _translate_thread(self, text: str):
        """Background thread for translation"""
        try:
            # Step 1: Extract glosses using Gemini
            self._update_status("Extracting glosses...")
            glosses = self._extract_glosses(text)
            
            if not glosses:
                self._show_error("Could not extract any valid glosses from the text.")
                return
            
            # Step 2: Filter to available glosses
            valid_glosses = [g for g in glosses if g.lower() in self.available_glosses]
            missing_glosses = [g for g in glosses if g.lower() not in self.available_glosses]
            
            if not valid_glosses:
                available_list = ", ".join(sorted(list(self.available_glosses)[:20])) + "..."
                self._show_error(
                    f"None of the extracted glosses are available in the dictionary.\n\n"
                    f"Extracted: {glosses}\n\n"
                    f"Available glosses include: {available_list}"
                )
                return
            
            # Update glosses display
            gloss_text = " → ".join(valid_glosses)
            if missing_glosses:
                gloss_text += f"\n(Skipped: {', '.join(missing_glosses)})"
            self._update_glosses(gloss_text)
            
            # Step 3: Synthesize motion
            self._update_status("Generating animation...")
            sequence = self.synthesizer.synthesize_phrase([g.lower() for g in valid_glosses])
            
            if sequence is None:
                self._show_error("Failed to generate animation sequence.")
                return
            
            # Step 4: Display video
            self._update_status(f"Ready - {sequence.shape[0]} frames")
            self._play_video(sequence)
            
            self.current_glosses = valid_glosses
            
        except Exception as e:
            self._show_error(f"Translation error: {str(e)}")
        finally:
            self.is_translating = False
            self.after(0, lambda: self.translate_btn.configure(state=tk.NORMAL))
    
    def _extract_glosses(self, text: str) -> List[str]:
        """
        Extract ASL glosses from English text using Gemini API.
        Falls back to simple word splitting if Gemini not available.
        """
        if self.gemini_model is None:
            # Fallback: simple word extraction
            return self._simple_gloss_extraction(text)
        
        # Use Gemini for intelligent extraction
        available_list = ", ".join(sorted(list(self.available_glosses)))
        
        prompt = f"""
        Objective:
        Convert an English sentence into a sequence of American Sign Language (ASL) glosses.
        ASL glosses are the base/root words used in sign language, without grammar particles.
        
        Instructions:
        - Input: An English sentence.
        - Processing: 
          1. Identify the key content words (nouns, verbs, adjectives, question words).
          2. Remove grammar words (articles, prepositions, auxiliary verbs) unless they carry meaning.
          3. Convert to base/root form (e.g., "running" -> "run", "children" -> "child").
          4. IMPORTANT: Only use glosses from this available list: {available_list}
          5. If a word is not in the list, try to find a synonym that IS in the list, or skip it.
        - Output: A Python list of glosses in the order they should be signed.
        
        Examples:
        - "Hello, how are you?" -> ["hello", "how", "you"]
        - "I want to go home" -> ["want", "go", "home"]
        - "What is your name?" -> ["what", "name", "you"]
        
        Input: {text}
        Output (Python list only, no explanation):
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                # Parse the response
                result_text = response.text.strip()
                # Try to extract list from response
                glosses = self._parse_gloss_list(result_text)
                if glosses:
                    return glosses
        except Exception as e:
            print(f"Gemini extraction error: {e}")
        
        # Fallback to simple extraction
        return self._simple_gloss_extraction(text)
    
    def _parse_gloss_list(self, text: str) -> List[str]:
        """Parse glosses from Gemini response"""
        import re
        
        # Try to find a Python list in the response
        # Pattern: ["word1", "word2", ...] or ['word1', 'word2', ...]
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            list_content = match.group(1)
            # Extract quoted strings
            glosses = re.findall(r'["\']([^"\']+)["\']', list_content)
            if glosses:
                return [g.strip().lower() for g in glosses]
        
        # Fallback: split by comma or space
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words[:10]  # Limit to 10 words
    
    def _simple_gloss_extraction(self, text: str) -> List[str]:
        """Simple fallback: extract words that exist in dictionary"""
        import re
        
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter to available glosses
        glosses = []
        for word in words:
            if word in self.available_glosses:
                glosses.append(word)
        
        return glosses
    
    def _update_status(self, status: str):
        """Update status label (thread-safe)"""
        self.after(0, lambda: self.status_var.set(status))
    
    def _update_glosses(self, text: str):
        """Update glosses label (thread-safe)"""
        self.after(0, lambda: self.glosses_var.set(text))
    
    def _show_error(self, message: str):
        """Show error message (thread-safe)"""
        self.after(0, lambda: messagebox.showerror("Translation Error", message))
        self.after(0, lambda: self.status_var.set("Error"))
    
    def _play_video(self, landmarks: np.ndarray):
        """Play video in the player (thread-safe)"""
        def _do_play():
            self.video_player.set_landmarks(landmarks, autoplay=True)
        self.after(0, _do_play)
    
    def set_gemini_model(self, model):
        """Set or update the Gemini model"""
        self.gemini_model = model
    
    def destroy(self):
        """Clean up resources"""
        if hasattr(self, 'video_player'):
            self.video_player.destroy()
        super().destroy()
