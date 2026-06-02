"""
Gemini API Service for sentence generation from recognized signs.
Optional service - can be disabled via ENABLE_GEMINI setting.
"""

from typing import List, Optional
import google.generativeai as genai

from app.config import settings


class GeminiService:
    """
    Service for generating natural language from sign sequences.
    Uses Google's Generative AI (Gemini) API.
    """

    _instance: Optional["GeminiService"] = None
    _enabled = False
    _initialized = False
    _model = None
    _last_error: Optional[str] = None

    def __new__(cls) -> "GeminiService":
        """Ensure singleton instantiation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Gemini service."""
        if not self._initialized:
            self._initialize()

    def _initialize(self):
        """Initialize and configure Gemini API."""
        self._initialized = True
        self._last_error = None

        if not settings.ENABLE_GEMINI:
            self._last_error = "Gemini service disabled (ENABLE_GEMINI=false)"
            print(f"[INFO] {self._last_error}")
            return

        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not api_key:
            self._last_error = "Gemini API key not found; set GEMINI_API_KEY or GOOGLE_API_KEY"
            print(f"[WARN] {self._last_error}")
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self._enabled = True
            print(f"[OK] Gemini service initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            self._last_error = f"Failed to initialize Gemini: {str(e)}"
            print(f"[WARN] {self._last_error}")
            self._enabled = False

    def generate_sentence(self, signs: List[str]) -> Optional[str]:
        """
        Generate a natural language sentence from a sequence of signs.

        Uses the same prompt structure as the desktop app (app_optimized.py)
        which has been proven to produce coherent English sentences from
        ASL gloss word lists.

        Args:
            signs: List of recognized signs/words

        Returns:
            Generated sentence or None if generation fails
        """
        if not self._enabled:
            self._last_error = self._last_error or "Gemini service is not enabled"
            return None

        if not signs:
            self._last_error = "No signs provided"
            return None

        try:
            # Use the proven prompt from the desktop application
            prompt = f"""
            Objective:
            Construct a coherent and meaningful English sentence from a list of recognized American Sign Language (ASL) words. The sentence should be simple and accurately convey the meaning.

            Instructions:
            - Input: A Python list of recognized ASL words.
            - Processing: Rearrange the words (if necessary) to form a grammatically correct sentence. Add necessary articles, prepositions, conjunctions, and auxiliary verbs. Ignore the word "TV" if present.
            - Output: A concise English sentence. Return ONLY the sentence, nothing else.

            Input: {signs}
            Output:
            """

            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.7,
                ),
            )

            if getattr(response, 'text', None):
                sentence = response.text.strip()
                print(f"[OK] Generated sentence: {sentence}")
                self._last_error = None
                return sentence

            self._last_error = "Empty Gemini response"
            print(f"[WARN] {self._last_error}")
            return None

        except Exception as e:
            self._last_error = f"Gemini generation failed: {str(e)}"
            print(f"[WARN] {self._last_error}")
            return None

    def is_enabled(self) -> bool:
        """Check if Gemini service is enabled and initialized."""
        return self._enabled

    def last_error(self) -> Optional[str]:
        """Return the last initialization/generation error for diagnostics."""
        return self._last_error

    def validate_signs(self, signs: List[str]) -> bool:
        """
        Validate sign list before sending to Gemini.

        Args:
            signs: List of signs

        Returns:
            bool: True if valid, False otherwise
        """
        if not signs:
            return False

        if len(signs) > 100:  # Max 100 signs
            return False

        # Check for valid characters
        for sign in signs:
            if not sign or not isinstance(sign, str):
                return False

        return True


# Singleton instance
gemini_service = GeminiService()
