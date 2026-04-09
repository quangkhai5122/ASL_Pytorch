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

        if not settings.ENABLE_GEMINI:
            print("[INFO] Gemini service disabled (ENABLE_GEMINI=false)")
            return

        if not settings.GEMINI_API_KEY:
            print("[WARN] Gemini API key not found, service disabled")
            return

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._enabled = True
            print(f"[OK] Gemini service initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            print(f"[WARN] Failed to initialize Gemini: {str(e)}")
            self._enabled = False

    def generate_sentence(self, signs: List[str]) -> Optional[str]:
        """
        Generate a natural language sentence from a sequence of signs.

        Args:
            signs: List of recognized signs/words

        Returns:
            Generated sentence or None if generation fails
        """
        if not self._enabled:
            return None

        if not signs:
            return None

        try:
            # Create prompt
            signs_str = " ".join(signs)
            prompt = f"""Given a sequence of sign language words, generate a natural, grammatically correct English sentence:

Signs: {signs_str}

Please generate a coherent English sentence that represents these signs. If the signs form a complete thought, maintain the original meaning. Be concise and natural.

Response should be ONLY the sentence, nothing else."""

            # Get model and generate
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.7,
                ),
            )

            sentence = response.text.strip()
            print(f"[OK] Generated sentence: {sentence}")
            return sentence

        except Exception as e:
            print(f"[WARN] Gemini generation failed: {str(e)}")
            return None

    def is_enabled(self) -> bool:
        """Check if Gemini service is enabled and initialized."""
        return self._enabled

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
