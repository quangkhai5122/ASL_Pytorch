"""
Translate Service — Text → ASL Gloss

Pipeline:
  1. Load available gloss list from WLASL_Skeleton parquet files.
  2. Send user text + gloss list to Gemini API.
  3. Gemini strips grammar words, maps remaining words to glosses.
  4. Fallback: simple stop-word removal + direct dictionary lookup.
"""

from __future__ import annotations

import os
import re
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Ensure data is available (silent if already downloaded)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _LEMMATIZER_AVAILABLE = True
except Exception:
    _lemmatizer = None  # type: ignore
    _LEMMATIZER_AVAILABLE = False

from app.config import settings


# ---------------------------------------------------------------------------
# Stop words (English grammar words to remove in fallback mode)
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    # Articles
    "a", "an", "the",
    # Auxiliary / linking verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    # Prepositions
    "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "as", "into", "through",
    # Conjunctions
    "and", "or", "but", "if", "then",
    # Demonstratives
    "that", "this", "these", "those",
    # Filler / degree adverbs
    "so", "up", "out", "just", "also", "very", "quite", "really", "rather",
    # NOTE: Pronouns (i, me, you, he, she, we, us, they, it) are intentionally
    #       kept because ASL has real signs for them (pointing gestures).
}

# ---------------------------------------------------------------------------
# Synonym map — words NOT in WLASL dict → nearest available gloss
# Used by fallback when lemmatization still doesn't find a match.
# Extend this list whenever users report missing words.
# ---------------------------------------------------------------------------
_SYNONYM_MAP: dict = {
    # Travel / location
    "abroad":       "international",
    "overseas":     "international",
    "foreign":      "international",
    "trip":         "travel",
    "vacation":     "holiday",
    "journey":      "travel",
    # Emotions
    "happy":        "excited",
    "unhappy":      "sad",
    "angry":        "mad",
    "scared":       "afraid",
    "frightened":   "afraid",
    "glad":         "excited",
    # Actions
    "speak":        "talk",
    "chat":         "talk",
    "discuss":      "talk",
    "watch":        "see",
    "observe":      "see",
    "purchase":     "buy",
    "obtain":       "get",
    "receive":      "get",
    "assist":       "help",
    "require":      "need",
    "desire":       "want",
    "wish":         "want",
    "begin":        "start",
    "finish":       "stop",
    "complete":     "stop",
    "depart":       "go",
    "arrive":       "come",
    "return":       "come",
    # People / relations
    "classmate":    "friend",
    "colleague":    "friend",
    "instructor":   "teacher",
    "professor":    "teacher",
    "pupil":        "student",
    # Common nouns
    "automobile":   "car",
    "vehicle":      "car",
    "residence":    "home",
    "job":          "work",
    "occupation":   "work",
    "illness":      "sick",
    "disease":      "sick",
}

# ---------------------------------------------------------------------------
# Gemini prompt template
# ---------------------------------------------------------------------------
_GLOSS_PROMPT = """You are an ASL (American Sign Language) expert.

Convert the following English sentence into an ordered list of ASL glosses.

Rules:
- Keep only content words: nouns, main verbs, adjectives, question words, numbers.
- Remove grammar words: articles (a, an, the), auxiliary verbs (is, are, was, were, be, been, have, has, do, does, will, can, could, should, may, might), prepositions (in, on, at, to, for, with, of, by), conjunctions (and, or, but, if).
- Use base/root form (run not running; child not children; good not better).
- ONLY use words from the available glosses list below.
- If a word is not in the list, find the closest synonym that IS in the list. If no synonym exists, skip the word.
- Return ONLY a Python list, for example: ["hello", "name", "you"]

Available glosses:
{gloss_list}

English sentence: {text}

ASL glosses (Python list only):"""


class TranslateService:
    """
    Singleton service that translates English text into a sequence of
    ASL glosses using Gemini API (with rule-based fallback).
    """

    _instance: Optional["TranslateService"] = None
    _initialized: bool = False

    # Cached state
    _available_glosses: List[str] = []       # all gloss names (lowercase, sorted)
    _gloss_set: set = set()                  # for O(1) lookup
    _gemini_model = None                     # genai.GenerativeModel or None
    _data_dir: Path = Path("/Users/macos/ASL_Pytorch/data/WLASL_Skeleton")

    # -----------------------------------------------------------------------
    def __new__(cls) -> "TranslateService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _initialize(self):
        """Load gloss list and optionally set up Gemini."""
        self._initialized = True
        self._load_gloss_list()
        self._setup_gemini()

    def _load_gloss_list(self):
        """Scan WLASL_Skeleton directory for available .parquet glosses."""
        data_dir = self._data_dir

        # Allow override via env variable
        env_dir = os.environ.get("WLASL_SKELETON_DIR")
        if env_dir:
            data_dir = Path(env_dir)

        if not data_dir.exists():
            print(f"[WARN] WLASL_Skeleton dir not found: {data_dir}. Gloss list will be empty.")
            self._available_glosses = []
            self._gloss_set = set()
            return

        parquet_files = list(data_dir.glob("*.parquet"))
        glosses = sorted(f.stem.lower() for f in parquet_files)
        self._available_glosses = glosses
        self._gloss_set = set(glosses)
        print(f"[OK] TranslateService: loaded {len(glosses)} glosses from {data_dir}")

    def _setup_gemini(self):
        """Configure Gemini API if key is available."""
        if not settings.ENABLE_GEMINI:
            print("[INFO] TranslateService: Gemini disabled (ENABLE_GEMINI=false)")
            return

        api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
        if not api_key:
            print("[WARN] TranslateService: GEMINI_API_KEY/GOOGLE_API_KEY not set — using fallback")
            return

        try:
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
            print(f"[OK] TranslateService: Gemini ready ({settings.GEMINI_MODEL})")
        except Exception as exc:
            print(f"[WARN] TranslateService: Gemini setup failed — {exc}")
            self._gemini_model = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def translate(self, text: str) -> dict:
        """
        Translate English text to ASL glosses.

        Returns:
            {
                "glosses":          ["hello", "name", "you"],   # valid glosses
                "raw_glosses":      ["hello", "name", "you"],   # all Gemini output
                "missing_glosses":  ["please"],                  # not in dict
                "method":           "gemini" | "fallback",
                "available_count":  1809,
            }
        """
        text = text.strip()
        if not text:
            return self._empty_result()

        # Try Gemini first
        if self._gemini_model is not None:
            result = self._translate_with_gemini(text)
            if result is not None:
                return result

        # Fallback
        return self._translate_fallback(text)

    def get_available_glosses(self) -> List[str]:
        """Return the full sorted list of available glosses."""
        return self._available_glosses

    def get_gloss_count(self) -> int:
        return len(self._available_glosses)

    def is_gemini_enabled(self) -> bool:
        return self._gemini_model is not None

    # -----------------------------------------------------------------------
    # Gemini path
    # -----------------------------------------------------------------------

    def _translate_with_gemini(self, text: str) -> Optional[dict]:
        """Call Gemini and parse the returned gloss list. Returns None on failure."""
        # Build a compact gloss list string for the prompt (avoid 100k-token prompts)
        gloss_list_str = ", ".join(self._available_glosses)

        prompt = _GLOSS_PROMPT.format(gloss_list=gloss_list_str, text=text)

        try:
            response = self._gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=256,
                    temperature=0.1,   # low temp for deterministic output
                ),
            )
            raw_text = (response.text or "").strip()
            raw_glosses = self._parse_list_response(raw_text)

            if not raw_glosses:
                return None

            valid, missing = self._split_valid(raw_glosses)
            return {
                "glosses": valid,
                "raw_glosses": raw_glosses,
                "missing_glosses": missing,
                "method": "gemini",
                "available_count": len(self._available_glosses),
            }

        except Exception as exc:
            print(f"[WARN] TranslateService Gemini call failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Fallback path
    # -----------------------------------------------------------------------

    def _translate_fallback(self, text: str) -> dict:
        """
        Rule-based fallback:
          1. Lowercase & tokenise.
          2. Remove stop words.
          3. Lemmatize to base form (friends→friend, studying→study).
          4. Keep tokens that exist in the gloss dictionary.
        """
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        candidates = [t for t in tokens if t not in _STOP_WORDS]

        raw_glosses = candidates[:]

        # Lemmatize each candidate to base form
        if _LEMMATIZER_AVAILABLE:
            lemmatized = []
            for word in candidates:
                # Try noun form first, then verb form
                base = _lemmatizer.lemmatize(word, pos='n')
                if base == word:  # no change as noun → try verb
                    base = _lemmatizer.lemmatize(word, pos='v')
                if base == word:  # try adjective
                    base = _lemmatizer.lemmatize(word, pos='a')
                lemmatized.append(base)
            candidates = lemmatized

        # Apply synonym map for words still not in the dictionary
        candidates = [_SYNONYM_MAP.get(w, w) for w in candidates]

        valid, missing = self._split_valid(candidates)

        return {
            "glosses": valid,
            "raw_glosses": raw_glosses,
            "missing_glosses": missing,
            "method": "fallback",
            "available_count": len(self._available_glosses),
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _parse_list_response(self, text: str) -> List[str]:
        """Extract a Python list of strings from Gemini's raw response."""
        # Try: ["word1", "word2", ...]  or  ['word1', 'word2', ...]
        match = re.search(r"\[([^\]]*)\]", text, re.DOTALL)
        if match:
            inner = match.group(1)
            words = re.findall(r"""["']([^"']+)["']""", inner)
            if words:
                return [w.strip().lower() for w in words]

        # Fallback: comma-separated bare words
        words = re.findall(r"\b[a-zA-Z][a-zA-Z\s]*[a-zA-Z]\b", text)
        return [w.strip().lower() for w in words if w.strip()]

    def _split_valid(self, glosses: List[str]):
        """Split glosses into (valid, missing) based on dictionary."""
        valid = [g for g in glosses if g in self._gloss_set]
        missing = [g for g in glosses if g not in self._gloss_set]
        return valid, missing

    @staticmethod
    def _empty_result() -> dict:
        return {
            "glosses": [],
            "raw_glosses": [],
            "missing_glosses": [],
            "method": "none",
            "available_count": 0,
        }


# Singleton instance (imported by other modules)
translate_service = TranslateService()
