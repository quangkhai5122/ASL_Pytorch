"""
Translate Router — /api/v1/translate
Provides Text → ASL Gloss translation endpoint.
"""

import os
import tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from app.schemas.request import TranslateGlossRequest
from app.services.translate_service import translate_service
from app.services.synthesize_service import get_synthesize_service

router = APIRouter(prefix="/api/v1/translate", tags=["translate"])



@router.post("/gloss")
async def translate_to_gloss(request: TranslateGlossRequest):
    """
    Translate an English sentence into an ordered list of ASL glosses.

    **Pipeline:**
    1. If `use_gemini=true` and Gemini API key is configured → Gemini extracts glosses
       (removes grammar words, maps to root forms, filters to available dictionary).
    2. If Gemini is unavailable or `use_gemini=false` → simple rule-based fallback
       (stop-word removal + direct dictionary lookup).

    **Returns:**
    - `glosses` — final valid gloss list (all exist in the WLASL dictionary)
    - `raw_glosses` — Gemini's raw output before dictionary filtering
    - `missing_glosses` — words that Gemini suggested but are not in the dictionary
    - `method` — `"gemini"` | `"fallback"`
    - `available_count` — total number of glosses in the dictionary
    """

    try:
        # Temporarily disable Gemini if caller requests fallback
        if not request.use_gemini:
            # Swap to fallback by temporarily patching the model reference
            original_model = translate_service._gemini_model
            translate_service._gemini_model = None
            result = translate_service.translate(request.text)
            translate_service._gemini_model = original_model
        else:
            result = translate_service.translate(request.text)

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/glosses")
async def list_available_glosses(prefix: str = ""):
    """
    Return the list of all available ASL glosses in the dictionary.

    Optionally filter by `prefix` query param (e.g. `?prefix=he` → hello, help, here …).
    """
    glosses = translate_service.get_available_glosses()

    if prefix:
        prefix_lower = prefix.lower()
        glosses = [g for g in glosses if g.startswith(prefix_lower)]

    return {
        "glosses": glosses,
        "count": len(glosses),
        "total_available": translate_service.get_gloss_count(),
        "gemini_enabled": translate_service.is_gemini_enabled(),
    }


# ---------------------------------------------------------------------------
# Synthesize endpoint — Skeleton animation
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    glosses: List[str]
    fps: int = 25


@router.post("/synthesize")
async def synthesize_glosses(request: SynthesizeRequest):
    """
    Synthesize a smooth skeleton animation from a list of ASL glosses.

    **Pipeline (mirrors text_to_sign_tab.py desktop app):**
    1. `SignDictionary.load_gloss()` — load landmark data from WLASL_Skeleton .parquet files
    2. `MotionSynthesizer.synthesize_phrase()` — Hermite spline transitions + IK constraints + smoothing
    3. Return frames (T × 153 × 3) as JSON for frontend Canvas 2D rendering

    **Returns:**
    - `frames` — List[T × 153 × 3] of float16 landmark coordinates (normalized 0-1)
    - `n_frames` — total frame count
    - `fps` — target playback FPS
    - `n_landmarks` — 153 (102 face + 21 left hand + 9 pose + 21 right hand)
    - `glosses_used` — glosses that were found and synthesized
    - `missing_glosses` — glosses not found in the WLASL dictionary

    **Landmark layout (153 points):**
    - [0:40]   Lips (40)
    - [40:76]  Face oval (36)
    - [76:86]  Eyebrows (10)
    - [86:102] Eyes (16)
    - [102:123] Left hand (21)
    - [123:132] Pose: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip (9)
    - [132:153] Right hand (21)
    """
    if not request.glosses:
        raise HTTPException(status_code=400, detail="glosses list is empty")

    svc = get_synthesize_service()

    if not svc.is_available:
        raise HTTPException(
            status_code=503,
            detail="SynthesizeService unavailable — SignDictionary/MotionSynthesizer failed to load"
        )

    result = svc.synthesize(request.glosses, fps=request.fps)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result.get("error", "Synthesis failed"))

    return result



@router.post("/gloss")
async def translate_to_gloss(request: TranslateGlossRequest):
    """
    Translate an English sentence into an ordered list of ASL glosses.

    **Pipeline:**
    1. If `use_gemini=true` and Gemini API key is configured → Gemini extracts glosses
       (removes grammar words, maps to root forms, filters to available dictionary).
    2. If Gemini is unavailable or `use_gemini=false` → simple rule-based fallback
       (stop-word removal + direct dictionary lookup).

    **Returns:**
    - `glosses` — final valid gloss list (all exist in the WLASL dictionary)
    - `raw_glosses` — Gemini's raw output before dictionary filtering
    - `missing_glosses` — words that Gemini suggested but are not in the dictionary
    - `method` — `"gemini"` | `"fallback"`
    - `available_count` — total number of glosses in the dictionary
    """

    try:
        # Temporarily disable Gemini if caller requests fallback
        if not request.use_gemini:
            # Swap to fallback by temporarily patching the model reference
            original_model = translate_service._gemini_model
            translate_service._gemini_model = None
            result = translate_service.translate(request.text)
            translate_service._gemini_model = original_model
        else:
            result = translate_service.translate(request.text)

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/glosses")
async def list_available_glosses(prefix: str = ""):
    """
    Return the list of all available ASL glosses in the dictionary.

    Optionally filter by `prefix` query param (e.g. `?prefix=he` → hello, help, here …).
    """
    glosses = translate_service.get_available_glosses()

    if prefix:
        prefix_lower = prefix.lower()
        glosses = [g for g in glosses if g.startswith(prefix_lower)]

    return {
        "glosses": glosses,
        "count": len(glosses),
        "total_available": translate_service.get_gloss_count(),
        "gemini_enabled": translate_service.is_gemini_enabled(),
    }


# ---------------------------------------------------------------------------
# GIF Export endpoint
# ---------------------------------------------------------------------------

class ExportGifRequest(BaseModel):
    glosses: List[str]
    fps: int = 15          # lower FPS → smaller GIF file


@router.post("/export-gif")
async def export_gif(request: ExportGifRequest):
    """
    Synthesize skeleton animation and return as an animated GIF file.

    **Pipeline:**
    1. `MotionSynthesizer.synthesize_phrase()` — same as /synthesize
    2. `SignVisualizer.create_animation()` — renders skeleton with PIL, saves as GIF
    3. Return GIF as file download

    **Note:** GIF generation takes a few seconds (renders every frame with PIL).
    Reduce `fps` (default 15) for smaller file size.
    """
    if not request.glosses:
        raise HTTPException(status_code=400, detail="glosses list is empty")

    svc = get_synthesize_service()
    if not svc.is_available:
        raise HTTPException(status_code=503, detail="SynthesizeService unavailable")

    # Step 1: Synthesize landmarks
    result = svc.synthesize(request.glosses, fps=request.fps)
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result.get("error", "Synthesis failed"))

    try:
        import numpy as np
        import sys
        from pathlib import Path
        _ROOT = Path(__file__).resolve().parents[5]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from scripts.sign_visualizer import SignVisualizer

        # Reconstruct numpy array from JSON frames
        sequence = np.array(result["frames"], dtype=np.float32)

        # Write to temp GIF file
        tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        tmp_path = tmp.name
        tmp.close()

        visualizer = SignVisualizer(fps=request.fps)
        visualizer.create_animation(sequence, tmp_path)

        # Return GIF, delete temp file after response
        filename = "_".join(result["glosses_used"][:5]) + ".gif"
        return FileResponse(
            path=tmp_path,
            media_type="image/gif",
            filename=filename,
            background=None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"GIF generation failed: {e}")
