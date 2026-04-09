"""
Prediction API routes.
Handles frame prediction, batch prediction, and video file processing.

All prediction routes use model_service.predict_from_landmarks() which
handles the full pipeline (PreprocessLayer → Model) internally.
"""

import base64
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from app.config import settings
from app.dependencies import (
    get_current_user,
    get_model_service,
    get_landmark_service,
    get_video_service,
    get_gemini_service,
)
from app.schemas.request import (
    FramePredictRequest,
    BatchPredictRequest,
    GenerateSentenceRequest,
)
from app.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    VideoPredictionResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1/predict", tags=["prediction"])


def base64_to_frame(frame_base64: str) -> Optional[np.ndarray]:
    """
    Decode base64 frame to BGR numpy array.

    Args:
        frame_base64: Base64 encoded JPEG/PNG frame

    Returns:
        BGR numpy array or None if decoding fails
    """
    try:
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding frame: {str(e)}")
        return None


@router.post("/frame", response_model=PredictionResponse, responses={401: {"model": ErrorResponse}})
async def predict_frame(
    request: FramePredictRequest,
    username: str = Depends(get_current_user),
    model_service=Depends(get_model_service),
    landmark_service=Depends(get_landmark_service),
) -> PredictionResponse:
    """
    Predict sign from a single frame (or small batch of frames).

    NOTE: Single-frame prediction has limited accuracy because the model
    relies on temporal information. For best results, use the WebSocket
    streaming endpoint which accumulates a temporal window of frames.

    For a single frame, we replicate it to fill the minimum window size
    to produce a rough prediction.
    """
    start_time = time.time()

    try:
        # Decode frame
        frame = base64_to_frame(request.frame_base64)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid frame format. Use base64 encoded JPEG/PNG.",
            )

        # Extract landmarks (GISLR order, NaN fill)
        landmark_result = landmark_service.extract_landmarks(frame)
        if not landmark_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Landmark extraction failed: {landmark_result.get('error')}",
            )

        landmarks = landmark_result["landmarks"]  # [543, 3] with NaN

        # For single-frame: replicate to fill minimum window
        # This gives a rough prediction; for accuracy, use WebSocket streaming
        min_frames = settings.MIN_FRAMES_FOR_INFER
        landmarks_seq = np.stack([landmarks] * min_frames, axis=0)  # [min_frames, 543, 3]

        # Run full pipeline (PreprocessLayer → Model)
        result = model_service.predict_from_landmarks(landmarks_seq, return_top_k=5)

        processing_time_ms = (time.time() - start_time) * 1000

        response_landmarks = (
            landmark_result["landmarks"].tolist() if request.extract_landmarks else None
        )

        return PredictionResponse(
            sign=result["sign"],
            confidence=result["confidence"],
            top5=result["top5"],
            processing_time_ms=round(processing_time_ms, 2),
            landmarks=response_landmarks,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch", response_model=BatchPredictionResponse, responses={401: {"model": ErrorResponse}}
)
async def predict_batch(
    request: BatchPredictRequest,
    username: str = Depends(get_current_user),
    model_service=Depends(get_model_service),
    gemini_service=Depends(get_gemini_service),
) -> BatchPredictionResponse:
    """
    Predict signs from a pre-computed landmark sequence.

    Expects landmarks as a temporal window [T, 543, 3] (raw GISLR order).
    The backend handles preprocessing internally.
    """
    start_time = time.time()

    try:
        landmarks_array = np.array(request.landmarks, dtype=np.float32)

        # Validate shape: should be [T, 543, 3]
        if len(landmarks_array.shape) != 3 or landmarks_array.shape[1] != 543 or landmarks_array.shape[2] != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid shape. Expected [T, 543, 3], got {landmarks_array.shape}",
            )

        # Run inference
        result = model_service.predict_from_landmarks(landmarks_array, return_top_k=5)

        processing_time_sec = time.time() - start_time

        signs = [result["sign"]]
        confidences = [result["confidence"]]

        # Generate sentence if enabled
        sentence = None
        if request.enable_gemini and gemini_service.is_enabled():
            sentence = gemini_service.generate_sentence(signs)

        return BatchPredictionResponse(
            signs=signs,
            confidences=confidences,
            sentence=sentence,
            top5_per_sign=[result["top5"]],
            frames_processed=landmarks_array.shape[0],
            processing_time_sec=round(processing_time_sec, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.post("/video", response_model=VideoPredictionResponse, responses={401: {"model": ErrorResponse}})
async def predict_video(
    file: UploadFile = File(...),
    enable_gemini: bool = True,
    username: str = Depends(get_current_user),
    model_service=Depends(get_model_service),
    video_service=Depends(get_video_service),
    landmark_service=Depends(get_landmark_service),
    gemini_service=Depends(get_gemini_service),
) -> VideoPredictionResponse:
    """
    Predict signs from uploaded video file.

    Extracts frames, processes landmarks using a sliding window approach
    (matching the desktop app), and runs inference on temporal windows.
    """
    start_time = time.time()
    temp_file_path = None

    try:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # Validate video
        is_valid, error_msg = video_service.validate_video_file(temp_file_path)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid video: {error_msg}",
            )

        # Get video info
        video_info = video_service.get_video_info(temp_file_path)
        if video_info.get("error"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read video: {video_info['error']}",
            )

        # Validate duration
        is_valid, error_msg = video_service.validate_duration(video_info["duration_sec"])
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

        # Extract all frames
        frames, metadata = video_service.extract_frames_batch(
            temp_file_path, max_frames=settings.MAX_VIDEO_DURATION_SECONDS * 30
        )

        if not frames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No frames extracted from video",
            )

        # Extract landmarks from all frames
        all_landmarks = []
        for frame in frames:
            landmark_result = landmark_service.extract_landmarks(frame)
            if landmark_result["success"]:
                all_landmarks.append(landmark_result["landmarks"])
            else:
                # Use NaN-filled array for failed frames
                all_landmarks.append(np.full((543, 3), np.nan, dtype=np.float32))

        all_landmarks_np = np.stack(all_landmarks, axis=0)  # [T, 543, 3]

        # Run inference on the full sequence
        # The PreprocessLayer will handle downsampling/padding to 64 frames
        result = model_service.predict_from_landmarks(all_landmarks_np, return_top_k=5)

        processing_time_sec = time.time() - start_time

        all_signs = [result["sign"]]
        all_confidences = [result["confidence"]]

        # Generate sentence
        sentence = None
        if enable_gemini and gemini_service.is_enabled():
            sentence = gemini_service.generate_sentence(all_signs)

        return VideoPredictionResponse(
            signs=all_signs,
            confidences=all_confidences,
            sentence=sentence,
            frames_processed=len(frames),
            processing_time_sec=round(processing_time_sec, 2),
            video_duration_sec=video_info["duration_sec"],
            fps=video_info["fps"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {str(e)}",
        )
    finally:
        if temp_file_path:
            import os
            try:
                os.remove(temp_file_path)
            except:
                pass


@router.post("/generate-sentence")
async def generate_sentence(
    request: GenerateSentenceRequest,
    username: str = Depends(get_current_user),
    gemini_service=Depends(get_gemini_service),
):
    """
    Generate a natural language sentence from a list of recognized signs.
    """
    if not request.signs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No signs provided",
        )

    if not gemini_service.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service is not enabled",
        )

    sentence = gemini_service.generate_sentence(request.signs)
    if sentence is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentence generation failed",
        )

    return {"sentence": sentence, "signs": request.signs}
