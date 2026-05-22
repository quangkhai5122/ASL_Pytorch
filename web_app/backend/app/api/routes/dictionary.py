import os
import glob
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from app.config import settings

router = APIRouter(prefix=f"{settings.API_V1_STR}/dictionary", tags=["dictionary"])

# The data directory is mounted at /app/data inside docker, or relative in local dev
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
if not os.path.exists(DATA_DIR):
    # Fallback to local path relative to this file
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../data"))

VIDEO_DIR = os.path.join(DATA_DIR, "WLASL_Only1Video")
SKELETON_DIR = os.path.join(DATA_DIR, "WLASL_Skeleton")

# =============================================================================
# Avatar landmark configuration (mirrors scripts/slp_config.py)
# =============================================================================
# Face landmarks: 40 lips + 36 oval + 10 eyebrows + 16 eyes = 102
LIPS_RAW_IDXS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
FACE_OVAL_RAW_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYEBROW_RAW_IDXS = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_RAW_IDXS = [336, 296, 334, 293, 300]
LEFT_EYE_RAW_IDXS = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE_RAW_IDXS = [362, 385, 387, 263, 373, 380, 382, 249]

FACE_RAW_IDXS = (LIPS_RAW_IDXS + FACE_OVAL_RAW_IDXS +
                 LEFT_EYEBROW_RAW_IDXS + RIGHT_EYEBROW_RAW_IDXS +
                 LEFT_EYE_RAW_IDXS + RIGHT_EYE_RAW_IDXS)  # 102

LEFT_HAND_RAW_IDXS = list(range(468, 489))   # 21 points
RIGHT_HAND_RAW_IDXS = list(range(522, 543))  # 21 points
POSE_SUBSET_LOCAL = [0, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_RAW_IDXS = [489 + i for i in POSE_SUBSET_LOCAL]  # 9 points

# All 153 avatar landmark indices
AVATAR_RAW_IDXS = FACE_RAW_IDXS + LEFT_HAND_RAW_IDXS + POSE_RAW_IDXS + RIGHT_HAND_RAW_IDXS
N_AVATAR = len(AVATAR_RAW_IDXS)  # 153

# Segment boundaries within the 153-point avatar array
N_FACE = len(FACE_RAW_IDXS)          # 102
N_HAND = 21
N_POSE = len(POSE_SUBSET_LOCAL)       # 9

N_LIPS = len(LIPS_RAW_IDXS)          # 40
N_FACE_OVAL = len(FACE_OVAL_RAW_IDXS)  # 36
N_EYEBROWS = len(LEFT_EYEBROW_RAW_IDXS) + len(RIGHT_EYEBROW_RAW_IDXS)  # 10
N_EYES = len(LEFT_EYE_RAW_IDXS) + len(RIGHT_EYE_RAW_IDXS)  # 16

# Motion detection thresholds
MOTION_THRESHOLD = 0.005
MIN_MOTION_FRAMES = 10

# Type offsets in the parquet file (global MediaPipe indices)
TYPE_OFFSETS = {
    'face': 0,
    'left_hand': 468,
    'pose': 489,
    'right_hand': 522,
}


def _parquet_to_full_array(df: pd.DataFrame) -> np.ndarray:
    """Convert parquet DataFrame to numpy array (T, 543, 3)."""
    frames = sorted(df['frame'].unique())
    n_frames = len(frames)
    frame_map = {f: i for i, f in enumerate(frames)}

    full = np.full((n_frames, 543, 3), np.nan, dtype=np.float32)
    for type_name, offset in TYPE_OFFSETS.items():
        mask = df['type'] == type_name
        if not mask.any():
            continue
        sub = df[mask]
        fi = sub['frame'].map(frame_map).values
        li = sub['landmark_index'].values + offset
        coords = sub[['x', 'y', 'z']].values
        full[fi, li] = coords
    return full


def _extract_avatar(full_data: np.ndarray) -> np.ndarray:
    """Extract the 153 avatar landmarks from the full 543 array."""
    return full_data[:, AVATAR_RAW_IDXS, :]


def _interpolate_nan(data: np.ndarray) -> np.ndarray:
    """Interpolate NaN values over the time axis."""
    T, N, D = data.shape
    flat = data.reshape(T, N * D)
    df = pd.DataFrame(flat)
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    df = df.fillna(0.5)
    return df.values.reshape(T, N, D).astype(np.float32)


def _trim_static_frames(data: np.ndarray) -> np.ndarray:
    """Trim frames with no hand motion at start/end."""
    T = data.shape[0]
    if T <= MIN_MOTION_FRAMES:
        return data

    # Hand indices within the 153-point avatar array
    left_hand_range = list(range(N_FACE, N_FACE + N_HAND))
    right_hand_range = list(range(N_FACE + N_HAND + N_POSE, N_FACE + 2 * N_HAND + N_POSE))
    hand_indices = left_hand_range + right_hand_range
    hand_data = data[:, hand_indices, :2]

    motion = np.zeros(T)
    for i in range(1, T):
        diff = hand_data[i] - hand_data[i - 1]
        valid = diff[~np.isnan(diff)]
        if len(valid) > 0:
            motion[i] = np.mean(np.abs(valid))

    start = 0
    for i in range(T):
        if motion[i] > MOTION_THRESHOLD:
            start = max(0, i - 2)
            break

    end = T
    for i in range(T - 1, -1, -1):
        if motion[i] > MOTION_THRESHOLD:
            end = min(T, i + 3)
            break

    if end - start < MIN_MOTION_FRAMES:
        return data
    return data[start:end]


def _is_hand_valid(hand_pts: np.ndarray) -> bool:
    """Check if 21-point hand data is real tracking, not placeholder."""
    if np.isnan(hand_pts).all():
        return False
    if np.allclose(hand_pts[:, :2], 0):
        return False
    # Check if all points are identical (rest-pose filler)
    first = hand_pts[0, :2]
    for i in range(1, len(hand_pts)):
        if not np.allclose(hand_pts[i, :2], first, atol=1e-6):
            return True
    return False  # All same → invalid


def _load_gloss_full(parquet_path: str) -> np.ndarray:
    """Load a parquet file and return the processed (T, 153, 3) avatar array."""
    df = pd.read_parquet(parquet_path)
    full = _parquet_to_full_array(df)
    avatar = _extract_avatar(full)
    avatar = _interpolate_nan(avatar)
    avatar = _trim_static_frames(avatar)
    return avatar


@router.get("/")
async def list_words():
    """List all available words in the dictionary"""
    video_words = set()
    if os.path.exists(VIDEO_DIR):
        for filepath in glob.glob(os.path.join(VIDEO_DIR, "**", "*.mp4"), recursive=True):
            word = os.path.splitext(os.path.basename(filepath))[0].lower()
            video_words.add(word)
            
    skeleton_words = set()
    if os.path.exists(SKELETON_DIR):
        for filepath in glob.glob(os.path.join(SKELETON_DIR, "**", "*.parquet"), recursive=True):
            word = os.path.splitext(os.path.basename(filepath))[0].lower()
            skeleton_words.add(word)
            
    all_words = sorted(list(video_words | skeleton_words))
    
    # Return mapping showing what data is available for each word
    results = []
    for word in all_words:
        results.append({
            "word": word,
            "has_video": word in video_words,
            "has_skeleton": word in skeleton_words
        })
        
    return {"words": results}

@router.get("/{word}/skeleton")
async def get_skeleton(word: str):
    """
    Get full 153-landmark skeleton animation data for a word.
    
    Returns segmented data: face (102 pts), left_hand (21), pose (9), right_hand (21).
    Face is further split into: lips (40), oval (36), eyebrows (10), eyes (16).
    """
    word = word.lower()
    
    # Find the parquet file
    parquet_path = None
    if os.path.exists(SKELETON_DIR):
        for filepath in glob.glob(os.path.join(SKELETON_DIR, "**", "*.parquet"), recursive=True):
            if os.path.splitext(os.path.basename(filepath))[0].lower() == word:
                parquet_path = filepath
                break
                
    if not parquet_path:
        raise HTTPException(status_code=404, detail=f"Skeleton data for '{word}' not found")
        
    try:
        avatar_data = _load_gloss_full(parquet_path)  # (T, 153, 3)
        T = avatar_data.shape[0]
        
        frames_data = []
        for i in range(T):
            frame = avatar_data[i]  # (153, 3)
            
            # Split into segments (only send x, y to reduce payload)
            face_all = frame[:N_FACE, :2]  # (102, 2)
            left_hand = frame[N_FACE:N_FACE + N_HAND, :2]  # (21, 2)
            pose = frame[N_FACE + N_HAND:N_FACE + N_HAND + N_POSE, :2]  # (9, 2)
            right_hand = frame[N_FACE + N_HAND + N_POSE:, :2]  # (21, 2)
            
            # Sub-split face components
            lips = face_all[:N_LIPS]                       # (40, 2)
            oval = face_all[N_LIPS:N_LIPS + N_FACE_OVAL]  # (36, 2)
            eyebrows = face_all[N_LIPS + N_FACE_OVAL:N_LIPS + N_FACE_OVAL + N_EYEBROWS]  # (10, 2)
            eyes = face_all[N_LIPS + N_FACE_OVAL + N_EYEBROWS:]  # (16, 2)
            
            # Check hand validity
            lh_valid = _is_hand_valid(frame[N_FACE:N_FACE + N_HAND])
            rh_valid = _is_hand_valid(frame[N_FACE + N_HAND + N_POSE:])
            
            frames_data.append({
                "frame": i,
                "lips": lips.tolist(),
                "oval": oval.tolist(),
                "eyebrows": eyebrows.tolist(),
                "eyes": eyes.tolist(),
                "left_hand": left_hand.tolist(),
                "pose": pose.tolist(),
                "right_hand": right_hand.tolist(),
                "hand_valid": {"left": bool(lh_valid), "right": bool(rh_valid)},
            })
            
        return {
            "frames": frames_data,
            "total_frames": T,
            "fps": 20,
            "segments": {
                "lips": N_LIPS,
                "oval": N_FACE_OVAL,
                "eyebrows": N_EYEBROWS,
                "eyes": N_EYES,
                "left_hand": N_HAND,
                "pose": N_POSE,
                "right_hand": N_HAND,
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing skeleton: {str(e)}")
