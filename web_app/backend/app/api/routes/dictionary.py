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
    """Get skeleton animation data for a specific word"""
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
        df = pd.read_parquet(parquet_path)
        
        # We need specific joints for the frontend SKELETON_JOINTS:
        # head (nose: 0), leftShoulder (11), rightShoulder (12), 
        # leftElbow (13), rightElbow (14), leftWrist (15), rightWrist (16)
        # We also need left_hip (23) and right_hip (24) to calculate hip center
        
        target_pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        
        # Filter for pose landmarks
        pose_df = df[(df['type'] == 'pose') & (df['landmark_index'].isin(target_pose_indices))]
        
        frames_data = []
        
        if not pose_df.empty:
            frames = sorted(pose_df['frame'].unique())
            
            for frame_id in frames:
                frame_df = pose_df[pose_df['frame'] == frame_id]
                
                # Create a lookup for current frame points
                points = {}
                for _, row in frame_df.iterrows():
                    x_val = float(row['x']) if np.isfinite(row['x']) else 0.5
                    y_val = float(row['y']) if np.isfinite(row['y']) else 0.5
                    points[int(row['landmark_index'])] = {"x": x_val, "y": y_val}
                
                # Construct frontend-friendly joints
                joints = []
                
                # helper to safely get point or use default 0.5
                def get_pt(idx):
                    return points.get(idx, {"x": 0.5, "y": 0.5})
                    
                nose = get_pt(0)
                ls = get_pt(11)
                rs = get_pt(12)
                le = get_pt(13)
                re = get_pt(14)
                lw = get_pt(15)
                rw = get_pt(16)
                lh = get_pt(23)
                rh = get_pt(24)
                
                # Calculate neck (between shoulders)
                neck_x = (ls['x'] + rs['x']) / 2.0
                neck_y = (ls['y'] + rs['y']) / 2.0
                
                # Calculate hip (between hips)
                hip_x = (lh['x'] + rh['x']) / 2.0
                hip_y = (lh['y'] + rh['y']) / 2.0
                
                joints.extend([
                    {"id": "head", "x": nose['x'], "y": nose['y']},
                    {"id": "neck", "x": neck_x, "y": neck_y},
                    {"id": "leftShoulder", "x": ls['x'], "y": ls['y']},
                    {"id": "rightShoulder", "x": rs['x'], "y": rs['y']},
                    {"id": "leftElbow", "x": le['x'], "y": le['y']},
                    {"id": "rightElbow", "x": re['x'], "y": re['y']},
                    {"id": "leftWrist", "x": lw['x'], "y": lw['y']},
                    {"id": "rightWrist", "x": rw['x'], "y": rw['y']},
                    {"id": "hip", "x": hip_x, "y": hip_y},
                ])
                
                frames_data.append({
                    "frame": int(frame_id),
                    "joints": joints
                })
                
        return {"frames": frames_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing skeleton: {str(e)}")
