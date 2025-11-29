import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os

try:
    from scripts.dataset import ASLParquetDataset
    from scripts.config import (
        LIPS_IDXS, LEFT_HAND_IDXS, POSE_IDXS,
        LIPS_START, LEFT_HAND_START, POSE_START,
        INPUT_SIZE, N_COLS, N_DIMS
    )
except ImportError:
    from dataset import ASLParquetDataset
    from config import (
        LIPS_IDXS, LEFT_HAND_IDXS, POSE_IDXS,
        LIPS_START, LEFT_HAND_START, POSE_START,
        INPUT_SIZE, N_COLS, N_DIMS
    )

def calculate_stats(csv_path='data/train.csv', data_root='data/', batch_size=64, num_workers=0):
    print(f"Calculating stats for dataset: {csv_path}")
    dataset = ASLParquetDataset(csv_path=csv_path, data_root=data_root)
    
    # Use a simple collate function to stack frames
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    
    # Initialize accumulators
    # We need to calculate mean and std for each landmark type (Lips, Hand, Pose)
    # Shape: (N_POINTS, 2) -> We only care about X, Y. Z is usually ignored or handled separately.
    # But the model code expects (N_POINTS, 2) for mean/std.
    
    # To avoid OOM with large datasets, we use Welford's online algorithm or simply sum and sum_sq.
    # Since we have many frames, simple sum might overflow if not careful, but float64 should be fine.
    
    # Accumulators for X, Y coordinates
    lips_sum = np.zeros((len(LIPS_IDXS), 2), dtype=np.float64)
    lips_sq_sum = np.zeros((len(LIPS_IDXS), 2), dtype=np.float64)
    lips_count = 0
    
    hand_sum = np.zeros((len(LEFT_HAND_IDXS), 2), dtype=np.float64)
    hand_sq_sum = np.zeros((len(LEFT_HAND_IDXS), 2), dtype=np.float64)
    hand_count = 0
    
    pose_sum = np.zeros((len(POSE_IDXS), 2), dtype=np.float64)
    pose_sq_sum = np.zeros((len(POSE_IDXS), 2), dtype=np.float64)
    pose_count = 0
    
    # Define slicing indices
    LIPS_END = LIPS_START + 40
    LEFT_HAND_END = LEFT_HAND_START + 21
    POSE_END = POSE_START + 5
    
    print("Iterating through dataset...")
    for i, (frames, non_empty_idxs, labels) in enumerate(tqdm(dataloader)):
        # frames shape: (B, INPUT_SIZE, N_COLS, 3)
        # We only care about non-empty frames and X, Y (indices 0, 1)
        
        # Mask for valid frames
        # non_empty_idxs shape: (B, INPUT_SIZE)
        # -1 indicates padding
        mask = (non_empty_idxs != -1) # (B, INPUT_SIZE)
        
        # Flatten batch and time dimensions
        # We want to select only valid frames
        valid_frames = frames[mask] # (N_VALID_FRAMES, N_COLS, 3)
        
        if valid_frames.shape[0] == 0:
            continue
            
        # Extract X, Y
        valid_frames_xy = valid_frames[:, :, :2].numpy() # (N, N_COLS, 2)
        
        # 1. LIPS
        lips_data = valid_frames_xy[:, LIPS_START:LIPS_END, :]
        # Filter out 0.0 (missing data often filled with 0 in preprocessing if NaN)
        # However, preprocessing fills NaNs with 0.0. We should be careful.
        # If we assume 0.0 is missing, we might ignore valid 0.0. 
        # But in normalized coordinates (before this step), 0.0 might be valid.
        # Let's assume the preprocessing output is what the model sees, so we calculate stats on that.
        # BUT, if we include padding zeros in Mean/Std calculation, it will bias towards 0.
        # We already filtered out padding frames (mask).
        # Within a valid frame, if a landmark is missing (NaN originally), it became 0.0.
        # Ideally we should ignore those. But checking for exactly 0.0 is risky.
        # For now, let's calculate stats on all valid frames.
        
        lips_sum += np.sum(lips_data, axis=0)
        lips_sq_sum += np.sum(lips_data ** 2, axis=0)
        lips_count += lips_data.shape[0]
        
        # 2. HANDS
        hand_data = valid_frames_xy[:, LEFT_HAND_START:LEFT_HAND_END, :]
        hand_sum += np.sum(hand_data, axis=0)
        hand_sq_sum += np.sum(hand_data ** 2, axis=0)
        hand_count += hand_data.shape[0]
        
        # 3. POSE
        pose_data = valid_frames_xy[:, POSE_START:POSE_END, :]
        pose_sum += np.sum(pose_data, axis=0)
        pose_sq_sum += np.sum(pose_data ** 2, axis=0)
        pose_count += pose_data.shape[0]

        if i >= 100:
            print("Reached 100 batches, stopping...")
            break

    print("Computing final stats...")
    
    def compute_mean_std(sum_val, sq_sum_val, count):
        if count == 0:
            return np.zeros_like(sum_val), np.ones_like(sum_val)
        mean = sum_val / count
        var = (sq_sum_val / count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 0))
        return mean, std

    lips_mean, lips_std = compute_mean_std(lips_sum, lips_sq_sum, lips_count)
    hand_mean, hand_std = compute_mean_std(hand_sum, hand_sq_sum, hand_count)
    pose_mean, pose_std = compute_mean_std(pose_sum, pose_sq_sum, pose_count)
    
    # Save to JSON
    stats = {
        "LIPS_MEAN": lips_mean.tolist(),
        "LIPS_STD": lips_std.tolist(),
        "LEFT_HANDS_MEAN": hand_mean.tolist(),
        "LEFT_HANDS_STD": hand_std.tolist(),
        "POSE_MEAN": pose_mean.tolist(),
        "POSE_STD": pose_std.tolist()
    }
    
    with open("mean_std.json", "w") as f:
        json.dump(stats, f)
        
    print("Stats saved to mean_std.json")
    
    # Print for copy-pasting
    print("\nCopy this to config.py:\n")
    print(f"LIPS_MEAN = np.array({lips_mean.tolist()}, dtype=np.float32)")
    print(f"LIPS_STD = np.array({lips_std.tolist()}, dtype=np.float32)")
    print(f"LEFT_HANDS_MEAN = np.array({hand_mean.tolist()}, dtype=np.float32)")
    print(f"LEFT_HANDS_STD = np.array({hand_std.tolist()}, dtype=np.float32)")
    print(f"POSE_MEAN = np.array({pose_mean.tolist()}, dtype=np.float32)")
    print(f"POSE_STD = np.array({pose_std.tolist()}, dtype=np.float32)")

if __name__ == "__main__":
    calculate_stats()
