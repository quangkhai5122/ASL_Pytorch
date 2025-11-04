import torch
import torch.nn as nn
import numpy as np

try:
    from scripts.config import (
        N_DIMS, INPUT_SIZE, N_COLS,
        LIPS_IDXS, LEFT_HAND_IDXS, POSE_IDXS,
        LANDMARK_IDXS_LEFT_DOMINANT0, LANDMARK_IDXS_RIGHT_DOMINANT0,
        LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0
    )
except ImportError:
    from config import (
        N_DIMS, INPUT_SIZE, N_COLS,
        LIPS_IDXS, LEFT_HAND_IDXS, POSE_IDXS,
        LANDMARK_IDXS_LEFT_DOMINANT0, LANDMARK_IDXS_RIGHT_DOMINANT0,
        LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0
    )

class PreprocessLayer(nn.Module):
    """
    PyTorch implementation of the TensorFlow PreprocessLayer.
    Handles dominant hand detection, filtering, normalization (mirroring), and resizing.
    """
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        
        # Pre-calculate the normalization correction tensor
        correction_x = torch.cat([
            torch.zeros(len(LIPS_IDXS)),
            torch.full((len(LEFT_HAND_IDXS),), 0.50),
            torch.full((len(POSE_IDXS),), 0.50)
        ])
        correction_y = torch.zeros(N_COLS)
        correction_z = torch.zeros(N_COLS)
        
        # Shape (N_COLS, 3)
        normalisation_correction = torch.stack([correction_x, correction_y, correction_z], dim=1)
        
        # Register buffers (constants that are part of the model but not trained)
        self.register_buffer('normalisation_correction', normalisation_correction)
        self.register_buffer('LEFT_HAND_IDXS0', torch.tensor(LEFT_HAND_IDXS0, dtype=torch.long))
        self.register_buffer('RIGHT_HAND_IDXS0', torch.tensor(RIGHT_HAND_IDXS0, dtype=torch.long))
        self.register_buffer('LANDMARK_IDXS_LEFT_DOMINANT0', torch.tensor(LANDMARK_IDXS_LEFT_DOMINANT0, dtype=torch.long))
        self.register_buffer('LANDMARK_IDXS_RIGHT_DOMINANT0', torch.tensor(LANDMARK_IDXS_RIGHT_DOMINANT0, dtype=torch.long))

    def pad_edge_3d(self, t, repeats, side):
        if repeats <= 0:
            return t
        repeats = int(repeats)
        if side == 'LEFT':
            return torch.cat((t[:1].repeat(repeats, 1, 1), t), dim=0)
        elif side == 'RIGHT':
            return torch.cat((t, t[-1:].repeat(repeats, 1, 1)), dim=0)

    def pad_edge_1d(self, t, repeats, side):
        if repeats <= 0:
            return t
        repeats = int(repeats)
        if side == 'LEFT':
            return torch.cat((t[:1].repeat(repeats), t), dim=0)
        elif side == 'RIGHT':
            return torch.cat((t, t[-1:].repeat(repeats)), dim=0)

    # Processes a SINGLE video sequence (N_FRAMES, N_ROWS, N_DIMS)
    def forward(self, data0):
        
        N_FRAMES0 = data0.shape[0]

        # 1. Find dominant hand
        def count_presence(tensor):
            return torch.sum(torch.where(torch.isnan(tensor), 0., 1.))

        left_hand_data = data0[:, self.LEFT_HAND_IDXS0, :]
        right_hand_data = data0[:, self.RIGHT_HAND_IDXS0, :]
        
        left_dominant = count_presence(left_hand_data) >= count_presence(right_hand_data)
        
        # 2. Filter frames based on dominant hand presence
        if left_dominant:
            dominant_hand_data = left_hand_data
        else:
            dominant_hand_data = right_hand_data
            
        frames_hands_non_nan_sum = torch.sum(torch.where(torch.isnan(dominant_hand_data), 0., 1.), dim=[1, 2])
        non_empty_frames_idxs = torch.where(frames_hands_non_nan_sum > 0)[0]
        
        if len(non_empty_frames_idxs) == 0:
            # Handle case where no frames have the dominant hand
            data = torch.zeros((INPUT_SIZE, N_COLS, N_DIMS), dtype=torch.float32, device=data0.device)
            non_empty_frames_idxs = torch.full((INPUT_SIZE,), -1.0, dtype=torch.float32, device=data0.device)
            return data, non_empty_frames_idxs

        # Filter data
        data = data0[non_empty_frames_idxs]
        
        # Normalize indices to start with 0
        non_empty_frames_idxs = non_empty_frames_idxs.float()
        non_empty_frames_idxs -= torch.min(non_empty_frames_idxs)
        
        N_FRAMES = data.shape[0]
        
        # 3. Gather Relevant Landmark Columns and Normalize (Mirroring)
        if left_dominant:
            data = data[:, self.LANDMARK_IDXS_LEFT_DOMINANT0, :]
        else:
            data = data[:, self.LANDMARK_IDXS_RIGHT_DOMINANT0, :]
            # Apply normalization correction (Flip X coordinates)
            multiplier = torch.where(self.normalisation_correction != 0, -1.0, 1.0)
            data = self.normalisation_correction + (data - self.normalisation_correction) * multiplier

        # 4. Resize/Downsample/Pad to INPUT_SIZE
        
        if N_FRAMES < INPUT_SIZE:
            # Case 1: Video is shorter -> Pad
            pad_size = INPUT_SIZE - N_FRAMES
            
            # Pad indices with -1
            non_empty_frames_idxs = torch.nn.functional.pad(non_empty_frames_idxs, (0, pad_size), value=-1.0)
            
            # Pad Data With Zeros
            data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, pad_size), value=0.0)
            
            # Fill NaN Values With 0
            data = torch.where(torch.isnan(data), 0.0, data)
            
            return data, non_empty_frames_idxs
        
        else:
            # Case 2: Video is longer or equal -> Downsample using Mean Pooling
            if N_FRAMES < INPUT_SIZE**2:
                # Use original N_FRAMES0 for calculation
                repeats = (INPUT_SIZE * INPUT_SIZE) // N_FRAMES0
                if repeats > 1:
                   data = torch.repeat_interleave(data, repeats=repeats, dim=0)
                   non_empty_frames_idxs = torch.repeat_interleave(non_empty_frames_idxs, repeats=repeats, dim=0)

            # Calculate pool size
            current_len = data.shape[0]
            # Use torch.div with rounding_mode='floor' for floordiv behavior
            pool_size = torch.div(current_len, INPUT_SIZE, rounding_mode='floor')
            
            if current_len % INPUT_SIZE > 0:
                pool_size += 1
                
            # Calculate padding size
            if pool_size == 1:
                 pad_size = (pool_size * INPUT_SIZE) - current_len
            else:
                 # This logic (mod current_len) is kept exactly as requested.
                 pad_size = (pool_size * INPUT_SIZE) % current_len

            # Pad Start/End with Edge Padding
            pad_left = torch.div(pad_size, 2, rounding_mode='floor') + torch.div(INPUT_SIZE, 2, rounding_mode='floor')
            pad_right = torch.div(pad_size, 2, rounding_mode='floor') + torch.div(INPUT_SIZE, 2, rounding_mode='floor')
            
            if pad_size % 2 > 0:
                pad_right += 1
            
            # Apply Edge Padding
            data = self.pad_edge_3d(data, pad_left, 'LEFT')
            data = self.pad_edge_3d(data, pad_right, 'RIGHT')
            
            non_empty_frames_idxs = self.pad_edge_1d(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge_1d(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            try:
                # Reshape: (INPUT_SIZE, -1, N_COLS, N_DIMS)
                data = data.reshape(INPUT_SIZE, -1, N_COLS, N_DIMS)
                non_empty_frames_idxs = non_empty_frames_idxs.reshape(INPUT_SIZE, -1)
            except RuntimeError as e:
                # Handle potential reshape errors if the logic doesn't guarantee divisibility
                print(f"Warning: Reshape failed during preprocessing. Len:{data.shape[0]}. Returning zeros. Error: {e}")
                data = torch.zeros((INPUT_SIZE, N_COLS, N_DIMS), dtype=torch.float32, device=data0.device)
                non_empty_frames_idxs = torch.full((INPUT_SIZE,), -1.0, dtype=torch.float32, device=data0.device)
                return data, non_empty_frames_idxs
            
            # Mean Pool 
            data = torch.nanmean(data, dim=1)
            non_empty_frames_idxs = torch.nanmean(non_empty_frames_idxs, dim=1)
            
            # Fill remaining NaN Values With 0
            data = torch.where(torch.isnan(data), 0.0, data)
            
            return data, non_empty_frames_idxs