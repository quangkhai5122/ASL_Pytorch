import numpy as np
import torch

# =============================================================================
# General Configuration
# =============================================================================
N_ROWS = 543
N_DIMS = 3
SEED = 42
NUM_CLASSES = 250
INPUT_SIZE = 64

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Training Hyperparameters
# =============================================================================
# Cross-Validation
N_FOLDS = 5 

# Training Batch Size (Balanced Sampler)
# Batch size = NUM_CLASSES * BATCH_ALL_SIGNS_N (ví dụ: 250 * 4 = 1000)
BATCH_ALL_SIGNS_N = 4 

# Validation Batch Size (Standard Sampler)
VAL_BATCH_SIZE = 64
N_EPOCHS_PER_FOLD = 100 
LR_MAX = 1e-3
N_WARMUP_EPOCHS = 0

# Optimizer specific
OPTIMIZER_LR = 1e-3
OPTIMIZER_WD = 1e-5  # Initial weight decay (will be overridden by adaptive callback)
WD_RATIO = 0.05  # Weight decay ratio for adaptive weight decay (weight_decay = lr * WD_RATIO)
CLIP_NORM = 1.0

# Loss specific
LABEL_SMOOTHING = 0.25

# =============================================================================
# Landmark Indices 
# =============================================================================
LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])

# Landmark indices in original data (0-indexed)
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

# Combined indices
LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))

N_COLS = LANDMARK_IDXS_LEFT_DOMINANT0.size # 66

# Landmark indices in processed data (relative indices)
LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_POSE_IDXS0)).squeeze()

# Start indices for slicing the processed data
LIPS_START = 0
LEFT_HAND_START = LIPS_IDXS.size # 40
POSE_START = LEFT_HAND_START + LEFT_HAND_IDXS.size # 61

# =============================================================================
# Model Architecture Hyperparameters
# =============================================================================
# Dense layer units for landmarks
LIPS_UNITS = 384
HANDS_UNITS = 384
POSE_UNITS = 384

# Final embedding and transformer embedding size
UNITS = 512

# Transformer
NUM_BLOCKS = 2
MLP_RATIO = 2
NUM_HEADS = 8 

# Dropout
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

# Augmentation
FRAME_MASK_RATIO = 0.25 # Random Frame Masking probability (1 - 0.25 = 75% keep rate)

# =============================================================================
# Normalization Constants (Placeholders from Notebook)
# =============================================================================
# Kept exactly as defined in the notebook (all zeros).

def get_normalization_constants():
    # LIPS
    LIPS_MEAN = np.zeros((LIPS_IDXS.size, 2), dtype=np.float32)
    LIPS_STD = np.zeros((LIPS_IDXS.size, 2), dtype=np.float32)

    # HANDS
    LEFT_HANDS_MEAN = np.zeros((LEFT_HAND_IDXS.size, 2), dtype=np.float32)
    LEFT_HANDS_STD = np.zeros((LEFT_HAND_IDXS.size, 2), dtype=np.float32)

    # POSE
    POSE_MEAN = np.zeros((POSE_IDXS.size, 2), dtype=np.float32)
    POSE_STD = np.zeros((POSE_IDXS.size, 2), dtype=np.float32)
    
    return LIPS_MEAN, LIPS_STD, LEFT_HANDS_MEAN, LEFT_HANDS_STD, POSE_MEAN, POSE_STD

LIPS_MEAN, LIPS_STD, LEFT_HANDS_MEAN, LEFT_HANDS_STD, POSE_MEAN, POSE_STD = get_normalization_constants()