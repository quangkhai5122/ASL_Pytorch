"""
Configuration cho Sign Language Production (SLP) với WLASL_Mediapipe
"""
import numpy as np

# =============================================================================
# RAW LANDMARK INDICES (MediaPipe Holistic - 543 total)
# Order: Face(468) -> Left Hand(21) -> Pose(33) -> Right Hand(21)
# =============================================================================

# Face Landmarks (468 points, indices 0-467)
# Lips (40 points) - cho chuyển động miệng
LIPS_RAW_IDXS = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])

# Face Oval (36 points) - đường viền khuôn mặt
FACE_OVAL_RAW_IDXS = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
])

# Eyebrows (10 points: 5 Left + 5 Right)
LEFT_EYEBROW_RAW_IDXS = np.array([70, 63, 105, 66, 107])
RIGHT_EYEBROW_RAW_IDXS = np.array([336, 296, 334, 293, 300])
EYEBROWS_RAW_IDXS = np.concatenate((LEFT_EYEBROW_RAW_IDXS, RIGHT_EYEBROW_RAW_IDXS))

# Eyes (16 points: 8 Left + 8 Right)
LEFT_EYE_RAW_IDXS = np.array([33, 160, 158, 133, 153, 144, 163, 7])
RIGHT_EYE_RAW_IDXS = np.array([362, 385, 387, 263, 373, 380, 382, 249])
EYES_RAW_IDXS = np.concatenate((LEFT_EYE_RAW_IDXS, RIGHT_EYE_RAW_IDXS))

# Tổng hợp Face features
FACE_RAW_IDXS = np.concatenate((LIPS_RAW_IDXS, FACE_OVAL_RAW_IDXS, EYEBROWS_RAW_IDXS, EYES_RAW_IDXS))
# 40 + 36 + 10 + 16 = 102 face points

# Hands (21 points each)
# Trong parquet: type='left_hand' có landmark_index 0-20, type='right_hand' có landmark_index 0-20
# Global indices: left_hand = 468-488, right_hand = 522-542
LEFT_HAND_RAW_OFFSET = 468
RIGHT_HAND_RAW_OFFSET = 522
LEFT_HAND_RAW_IDXS = np.arange(468, 489)   # 21 points
RIGHT_HAND_RAW_IDXS = np.arange(522, 543)  # 21 points

# Pose (33 points, indices 489-521)
# Chỉ lấy subset cần thiết
POSE_RAW_OFFSET = 489
# 0: Nose, 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow, 
# 15: L_Wrist, 16: R_Wrist, 23: L_Hip, 24: R_Hip
POSE_SUBSET_LOCAL = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24])
POSE_RAW_IDXS = POSE_RAW_OFFSET + POSE_SUBSET_LOCAL  # 9 points

# =============================================================================
# AVATAR SUBSET (Reduced landmarks for visualization)
# Total: 102 (face) + 21 (left_hand) + 9 (pose) + 21 (right_hand) = 153 points
# =============================================================================

AVATAR_RAW_IDXS = np.concatenate((
    FACE_RAW_IDXS,       # 0-101   (102 points)
    LEFT_HAND_RAW_IDXS,  # 102-122 (21 points)
    POSE_RAW_IDXS,       # 123-131 (9 points)
    RIGHT_HAND_RAW_IDXS  # 132-152 (21 points)
))

N_AVATAR_LANDMARKS = len(AVATAR_RAW_IDXS)  # 153

# =============================================================================
# RELATIVE INDICES trong Avatar array (153 points)
# =============================================================================

# Face components (relative to avatar array)
N_LIPS = len(LIPS_RAW_IDXS)           # 40
N_FACE_OVAL = len(FACE_OVAL_RAW_IDXS) # 36
N_EYEBROWS = len(EYEBROWS_RAW_IDXS)   # 10
N_EYES = len(EYES_RAW_IDXS)           # 16
N_FACE = N_LIPS + N_FACE_OVAL + N_EYEBROWS + N_EYES  # 102

N_HAND = 21
N_POSE = len(POSE_SUBSET_LOCAL)  # 9

# Relative indices trong avatar array
IDX_LIPS = np.arange(0, N_LIPS)  # 0-39
IDX_FACE_OVAL = np.arange(N_LIPS, N_LIPS + N_FACE_OVAL)  # 40-75
IDX_EYEBROWS = np.arange(N_LIPS + N_FACE_OVAL, N_LIPS + N_FACE_OVAL + N_EYEBROWS)  # 76-85
IDX_EYES = np.arange(N_LIPS + N_FACE_OVAL + N_EYEBROWS, N_FACE)  # 86-101

IDX_FACE = np.arange(0, N_FACE)  # 0-101
IDX_LEFT_HAND = np.arange(N_FACE, N_FACE + N_HAND)  # 102-122
IDX_POSE = np.arange(N_FACE + N_HAND, N_FACE + N_HAND + N_POSE)  # 123-131
IDX_RIGHT_HAND = np.arange(N_FACE + N_HAND + N_POSE, N_FACE + 2*N_HAND + N_POSE)  # 132-152

# Pose specific indices trong avatar array
# Pose subset order: [Nose, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hip, R_Hip]
POSE_NOSE_IDX = IDX_POSE[0]          # 123
POSE_L_SHOULDER_IDX = IDX_POSE[1]    # 124
POSE_R_SHOULDER_IDX = IDX_POSE[2]    # 125
POSE_L_ELBOW_IDX = IDX_POSE[3]       # 126
POSE_R_ELBOW_IDX = IDX_POSE[4]       # 127
POSE_L_WRIST_IDX = IDX_POSE[5]       # 128
POSE_R_WRIST_IDX = IDX_POSE[6]       # 129
POSE_L_HIP_IDX = IDX_POSE[7]         # 130
POSE_R_HIP_IDX = IDX_POSE[8]         # 131

# Hand wrist indices (index 0 của mỗi hand)
LEFT_HAND_WRIST_IDX = IDX_LEFT_HAND[0]   # 102
RIGHT_HAND_WRIST_IDX = IDX_RIGHT_HAND[0] # 132

# =============================================================================
# SMOOTHING CONFIGURATION
# =============================================================================

SMOOTHING_WINDOW_FACE = 15   # Low frequency - face contours
SMOOTHING_WINDOW_LIPS = 5    # High frequency - lips
SMOOTHING_WINDOW_HANDS = 5   # High frequency - fingers
SMOOTHING_WINDOW_POSE = 21   # Low frequency - body anchors

# =============================================================================
# MOTION DETECTION (để trim frames không có motion)
# =============================================================================

MOTION_THRESHOLD = 0.005  # Ngưỡng motion để detect frame có chuyển động
MIN_MOTION_FRAMES = 10    # Số frame tối thiểu phải giữ lại

# =============================================================================
# VISUALIZATION CONNECTIONS
# =============================================================================

# Pose skeleton connections (relative indices)
POSE_CONNECTIONS = [
    (POSE_L_SHOULDER_IDX, POSE_R_SHOULDER_IDX),  # Shoulders
    (POSE_L_SHOULDER_IDX, POSE_L_ELBOW_IDX),
    (POSE_L_ELBOW_IDX, POSE_L_WRIST_IDX),
    (POSE_R_SHOULDER_IDX, POSE_R_ELBOW_IDX),
    (POSE_R_ELBOW_IDX, POSE_R_WRIST_IDX),
    (POSE_L_SHOULDER_IDX, POSE_L_HIP_IDX),
    (POSE_R_SHOULDER_IDX, POSE_R_HIP_IDX),
    (POSE_L_HIP_IDX, POSE_R_HIP_IDX),
]

# Pose to Hand connections
POSE_HAND_CONNECTIONS = [
    (POSE_L_WRIST_IDX, LEFT_HAND_WRIST_IDX),
    (POSE_R_WRIST_IDX, RIGHT_HAND_WRIST_IDX),
]

# Hand finger chains (local indices 0-20)
HAND_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],      # Thumb
    [0, 5, 6, 7, 8],      # Index
    [0, 9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],  # Ring
    [0, 17, 18, 19, 20],  # Pinky
]

FINGER_COLORS = ['#DDDD00', '#00DD00', '#00DDDD', '#0000DD', '#DD00DD']

# Eye indices (local to IDX_EYES, 8 points each)
IDX_LEFT_EYE_LOCAL = np.arange(0, 8)   # First 8 in IDX_EYES
IDX_RIGHT_EYE_LOCAL = np.arange(8, 16) # Last 8 in IDX_EYES

# Eyebrow indices (local to IDX_EYEBROWS, 5 points each)
IDX_LEFT_EYEBROW_LOCAL = np.arange(0, 5)
IDX_RIGHT_EYEBROW_LOCAL = np.arange(5, 10)

# =============================================================================
# ANATOMICAL CONSTRAINTS (để đảm bảo tính hợp lý khi nối)
# =============================================================================

# Chiều dài cánh tay ước tính (normalized coordinates)
# Dựa trên tỷ lệ cơ thể trung bình
ARM_UPPER_LENGTH = 0.15   # Shoulder -> Elbow
ARM_LOWER_LENGTH = 0.12   # Elbow -> Wrist
HAND_LENGTH = 0.08        # Wrist -> Fingertip

# Rest pose offsets (relative to shoulder, khi tay buông xuống)
REST_POSE_LEFT = {
    'elbow_offset': np.array([0.02, 0.15, 0.0]),   # Slightly out and down
    'wrist_offset': np.array([0.02, 0.27, 0.0]),   # Further down
}
REST_POSE_RIGHT = {
    'elbow_offset': np.array([-0.02, 0.15, 0.0]),
    'wrist_offset': np.array([-0.02, 0.27, 0.0]),
}

# Ngưỡng để xác định hand có valid data không
HAND_VALID_THRESHOLD = 0.1  # Nếu variance của hand < threshold -> không có data
