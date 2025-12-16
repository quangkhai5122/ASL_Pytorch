"""
Sign Dictionary - Đọc và xử lý dữ liệu từ WLASL_Skeleton
Mỗi gloss có 1 file parquet duy nhất: {gloss}.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.slp_config import (
    AVATAR_RAW_IDXS, N_AVATAR_LANDMARKS,
    FACE_RAW_IDXS, LEFT_HAND_RAW_IDXS, RIGHT_HAND_RAW_IDXS, POSE_RAW_IDXS,
    POSE_RAW_OFFSET, POSE_SUBSET_LOCAL,
    IDX_LEFT_HAND, IDX_RIGHT_HAND,
    MOTION_THRESHOLD, MIN_MOTION_FRAMES
)


class SignDictionary:
    """
    Dictionary để quản lý và truy xuất dữ liệu từ WLASL_Skeleton.
    Mỗi gloss tương ứng với 1 file .parquet.
    """
    
    def __init__(self, data_dir: str = "data/WLASL_Skeleton"):
        """
        Args:
            data_dir: Đường dẫn đến thư mục chứa các file .parquet
        """
        self.data_dir = Path(data_dir)
        self._load_glossary()
        
    def _load_glossary(self):
        """Load danh sách các gloss có sẵn từ thư mục"""
        parquet_files = list(self.data_dir.glob("*.parquet"))
        self.glossary = {f.stem.lower(): f for f in parquet_files}
        print(f"Loaded {len(self.glossary)} glosses from {self.data_dir}")
        
    def has_gloss(self, gloss: str) -> bool:
        """Kiểm tra gloss có trong dictionary không"""
        return gloss.lower() in self.glossary
    
    def get_available_glosses(self) -> list:
        """Trả về danh sách các gloss có sẵn"""
        return sorted(self.glossary.keys())
    
    def load_gloss(self, gloss: str) -> np.ndarray:
        """
        Load và xử lý dữ liệu cho một gloss.
        
        Args:
            gloss: Tên gloss (case-insensitive)
            
        Returns:
            np.ndarray: Shape (T, 153, 3) - Avatar landmarks đã được xử lý
            None nếu không tìm thấy gloss
        """
        gloss_lower = gloss.lower()
        
        if gloss_lower not in self.glossary:
            print(f"Warning: Gloss '{gloss}' not found in dictionary.")
            return None
            
        file_path = self.glossary[gloss_lower]
        
        try:
            # Load parquet
            df = pd.read_parquet(file_path)
            
            # Convert to full array (T, 543, 3)
            full_data = self._parquet_to_array(df)
            
            # Extract avatar subset (T, 153, 3)
            avatar_data = self._extract_avatar_subset(full_data)
            
            # Interpolate NaN values
            avatar_data = self._interpolate_nan(avatar_data)
            
            # Trim static frames at start/end
            avatar_data = self._trim_static_frames(avatar_data)
            
            return avatar_data
            
        except Exception as e:
            print(f"Error loading gloss '{gloss}': {e}")
            return None
    
    def _parquet_to_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert parquet DataFrame to numpy array (T, 543, 3)
        
        Parquet structure:
        - frame: frame index
        - type: 'face', 'left_hand', 'pose', 'right_hand'
        - landmark_index: 0-467 (face), 0-20 (hands), 0-32 (pose)
        - x, y, z: coordinates
        """
        frames = sorted(df['frame'].unique())
        n_frames = len(frames)
        frame_to_idx = {f: i for i, f in enumerate(frames)}
        
        # Initialize với NaN
        full_data = np.full((n_frames, 543, 3), np.nan, dtype=np.float32)
        
        # Type offsets để map về global indices
        type_offsets = {
            'face': 0,
            'left_hand': 468,
            'pose': 489,
            'right_hand': 522
        }
        
        # Fill data từng type
        for type_name, offset in type_offsets.items():
            mask = df['type'] == type_name
            if not mask.any():
                continue
                
            sub = df[mask]
            frame_indices = sub['frame'].map(frame_to_idx).values
            landmark_indices = sub['landmark_index'].values + offset
            coords = sub[['x', 'y', 'z']].values
            
            full_data[frame_indices, landmark_indices] = coords
            
        return full_data
    
    def _extract_avatar_subset(self, full_data: np.ndarray) -> np.ndarray:
        """
        Extract avatar landmark subset từ full 543 landmarks.
        
        Args:
            full_data: Shape (T, 543, 3)
            
        Returns:
            Shape (T, 153, 3)
        """
        T = full_data.shape[0]
        avatar_data = np.full((T, N_AVATAR_LANDMARKS, 3), np.nan, dtype=np.float32)
        
        # Map indices
        # Face: FACE_RAW_IDXS -> 0:102
        # Left Hand: 468-488 -> 102:123
        # Pose subset: POSE_RAW_IDXS -> 123:132
        # Right Hand: 522-542 -> 132:153
        
        n_face = len(FACE_RAW_IDXS)
        n_hand = len(LEFT_HAND_RAW_IDXS)
        n_pose = len(POSE_RAW_IDXS)
        
        # Face (select specific indices from face mesh)
        avatar_data[:, :n_face, :] = full_data[:, FACE_RAW_IDXS, :]
        
        # Left Hand
        start = n_face
        avatar_data[:, start:start+n_hand, :] = full_data[:, LEFT_HAND_RAW_IDXS, :]
        
        # Pose subset
        start = n_face + n_hand
        avatar_data[:, start:start+n_pose, :] = full_data[:, POSE_RAW_IDXS, :]
        
        # Right Hand
        start = n_face + n_hand + n_pose
        avatar_data[:, start:start+n_hand, :] = full_data[:, RIGHT_HAND_RAW_IDXS, :]
        
        return avatar_data
    
    def _interpolate_nan(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values theo thời gian.
        
        Args:
            data: Shape (T, N, 3)
            
        Returns:
            Shape (T, N, 3) với NaN đã được interpolate
        """
        T, N, D = data.shape
        
        # Reshape để dễ xử lý
        flat_data = data.reshape(T, N * D)
        df = pd.DataFrame(flat_data)
        
        # Interpolate theo thời gian
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        
        # Fill remaining NaN (nếu toàn bộ cột là NaN) với 0.5 (center)
        df = df.fillna(0.5)
        
        result = df.values.reshape(T, N, D).astype(np.float32)
        return result
    
    def _trim_static_frames(self, data: np.ndarray) -> np.ndarray:
        """
        Trim các frame tĩnh ở đầu và cuối dựa trên motion của hands.
        
        Args:
            data: Shape (T, 153, 3)
            
        Returns:
            Shape (T', 153, 3) với T' <= T
        """
        T = data.shape[0]
        
        if T <= MIN_MOTION_FRAMES:
            return data
            
        # Tính motion dựa trên hands (chuyển động chính)
        hand_indices = np.concatenate([IDX_LEFT_HAND, IDX_RIGHT_HAND])
        hand_data = data[:, hand_indices, :2]  # Chỉ dùng x, y
        
        # Tính motion frame-by-frame
        motion = np.zeros(T)
        for i in range(1, T):
            diff = hand_data[i] - hand_data[i-1]
            # Bỏ qua NaN
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) > 0:
                motion[i] = np.mean(np.abs(valid_diff))
        
        # Tìm frame đầu tiên có motion > threshold
        start_idx = 0
        for i in range(T):
            if motion[i] > MOTION_THRESHOLD:
                # Lùi lại 1-2 frame để giữ pose bắt đầu
                start_idx = max(0, i - 2)
                break
        
        # Tìm frame cuối cùng có motion > threshold
        end_idx = T
        for i in range(T-1, -1, -1):
            if motion[i] > MOTION_THRESHOLD:
                # Thêm 1-2 frame để giữ pose kết thúc
                end_idx = min(T, i + 3)
                break
        
        # Đảm bảo có đủ frames
        if end_idx - start_idx < MIN_MOTION_FRAMES:
            # Không trim nếu quá ngắn
            return data
            
        return data[start_idx:end_idx]
    
    def get_gloss_info(self, gloss: str) -> dict:
        """
        Lấy thông tin về một gloss (cho debugging).
        
        Args:
            gloss: Tên gloss
            
        Returns:
            dict với thông tin về gloss
        """
        gloss_lower = gloss.lower()
        
        if gloss_lower not in self.glossary:
            return {"error": f"Gloss '{gloss}' not found"}
            
        file_path = self.glossary[gloss_lower]
        df = pd.read_parquet(file_path)
        
        info = {
            "gloss": gloss,
            "file": str(file_path),
            "n_frames": df['frame'].nunique(),
            "frame_range": (df['frame'].min(), df['frame'].max()),
        }
        
        # Check NaN ratio per type
        for type_name in ['face', 'left_hand', 'pose', 'right_hand']:
            sub = df[df['type'] == type_name]
            if len(sub) > 0:
                nan_ratio = sub['x'].isna().sum() / len(sub)
                info[f"{type_name}_nan_ratio"] = f"{100*nan_ratio:.1f}%"
                
        return info


# Test
if __name__ == "__main__":
    dictionary = SignDictionary()
    
    # Test với một vài gloss
    for gloss in ["hello", "thank", "you", "goodbye"]:
        print(f"\n=== {gloss.upper()} ===")
        info = dictionary.get_gloss_info(gloss)
        print(info)
        
        data = dictionary.load_gloss(gloss)
        if data is not None:
            print(f"Loaded shape: {data.shape}")
            print(f"NaN ratio: {100*np.isnan(data).sum()/data.size:.2f}%")
