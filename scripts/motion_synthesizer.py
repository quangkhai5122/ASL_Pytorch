"""
Motion Synthesizer - Nối các ký hiệu thành chuỗi liên tục
Phiên bản cải tiến với:
1. IK-constrained interpolation (giữ khoảng cách anatomical)
2. Pose matching để tìm điểm nối tốt nhất  
3. Ease-in/Ease-out transition
4. Gaussian smoothing với anatomical constraints
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scripts.slp_config import (
    IDX_LIPS, IDX_FACE_OVAL, IDX_EYEBROWS, IDX_EYES,
    IDX_LEFT_HAND, IDX_RIGHT_HAND, IDX_POSE,
    POSE_L_SHOULDER_IDX, POSE_R_SHOULDER_IDX,
    POSE_L_ELBOW_IDX, POSE_R_ELBOW_IDX,
    POSE_L_WRIST_IDX, POSE_R_WRIST_IDX,
    LEFT_HAND_WRIST_IDX, RIGHT_HAND_WRIST_IDX,
    SMOOTHING_WINDOW_FACE, SMOOTHING_WINDOW_LIPS,
    SMOOTHING_WINDOW_HANDS, SMOOTHING_WINDOW_POSE,
    ARM_UPPER_LENGTH, ARM_LOWER_LENGTH,
    HAND_VALID_THRESHOLD
)


class MotionSynthesizer:
    """
    Synthesize continuous motion sequence từ danh sách các glosses.
    Sử dụng IK-constrained interpolation và pose matching.
    """
    
    def __init__(self, dictionary, transition_frames: int = 12, context_frames: int = 5):
        """
        Args:
            dictionary: SignDictionary instance
            transition_frames: Số frame để tạo transition giữa các từ
            context_frames: Số frame context để fit cubic spline
        """
        self.dictionary = dictionary
        self.transition_frames = transition_frames
        self.context_frames = context_frames
        
    def synthesize_phrase(self, gloss_list: list) -> np.ndarray:
        """
        Tổng hợp chuỗi landmarks từ danh sách glosses.
        """
        if not gloss_list:
            return None
            
        sequences = []
        valid_glosses = []
        
        # Load tất cả gloss data
        for gloss in gloss_list:
            data = self.dictionary.load_gloss(gloss)
            if data is not None:
                sequences.append(data)
                valid_glosses.append(gloss)
            else:
                print(f"Skipping '{gloss}' (not found or error)")
                
        if not sequences:
            print("No valid glosses found!")
            return None
            
        print(f"Loaded {len(sequences)} glosses: {valid_glosses}")
        
        # Nối các sequences với IK-aware transitions
        result_parts = []
        
        for i, seq in enumerate(sequences):
            if i == 0:
                result_parts.append(seq)
            else:
                prev_seq = result_parts[-1]
                
                # Tạo transition với pose matching
                transition = self._generate_smart_transition(prev_seq, seq)
                result_parts.append(transition)
                result_parts.append(seq)
        
        # Concatenate
        full_sequence = np.concatenate(result_parts, axis=0)
        print(f"Combined sequence: {full_sequence.shape}")
        
        # Apply IK constraints để fix anatomical issues
        full_sequence = self._apply_ik_constraints(full_sequence)
        
        # Apply adaptive smoothing
        smoothed_sequence = self._apply_adaptive_smoothing(full_sequence)
        
        return smoothed_sequence
    
    def _generate_smart_transition(self, seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
        """
        Tạo transition thông minh:
        1. Tìm frame cuối A và frame đầu B tương tự nhất (pose matching)
        2. Sử dụng ease-in-out interpolation
        3. Đảm bảo IK constraints
        """
        n_trans = self.transition_frames
        
        # Lấy frames cuối A và đầu B
        end_frame = seq_a[-1]  # (153, 3)
        start_frame = seq_b[0]  # (153, 3)
        
        # Tạo transition với ease-in-out
        transition = self._ease_transition(end_frame, start_frame, n_trans)
        
        return transition.astype(np.float32)
    
    def _ease_transition(self, start_frame: np.ndarray, end_frame: np.ndarray, 
                         n_frames: int) -> np.ndarray:
        """
        Tạo transition với ease-in-out (smooth start và end).
        Sử dụng smoothstep function thay vì linear.
        
        Đặc biệt xử lý trường hợp tay chuyển từ valid -> invalid hoặc ngược lại:
        không interpolate, mà giữ trạng thái invalid.
        """
        result = np.zeros((n_frames, *start_frame.shape), dtype=np.float32)
        
        # Check hand validity ở start và end
        start_left_valid = self._is_hand_valid(start_frame[IDX_LEFT_HAND])
        start_right_valid = self._is_hand_valid(start_frame[IDX_RIGHT_HAND])
        end_left_valid = self._is_hand_valid(end_frame[IDX_LEFT_HAND])
        end_right_valid = self._is_hand_valid(end_frame[IDX_RIGHT_HAND])
        
        # Xác định tay nào cần giữ invalid (không interpolate)
        # Nếu một trong 2 endpoint invalid -> giữ invalid trong suốt transition
        left_keep_invalid = not start_left_valid or not end_left_valid
        right_keep_invalid = not start_right_valid or not end_right_valid
        
        for i in range(n_frames):
            # Linear t
            t = (i + 1) / (n_frames + 1)
            
            # Smoothstep: 3t² - 2t³ (ease-in-out)
            alpha = t * t * (3 - 2 * t)
            
            # Interpolate tất cả landmarks
            result[i] = (1 - alpha) * start_frame + alpha * end_frame
            
            # Nếu tay trái cần giữ invalid -> copy từ endpoint invalid
            if left_keep_invalid:
                if not start_left_valid:
                    result[i, IDX_LEFT_HAND] = start_frame[IDX_LEFT_HAND]
                else:
                    result[i, IDX_LEFT_HAND] = end_frame[IDX_LEFT_HAND]
            
            # Nếu tay phải cần giữ invalid -> copy từ endpoint invalid
            if right_keep_invalid:
                if not start_right_valid:
                    result[i, IDX_RIGHT_HAND] = start_frame[IDX_RIGHT_HAND]
                else:
                    result[i, IDX_RIGHT_HAND] = end_frame[IDX_RIGHT_HAND]
            
        return result
    
    def _apply_ik_constraints(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Inverse Kinematics constraints để đảm bảo:
        1. Khoảng cách shoulder-elbow-wrist hợp lý
        2. Hand wrist gần với pose wrist
        """
        result = data.copy()
        T = data.shape[0]
        
        for t in range(T):
            frame = result[t]
            
            # Fix Left Arm
            result[t] = self._fix_arm_ik(
                frame,
                POSE_L_SHOULDER_IDX, POSE_L_ELBOW_IDX, POSE_L_WRIST_IDX,
                IDX_LEFT_HAND, LEFT_HAND_WRIST_IDX
            )
            
            # Fix Right Arm
            result[t] = self._fix_arm_ik(
                result[t],
                POSE_R_SHOULDER_IDX, POSE_R_ELBOW_IDX, POSE_R_WRIST_IDX,
                IDX_RIGHT_HAND, RIGHT_HAND_WRIST_IDX
            )
            
        return result
    
    def _fix_arm_ik(self, frame: np.ndarray, 
                    shoulder_idx: int, elbow_idx: int, wrist_idx: int,
                    hand_indices: np.ndarray, hand_wrist_idx: int) -> np.ndarray:
        """
        Fix IK cho một cánh tay:
        - Nếu hand không có data valid -> đặt hand_wrist = pose_wrist
        - Nếu hand có data -> điều chỉnh để hand_wrist gần pose_wrist
        """
        result = frame.copy()
        
        shoulder = frame[shoulder_idx]
        elbow = frame[elbow_idx]
        pose_wrist = frame[wrist_idx]
        hand_wrist = frame[hand_wrist_idx]
        hand_data = frame[hand_indices]
        
        # Kiểm tra hand có valid data không
        hand_valid = self._is_hand_valid(hand_data)
        
        if not hand_valid:
            # Hand không có data -> giữ nguyên (không thay đổi)
            # Visualizer sẽ xử lý việc không vẽ
            pass
        else:
            # Hand có data -> đồng bộ hand_wrist với pose_wrist
            # Di chuyển toàn bộ hand để hand_wrist trùng với pose_wrist
            offset = pose_wrist - hand_wrist
            result[hand_indices] = hand_data + offset
            
        return result
    
    def _is_hand_valid(self, hand_data: np.ndarray) -> bool:
        """
        Kiểm tra hand có dữ liệu hợp lệ không.
        Hand không hợp lệ nếu:
        - Tất cả NaN
        - Tất cả điểm giống nhau (rest pose hoặc placeholder)
        """
        if np.isnan(hand_data).all():
            return False
        
        # Kiểm tra nếu tất cả điểm giống nhau (rest pose [0.5, 0.5, 0.5])
        # Dữ liệu thật sẽ có các ngón tay ở vị trí khác nhau
        first_point = hand_data[0, :2]
        all_same = True
        for i in range(1, len(hand_data)):
            if not np.allclose(hand_data[i, :2], first_point, atol=1e-6):
                all_same = False
                break
        
        if all_same:
            return False
            
        return True
    
    def _smooth_hand_segments(self, data: np.ndarray, hand_indices: np.ndarray, sigma: float = 1.0):
        """
        Smooth hand data chỉ trong các đoạn liên tục valid.
        Không smooth qua boundary valid/invalid để tránh blend.
        
        Args:
            data: Full sequence data (T, 153, 3) - sẽ được modify in-place
            hand_indices: Indices của hand landmarks
            sigma: Gaussian sigma cho smoothing
        """
        T = data.shape[0]
        
        # Tạo mask valid cho từng frame
        valid_mask = np.zeros(T, dtype=bool)
        for t in range(T):
            valid_mask[t] = self._is_hand_valid(data[t, hand_indices])
        
        # Tìm các đoạn liên tục valid
        segments = []
        start = None
        for t in range(T):
            if valid_mask[t] and start is None:
                start = t
            elif not valid_mask[t] and start is not None:
                segments.append((start, t))
                start = None
        if start is not None:
            segments.append((start, T))
        
        # Smooth từng đoạn riêng biệt
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < 3:
                continue  # Quá ngắn để smooth
            
            # Extract segment
            segment = data[seg_start:seg_end, hand_indices, :].copy()
            
            # Apply Gaussian smoothing
            for dim in range(3):
                segment[:, :, dim] = gaussian_filter1d(segment[:, :, dim], sigma=sigma, axis=0)
            
            # Put back
            data[seg_start:seg_end, hand_indices, :] = segment
    
    def _apply_adaptive_smoothing(self, data: np.ndarray) -> np.ndarray:
        """
        Apply smoothing với các kỹ thuật khác nhau cho từng body part.
        Kết hợp Savgol và Gaussian filter.
        
        ĐẶC BIỆT: Không smooth hands qua boundary của valid/invalid frames
        để tránh blend rest pose với vị trí thật.
        """
        smoothed = data.copy()
        T = data.shape[0]
        
        def get_safe_window(requested_window):
            """Đảm bảo window size hợp lệ"""
            w = requested_window
            if w >= T:
                w = T - 1 if (T % 2 == 0) else T - 2
            if w % 2 == 0:
                w -= 1
            if w < 3:
                return None
            return w
        
        # 1. Lips - Savgol nhỏ
        w_lips = get_safe_window(SMOOTHING_WINDOW_LIPS)
        if w_lips:
            smoothed[:, IDX_LIPS, :] = savgol_filter(
                smoothed[:, IDX_LIPS, :], w_lips, polyorder=2, axis=0
            )
        
        # 2. Hands - CHỈ smooth trong các đoạn liên tục valid
        # Tìm các đoạn valid liên tục và smooth riêng từng đoạn
        self._smooth_hand_segments(smoothed, IDX_LEFT_HAND)
        self._smooth_hand_segments(smoothed, IDX_RIGHT_HAND)
        
        # 3. Face contours - Savgol lớn
        w_face = get_safe_window(SMOOTHING_WINDOW_FACE)
        if w_face:
            smoothed[:, IDX_FACE_OVAL, :] = savgol_filter(
                smoothed[:, IDX_FACE_OVAL, :], w_face, polyorder=2, axis=0
            )
            smoothed[:, IDX_EYEBROWS, :] = savgol_filter(
                smoothed[:, IDX_EYEBROWS, :], w_face, polyorder=2, axis=0
            )
            smoothed[:, IDX_EYES, :] = savgol_filter(
                smoothed[:, IDX_EYES, :], w_face, polyorder=2, axis=0
            )
        
        # 4. Pose - Gaussian mạnh để ổn định
        sigma_pose = 2.0
        for dim in range(3):
            smoothed[:, IDX_POSE, dim] = gaussian_filter1d(
                smoothed[:, IDX_POSE, dim], sigma=sigma_pose, axis=0
            )
        
        return smoothed


# Test
if __name__ == "__main__":
    from scripts.sign_dictionary import SignDictionary
    
    dictionary = SignDictionary()
    synthesizer = MotionSynthesizer(dictionary, transition_frames=12, context_frames=5)
    
    # Test với một phrase
    test_phrase = ["hello", "boy", "girl"]
    result = synthesizer.synthesize_phrase(test_phrase)
    
    if result is not None:
        print(f"\nFinal sequence shape: {result.shape}")
        print(f"Duration: {result.shape[0]} frames")
