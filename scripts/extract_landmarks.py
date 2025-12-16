"""
Script trích xuất landmark từ video sử dụng MediaPipe
Output: file .parquet với cấu trúc giống Google GISLR competition

Cấu trúc output:
- frame: số thứ tự frame
- row_id: "{frame}-{type}-{landmark_index}"
- type: 'face', 'left_hand', 'right_hand', 'pose'
- landmark_index: chỉ số landmark (0-467 cho face, 0-20 cho hand, 0-32 cho pose)
- x, y, z: tọa độ (làm tròn 6 chữ số)
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Cấu hình
INPUT_DIR = Path("data/WLASL_Only1Video")
OUTPUT_DIR = Path("data/WLASL_Skeleton")

# Số lượng landmark theo từng loại
FACE_LANDMARKS = 468
HAND_LANDMARKS = 21
POSE_LANDMARKS = 33


def init_mediapipe():
    """Khởi tạo các model MediaPipe"""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True, # False
        model_complexity=2, # 1
        refine_face_landmarks=True, # True
        min_detection_confidence=0.4, # 0.5
        min_tracking_confidence=0.4 # 0.5
    )
    return holistic


def extract_landmarks_from_results(results, frame_idx):
    """
    Trích xuất landmarks từ kết quả MediaPipe
    Trả về list các dict với cấu trúc: frame, row_id, type, landmark_index, x, y, z
    """
    rows = []
    
    # Face landmarks (468 điểm)
    if results.face_landmarks:
        for idx, lm in enumerate(results.face_landmarks.landmark):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-face-{idx}",
                'type': 'face',
                'landmark_index': idx,
                'x': round(lm.x, 6),
                'y': round(lm.y, 6),
                'z': round(lm.z, 6)
            })
    else:
        # Nếu không phát hiện, điền NaN
        for idx in range(FACE_LANDMARKS):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-face-{idx}",
                'type': 'face',
                'landmark_index': idx,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
    
    # Left hand landmarks (21 điểm)
    if results.left_hand_landmarks:
        for idx, lm in enumerate(results.left_hand_landmarks.landmark):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-left_hand-{idx}",
                'type': 'left_hand',
                'landmark_index': idx,
                'x': round(lm.x, 6),
                'y': round(lm.y, 6),
                'z': round(lm.z, 6)
            })
    else:
        for idx in range(HAND_LANDMARKS):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-left_hand-{idx}",
                'type': 'left_hand',
                'landmark_index': idx,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
    
    # Pose landmarks (33 điểm)
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-pose-{idx}",
                'type': 'pose',
                'landmark_index': idx,
                'x': round(lm.x, 6),
                'y': round(lm.y, 6),
                'z': round(lm.z, 6)
            })
    else:
        for idx in range(POSE_LANDMARKS):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-pose-{idx}",
                'type': 'pose',
                'landmark_index': idx,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
    
    # Right hand landmarks (21 điểm)
    if results.right_hand_landmarks:
        for idx, lm in enumerate(results.right_hand_landmarks.landmark):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-right_hand-{idx}",
                'type': 'right_hand',
                'landmark_index': idx,
                'x': round(lm.x, 6),
                'y': round(lm.y, 6),
                'z': round(lm.z, 6)
            })
    else:
        for idx in range(HAND_LANDMARKS):
            rows.append({
                'frame': frame_idx,
                'row_id': f"{frame_idx}-right_hand-{idx}",
                'type': 'right_hand',
                'landmark_index': idx,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
    
    return rows


def process_video(video_path, holistic):
    """
    Xử lý một video và trích xuất tất cả landmarks
    """
    cap = cv2.VideoCapture(str(video_path))
    all_rows = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển BGR sang RGB cho MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý frame
        results = holistic.process(frame_rgb)
        
        # Trích xuất landmarks
        rows = extract_landmarks_from_results(results, frame_idx)
        all_rows.extend(rows)
        
        frame_idx += 1
    
    cap.release()
    
    return all_rows, frame_idx


def save_parquet(rows, output_path):
    """
    Lưu dữ liệu vào file parquet với đúng định dạng
    """
    df = pd.DataFrame(rows)
    
    # Chuyển đổi kiểu dữ liệu
    df['frame'] = df['frame'].astype('int16')
    df['landmark_index'] = df['landmark_index'].astype('int16')
    df['x'] = df['x'].astype('float64')
    df['y'] = df['y'].astype('float64')
    df['z'] = df['z'].astype('float64')
    
    # Sắp xếp cột đúng thứ tự
    df = df[['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']]
    
    # Lưu file
    df.to_parquet(output_path, index=False)
    
    return df.shape[0]


def main():
    # Tạo thư mục output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách video
    video_files = list(INPUT_DIR.glob("*.mp4"))
    print(f"Tìm thấy {len(video_files)} video trong {INPUT_DIR}")
    
    if len(video_files) == 0:
        print("Không tìm thấy video nào!")
        return
    
    # Khởi tạo MediaPipe
    print("Đang khởi tạo MediaPipe Holistic...")
    holistic = init_mediapipe()
    
    # Thống kê
    stats = {
        'total_videos': len(video_files),
        'processed': 0,
        'failed': 0,
        'total_frames': 0,
        'total_landmarks': 0
    }
    
    failed_videos = []
    
    # Xử lý từng video
    for video_path in tqdm(video_files, desc="Đang xử lý"):
        try:
            # Tên output file (dùng tên video làm tên file)
            output_name = video_path.stem + ".parquet"
            output_path = OUTPUT_DIR / output_name
            
            # Bỏ qua nếu đã xử lý
            if output_path.exists():
                stats['processed'] += 1
                continue
            
            # Xử lý video
            rows, num_frames = process_video(video_path, holistic)
            
            if len(rows) == 0:
                print(f"\nWarning: Không có landmark nào cho {video_path.name}")
                failed_videos.append(video_path.name)
                stats['failed'] += 1
                continue
            
            # Lưu file parquet
            num_rows = save_parquet(rows, output_path)
            
            stats['processed'] += 1
            stats['total_frames'] += num_frames
            stats['total_landmarks'] += num_rows
            
        except Exception as e:
            print(f"\nLỗi xử lý {video_path.name}: {str(e)}")
            failed_videos.append(video_path.name)
            stats['failed'] += 1
    
    # Đóng MediaPipe
    holistic.close()
    
    # In kết quả
    print("\n" + "="*60)
    print("KẾT QUẢ TRÍCH XUẤT LANDMARK")
    print("="*60)
    print(f"Tổng số video: {stats['total_videos']}")
    print(f"Xử lý thành công: {stats['processed']}")
    print(f"Thất bại: {stats['failed']}")
    print(f"Tổng số frame: {stats['total_frames']}")
    print(f"Tổng số landmark rows: {stats['total_landmarks']}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    if failed_videos:
        print(f"\nVideo thất bại ({len(failed_videos)}):")
        for v in failed_videos[:10]:
            print(f"  - {v}")
        if len(failed_videos) > 10:
            print(f"  ... và {len(failed_videos) - 10} video khác")


if __name__ == "__main__":
    main()
