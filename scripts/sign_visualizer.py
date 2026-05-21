"""
Sign Visualizer - Tạo animation từ landmark sequences
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scripts.slp_config import (
    IDX_LIPS, IDX_FACE_OVAL, IDX_EYEBROWS, IDX_EYES,
    IDX_LEFT_HAND, IDX_RIGHT_HAND, IDX_POSE,
    IDX_LEFT_EYE_LOCAL, IDX_RIGHT_EYE_LOCAL,
    IDX_LEFT_EYEBROW_LOCAL, IDX_RIGHT_EYEBROW_LOCAL,
    POSE_CONNECTIONS, POSE_HAND_CONNECTIONS,
    HAND_FINGER_CHAINS, FINGER_COLORS
)


class SignVisualizer:
    """
    Visualize landmark sequences thành video/GIF.
    """
    
    def __init__(self, figsize=(8, 8), fps=20):
        """
        Args:
            figsize: Kích thước figure
            fps: Frames per second cho output
        """
        self.figsize = figsize
        self.fps = fps
        
    def create_animation(self, landmarks: np.ndarray, output_path: str):
        """
        Tạo animation từ landmarks và lưu thành file.
        
        Args:
            landmarks: Shape (T, 153, 3) - Avatar landmarks
            output_path: Đường dẫn output (.gif hoặc .mp4)
        """
        T = landmarks.shape[0]
        print(f"Creating animation with {T} frames...")

        # Setup figure
        plt.ioff()
        fig, ax = plt.subplots(figsize=self.figsize)

        # Canvas dimensions
        height, width = 600, 600
        fig.set_size_inches(width / 100, height / 100)

        # Output format
        is_gif = output_path.lower().endswith('.gif')
        frames_buffer = []
        writer = None

        if not is_gif:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        # Render each frame
        for i in range(T):
            ax.clear()
            self._setup_axes(ax)

            frame_data = landmarks[i]
            self._draw_frame(ax, frame_data)

            # Convert matplotlib canvas → numpy RGB
            fig.canvas.draw()
            try:
                # matplotlib >= 3.8: use buffer_rgba()
                buf = fig.canvas.buffer_rgba()
                img = np.frombuffer(buf, dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                img = img.reshape(h, w, 4)[:, :, :3]  # RGBA → RGB
            except AttributeError:
                # fallback for older matplotlib
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if is_gif:
                frames_buffer.append(img)
            else:
                import cv2
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img_bgr)

            if i % 20 == 0:
                print(f"  Rendered {i}/{T} frames", end='\r')

        # Save output
        if is_gif:
            print(f"\nSaving GIF to {output_path}...")
            try:
                import imageio.v2 as iio
                iio.mimsave(output_path, frames_buffer, fps=self.fps, loop=0)
            except ImportError:
                # Fallback: use Pillow to save GIF (no imageio needed)
                from PIL import Image as PILImage
                pil_frames = [PILImage.fromarray(f) for f in frames_buffer]
                duration_ms = int(1000 / self.fps)
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
        else:
            writer.release()
            print(f"\nSaved video to {output_path}")

        plt.close(fig)
        print("Done!")

        
    def _setup_axes(self, ax):
        """Setup axes cho visualization"""
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(1.1, -0.1)  # Flip Y axis (image coordinates)
        ax.set_aspect('equal')
        ax.axis('off')
        
    def _draw_frame(self, ax, frame: np.ndarray):
        """
        Vẽ một frame.
        
        Args:
            ax: Matplotlib axes
            frame: Shape (153, 3) - Landmarks cho một frame
        """
        # 1. Draw Face
        self._draw_face(ax, frame)
        
        # 2. Draw Pose skeleton
        self._draw_pose(ax, frame)
        
        # 3. Draw Hands
        self._draw_hand(ax, frame, IDX_LEFT_HAND)
        self._draw_hand(ax, frame, IDX_RIGHT_HAND)
        
    def _draw_face(self, ax, frame: np.ndarray):
        """Vẽ khuôn mặt"""
        face_color = 'darkred'
        face_lw = 2
        
        # Face Oval - đường viền khuôn mặt
        oval_pts = frame[IDX_FACE_OVAL]
        if not self._is_invalid(oval_pts):
            xs, ys = oval_pts[:, 0], oval_pts[:, 1]
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                   color=face_color, linewidth=face_lw)
        
        # Lips - môi (closed loop)
        lips_pts = frame[IDX_LIPS]
        if not self._is_invalid(lips_pts):
            xs, ys = lips_pts[:, 0], lips_pts[:, 1]
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                   color=face_color, linewidth=face_lw)
        
        # Eyebrows - lông mày (2 đường riêng biệt)
        eyebrow_pts = frame[IDX_EYEBROWS]
        if not self._is_invalid(eyebrow_pts):
            # Left eyebrow
            left_eb = eyebrow_pts[IDX_LEFT_EYEBROW_LOCAL]
            ax.plot(left_eb[:, 0], left_eb[:, 1], color=face_color, linewidth=face_lw)
            # Right eyebrow
            right_eb = eyebrow_pts[IDX_RIGHT_EYEBROW_LOCAL]
            ax.plot(right_eb[:, 0], right_eb[:, 1], color=face_color, linewidth=face_lw)
        
        # Eyes - mắt (2 vòng closed)
        eye_pts = frame[IDX_EYES]
        if not self._is_invalid(eye_pts):
            # Left eye
            left_eye = eye_pts[IDX_LEFT_EYE_LOCAL]
            xs, ys = left_eye[:, 0], left_eye[:, 1]
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                   color=face_color, linewidth=face_lw)
            # Right eye
            right_eye = eye_pts[IDX_RIGHT_EYE_LOCAL]
            xs, ys = right_eye[:, 0], right_eye[:, 1]
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), 
                   color=face_color, linewidth=face_lw)
    
    def _draw_pose(self, ax, frame: np.ndarray):
        """Vẽ skeleton pose"""
        pose_color = 'red'
        pose_lw = 3
        
        # Check hand validity for pose-to-hand connections
        left_hand_valid = self._is_hand_valid(frame, IDX_LEFT_HAND)
        right_hand_valid = self._is_hand_valid(frame, IDX_RIGHT_HAND)
        
        # Draw pose connections
        for idx1, idx2 in POSE_CONNECTIONS:
            p1 = frame[idx1]
            p2 = frame[idx2]
            
            if self._is_invalid_point(p1) or self._is_invalid_point(p2):
                continue
                
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=pose_color, linewidth=pose_lw)
        
        # Draw pose-to-hand connections - only if corresponding hand is valid
        for idx1, idx2 in POSE_HAND_CONNECTIONS:
            # Skip left wrist-to-hand connection if left hand invalid
            if idx2 == IDX_LEFT_HAND[0] and not left_hand_valid:
                continue
            # Skip right wrist-to-hand connection if right hand invalid
            if idx2 == IDX_RIGHT_HAND[0] and not right_hand_valid:
                continue
                
            p1 = frame[idx1]
            p2 = frame[idx2]
            
            if self._is_invalid_point(p1) or self._is_invalid_point(p2):
                continue
                
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=pose_color, linewidth=pose_lw)
    
    def _draw_hand(self, ax, frame: np.ndarray, hand_indices: np.ndarray):
        """
        Vẽ một bàn tay với các ngón tay màu khác nhau.
        
        Args:
            ax: Matplotlib axes
            frame: Full frame data
            hand_indices: Indices của hand trong frame (21 points)
        """
        # Check if hand data is valid first
        if not self._is_hand_valid(frame, hand_indices):
            return
            
        hand_pts = frame[hand_indices]
            
        # Draw each finger chain
        for chain, color in zip(HAND_FINGER_CHAINS, FINGER_COLORS):
            chain_pts = hand_pts[chain]
            
            # Skip if all NaN
            if self._is_invalid(chain_pts):
                continue
                
            ax.plot(chain_pts[:, 0], chain_pts[:, 1], 
                   color=color, linewidth=2)
    
    def _is_invalid(self, pts: np.ndarray) -> bool:
        """Check if points array is invalid (all zeros or all NaN)"""
        if np.isnan(pts).all():
            return True
        if np.allclose(pts[:, :2], 0):
            return True
        return False
    
    def _is_invalid_point(self, pt: np.ndarray) -> bool:
        """Check if a single point is invalid"""
        if np.isnan(pt).any():
            return True
        if np.allclose(pt[:2], 0):
            return True
        return False
    
    def _is_hand_valid(self, frame: np.ndarray, hand_indices: np.ndarray) -> bool:
        """
        Check if hand data is valid (has real tracking data, not placeholder).
        
        Args:
            frame: Full frame data
            hand_indices: Indices of hand landmarks (21 points)
            
        Returns:
            True if hand data appears to be valid tracking data
        """
        hand_pts = frame[hand_indices]
        
        # Check wrist point first
        wrist = hand_pts[0]
        if self._is_invalid_point(wrist):
            return False
        
        # Check if all points are zeros or NaN
        if self._is_invalid(hand_pts):
            return False
        
        # Check if all points are at the same position (rest pose)
        # This catches [0.5, 0.5, 0.5] placeholder data where all 21 points are identical
        # Real hand data will have different positions for fingertips
        first_point = hand_pts[0, :2]
        all_same = True
        for i in range(1, len(hand_pts)):
            if not np.allclose(hand_pts[i, :2], first_point, atol=1e-6):
                all_same = False
                break
        
        if all_same:
            return False
        
        return True


# Test
if __name__ == "__main__":
    from scripts.sign_dictionary import SignDictionary
    from scripts.motion_synthesizer import MotionSynthesizer
    
    dictionary = SignDictionary()
    synthesizer = MotionSynthesizer(dictionary)
    visualizer = SignVisualizer()
    
    # Test với một từ
    data = dictionary.load_gloss("hello")
    if data is not None:
        visualizer.create_animation(data, "test_hello.gif")
