import torch
import math

class SpatialAugmentation:
    """
    Applies spatial augmentations to landmarks: Rotation, Scaling, and Translation.
    Expects input shape: (T, N_LANDMARKS, 3) or (T, N_LANDMARKS, 2)
    """
    def __init__(self, rotate_range=20, scale_range=0.25, shift_range=0.15, p=0.7):
        self.rotate_range = rotate_range # Degrees
        self.scale_range = scale_range   # Fraction (e.g., 0.2 means 0.8 to 1.2)
        self.shift_range = shift_range   # Fraction of the coordinate space (assuming normalized ~0-1)
        self.p = p

    def __call__(self, x):
        """
        x: Tensor of shape (T, N, D) where D is 2 or 3.
        """
        if torch.rand(1) > self.p:
            return x
        
        # Clone to avoid modifying original data in place if shared
        x = x.clone()
        
        # 1. Rotation
        if self.rotate_range > 0:
            angle = (torch.rand(1) * 2 - 1) * self.rotate_range # [-range, range]
            rad = angle * (math.pi / 180.0)
            cos_a = torch.cos(rad)
            sin_a = torch.sin(rad)
            
            # Rotation matrix for 2D
            rot_mat = torch.tensor([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ], device=x.device, dtype=x.dtype)
            
            # Apply rotation to X, Y
            # Center of rotation? Ideally (0.5, 0.5) if normalized, or mean of points.
            # Let's use mean of points per frame or global mean.
            # Using 0.5, 0.5 is safer if we assume rough normalization, but let's use the frame centroid.
            # However, calculating centroid per frame might be jittery.
            # Let's assume the data is somewhat centered or use the mean of the first frame.
            
            # Simple approach: Rotate around (0,0) if data is centered, or just apply matrix.
            # Since we don't know exact coordinates range yet (likely raw pixels 0-1 or 0-width),
            # rotation around (0,0) might shift everything out of view.
            # Better to center, rotate, de-center.
            
            # Calculate centroid of the first valid frame
            # Mask out NaNs (0.0 in our case if preprocessed, but here we might be before preprocessing)
            # The augmentation is applied in Dataset __getitem__, which is BEFORE PreprocessLayer.
            # So data is raw coordinates (pixels).
            # We should calculate centroid from valid points.
            
            # Flatten to find global center of the sequence to keep temporal consistency
            valid_mask = ~torch.isnan(x)
            if valid_mask.any():
                # Compute mean of X and Y
                center_x = x[..., 0][valid_mask[..., 0]].mean()
                center_y = x[..., 1][valid_mask[..., 1]].mean()
                center = torch.tensor([center_x, center_y], device=x.device, dtype=x.dtype)
            else:
                center = torch.zeros(2, device=x.device, dtype=x.dtype)

            # Apply rotation to X, Y
            xy = x[..., :2] - center
            xy = torch.matmul(xy, rot_mat)
            x[..., :2] = xy + center

        # 2. Scaling
        if self.scale_range > 0:
            scale = 1.0 + (torch.rand(1) * 2 - 1) * self.scale_range # [1-range, 1+range]
            
            # Scale around center
            # We can reuse the center from rotation or recompute
            # For simplicity and speed, let's just multiply. 
            # But multiplying raw coordinates shifts them.
            # We need to center, scale, de-center.
            
            valid_mask = ~torch.isnan(x)
            if valid_mask.any():
                center_x = x[..., 0][valid_mask[..., 0]].mean()
                center_y = x[..., 1][valid_mask[..., 1]].mean()
                center = torch.tensor([center_x, center_y], device=x.device, dtype=x.dtype)
            else:
                center = torch.zeros(2, device=x.device, dtype=x.dtype)
                
            x[..., :2] = (x[..., :2] - center) * scale + center

        # 3. Translation (Shift)
        if self.shift_range > 0:
            # Shift is relative to the spread of data or absolute?
            # Raw coordinates are in pixels (e.g. 1920x1080). 
            # 0.1 shift range means 10% of... what?
            # Let's assume a reasonable shift in pixels, e.g., +/- 50 pixels?
            # Or estimate spread.
            
            valid_mask = ~torch.isnan(x)
            if valid_mask.any():
                max_x = x[..., 0][valid_mask[..., 0]].max()
                min_x = x[..., 0][valid_mask[..., 0]].min()
                width = max_x - min_x
                
                max_y = x[..., 1][valid_mask[..., 1]].max()
                min_y = x[..., 1][valid_mask[..., 1]].min()
                height = max_y - min_y
            else:
                width = 1.0
                height = 1.0
            
            shift_x = (torch.rand(1) * 2 - 1) * self.shift_range * width
            shift_y = (torch.rand(1) * 2 - 1) * self.shift_range * height
            
            x[..., 0] += shift_x
            x[..., 1] += shift_y

        return x
