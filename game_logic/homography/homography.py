import numpy as np
import cv2 

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

        # tactical → camera
        self.m_inv, _ = cv2.findHomography(target, source)
        if self.m_inv is None:
            raise ValueError("Inverse homography matrix could not be calculated.")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map points from target → source space"""
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m_inv)
        return points.reshape(-1, 2).astype(np.float32)



