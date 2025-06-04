"""Module for temporal smoothing of frame data (e.g., depth maps)."""

import numpy as np


class ExponentialMovingAverageSmoother:
    def __init__(self, alpha=0.5):
        """
        Initializes the EMA smoother.
        Args:
            alpha (float): Smoothing factor (0 < alpha <= 1).
                           Lower alpha means more smoothing (more history).
                           alpha=1 means no smoothing (current frame only).
        """
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 (exclusive) and 1 (inclusive).")
        self.alpha = alpha
        self.previous_frame_data = None
        self.is_first_frame = True

    def smooth(self, current_frame_data: np.ndarray) -> np.ndarray:
        """
        Applies EMA smoothing to the current frame data.
        Can be applied to depth maps or point coordinates.
        Assumes current_frame_data is a NumPy array.
        """
        if self.is_first_frame or self.previous_frame_data is None:
            self.previous_frame_data = current_frame_data.copy()
            self.is_first_frame = False
            return current_frame_data.copy()

        if self.previous_frame_data.shape != current_frame_data.shape:
            # Shape mismatch (e.g. different number of points, different resolution)
            # Reset smoother.
            # print( # Consider logging
            #     f"Warning: Frame data shape mismatch (prev: {self.previous_frame_data.shape}, curr: {current_frame_data.shape}). Resetting EMA smoother."
            # )
            self.previous_frame_data = current_frame_data.copy()
            return current_frame_data.copy()

        smoothed_data = (
            self.alpha * current_frame_data + (1 - self.alpha) * self.previous_frame_data
        )
        self.previous_frame_data = smoothed_data.copy()
        return smoothed_data

    def reset(self):
        """Resets the smoother's history."""
        self.previous_frame_data = None
        self.is_first_frame = True
