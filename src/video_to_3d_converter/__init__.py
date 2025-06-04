"""
Main package for the Real-time 2D Video to 3D Point-Cloud Converter.

This package provides the  class to process RGB video frames or image files
into 3D point clouds.
"""

import json  # For error handling

import numpy as np

from .depth_estimator import _load_model as _load_depth_model
from .depth_estimator import estimate_depth, load_rgb_frame
from .point_cloud_generator import combine_xyz_rgb, depth_to_points, save_pcd
from .temporal_smoother import ExponentialMovingAverageSmoother

# --- Error Handling (FR9) ---
ERROR_CODES = {
    "E_MODEL_LOAD": "Failed to load the specified depth estimation model.",
    "E_FILE_NOT_FOUND": "Input image file not found.",
    "E_NO_INPUT": "No input frame provided.",
    "E_INVALID_INTRINSICS": "Camera intrinsics are missing or invalid.",
    "E_OOM": "Out of memory during processing (typically GPU).",
    "E_UNKNOWN": "An unknown error occurred.",
}


class ConverterError(Exception):
    """Custom exception for converter errors."""

    def __init__(self, error_code: str, message: str = "", details: str = ""):
        self.error_code = error_code
        self.message = message if message else ERROR_CODES.get(error_code, "Unknown error.")
        self.details = details
        super().__init__(self.to_json())  # Call super with the JSON string representation

    def to_json(self) -> str:  # Ensure it returns string
        return json.dumps(
            {
                "error_code": self.error_code,
                "message": self.message,
                "details": str(self.details),  # Ensure details are string serializable
            }
        )


# --- Public API ---
class Converter:
    """
    Main class for converting 2D video frames to 3D point clouds.
    Provides methods to process individual frames or image files.
    """

    def __init__(
        self,
        model_variant: str,
        intrinsics: dict,
        use_temporal_smoother: bool = False,
        smoothing_alpha: float = 0.5,
    ):
        """
        Initializes the Converter.

        Args:
            model_variant (str): The depth estimation model variant to use (e.g., "MiDaS_small", "DPT_Large").
            intrinsics (dict): Camera intrinsics (fx, fy, cx, cy).
            use_temporal_smoother (bool): Whether to apply temporal smoothing to depth maps.
            smoothing_alpha (float): Alpha value for EMA smoother if used (0 < alpha <= 1).

        Raises:
            ConverterError: If the model fails to load or intrinsics are invalid.
        """
        if not all(k in intrinsics for k in ["fx", "fy", "cx", "cy"]):
            raise ConverterError("E_INVALID_INTRINSICS", details="fx, fy, cx, cy are required.")

        self.model_variant = model_variant
        self.intrinsics = intrinsics
        self.use_temporal_smoother = use_temporal_smoother
        self.smoother = None

        try:
            # Ensure the global model in depth_estimator is loaded/updated
            _load_depth_model(self.model_variant)
        except Exception as e:
            # Catch specific exceptions from _load_model if possible, or general ones
            raise ConverterError(
                "E_MODEL_LOAD", details=f"Failed during _load_depth_model: {str(e)}"
            ) from e

        if self.use_temporal_smoother:
            try:
                self.smoother = ExponentialMovingAverageSmoother(alpha=smoothing_alpha)
            except ValueError as e:  # Catch specific error from smoother init
                raise ConverterError(
                    "E_INVALID_INTRINSICS", message="Invalid smoothing_alpha value.", details=str(e)
                ) from e

    def process_rgb_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Processes a single RGB frame to generate an XYZRGB point cloud.

        Args:
            rgb_frame (np.ndarray): The input RGB frame as a NumPy array (H, W, C).

        Returns:
            np.ndarray: Nx4 array representing the XYZRGB point cloud.
                        Returns an empty array if no points are generated.

        Raises:
            ConverterError: For processing errors (e.g., OOM).
        """
        if not isinstance(rgb_frame, np.ndarray):  # Basic type check
            raise ConverterError("E_NO_INPUT", message="Input rgb_frame must be a NumPy array.")
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:  # Basic shape check
            raise ConverterError("E_NO_INPUT", message="Input rgb_frame must be HxWxC.")

        try:
            # 1. Estimate Depth
            depth_map = estimate_depth(rgb_frame, model_variant=self.model_variant)

            # 2. (Optional) Temporal Smoothing
            if self.use_temporal_smoother and self.smoother:
                depth_map = self.smoother.smooth(depth_map)

            # 3. Generate Point Cloud
            points_xyz = depth_to_points(depth_map, self.intrinsics)
            if points_xyz.shape[0] == 0:
                return np.empty((0, 4), dtype=np.float32)

            points_xyzrgb = combine_xyz_rgb(points_xyz, rgb_frame, self.intrinsics)
            return points_xyzrgb

        except RuntimeError as e:
            if (
                "out of memory" in str(e).lower() or "cuda" in str(e).lower()
            ):  # Broader check for GPU errors
                raise ConverterError("E_OOM", details=str(e)) from e
            # Other runtime errors from PyTorch/NumPy etc.
            raise ConverterError(
                "E_UNKNOWN", message=f"Runtime error during processing: {e}", details=str(e)
            ) from e
        except ConverterError:  # Re-raise known converter errors
            raise
        except Exception as e:  # Catch any other unexpected errors
            raise ConverterError(
                "E_UNKNOWN", message=f"Unexpected error during processing: {e}", details=str(e)
            ) from e

    def process_image_file(self, image_path: str) -> np.ndarray:
        """
        Loads an image from a file and processes it to generate a point cloud.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            np.ndarray: Nx4 XYZRGB point cloud.

        Raises:
            ConverterError: If file not found or during processing.
        """
        if not image_path or not isinstance(image_path, str):
            raise ConverterError(
                "E_FILE_NOT_FOUND", message="Image path must be a non-empty string."
            )
        try:
            rgb_frame = load_rgb_frame(image_path)
        except FileNotFoundError as e:  # This is already raised by load_rgb_frame
            raise ConverterError(
                "E_FILE_NOT_FOUND", details=f"Image file not found at: {image_path}"
            ) from e
        except ValueError as e:  # Catch other load_rgb_frame errors like empty path
            raise ConverterError(
                "E_FILE_NOT_FOUND", message="Invalid image path provided.", details=str(e)
            ) from e
        # Catch any other unexpected error during image loading
        except Exception as e:
            raise ConverterError("E_UNKNOWN", message="Error loading image file.", details=str(e)) from e

        return self.process_rgb_frame(rgb_frame)

    def reset_smoother(self):
        """Resets the temporal smoother's history if it's being used."""
        if self.smoother:
            self.smoother.reset()
            # print("Temporal smoother has been reset.") # Consider logging
        # else:
            # print("Temporal smoother is not in use.") # Consider logging


__all__ = ["Converter", "ConverterError", "ERROR_CODES", "save_pcd"]  # Expose save_pcd utility
