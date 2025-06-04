import os
import unittest

import numpy as np
import torch  # For checking cuda availability
from PIL import Image

from video_to_3d_converter.depth_estimator import _load_model, estimate_depth, load_rgb_frame


class TestDepthEstimator(unittest.TestCase):
    DUMMY_IMAGE_PATH = "dummy_test_image.png"
    # Use MiDaS_small as the default for most tests to ensure speed and lower resource usage.
    # DPT_Large will be tested specifically.
    DEFAULT_MODEL_VARIANT = "MiDaS_small"
    ALTERNATE_MODEL_VARIANT = "DPT_Large"  # A more resource-intensive model for FR3

    @classmethod
    def setUpClass(cls):
        # Create a small dummy image for testing
        img = Image.new("RGB", (160, 120), color="red")  # Standard small size for tests
        img.save(cls.DUMMY_IMAGE_PATH)

        # Attempt to pre-load the default small model to speed up tests
        # and to catch loading issues early.
        print(f"Attempting to pre-load MiDaS model: {cls.DEFAULT_MODEL_VARIANT} for tests...")
        try:
            _load_model(cls.DEFAULT_MODEL_VARIANT)
            print(f"Successfully pre-loaded {cls.DEFAULT_MODEL_VARIANT}.")
        except Exception as e:
            print(
                f"Warning: Could not pre-load MiDaS model ({cls.DEFAULT_MODEL_VARIANT}): {e}. "
                "Depth estimation tests involving this model may be skipped or fail if loading also fails there."
            )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.DUMMY_IMAGE_PATH):
            os.remove(cls.DUMMY_IMAGE_PATH)

    def test_load_rgb_frame(self):
        frame = load_rgb_frame(self.DUMMY_IMAGE_PATH)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (120, 160, 3))  # H, W, C

    def test_load_non_existent_frame(self):
        with self.assertRaises(FileNotFoundError):
            load_rgb_frame("non_existent_image.png")

    def _run_depth_estimation_test(self, model_variant_to_test):
        frame = load_rgb_frame(self.DUMMY_IMAGE_PATH)
        try:
            depth_map = estimate_depth(frame, model_variant=model_variant_to_test)
            self.assertIsInstance(depth_map, np.ndarray)
            # Depth map should be 2D (H, W)
            self.assertEqual(depth_map.shape, (120, 160))
            self.assertTrue(
                depth_map.dtype == np.float32 or depth_map.dtype == np.float16
            )  # Common depth map types
        except RuntimeError as e:  # Catch RuntimeError from model loading issues
            # Skip test if model loading fails (e.g. out of memory, network issue)
            self.skipTest(
                f"Skipping depth estimation for {model_variant_to_test} due to runtime error: {e}"
            )
        except Exception as e:  # Catch any other unexpected errors
            self.fail(
                f"Depth estimation for {model_variant_to_test} failed with unexpected error: {e}"
            )

    def test_estimate_depth_default_model(self):
        """Test with the default (small) model variant."""
        self._run_depth_estimation_test(self.DEFAULT_MODEL_VARIANT)

    def test_estimate_depth_alternate_model(self):
        """Test with an alternate (larger) model variant as per FR3."""
        # Only run if CUDA is available, as DPT_Large is heavy
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.ALTERNATE_MODEL_VARIANT} test as CUDA is not available.")
        self._run_depth_estimation_test(self.ALTERNATE_MODEL_VARIANT)

    def test_model_switching_logic(self):
        """Test if the model correctly switches if a different variant is requested."""
        frame = load_rgb_frame(self.DUMMY_IMAGE_PATH)

        # First, run with the default model
        try:
            print(f"Testing with {self.DEFAULT_MODEL_VARIANT} first...")
            depth_map_small = estimate_depth(frame, model_variant=self.DEFAULT_MODEL_VARIANT)
            self.assertEqual(depth_map_small.shape, (120, 160))
        except RuntimeError as e:
            self.skipTest(
                f"Skipping model switching test; initial load of {self.DEFAULT_MODEL_VARIANT} failed: {e}"
            )
            return  # Cannot proceed with this test

        # Then, run with the alternate model (if CUDA available)
        if not torch.cuda.is_available():
            print("Skipping DPT_Large part of model switching test as CUDA is not available.")
            return

        try:
            print(f"Testing with {self.ALTERNATE_MODEL_VARIANT} next...")
            depth_map_large = estimate_depth(frame, model_variant=self.ALTERNATE_MODEL_VARIANT)
            self.assertEqual(depth_map_large.shape, (120, 160))

            # A basic check: the outputs should likely be different if models are different
            # This is not a perfect check but can catch if the model didn't actually switch.
            if np.array_equal(depth_map_small, depth_map_large):
                print(
                    f"Warning: Depth maps from {self.DEFAULT_MODEL_VARIANT} and {self.ALTERNATE_MODEL_VARIANT} are identical. "
                    "This might indicate an issue with model switching or the test image is too simple."
                )
        except RuntimeError as e:
            self.skipTest(
                f"Skipping model switching test; load of {self.ALTERNATE_MODEL_VARIANT} failed: {e}"
            )


if __name__ == "__main__":
    unittest.main()
