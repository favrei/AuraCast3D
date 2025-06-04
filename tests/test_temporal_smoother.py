import unittest

import numpy as np

from video_to_3d_converter.temporal_smoother import ExponentialMovingAverageSmoother


class TestTemporalSmoother(unittest.TestCase):
    def test_ema_smoother_initialization(self):
        smoother = ExponentialMovingAverageSmoother(alpha=0.3)
        self.assertEqual(smoother.alpha, 0.3)
        with self.assertRaises(ValueError):
            ExponentialMovingAverageSmoother(alpha=0)
        with self.assertRaises(ValueError):
            ExponentialMovingAverageSmoother(alpha=1.1)

    def test_ema_first_frame(self):
        smoother = ExponentialMovingAverageSmoother(alpha=0.5)
        frame1 = np.array([1.0, 2.0, 3.0])
        smoothed1 = smoother.smooth(frame1.copy())  # Pass copy
        np.testing.assert_array_equal(smoothed1, frame1, "First frame should not be modified.")
        np.testing.assert_array_equal(
            smoother.previous_frame_data, frame1, "First frame data not stored correctly."
        )
        self.assertFalse(
            smoother.is_first_frame, "is_first_frame flag not updated after first frame."
        )

    def test_ema_smoothing_logic(self):
        smoother = ExponentialMovingAverageSmoother(alpha=0.5)
        frame1 = np.array([10.0, 20.0], dtype=np.float32)
        _ = smoother.smooth(frame1.copy())  # First frame, populates previous_frame_data

        frame2 = np.array([30.0, 40.0], dtype=np.float32)
        smoothed2 = smoother.smooth(frame2.copy())
        expected2 = 0.5 * frame2 + 0.5 * frame1
        np.testing.assert_array_almost_equal(
            smoothed2, expected2, decimal=6, err_msg="Second frame EMA calculation incorrect."
        )
        np.testing.assert_array_almost_equal(
            smoother.previous_frame_data,
            expected2,
            decimal=6,
            err_msg="Smoothed data not stored correctly after second frame.",
        )

        frame3 = np.array([50.0, 60.0], dtype=np.float32)
        smoothed3 = smoother.smooth(frame3.copy())
        expected3 = 0.5 * frame3 + 0.5 * expected2
        np.testing.assert_array_almost_equal(
            smoothed3, expected3, decimal=6, err_msg="Third frame EMA calculation incorrect."
        )
        np.testing.assert_array_almost_equal(
            smoother.previous_frame_data,
            expected3,
            decimal=6,
            err_msg="Smoothed data not stored correctly after third frame.",
        )

    def test_ema_alpha_one(self):  # Alpha = 1 means current frame only
        smoother = ExponentialMovingAverageSmoother(alpha=1.0)
        frame1 = np.array([10.0, 20.0])
        _ = smoother.smooth(frame1.copy())

        frame2 = np.array([30.0, 40.0])
        smoothed2 = smoother.smooth(frame2.copy())
        np.testing.assert_array_equal(
            smoothed2, frame2, "With alpha=1, smoothed output should be current frame."
        )
        np.testing.assert_array_equal(
            smoother.previous_frame_data, frame2, "With alpha=1, history should be current frame."
        )

    def test_ema_reset(self):
        smoother = ExponentialMovingAverageSmoother(alpha=0.5)
        frame1 = np.array([1.0, 2.0])
        _ = smoother.smooth(frame1.copy())
        self.assertIsNotNone(smoother.previous_frame_data)
        self.assertFalse(smoother.is_first_frame)

        smoother.reset()
        self.assertIsNone(smoother.previous_frame_data)
        self.assertTrue(smoother.is_first_frame)

        frame2 = np.array([3.0, 4.0])
        smoothed2_after_reset = smoother.smooth(frame2.copy())
        np.testing.assert_array_equal(
            smoothed2_after_reset, frame2, "After reset, should behave like first frame."
        )
        np.testing.assert_array_equal(smoother.previous_frame_data, frame2)
        self.assertFalse(smoother.is_first_frame)

    def test_ema_shape_mismatch(self):
        smoother = ExponentialMovingAverageSmoother(alpha=0.5)
        frame1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        _ = smoother.smooth(frame1.copy())

        frame2_diff_shape = np.array([1, 2, 3], dtype=np.float32)
        smoothed2 = smoother.smooth(frame2_diff_shape.copy())  # Expect warning (not tested)

        # Smoother should reset and return the current frame as is
        np.testing.assert_array_equal(
            smoothed2, frame2_diff_shape, "Shape mismatch should return current frame."
        )
        np.testing.assert_array_equal(
            smoother.previous_frame_data,
            frame2_diff_shape,
            "History should be updated to current frame on shape mismatch.",
        )
        self.assertFalse(
            smoother.is_first_frame,
            "is_first_frame should be false after processing a frame, even with shape mismatch.",
        )


if __name__ == "__main__":
    unittest.main()
