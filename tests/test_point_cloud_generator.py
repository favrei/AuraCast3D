import os
import struct
import unittest

import numpy as np

from video_to_3d_converter.point_cloud_generator import combine_xyz_rgb, depth_to_points, save_pcd


def load_pcd_for_test(filepath: str) -> (dict, np.ndarray):
    header_dict = {}
    points_list = []  # Use a list to append rows
    num_points_from_header = 0
    expected_num_fields = 0  # Will be set by FIELDS line

    with open(filepath, "rb") as f:
        while True:
            try:
                line_bytes = f.readline()
                if not line_bytes:  # EOF
                    break
                line = line_bytes.decode("ascii").strip()
            except UnicodeDecodeError:
                # This might happen if we try to decode binary data part as ASCII
                # However, PCD header must be ASCII.
                print(
                    f"Warning: UnicodeDecodeError reading line, possible header corruption or premature end of header: {line_bytes[:100]}"
                )
                break  # Stop reading header on decode error

            if not line:
                continue  # Skip empty lines

            parts = line.split()
            if not parts:
                continue  # Skip lines that become empty after split

            header_dict[parts[0]] = parts[1:]

            if parts[0] == "FIELDS":
                expected_num_fields = len(parts) - 1
            elif parts[0] == "POINTS":
                num_points_from_header = int(parts[1])
            elif parts[0] == "DATA" and parts[1] == "binary":
                break  # End of header
            elif parts[0] == "DATA" and parts[1] != "binary":
                raise NotImplementedError(
                    f"Only 'DATA binary' is supported for test loader, found {parts[1]}"
                )

        if num_points_from_header > 0 and expected_num_fields > 0:
            # Assuming FIELDS x y z rgb, SIZE 4 4 4 4, TYPE F F F F (i.e., 4 float32 fields)
            # Each point is 4*4 = 16 bytes
            bytes_per_point = expected_num_fields * 4  # Assuming all are float32

            binary_data = f.read(num_points_from_header * bytes_per_point)
            if len(binary_data) < num_points_from_header * bytes_per_point:
                print(
                    f"Warning: Expected {num_points_from_header * bytes_per_point} bytes of data, got {len(binary_data)}"
                )
                # Adjust num_points_from_header if file is truncated
                num_points_from_header = len(binary_data) // bytes_per_point

            # Reconstruct points: iterate through binary data chunk by chunk
            for i in range(num_points_from_header):
                point_data = []
                for j in range(expected_num_fields):  # x, y, z, rgb
                    # struct.unpack always returns a tuple
                    value = struct.unpack(
                        "<f",
                        binary_data[
                            i * bytes_per_point + j * 4 : i * bytes_per_point + (j + 1) * 4
                        ],
                    )[0]
                    point_data.append(value)
                points_list.append(point_data)

    return header_dict, np.array(points_list, dtype=np.float32)


class TestPointCloudGenerator(unittest.TestCase):
    DUMMY_PCD_PATH = "dummy_test_cloud.pcd"
    # Define general intrinsics for a VGA-like setup, but test-specific ones will be used for small images
    FULL_IMAGE_INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 319.5, "cy": 239.5}
    # Test image dimensions
    TEST_IMG_HEIGHT = 48
    TEST_IMG_WIDTH = 64
    DEPTH_MAP_SHAPE = (TEST_IMG_HEIGHT, TEST_IMG_WIDTH)
    RGB_FRAME_SHAPE = (TEST_IMG_HEIGHT, TEST_IMG_WIDTH, 3)
    # Intrinsics tailored for the small test images
    TEST_INTRINSICS = {
        "fx": 50.0,
        "fy": 50.0,
        "cx": (TEST_IMG_WIDTH - 1) / 2.0,
        "cy": (TEST_IMG_HEIGHT - 1) / 2.0,
    }

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.DUMMY_PCD_PATH):
            os.remove(cls.DUMMY_PCD_PATH)

    def test_depth_to_points(self):
        depth_map = np.ones(self.DEPTH_MAP_SHAPE, dtype=np.float32) * 1.5
        depth_map[0, 0] = 0  # One invalid depth point
        points_xyz = depth_to_points(depth_map, self.TEST_INTRINSICS)  # Use TEST_INTRINSICS

        expected_num_points = self.DEPTH_MAP_SHAPE[0] * self.DEPTH_MAP_SHAPE[1] - 1
        self.assertEqual(points_xyz.shape[0], expected_num_points)
        self.assertEqual(points_xyz.shape[1], 3)  # XYZ

        # Test a single point projection: u=cx, v=cy, depth=Z_val => x=0, y=0, z=Z_val
        # For TEST_INTRINSICS, cx and cy are center of the small image
        test_u = int(round(self.TEST_INTRINSICS["cx"]))
        test_v = int(round(self.TEST_INTRINSICS["cy"]))

        depth_val = 2.0
        depth_map_single_pt = np.zeros(self.DEPTH_MAP_SHAPE, dtype=np.float32)
        depth_map_single_pt[test_v, test_u] = depth_val  # Set depth at (cy,cx)

        points_single = depth_to_points(
            depth_map_single_pt, self.TEST_INTRINSICS
        )  # Use TEST_INTRINSICS
        self.assertEqual(points_single.shape[0], 1)  # Should only find one valid point

        # Expected values based on u=32, cx=31.5, z=2.0, fx=50.0  => x = (32-31.5)*2.0/50.0 = 0.02
        # Expected values based on v=24, cy=23.5, z=2.0, fy=50.0  => y = (24-23.5)*2.0/50.0 = 0.02
        expected_x = (test_u - self.TEST_INTRINSICS["cx"]) * depth_val / self.TEST_INTRINSICS["fx"]
        expected_y = (test_v - self.TEST_INTRINSICS["cy"]) * depth_val / self.TEST_INTRINSICS["fy"]
        self.assertAlmostEqual(points_single[0, 0], expected_x, places=5)
        self.assertAlmostEqual(points_single[0, 1], expected_y, places=5)
        self.assertAlmostEqual(points_single[0, 2], depth_val, places=5)

    def test_combine_xyz_rgb(self):
        num_pts = 3
        img_h, img_w = self.RGB_FRAME_SHAPE[0], self.RGB_FRAME_SHAPE[1]

        # Use TEST_INTRINSICS for point generation to match test image size
        fx, fy = self.TEST_INTRINSICS["fx"], self.TEST_INTRINSICS["fy"]
        cx, cy = self.TEST_INTRINSICS["cx"], self.TEST_INTRINSICS["cy"]

        points_xyz = np.array(
            [
                [0.0, 0.0, 1.0],  # Projects to (cx, cy) -> center of TEST_IMAGE
                [(0 - cx) * 1.0 / fx, (0 - cy) * 1.0 / fy, 1.0],  # projects to (0,0) of TEST_IMAGE
                [
                    ((img_w - 1) - cx) * 2.0 / fx,
                    ((img_h - 1) - cy) * 2.0 / fy,
                    2.0,
                ],  # projects to (img_w-1, img_h-1) of TEST_IMAGE
            ],
            dtype=np.float32,
        )

        rgb_frame = np.zeros(self.RGB_FRAME_SHAPE, dtype=np.uint8)
        # Define colors at the specific pixels these points project to
        rgb_frame[int(round(cy)), int(round(cx))] = [255, 0, 0]  # Red at (cx,cy)
        rgb_frame[0, 0] = [0, 255, 0]  # Green at (0,0)
        rgb_frame[img_h - 1, img_w - 1] = [0, 0, 255]  # Blue at (img_w-1, img_h-1)

        points_xyzrgb = combine_xyz_rgb(
            points_xyz, rgb_frame, self.TEST_INTRINSICS
        )  # Use TEST_INTRINSICS
        self.assertEqual(points_xyzrgb.shape, (num_pts, 4))  # XYZ + packed RGB

        # Test colors by unpacking them
        r0, g0, b0 = self._unpack_rgb_float(points_xyzrgb[0, 3])
        self.assertEqual((r0, g0, b0), (255, 0, 0), "Point 0 color (center) incorrect")

        r1, g1, b1 = self._unpack_rgb_float(points_xyzrgb[1, 3])
        self.assertEqual((r1, g1, b1), (0, 255, 0), "Point 1 color (0,0) incorrect")

        r2, g2, b2 = self._unpack_rgb_float(points_xyzrgb[2, 3])
        self.assertEqual((r2, g2, b2), (0, 0, 255), "Point 2 color (max_extents) incorrect")

    def _unpack_rgb_float(self, packed_float):
        """Helper to unpack RGB from float32 to three uint8 values."""
        packed_int = struct.unpack("I", struct.pack("f", packed_float))[0]
        r = (packed_int >> 16) & 0xFF
        g = (packed_int >> 8) & 0xFF
        b = packed_int & 0xFF
        return r, g, b

    def test_save_and_load_pcd_binary(self):
        num_pts = 50
        xyz = (np.random.rand(num_pts, 3) - 0.5).astype(np.float32)
        xyz[:, 2] = np.abs(xyz[:, 2]) + 0.1  # Ensure positive depth

        r = np.random.randint(0, 256, num_pts, dtype=np.uint32)
        g = np.random.randint(0, 256, num_pts, dtype=np.uint32)
        b = np.random.randint(0, 256, num_pts, dtype=np.uint32)
        packed_int = (r << 16) | (g << 8) | b

        rgb_float32 = packed_int.astype(np.uint32).view(np.float32)  # Efficient conversion
        point_cloud_original = np.hstack((xyz, rgb_float32.reshape(-1, 1)))

        save_pcd(point_cloud_original, self.DUMMY_PCD_PATH)
        self.assertTrue(os.path.exists(self.DUMMY_PCD_PATH))

        header, points_loaded = load_pcd_for_test(self.DUMMY_PCD_PATH)

        self.assertEqual(int(header["POINTS"][0]), num_pts)
        self.assertEqual(header["FIELDS"], ["x", "y", "z", "rgb"])
        self.assertEqual(header["DATA"][0], "binary")
        self.assertEqual(header["SIZE"], ["4", "4", "4", "4"])
        self.assertEqual(header["TYPE"], ["F", "F", "F", "F"])

        self.assertTrue(
            np.allclose(points_loaded, point_cloud_original, atol=1e-6),
            f"Loaded PCD data does not match original.\nOriginal:\n{point_cloud_original[:3]}\nLoaded:\n{points_loaded[:3]}",
        )

    def test_empty_point_cloud_save_and_load(self):
        empty_cloud = np.empty((0, 4), dtype=np.float32)
        save_pcd(empty_cloud, self.DUMMY_PCD_PATH)
        self.assertTrue(os.path.exists(self.DUMMY_PCD_PATH))

        header, points_loaded = load_pcd_for_test(self.DUMMY_PCD_PATH)
        self.assertEqual(int(header["POINTS"][0]), 0)
        self.assertEqual(points_loaded.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
