"""Module for generating and saving XYZRGB point clouds from depth maps."""


import numpy as np


def depth_to_points(depth_map: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    Converts a depth map to a 3D point cloud (XYZ coordinates).
    FR4: Accept calibrated camera intrinsics (fx, fy, cx, cy).
    FR5: Generate XYZ coordinates via pinhole un-projection.

    Args:
        depth_map (np.ndarray): 2D array representing depth at each pixel.
        intrinsics (dict): Dictionary with keys 'fx', 'fy', 'cx', 'cy'.

    Returns:
        np.ndarray: Nx3 array of XYZ coordinates. Returns empty array if no valid points.
    """
    if not all(k in intrinsics for k in ["fx", "fy", "cx", "cy"]):
        raise ValueError("Intrinsics must contain 'fx', 'fy', 'cx', 'cy'.")

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    height, width = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

    z_values = depth_map.flatten()
    valid_mask = z_values > 0  # Assuming 0 or negative are invalid depth

    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    u_flat = u_coords.flatten()[valid_mask]
    v_flat = v_coords.flatten()[valid_mask]
    z_flat_valid = z_values[valid_mask]

    x_values = (u_flat - cx) * z_flat_valid / fx
    y_values = (v_flat - cy) * z_flat_valid / fy

    points_xyz = np.vstack((x_values, y_values, z_flat_valid)).T.astype(np.float32)
    return points_xyz


def combine_xyz_rgb(points_xyz: np.ndarray, rgb_frame: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    Combines XYZ point cloud with RGB color data.
    FR6: Attach RGB colour to each point.
    The depth_map_shape argument was removed as rgb_frame.shape can be used and
    original u,v are implicitly present by the order of points from depth_to_points if not filtered,
    or can be re-projected. This version re-projects for robustness.

    Args:
        points_xyz (np.ndarray): Nx3 array of XYZ coordinates.
        rgb_frame (np.ndarray): HxWxC array of RGB colors.
        intrinsics (dict): Camera intrinsics ('fx', 'fy', 'cx', 'cy').

    Returns:
        np.ndarray: Nx4 array where each row is (x, y, z, packed_rgb_float32).
    """
    if points_xyz.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    # Project XYZ back to image plane to get (u, v) coordinates for color sampling
    # Avoid division by zero for Z. If Z is very close to zero, u,v can be huge.
    # Points with Z=0 should have been filtered by depth_to_points, but as a safeguard:
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)

    u = (x * fx / z_safe) + cx
    v = (y * fy / z_safe) + cy

    # Clip coordinates to be within image bounds
    u_indices = np.clip(np.round(u).astype(int), 0, rgb_frame.shape[1] - 1)
    v_indices = np.clip(np.round(v).astype(int), 0, rgb_frame.shape[0] - 1)

    colors = rgb_frame[v_indices, u_indices]  # HxWxC, C should be 3 (RGB)

    r = colors[:, 0].astype(np.uint32)
    g = colors[:, 1].astype(np.uint32)
    b = colors[:, 2].astype(np.uint32)

    # Pack RGB into a single uint32, then view as float32
    packed_rgb_int = (r << 16) | (g << 8) | b

    # Use a view to convert uint32 to float32 without copying data, ensuring bit patterns are preserved
    packed_rgb_float32 = packed_rgb_int.astype(np.uint32).view(np.float32)

    points_xyzrgb = np.hstack((points_xyz.astype(np.float32), packed_rgb_float32.reshape(-1, 1)))
    return points_xyzrgb


def save_pcd(point_cloud_xyzrgb: np.ndarray, filepath: str):
    """
    Saves an XYZRGB point cloud to a binary PCD file (v0.7).
    FR8: Persist point cloud in binary PCD (little-endian, float32, XYZRGB).
    """
    num_points = point_cloud_xyzrgb.shape[0]

    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z rgb",  # Standard PCL field name for packed RGB
        "SIZE 4 4 4 4",  # Size of x, y, z, rgb in bytes
        "TYPE F F F F",  # Type: F=float
        "COUNT 1 1 1 1",  # Number of elements for each field
        f"WIDTH {num_points}",  # Number of points if organized as a single row
        "HEIGHT 1",  # Indicates an unorganized point cloud
        "VIEWPOINT 0 0 0 1 0 0 0",  # Default viewpoint
        f"POINTS {num_points}",  # Total number of points
        "DATA binary",
    ]
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        if num_points > 0:
            # Ensure the array is C-contiguous and of the correct type before tobytes()
            # XYZ should be float32, RGB is already packed into float32
            f.write(point_cloud_xyzrgb.astype(np.float32).tobytes())
